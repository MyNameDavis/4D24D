import os
import glob
import cv2
import numpy as np
import collections
import gc
import multiprocessing
from SIFT import extract_sift_features, compute_matches, stitch_mosaic

def cluster_film_scenes(features_dict, min_inliers, window_size=5):
    filenames = sorted(list(features_dict.keys()))
    N = len(filenames)
    
    print(f"\n--- Phase 1: Local Sliding Window Matching ---")
    W = window_size
    local_pairs = set()
    for i in range(N):
        for j in range(i + 1, min(i + 1 + W, N)):
            local_pairs.add((filenames[i], filenames[j]))
            
    print(f"Evaluating {len(local_pairs)} local sequential pairs...")
    local_connections = compute_matches(features_dict, local_pairs, min_inliers)
    
    graph = collections.defaultdict(list)
    for conn in local_connections:
        graph[conn['img1']].append(conn['img2'])
        graph[conn['img2']].append(conn['img1'])
        
    components = []
    visited = set()
    for img in filenames:
        if img not in visited:
            comp = set()
            q = collections.deque([img])
            while q:
                curr = q.popleft()
                if curr not in visited:
                    visited.add(curr)
                    comp.add(curr)
                    for neighbor in graph[curr]:
                        if neighbor not in visited:
                            q.append(neighbor)
            components.append(list(comp))
            
    print(f" -> Found {len(components)} initial components before outlier recovery.")
    
    print(f"\n--- Phase 2: Outlier Recovery (Component Merging) ---")
    cross_pairs = set()
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            comp_A = components[i]
            comp_B = components[j]
            
            for a in comp_A:
                for b in comp_B:
                    pair = tuple(sorted([a, b]))
                    if pair not in local_pairs:
                        cross_pairs.add(pair)
                        
    print(f"Evaluating {len(cross_pairs)} cross-component pairs to find outliers...")
    outlier_connections = compute_matches(features_dict, cross_pairs, min_inliers)
    
    all_connections = local_connections + outlier_connections
    
    final_graph = collections.defaultdict(list)
    for conn in all_connections:
        final_graph[conn['img1']].append(conn['img2'])
        final_graph[conn['img2']].append(conn['img1'])
        
    final_components = []
    visited = set()
    for img in filenames:
        if img not in visited:
            comp = set()
            q = collections.deque([img])
            while q:
                curr = q.popleft()
                if curr not in visited:
                    visited.add(curr)
                    comp.add(curr)
                    for neighbor in final_graph[curr]:
                        if neighbor not in visited:
                            q.append(neighbor)
            final_components.append(list(comp))

    print(f" -> Matching complete! Consolidated into {len(final_components)} distinct mosaics.")
    return final_components, all_connections

def is_black_frame(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        return False
    small = cv2.resize(img, (10, 10))
    return np.mean(small) < 10

def process_single_batch(b_idx, num_batches, batch_paths, flat_field_img, DOWNSCALE_FACTOR, MAX_FEATURES, MIN_INLIERS, WINDOW_SIZE, OUTPUT_DIR, shared_tracker):
    print(f"\n{'='*60}\nProcessing Film Scan Batch {b_idx + 1}/{num_batches} ({len(batch_paths)} images)\n{'='*60}")
    
    features_dict = extract_sift_features(batch_paths, flat_field_img, downscale_factor=DOWNSCALE_FACTOR, max_features=MAX_FEATURES)
    components, connections = cluster_film_scenes(features_dict, min_inliers=MIN_INLIERS, window_size=WINDOW_SIZE)
    
    valid_components = [c for c in components if len(c) >= 2]
    print(f"\nFound {len(valid_components)} distinct mosaic scenes in this batch!")
    
    local_tracker = 0
    for idx, comp in enumerate(components):
        if len(comp) < 2:
            print(f"Skipping component {idx} (Only 1 image: {os.path.basename(comp[0])})")
            continue

        current_id = shared_tracker.value + local_tracker
        print(f"\n{'='*50}\nProcessing mosaic {current_id} ({len(comp)} tiles)\n{'='*50}")
        comp_set = set(comp)
        comp_connections = [c for c in connections if c['img1'] in comp_set and c['img2'] in comp_set]
        
        stitch_mosaic(comp, features_dict, comp_connections, OUTPUT_DIR, current_id, flat_field_img)
        local_tracker += 1
        
    with shared_tracker.get_lock():
        shared_tracker.value += local_tracker

def main():
    import argparse
    parser = argparse.ArgumentParser(description="4D24D Film Scanner Mosaic Batch Processor")
    parser.add_argument("-i", "--input", type=str, default="./image_seg_dataset_3d", help="Input directory")
    parser.add_argument("-o", "--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--downscale", type=float, default=1.0, help="Downscale factor for SIFT (default 1.0)")
    parser.add_argument("--max-features", type=int, default=2000, help="Max SIFT features per image")
    parser.add_argument("--min-inliers", type=int, default=15, help="Minimum inliers for a valid connection")
    parser.add_argument("--window-size", type=int, default=5, help="Sliding window size for sequential matching")
    args = parser.parse_args()

    INPUT_DIR = args.input
    OUTPUT_DIR = args.output
    DOWNSCALE_FACTOR = args.downscale
    MIN_INLIERS = args.min_inliers
    WINDOW_SIZE = args.window_size
    MAX_FEATURES = args.max_features if args.max_features > 0 else None

    ff_path = os.path.join(INPUT_DIR, "flat_field_calibration.jpg")
    if not os.path.exists(ff_path):
        ff_path = os.path.join(INPUT_DIR, "flat_field_reference.jpg")
        
    flat_field_img = None
    if os.path.exists(ff_path):
        print("Flat Field Reference detected! Applying illumination normalization...")
        flat_field_img = cv2.imread(ff_path)

    search_path = os.path.join(INPUT_DIR, "*.jpg")
    image_paths = sorted([p for p in glob.glob(search_path) if "flat_field" not in p])

    if len(image_paths) < 2:
        print(f"Need at least 2 images to match. Found {len(image_paths)} in {INPUT_DIR}.")
    else:
        print("Scanning dataset for black frame separators...")
        batches = []
        current_batch = []
        for path in image_paths:
            if is_black_frame(path):
                if len(current_batch) >= 2:
                    batches.append(current_batch)
                current_batch = []
                print(f" -> Found black frame separator: {os.path.basename(path)}")
            else:
                current_batch.append(path)
                
        if len(current_batch) >= 2:
            batches.append(current_batch)
            
        print(f"Dataset split into {len(batches)} physical film scan batches.")
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        eq_dir = os.path.join(OUTPUT_DIR, "debug")
        if os.path.exists(eq_dir):
            for f in glob.glob(os.path.join(eq_dir, "equations_*.json")):
                os.remove(f)
            
        shared_tracker = multiprocessing.Value('i', 0)
        
        for b_idx, batch_paths in enumerate(batches):
            proc = multiprocessing.Process(
                target=process_single_batch,
                args=(b_idx, len(batches), batch_paths, flat_field_img, DOWNSCALE_FACTOR, MAX_FEATURES, MIN_INLIERS, WINDOW_SIZE, OUTPUT_DIR, shared_tracker)
            )
            proc.start()
            proc.join()

        print("\n" + "="*80)
        print("Calculating Global Consensus Aspect Ratio...")
        import json
        all_equations = []
        for f in sorted(glob.glob(os.path.join(eq_dir, "equations_*.json"))):
            with open(f, "r") as eq_file:
                all_equations.extend(json.load(eq_file))
                
        if all_equations:
            unique_segments = list(set([eq['segment_key'] for eq in all_equations]))
            seg_idx_map = {seg: i for i, seg in enumerate(unique_segments)}
            num_segs = len(unique_segments)
            
            A_rows = []
            b_rows = []
            for eq in all_equations:
                row = np.zeros(1 + num_segs)
                row[0] = eq['coeff_R2']
                row[1 + seg_idx_map[eq['segment_key']]] = eq['coeff_beta']
                A_rows.append(row)
                b_rows.append(eq['rhs'])
                
            A = np.array(A_rows)
            b = np.array(b_rows)
            
            print(f" -> Solving global linear system with {A.shape[0]} equations across {num_segs} segments...")
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            R_sq = x[0]
            R_calc = np.sqrt(abs(R_sq))
            
            # Snap to closest standard film format
            try:
                with open("film_aspect_ratios.json", "r") as f:
                    aspect_ratios = json.load(f)
                    
                closest_name = min(aspect_ratios, key=lambda k: abs(aspect_ratios[k] - R_calc))
                R = aspect_ratios[closest_name]
                print(f"\n==================================================")
                print(f"RAW CALCULATED ASPECT RATIO: {R_calc:.4f}")
                print(f"SNAPPED UNIVERSAL ASPECT RATIO: {R:.4f} ({closest_name})")
                print(f"==================================================\n")
            except Exception as e:
                print(f"Warning: Could not load aspect ratios from json ({e}). Using raw calculated ratio.")
                R = R_calc
                print(f"\n==================================================")
                print(f"UNIVERSAL TRUE ASPECT RATIO: {R:.4f}")
                print(f"==================================================\n")
            
            print("Applying aspect ratio correction to all mosaics...")
            
            ENABLE_DEBUG_VISUALIZATIONS = False
            
            if ENABLE_DEBUG_VISUALIZATIONS:
                try:
                    from create_debug_overlay import create_overlay
                except ImportError:
                    create_overlay = lambda x: None
            
            for idx in range(shared_tracker.value):
                uncorrected_path = os.path.join(OUTPUT_DIR, f"mosaic_{idx:02d}_uncorrected.tif")
                if os.path.exists(uncorrected_path):
                    img = cv2.imread(uncorrected_path)
                    h_c, w_c = img.shape[:2]
                    
                    # The height is recorded perfectly by the line sensor, only the width (stepper motor axis) is warped.
                    # Therefore, we enforce the universal aspect ratio by purely rescaling the width.
                    corrected_w = int(h_c * R)
                    corrected_img = cv2.resize(img, (corrected_w, h_c), interpolation=cv2.INTER_LANCZOS4)
                    
                    full_crop_path = os.path.join(OUTPUT_DIR, f"mosaic_{idx:02d}.tif")
                    cv2.imwrite(full_crop_path, corrected_img)
                    print(f" -> Saved corrected mosaic {idx:02d} (Raw: {w_c}x{h_c}, Corrected: {corrected_w}x{h_c})")
                    
                    if ENABLE_DEBUG_VISUALIZATIONS:
                        seam_uncorrected = os.path.join(eq_dir, f"seams_{idx:02d}_uncorrected.jpg")
                        if os.path.exists(seam_uncorrected):
                            s_img = cv2.imread(seam_uncorrected)
                            s_img_corrected = cv2.resize(s_img, (corrected_w, h_c), interpolation=cv2.INTER_LANCZOS4)
                            cv2.imwrite(os.path.join(eq_dir, f"seams_{idx:02d}.jpg"), s_img_corrected)
                            
                        create_overlay(idx)
        else:
            print(" -> Error: No aspect ratio equations were generated!")

if __name__ == "__main__":
    # Needed for multiprocessing on MacOS
    multiprocessing.set_start_method('spawn')
    main()
