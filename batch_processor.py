import os
import glob
import collections
import cv2
import numpy as np
from SIFT import extract_sift_features, compute_matches, stitch_mosaic
from tqdm import tqdm

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
    pbar = tqdm(total=len(local_pairs), desc="Clustering Images", unit="pair")
    local_connections = compute_matches(features_dict, local_pairs, min_inliers, pbar)
    
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
    pbar.total += len(cross_pairs)
    pbar.refresh()
    outlier_connections = compute_matches(features_dict, cross_pairs, min_inliers, pbar)
    pbar.close()
    
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

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Process mosaics")
    parser.add_argument("--input", default="./image_seg_dataset", help="Input directory")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--downscale", type=float, default=0.5, help="Downscale factor (e.g. 0.5 for fast, 1.0 for high fidelity)")
    parser.add_argument("--min-inliers", type=int, default=40, help="RANSAC min inliers (e.g. 15 for fast, 40 for strict)")
    parser.add_argument("--window-size", type=int, default=5, help="Sliding window size (e.g. 5 for sequential scans, 15 for unordered)")
    parser.add_argument("--max-features", type=int, default=1000, help="Max SIFT features per image (0 for unlimited)")
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
            
        tracker = 0
        for b_idx, batch_paths in enumerate(batches):
            print(f"\n{'='*60}\nProcessing Film Scan Batch {b_idx + 1}/{len(batches)} ({len(batch_paths)} images)\n{'='*60}")
            
            features_dict = extract_sift_features(batch_paths, flat_field_img, downscale_factor=DOWNSCALE_FACTOR, max_features=MAX_FEATURES)
            components, connections = cluster_film_scenes(features_dict, min_inliers=MIN_INLIERS, window_size=WINDOW_SIZE)
            
            valid_components = [c for c in components if len(c) >= 2]
            print(f"\nFound {len(valid_components)} distinct mosaic scenes in this batch!")
            
            for idx, comp in enumerate(components):
                if len(comp) < 2:
                    print(f"Skipping component {idx} (Only 1 image: {os.path.basename(comp[0])})")
                    continue
        
                print(f"\n{'='*50}\nProcessing mosaic {tracker} ({len(comp)} tiles)\n{'='*50}")
                comp_set = set(comp)
                comp_connections = [c for c in connections if c['img1'] in comp_set and c['img2'] in comp_set]
                
                stitch_mosaic(comp, features_dict, comp_connections, OUTPUT_DIR, tracker, flat_field_img)
                tracker += 1
