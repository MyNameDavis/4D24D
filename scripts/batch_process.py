import os
import glob
import cv2
import numpy as np
import multiprocessing

import json
with open("param.json") as _f:
    PARAMS = json.load(_f)
from SIFT import extract_sift_features
from match import cluster_film_scenes, compute_exposure_gains, compute_canvas_bounds, blend_images_onto_canvas
from transform import orient_and_crop, map_all_sift_features
from aspect_ratio import extract_aspect_ratio_equations

from image_io import read_image

def is_black_frame(path):
    img = read_image(path, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        return False
    small = cv2.resize(img, (10, 10))
    return np.mean(small) < 10

def stitch_mosaic(comp, features_dict, connections, output_dir, idx, flat_field_img=None):
    if not connections:
        print("No connections to stitch.")
        return

    best_match = sorted(connections, key=lambda x: x['inliers'], reverse=True)[0]
    anchor = best_match['img1']
    
    global_transforms = {anchor: np.eye(3, dtype=np.float32)}

    global_gains = compute_exposure_gains(connections, features_dict, global_transforms, anchor, flat_field_img)
    canvas_w, canvas_h, T_shift = compute_canvas_bounds(global_transforms, features_dict)
    final_canvas, tile_polygons = blend_images_onto_canvas(global_transforms, global_gains, features_dict, flat_field_img, canvas_w, canvas_h, T_shift, idx)
    mapped_features = map_all_sift_features(global_transforms, features_dict, T_shift, canvas_w, canvas_h)
    
    cropped_canvas_data = orient_and_crop(final_canvas, mapped_features, idx, tile_polygons)
    if cropped_canvas_data is None:
        print(" -> Warning: orient_and_crop returned None.")
        return
        
    if isinstance(cropped_canvas_data, tuple):
        cropped_canvas, M_persp, (rx_min, ry_min) = cropped_canvas_data
    else:
        cropped_canvas = cropped_canvas_data
        M_persp = None
        rx_min, ry_min = 0, 0

    print("\nSaving final outputs...")
    equations = []
    if cropped_canvas is not None and cropped_canvas.size > 0:
        full_crop_path = os.path.join(output_dir, f"mosaic_{idx:02d}.tif")
        uncorrected_path = os.path.join(output_dir, f"mosaic_{idx:02d}_uncorrected.tif")
        # Save uncorrected first (will be overwritten if we did local correction, but we defer it now)
        cv2.imwrite(uncorrected_path, cropped_canvas)
        cv2.imwrite(full_crop_path, cropped_canvas)
        print(f" -> Cropped mosaic saved to {full_crop_path}")
        
        if PARAMS.get("TARGET_ASPECT_RATIO") is None:
            try:
                from aspect_ratio import extract_aspect_ratio_equations
                print(f" -> Analyzing film grain and SIFT point cloud to extract aspect ratio equations for mosaic {idx}...")
                c_h, c_w = cropped_canvas.shape[:2]
                equations = extract_aspect_ratio_equations(features_dict, connections, global_transforms, T_shift, M_persp, idx, c_w, c_h)
                
                import json
                os.makedirs(os.path.join(output_dir, "equations"), exist_ok=True)
                with open(os.path.join(output_dir, "equations", f"equations_{idx:02d}.json"), "w") as f:
                    json.dump(equations, f, indent=4)
                    
            except Exception as e:
                print(f" -> Warning: Equation extraction failed: {e}")
        else:
            print(f" -> Manual Aspect Ratio override detected. Skipping point cloud solver for mosaic {idx}.")
            
    else:
        print(" -> Warning: Cropped canvas is empty, skipping output.")
        
    return equations

def process_single_batch(b_idx, num_batches, batch_paths, flat_field_img, DOWNSCALE_FACTOR, MAX_FEATURES, MIN_INLIERS, WINDOW_SIZE, OUTPUT_DIR, shared_tracker):
    print(f"\n{'='*60}\nProcessing Film Scan Batch {b_idx + 1}/{num_batches} ({len(batch_paths)} images)\n{'='*60}")
    
    features_dict = extract_sift_features(batch_paths, flat_field_img, downscale_factor=DOWNSCALE_FACTOR, max_features=MAX_FEATURES)
    components, connections = cluster_film_scenes(features_dict, min_inliers=MIN_INLIERS, window_size=WINDOW_SIZE)
    
    valid_components = [c for c in components if len(c) >= 2]
    print(f"\nFound {len(valid_components)} distinct mosaic scenes in this batch!")
    
    local_tracker = 0
    for idx, comp in enumerate(components):
        if len(comp) < 2:
            print(f"Skipping mosaic {idx} (Only 1 image: {os.path.basename(comp[0])})")
            continue

        current_id = shared_tracker.value + local_tracker
        print(f"\n{'='*50}\nProcessing mosaic {current_id} ({len(comp)} tiles)\n{'='*50}")
        comp_set = set(comp)
        comp_connections = [c for c in connections if c['img1'] in comp_set and c['img2'] in comp_set]
        
        stitch_mosaic(comp, features_dict, comp_connections, OUTPUT_DIR, current_id, flat_field_img)
        local_tracker += 1
        
    shared_tracker.value += local_tracker
