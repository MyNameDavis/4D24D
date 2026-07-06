import os
import glob
import cv2
import numpy as np
import multiprocessing
import json

import json
with open("param.json") as _f:
    PARAMS = json.load(_f)
from batch_process import process_single_batch

def main():
    INPUT_DIR = PARAMS["INPUT_DIR"]
    OUTPUT_DIR = PARAMS["OUTPUT_DIR"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "equations"), exist_ok=True)

    ff_path = os.path.join(INPUT_DIR, "flat_field_calibration.jpg")
    flat_field_img = cv2.imread(ff_path) if os.path.exists(ff_path) else None

    search_path = os.path.join(INPUT_DIR, "*.tif")
    image_paths = sorted([p for p in glob.glob(search_path) if "flat_field" not in p])
    
    if not image_paths:
        search_path = os.path.join(INPUT_DIR, "*.dng")
        image_paths = sorted([p for p in glob.glob(search_path) if "flat_field" not in p])

    time_threshold = 2.0
    batches = []
    current_batch = []
    last_time = None
    
    for path in image_paths:
        ctime = os.path.getmtime(path)
        if last_time is None or (ctime - last_time) <= time_threshold:
            current_batch.append(path)
        else:
            batches.append(current_batch)
            current_batch = [path]
        last_time = ctime
    if current_batch:
        batches.append(current_batch)

    manager = multiprocessing.Manager()
    shared_tracker = manager.Value('i', 0)
    
    for b_idx, batch_paths in enumerate(batches):
        process_single_batch(b_idx, len(batches), batch_paths, flat_field_img, 
                             PARAMS["DOWNSCALE_FACTOR"], PARAMS["MAX_FEATURES"], 
                             PARAMS["MIN_INLIERS"], PARAMS["WINDOW_SIZE"], 
                             OUTPUT_DIR, shared_tracker)
                             
    print("\n" + "="*80)
    print("Calculating Global Consensus Aspect Ratio...")
    
    eq_dir = os.path.join(OUTPUT_DIR, "equations")
    eq_files = sorted(glob.glob(os.path.join(eq_dir, "equations_*.json")))
    
    global_R = []
    for eq_file in eq_files:
        try:
            with open(eq_file, "r") as f:
                data = json.load(f)
                R_calc = data["R"]
                w, h = data["width"], data["height"]
                global_R.append(R_calc)
                print(f" -> Measured physical film aspect ratio from mosaic bounding box: {R_calc:.4f} ({w}x{h})")
        except Exception as e:
            pass
            
    if global_R:
        avg_R = np.mean(global_R)
        print("="*50)
        print(f"RAW CALCULATED ASPECT RATIO: {avg_R:.4f}")
        
        try:
            with open("film_aspect_ratios.json", "r") as f:
                aspect_ratios = json.load(f)
                
            closest_name = None
            closest_ratio = None
            min_diff = float('inf')
            for name, ratio in aspect_ratios.items():
                if abs(avg_R - ratio) < min_diff:
                    min_diff = abs(avg_R - ratio)
                    closest_ratio = ratio
                    closest_name = name
                    
            print(f"SNAPPED UNIVERSAL ASPECT RATIO: {closest_ratio:.4f} ({closest_name})")
            print("="*50)
            R = closest_ratio
        except:
            print("Warning: Could not snap to standard ratios. Using raw calculated.")
            R = avg_R
        
        print(f"\nUNIVERSAL TRUE ASPECT RATIO: {R:.4f}")
        print("="*50)
        print("\nApplying aspect ratio correction to all mosaics...")
        
        for idx in range(shared_tracker.value):
            uncorrected_path = os.path.join(OUTPUT_DIR, f"mosaic_{idx:02d}_uncorrected.tif")
            if os.path.exists(uncorrected_path):
                img = cv2.imread(uncorrected_path)
                h_c, w_c = img.shape[:2]
                corrected_w = int(h_c * R)
                corrected_img = cv2.resize(img, (corrected_w, h_c), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"mosaic_{idx:02d}.tif"), corrected_img)
                print(f" -> Saved corrected mosaic {idx:02d} (Raw: {w_c}x{h_c}, Corrected: {corrected_w}x{h_c})")
    else:
        print(" -> Error: No aspect ratio equations were generated!")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
