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
    import os
    import glob
    from image_io import read_image
    
    INPUT_DIR = PARAMS["INPUT_DIR"]
    OUTPUT_DIR = PARAMS["OUTPUT_DIR"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "equations"), exist_ok=True)

    ff_path = os.path.join(INPUT_DIR, "flat_field_calibration.jpg")
    flat_field_img = read_image(ff_path) if os.path.exists(ff_path) else None

    # Load images in prioritized formats
    search_path_tif = os.path.join(INPUT_DIR, "*.tif")
    search_path_dng = os.path.join(INPUT_DIR, "*.dng")
    search_path_heic = os.path.join(INPUT_DIR, "*.heic")
    search_path_HEIC = os.path.join(INPUT_DIR, "*.HEIC")
    
    all_paths = glob.glob(search_path_tif) + glob.glob(search_path_dng) + glob.glob(search_path_heic) + glob.glob(search_path_HEIC)
    image_paths = sorted([p for p in all_paths if "flat_field" not in p])

    batches = []
    current_batch = []
    
    if PARAMS.get("USE_BLACK_FRAMES_AS_SEPARATORS", False):
        print("Grouping batches via black frame separators...")
        from batch_process import is_black_frame
        for path in image_paths:
            if is_black_frame(path):
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
            else:
                current_batch.append(path)
        if current_batch:
            batches.append(current_batch)
    else:
        time_threshold = 6.0
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
    all_equations = []
    for eq_file in eq_files:
        try:
            with open(eq_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_equations.extend(data)
                elif isinstance(data, dict) and "R" in data:
                    global_R.append(data["R"])
        except Exception as e:
            pass
            
    avg_R = None
    if all_equations:
        segment_keys = list(set(eq['segment_key'] for eq in all_equations))
        segment_index_map = {k: i+1 for i, k in enumerate(segment_keys)}
        
        def error_func(params, eq_list):
            R2 = params[0]
            errors = []
            for eq in eq_list:
                beta = params[segment_index_map[eq['segment_key']]]
                pred = eq['coeff_R2'] * R2 + eq['coeff_beta'] * beta
                errors.append(pred - eq['rhs'])
            return np.array(errors)
            
        initial_guess = np.ones(1 + len(segment_keys))
        initial_guess[0] = 1.96  # (1.4)^2
        
        from scipy.optimize import least_squares
        res = least_squares(error_func, initial_guess, args=(all_equations,))
        avg_R = np.sqrt(res.x[0])
        
    elif global_R:
        avg_R = np.mean(global_R)
        
    target_R = PARAMS.get("TARGET_ASPECT_RATIO")
    if target_R is not None:
        print("="*50)
        print(f"USING MANUAL TARGET ASPECT RATIO: {target_R:.4f}")
        R = target_R
    elif avg_R is not None:
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
    else:
        R = None
        print(" -> Error: No aspect ratio equations were generated and no manual override set!")

    if R is not None:
        print(f"\nUNIVERSAL TRUE ASPECT RATIO: {R:.4f}")
        print("="*50)
        print("\nApplying aspect ratio correction to all mosaics...")
        
        max_h = PARAMS.get("MAX_OUTPUT_HEIGHT")
        
        for idx in range(shared_tracker.value):
            uncorrected_path = os.path.join(OUTPUT_DIR, f"mosaic_{idx:02d}_uncorrected.tif")
            if os.path.exists(uncorrected_path):
                img = read_image(uncorrected_path)
                h_c, w_c = img.shape[:2]
                
                final_h = h_c
                interp = cv2.INTER_LINEAR
                if max_h and h_c > max_h:
                    final_h = max_h
                    interp = cv2.INTER_AREA
                    
                corrected_w = int(final_h * R)
                corrected_img = cv2.resize(img, (corrected_w, final_h), interpolation=interp)
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"mosaic_{idx:02d}.tif"), corrected_img)
                print(f" -> Saved corrected mosaic {idx:02d} (Raw: {w_c}x{h_c}, Corrected: {corrected_w}x{final_h})")
    else:
        print(" -> Error: No aspect ratio equations were generated!")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
