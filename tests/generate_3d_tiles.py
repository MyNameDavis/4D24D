import os
import cv2
import math
import random
import numpy as np
import json

def generate_flat_field(w, h):
    x = np.linspace(-w/2, w/2, w)
    y = np.linspace(-h/2, h/2, h)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    max_R = np.sqrt((w/2)**2 + (h/2)**2)
    mask = 1.0 - 0.4 * (R / max_R)
    return np.stack([mask]*3, axis=2).astype(np.float32)

def generate_backlight_scene(img, canvas_margin=500, backlight_val=240):
    """Places the film on a large, bright backlight canvas."""
    h, w = img.shape[:2]
    canvas_h, canvas_w = h + 2*canvas_margin, w + 2*canvas_margin
    
    # Create canvas filled with the backlight color
    canvas = np.full((canvas_h, canvas_w, 3), backlight_val, dtype=np.uint8)
    
    # Apply a very subtle, high-resolution unique noise pattern to the backlight itself
    # This ensures each film scan has a uniquely textured background, preventing false cross-matches
    noise = np.random.normal(0, 1.5, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Place film in center
    canvas[canvas_margin:canvas_margin+h, canvas_margin:canvas_margin+w] = img
    
    return canvas

def save_calibration_reference(tile_w, tile_h, output_dir, backlight_val=240):
    """
    Generates a calibration reference image (The 'Flat Field').
    This represents the backlight captured by the sensor without film.
    """
    # 1. Start with pure backlight intensity
    calibration_img = np.full((tile_h, tile_w, 3), backlight_val, dtype=np.float32)
    
    # 2. Apply the same vignetting mask used in tile processing
    mask = generate_flat_field(tile_w, tile_h)
    calibration_img = calibration_img * mask
    
    # 3. Save directly (Real calibration frames average many shots to eliminate photon noise)
    out_path = os.path.join(output_dir, "flat_field_calibration.jpg")
    cv2.imwrite(out_path, calibration_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print(f"Calibration reference saved to: {out_path}")

import glob
import shutil

def process_all_photos(input_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    all_tiles = []
    
    photo_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    
    # Save a single calibration reference for the dataset
    save_calibration_reference(1000, 1000, output_dir)
    
    for photo_path in photo_files:
        img = cv2.imread(photo_path)
        
        # Add TRUE film grain to the physical film frame BEFORE any camera warping
        # This simulates the physical silver halide crystals on the film emulsion
        film_grain = np.random.normal(0, 8.0, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + film_grain, 0, 255).astype(np.uint8)
        
        h, w = img.shape[:2]
        scene = generate_backlight_scene(img, canvas_margin=1000)
        scene_h, scene_w = scene.shape[:2]
        
        # Randomize grid for this photo
        rows = random.randint(2, 4)
        cols = random.randint(2, 4)
        overlap = random.uniform(0.4, 0.7)
        max_rot = random.uniform(1.0, 3.0)
        
        tile_h = int(scene_h / (rows - (rows - 1) * overlap))
        tile_w = int(scene_w / (cols - (cols - 1) * overlap))
        
        flat_field = generate_flat_field(tile_w, tile_h)
        
        step_y = int(tile_h * (1 - overlap))
        step_x = int(tile_w * (1 - overlap))
        
        print(f"Processing {os.path.basename(photo_path)} -> Grid: {rows}x{cols}")
        
        for r in range(rows):
            for c in range(cols):
                y_start = r * step_y
                x_start = c * step_x
                roi = scene[y_start:y_start+tile_h, x_start:x_start+tile_w]
                
                dx = tile_w * 0.05
                dy = tile_h * 0.05
                pts1 = np.float32([[0, 0], [tile_w, 0], [tile_w, tile_h], [0, tile_h]])
                pts2 = np.float32([
                    [random.uniform(-dx, dx), random.uniform(-dy, dy)],
                    [tile_w + random.uniform(-dx, dx), random.uniform(-dy, dy)],
                    [tile_w + random.uniform(-dx, dx), tile_h + random.uniform(-dy, dy)],
                    [random.uniform(-dx, dx), tile_h + random.uniform(-dy, dy)]
                ])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                frame = cv2.warpPerspective(roi, M, (tile_w, tile_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(240,240,240))
                
                frame = frame.astype(np.float32) * random.uniform(0.85, 1.15)
                frame = frame * flat_field
                
                # Add subtle sensor photon noise
                photon_noise = np.random.normal(0, 1.5, frame.shape)
                frame = frame + photon_noise
                
                frame = np.clip(frame, 0, 255).astype(np.uint8)
                
                all_tiles.append({
                    "type": "tile",
                    "frame": frame,
                    "metadata": {
                        "original_photo": os.path.basename(photo_path),
                        "film_w": w,
                        "film_h": h,
                        "canvas_margin": 1000,
                        "scene_to_roi_offset": [x_start, y_start],
                        "roi_to_frame_M": M.tolist()
                    }
                })
                
        # Add a black frame divider
        black_frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        all_tiles.append({
            "type": "divider",
            "frame": black_frame
        })

    # Save sequentially with anonymous names
    ground_truth = {}
    for i, item in enumerate(all_tiles):
        filename = f"img_{i:03d}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), item["frame"])
        if item["type"] == "tile":
            ground_truth[filename] = item["metadata"]
            
    with open(os.path.join(output_dir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=4)
        
    print(f"\nSaved {len(all_tiles)} total anonymized tiles to {output_dir}")

if __name__ == "__main__":
    process_all_photos("./sample_photos", "./image_seg_dataset_3d")