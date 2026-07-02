import os
import cv2
import math
import random
import numpy as np

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
    
    # 3. Add base sensor noise (optional, but realistic)
    noise = np.random.normal(0, 1.0, calibration_img.shape)
    calibration_img = np.clip(calibration_img + noise, 0, 255).astype(np.uint8)
    
    # 4. Save
    out_path = os.path.join(output_dir, "flat_field_calibration.jpg")
    cv2.imwrite(out_path, calibration_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print(f"Calibration reference saved to: {out_path}")

def process_dslr_scans(img_path, output_dir, rows=3, cols=3, overlap=0.4, max_rot=2.0):
    """
    Simulates a DSLR camera moving over a piece of film.
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    img = cv2.imread(img_path)
    # Create the backlight environment
    scene = generate_backlight_scene(img, canvas_margin=1000)
    
    # Calculate crop dimensions to achieve desired overlap
    # We want rows and cols to cover the actual image region plus some extra
    scene_h, scene_w = scene.shape[:2]
    
    # Calculate tile size based on the scene resolution and grid requests
    # To guarantee coverage:
    tile_h = int(scene_h / (rows - (rows - 1) * overlap))
    tile_w = int(scene_w / (cols - (cols - 1) * overlap))

    save_calibration_reference(tile_w, tile_h, output_dir)
    
    flat_field = generate_flat_field(tile_w, tile_h)
    
    # Calculate steps
    step_y = int(tile_h * (1 - overlap))
    step_x = int(tile_w * (1 - overlap))
    
    print(f"Scanning grid: {rows}x{cols} ({rows*cols} captures)")
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            # Calculate top-left of the camera frame
            y_start = r * step_y
            x_start = c * step_x
            
            roi = scene[y_start:y_start+tile_h, x_start:x_start+tile_w]
            
            # --- Apply Camera Transformation ---
            angle = random.uniform(-max_rot, max_rot)
            # Use warpAffine to rotate around center of the camera sensor
            M = cv2.getRotationMatrix2D((tile_w/2, tile_h/2), angle, 1.0)
            frame = cv2.warpAffine(roi, M, (tile_w, tile_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(240,240,240))
            
            # --- Apply Lens Flaws (Applied to whole frame) ---
            # 1. Exposure Flicker
            frame = frame.astype(np.float32) * random.uniform(0.85, 1.15)
            
            # 2. Vignetting (Applies to backlight AND film)
            frame = frame * flat_field
            
            # 3. Sensor Noise
            frame = frame + np.random.normal(0, 2.0, frame.shape)
            
            # Finalize
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            cv2.imwrite(os.path.join(output_dir, f"scan_{count:03d}.jpg"), frame)
            count += 1
            
    print(f"Generated {count} scans in {output_dir}")


if __name__ == "__main__":
    process_dslr_scans(
        "./sample_photos/pexels-nam-quan-nguy-n-459228913-15839630.jpg", 
        "./image_seg_dataset", 
        rows=4, 
        cols=4, 
        overlap=0.4
    )