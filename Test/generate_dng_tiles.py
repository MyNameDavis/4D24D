import os
import glob
import cv2
import numpy as np
import random
import rawpy

def generate_backlight_scene(img, canvas_margin=1000):
    """Places the image segment onto a backlight with realistic lighting gradients."""
    h, w = img.shape[:2]
    scene_h = h + canvas_margin * 2
    scene_w = w + canvas_margin * 2
    
    # Base backlight color (slightly off-white)
    scene = np.full((scene_h, scene_w, 3), 240, dtype=np.uint8)
    
    # Apply a gentle 2D gradient to simulate uneven backlight illumination
    X, Y = np.meshgrid(np.linspace(-1, 1, scene_w), np.linspace(-1, 1, scene_h))
    gradient = np.exp(-(X**2 + Y**2) * 0.5) * 30
    gradient = gradient.astype(np.float32)
    
    scene = np.clip(scene + gradient[:, :, np.newaxis], 0, 255).astype(np.uint8)
    
    # Place the film segment in the center
    scene[canvas_margin:canvas_margin+h, canvas_margin:canvas_margin+w] = img
    return scene, canvas_margin

def generate_dng_dataset():
    input_dir = "./sample_dng"
    output_dir = "./image_seg_dataset_dng"
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean previous output
    for f in glob.glob(os.path.join(output_dir, "*.jpg")):
        os.remove(f)
        
    dng_files = sorted(glob.glob(os.path.join(input_dir, "*.dng")))
    if not dng_files:
        print("No DNG files found.")
        return
        
    # Process only 2 DNG files to save time
    dng_files = dng_files[:2]
    
    target_w, target_h = 4272, 2848
    canvas_margin = 1000
    
    tile_count = 0
    for file_idx, dng_path in enumerate(dng_files):
        print(f"Processing {dng_path}...")
        
        # 1. Read the uncompressed DNG file
        with rawpy.imread(dng_path) as raw:
            rgb_img = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
            
        base_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        H, W = base_img.shape[:2]
        
        # 2. Add real seamless 35mm film grain overlay
        grain_masks = glob.glob('seamless_grain_masks/*.tif')
        selected_mask = random.choice(grain_masks)
        print(f" -> Adding real 35mm seamless film grain ({selected_mask})...")
        tiled_grain = cv2.imread(selected_mask)
        
        # Crop the mask to match the DNG size exactly, with a random offset
        max_y = tiled_grain.shape[0] - H
        max_x = tiled_grain.shape[1] - W
        start_y = random.randint(0, max_y)
        start_x = random.randint(0, max_x)
        grain_crop = tiled_grain[start_y:start_y+H, start_x:start_x+W]
        
        # The overlay is centered at 128
        film_grain = grain_crop.astype(np.float32) - 128.0
        # Boost intensity slightly since it's uncompressed 16-bit
        film_grain = film_grain * 1.5
        base_img = np.clip(base_img.astype(np.float32) + film_grain, 0, 255).astype(np.uint8)
        
        # 3. Place the ENTIRE physical film onto the light table backlight
        scene, margin = generate_backlight_scene(base_img, canvas_margin=canvas_margin)
        scene_h, scene_w = scene.shape[:2]
        
        # 4. Simulate a DSLR Camera Sensor (4272x2848) panning across the light table
        # We will take 4 shots that capture the 4 corners of the film
        # To ensure the film corners are in the camera shots, we overlap the margin
        x_offsets = [margin - 500, scene_w - target_w - (margin - 500)]
        y_offsets = [margin - 500, scene_h - target_h - (margin - 500)]
        
        for r, y in enumerate(y_offsets):
            for c, x in enumerate(x_offsets):
                # To prevent cv2.warpPerspective from leaving an artificial background border,
                # we extract a slightly larger ROI, warp it, and then crop out the exact camera frame!
                pad = 400
                roi_y1 = max(0, y - pad)
                roi_y2 = min(scene_h, y + target_h + pad)
                roi_x1 = max(0, x - pad)
                roi_x2 = min(scene_w, x + target_w + pad)
                roi = scene[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                
                # Local coordinates of the camera sensor within the padded ROI
                cx = x - roi_x1
                cy = y - roi_y1
                
                # 5. Warp to simulate DSLR camera tilt/misalignment for this specific shot
                dx = target_w * 0.05
                dy = target_h * 0.05
                
                pts1 = np.float32([[cx, cy], [cx+target_w, cy], [cx+target_w, cy+target_h], [cx, cy+target_h]])
                pts2 = np.float32([
                    [cx + random.uniform(-dx, dx), cy + random.uniform(-dy, dy)],
                    [cx+target_w + random.uniform(-dx, dx), cy + random.uniform(-dy, dy)],
                    [cx+target_w + random.uniform(-dx, dx), cy+target_h + random.uniform(-dy, dy)],
                    [cx + random.uniform(-dx, dx), cy+target_h + random.uniform(-dy, dy)]
                ])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                
                warped_roi = cv2.warpPerspective(roi, M, (roi.shape[1], roi.shape[0]), flags=cv2.INTER_LANCZOS4)
                
                # Extract the final camera frame
                warped = warped_roi[cy:cy+target_h, cx:cx+target_w]
                
                # 6. Add photon noise (sensor noise) after warp
                photon_noise = np.random.normal(0, 1.5, warped.shape).astype(np.float32)
                final_img = np.clip(warped.astype(np.float32) + photon_noise, 0, 255).astype(np.uint8)
                
                out_path = os.path.join(output_dir, f"segment_{tile_count:02d}.tif")
                cv2.imwrite(out_path, final_img)
                print(f" -> Saved {out_path}")
                tile_count += 1

    print(f"Generated {tile_count} DNG tiles successfully in '{output_dir}'.")

if __name__ == "__main__":
    generate_dng_dataset()
