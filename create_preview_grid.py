import os
import cv2
import math
import glob
import numpy as np

def create_preview_grid(input_dir, output_file, scale_factor=0.2):
    """
    Reads a directory of image tiles, downscales them, and arranges them 
    into an unordered visual grid for previewing.
    """
    # 1. Grab all TIFF files and sort them (though the visual layout is unordered)
    search_path = os.path.join(input_dir, "*.jpg")
    file_list = sorted(glob.glob(search_path))
    num_images = len(file_list)
    
    if num_images == 0:
        print(f"No .jpg files found in {input_dir}.")
        return

    print(f"Found {num_images} tiles. Generating preview grid...")

    # 2. Calculate a roughly square grid layout
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    # 3. Read the first image to establish the thumbnail dimensions
    first_img = cv2.imread(file_list[0])
    h, w = first_img.shape[:2]
    thumb_w = int(w * scale_factor)
    thumb_h = int(h * scale_factor)

    # 4. Create a blank dark gray master canvas
    canvas_w = cols * thumb_w
    canvas_h = rows * thumb_h
    # 40 represents a dark gray background for empty grid slots
    canvas = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8) 

    # 5. Process and place each tile
    for idx, file_path in enumerate(file_list):
        img = cv2.imread(file_path)
        
        # Downscale for memory efficiency
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        
        # Draw a 2-pixel white border inside the thumbnail to delineate edges
        cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), (255, 255, 255), 2)
        
        # Calculate X and Y coordinates on the master canvas
        row_idx = idx // cols
        col_idx = idx % cols
        
        y_start = row_idx * thumb_h
        x_start = col_idx * thumb_w
        
        # Drop the thumbnail into its grid slot
        canvas[y_start : y_start + thumb_h, x_start : x_start + thumb_w] = thumb

    # 6. Save the final preview
    cv2.imwrite(output_file, canvas)
    print(f"Success. Preview saved to {output_file} ({cols}x{rows} grid)")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    INPUT_DIR = "./image_seg_dataset"         # Directory containing the generated tiles
    OUTPUT_PREVIEW = "preview_grid.jpg"  # Output filename
    SCALE_FACTOR = 0.25                  # 25% scale to keep RAM usage low
    # ---------------------

    create_preview_grid(
        input_dir=INPUT_DIR,
        output_file=OUTPUT_PREVIEW,
        scale_factor=SCALE_FACTOR
    )