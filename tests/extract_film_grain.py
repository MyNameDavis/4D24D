import cv2
import numpy as np
import os

def extract_and_tile_grain(video_path, output_dir, num_frames=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
        
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Reached end of video at frame {i}")
            break
            
        print(f"Tiling frame {i}...")
        
        # Mirror horizontally to perfectly seamlessly tile
        frame_flipped_h = cv2.flip(frame, 1)
        row1 = np.hstack((frame, frame_flipped_h))
        
        # Mirror vertically to seamlessly tile downwards
        row2 = cv2.flip(row1, 0)
        
        # Combine rows
        tiled_grain = np.vstack((row1, row2))
        
        output_path = os.path.join(output_dir, f"seamless_grain_mask_{i:02d}.tif")
        cv2.imwrite(output_path, tiled_grain)
        print(f"Saved {output_path}")

    cap.release()

import glob
if __name__ == "__main__":
    video_files = glob.glob("Filmgrain_4KDCI*.mov")
    if not video_files:
        print("Error: video file not found.")
        exit(1)
    video_path = video_files[0]
        
    output_dir = "seamless_grain_masks"
    extract_and_tile_grain(video_path, output_dir, 10)
