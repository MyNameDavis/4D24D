import os
import cv2
import numpy as np

def create_seamless_tile(frame, target_w=8000, target_h=8000, blend_px=200):
    h, w = frame.shape[:2]
    
    # We will tile the image horizontally and vertically
    # To mitigate harsh boundaries, we'll overlap tiles and crossfade them
    
    out = np.zeros((target_h, target_w, 3), dtype=np.float32)
    weight_sum = np.zeros((target_h, target_w, 1), dtype=np.float32)
    
    # Precompute a 2D window (like a Tukey window) for the tile
    # A flat center with a linear fade at the edges
    y_fade = np.ones(h, dtype=np.float32)
    y_fade[:blend_px] = np.linspace(0, 1, blend_px)
    y_fade[-blend_px:] = np.linspace(1, 0, blend_px)
    
    x_fade = np.ones(w, dtype=np.float32)
    x_fade[:blend_px] = np.linspace(0, 1, blend_px)
    x_fade[-blend_px:] = np.linspace(1, 0, blend_px)
    
    window = np.outer(y_fade, x_fade)
    window = np.expand_dims(window, axis=2)
    
    frame_f = frame.astype(np.float32)
    
    # Tile it over the target canvas
    y_step = h - blend_px
    x_step = w - blend_px
    
    for y in range(0, target_h, y_step):
        for x in range(0, target_w, x_step):
            y1 = y
            y2 = min(y + h, target_h)
            x1 = x
            x2 = min(x + w, target_w)
            
            src_y2 = y2 - y1
            src_x2 = x2 - x1
            
            out[y1:y2, x1:x2] += frame_f[0:src_y2, 0:src_x2] * window[0:src_y2, 0:src_x2]
            weight_sum[y1:y2, x1:x2] += window[0:src_y2, 0:src_x2]
            
    # Normalize
    weight_sum[weight_sum == 0] = 1.0
    out = (out / weight_sum)
    return np.clip(out, 0, 255).astype(np.uint8)

def main():
    os.makedirs('seamless_grain_masks', exist_ok=True)
    
    # Clean previous masks just in case
    for f in os.listdir('seamless_grain_masks'):
        if f.endswith('.tif'):
            os.remove(os.path.join('seamless_grain_masks', f))
            
    cap = cv2.VideoCapture('film_grain.mov')
    
    num_frames_to_extract = 5
    extracted = 0
    
    print("Extracting frames from film_grain.mov...")
    while extracted < num_frames_to_extract:
        ret, frame = cap.read()
        if not ret:
            break
            
        print(f" -> Tiling frame {extracted+1}/{num_frames_to_extract}...")
        
        # The frame is in color, but film grain is usually monochrome or slightly color tinted.
        # The user's original grain might be grayscale, but we'll preserve it as-is.
        tiled = create_seamless_tile(frame, target_w=8000, target_h=8000, blend_px=200)
        
        out_path = f"seamless_grain_masks/grain_mask_{extracted:02d}.tif"
        cv2.imwrite(out_path, tiled)
        print(f" -> Saved {out_path}")
        extracted += 1
        
        # Skip a few frames to get more variation
        for _ in range(5):
            cap.read()

    cap.release()
    print("Finished extracting and tiling film grain masks!")

if __name__ == "__main__":
    main()
