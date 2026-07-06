import cv2
import numpy as np
import os
import glob

def create_overlay(idx):
    output_dir = "./output/debug_overlays"
    os.makedirs(output_dir, exist_ok=True)
    
    sample_photos = sorted(glob.glob("../sample_photos/*.jpg"))
    if idx >= len(sample_photos):
        return
        
    orig_path = sample_photos[idx]
    uncorrected_path = f"./output/mosaic_{idx:02d}_uncorrected.tif"
    corrected_path = f"./output/mosaic_{idx:02d}.tif"
    
    if not os.path.exists(uncorrected_path) or not os.path.exists(corrected_path):
        return
        
    orig = cv2.imread(orig_path)
    uncorrected = cv2.imread(uncorrected_path)
    corrected = cv2.imread(corrected_path)
    
    if orig is None or uncorrected is None or corrected is None:
        return
        
    print(f" -> Generating debug stretch comparison overlay for {os.path.basename(orig_path)} (Mosaic {idx:02d})...")
        
    # Scale all images to the same height (e.g. 1000px)
    target_h = 1000
    
    def resize_to_h(img, th):
        tw = int(img.shape[1] * (th / img.shape[0]))
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        
    orig_r = resize_to_h(orig, target_h)
    uncorrected_r = resize_to_h(uncorrected, target_h)
    corrected_r = resize_to_h(corrected, target_h)
    
    # Create an OVERLAY by taking the maximum width, creating blank canvases, and layering them
    max_w = max(orig_r.shape[1], uncorrected_r.shape[1], corrected_r.shape[1])
    
    canvas_orig = np.zeros((target_h, max_w, 3), dtype=np.uint8)
    canvas_uncorrected = np.zeros((target_h, max_w, 3), dtype=np.uint8)
    canvas_corrected = np.zeros((target_h, max_w, 3), dtype=np.uint8)
    
    # Center them horizontally
    def center_paste(canvas, img):
        offset_x = (canvas.shape[1] - img.shape[1]) // 2
        canvas[:, offset_x:offset_x+img.shape[1]] = img
        return canvas
        
    canvas_orig = center_paste(canvas_orig, orig_r)
    canvas_uncorrected = center_paste(canvas_uncorrected, uncorrected_r)
    canvas_corrected = center_paste(canvas_corrected, corrected_r)
    
    # Draw bounding boxes
    cv2.rectangle(canvas_orig, ((max_w - orig_r.shape[1])//2, 0), ((max_w + orig_r.shape[1])//2, target_h-1), (0, 255, 0), 4)
    cv2.rectangle(canvas_uncorrected, ((max_w - uncorrected_r.shape[1])//2, 0), ((max_w + uncorrected_r.shape[1])//2, target_h-1), (0, 0, 255), 4)
    cv2.rectangle(canvas_corrected, ((max_w - corrected_r.shape[1])//2, 0), ((max_w + corrected_r.shape[1])//2, target_h-1), (255, 0, 255), 4)
    
    # Side by side view
    cv2.putText(canvas_orig, f"Original (H: {orig.shape[0]}, W: {orig.shape[1]}) Aspect: {orig.shape[1]/orig.shape[0]:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(canvas_uncorrected, f"Pre-Correction (H: {uncorrected.shape[0]}, W: {uncorrected.shape[1]}) Aspect: {uncorrected.shape[1]/uncorrected.shape[0]:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(canvas_corrected, f"Post-Correction (H: {corrected.shape[0]}, W: {corrected.shape[1]}) Aspect: {corrected.shape[1]/corrected.shape[0]:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    spacer = np.zeros((10, max_w, 3), dtype=np.uint8)
    combined = np.vstack((canvas_orig, spacer, canvas_uncorrected, spacer, canvas_corrected))
    
    cv2.imwrite(os.path.join(output_dir, f"comparison_{idx:02d}.jpg"), combined)
        
    # Actual alpha overlay
    overlay = cv2.addWeighted(canvas_orig, 0.5, canvas_corrected, 0.5, 0)
    cv2.putText(overlay, "Overlay: Original (Green) vs Post-Correction (Magenta)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(output_dir, f"overlay_{idx:02d}.jpg"), overlay)

def create_overlays():
    sample_photos = sorted(glob.glob("../sample_photos/*.jpg"))
    for idx in range(len(sample_photos)):
        create_overlay(idx)
    print(f"Saved overlays to ./output/debug_overlays")

if __name__ == "__main__":
    create_overlays()
