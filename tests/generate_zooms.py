import cv2
import numpy as np
import glob
import os

files = glob.glob("output/debug/cropping_*.jpg")
for f in sorted(files):
    img = cv2.imread(f)
    if img is None:
        continue
    
    # The green dots have exact BGR color (0, 255, 0)
    # We can create a mask for this exact color
    # The green dots are bright green, so we use a range due to JPEG compression
    lower_green = np.array([0, 200, 0])
    upper_green = np.array([100, 255, 100])
    green_mask = cv2.inRange(img, lower_green, upper_green)
    
    # Find contours of the green dots
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    corners = []
    for cnt in contours:
        # We know the radius is 10, so area is approx 314. Lines are also green (thickness 4).
        # We can just look for the centers of contours, but lines might merge.
        # Actually, let's just use the bounding box of the contour if it's large, or better, HoughCircles!
        pass
    
    # A more robust way: use Harris corner detection or just find the intersections of the red/blue lines?
    # Wait, the green circles are very distinct. Let's use morphological operations to isolate the circles from the lines.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    circles_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(circles_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    corners = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            corners.append((cX, cY))
            
    if len(corners) != 4:
        print(f"Failed to find exactly 4 corners in {f}, found {len(corners)}")
        continue
        
    # Sort corners: top-left, top-right, bottom-right, bottom-left
    corners = np.array(corners)
    rect = np.zeros((4, 2), dtype=np.int32)
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)] # Top-left
    rect[2] = corners[np.argmax(s)] # Bottom-right
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)] # Top-right
    rect[3] = corners[np.argmax(diff)] # Bottom-left
    
    zoom_size = 200
    zooms = []
    for pt in rect:
        x, y = pt[0], pt[1]
        y_start = max(0, y - zoom_size)
        y_end = min(img.shape[0], y + zoom_size)
        x_start = max(0, x - zoom_size)
        x_end = min(img.shape[1], x + zoom_size)
        
        patch = img[y_start:y_end, x_start:x_end]
        
        # Pad if patch is at the edge
        padded_patch = np.zeros((2*zoom_size, 2*zoom_size, 3), dtype=np.uint8)
        py_start = zoom_size - (y - y_start)
        px_start = zoom_size - (x - x_start)
        padded_patch[py_start:py_start+patch.shape[0], px_start:px_start+patch.shape[1]] = patch
        
        # Draw a crosshair in the center of the patch to show the exact corner
        cv2.line(padded_patch, (zoom_size, zoom_size - 20), (zoom_size, zoom_size + 20), (255, 255, 255), 2)
        cv2.line(padded_patch, (zoom_size - 20, zoom_size), (zoom_size + 20, zoom_size), (255, 255, 255), 2)
        
        zooms.append(padded_patch)
        
    top_row = np.hstack((zooms[0], zooms[1]))
    bottom_row = np.hstack((zooms[3], zooms[2]))
    grid = np.vstack((top_row, bottom_row))
    
    out_name = f"output/debug/corners_zoom_{os.path.basename(f).split('_')[1]}"
    cv2.imwrite(out_name, grid)
    print(f"Saved {out_name}")
