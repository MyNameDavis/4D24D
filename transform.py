import cv2
import numpy as np
import math
from scipy import stats
import json
with open("param.json") as _f:
    PARAMS = json.load(_f)

def calculate_1d_entropy(slice_1d):
    """
    Calculates the Shannon Entropy of a 1D pixel slice.
    Expects slice_1d to be an array of BGR or grayscale pixels.
    """
    if len(slice_1d.shape) == 3 and slice_1d.shape[2] == 3:
        # Convert BGR slice to grayscale using mean to avoid cv2.cvtColor shape issues
        gray_slice = np.mean(slice_1d, axis=2).astype(np.uint8)
    else:
        gray_slice = slice_1d.astype(np.uint8)

def create_canvas_valid_mask(image, tile_polygons):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if tile_polygons is not None:
        for poly in tile_polygons:
            cv2.fillPoly(mask, [np.int32(poly)], 255)
        for poly in tile_polygons:
            cv2.polylines(mask, [np.int32(poly)], True, 0, thickness=PARAMS['EXCLUSION_THICKNESS'])
    else:
        mask.fill(255)
    return mask

def find_rough_bounds_shannon_scan(image, valid_mask, idx, threshold=5.5):
    print(f" -> Scanning 16x16 blocks inward from edges using Shannon entropy (>{threshold})...")
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    H, W = gray.shape
    

    
    def entropy_16x16(block):
        if block.size == 0: return 0
        hist = cv2.calcHist([block], [0], None, [256], [0, 256])
        hist = hist.ravel() / block.size
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
        
    top = 0
    for y in range(0, H-16, 16):
        found = False
        for x in range(0, W-16, 16):
            if not valid_mask[y+8, x+8]: continue
            if entropy_16x16(gray[y:y+16, x:x+16]) > threshold:
                top = y
                found = True
                break
        if found: break
        
    bottom = H
    for y in range(H-16, -1, -16):
        found = False
        for x in range(0, W-16, 16):
            if not valid_mask[y+8, x+8]: continue
            if entropy_16x16(gray[y:y+16, x:x+16]) > threshold:
                bottom = y + 16
                found = True
                break
        if found: break
        
    left = 0
    for x in range(0, W-16, 16):
        found = False
        for y in range(top, bottom, 16):
            if not valid_mask[y+8, x+8]: continue
            if entropy_16x16(gray[y:y+16, x:x+16]) > threshold:
                left = x
                found = True
                break
        if found: break
        
    right = W
    for x in range(W-16, -1, -16):
        found = False
        for y in range(top, bottom, 16):
            if not valid_mask[y+8, x+8]: continue
            if entropy_16x16(gray[y:y+16, x:x+16]) > threshold:
                right = x + 16
                found = True
                break
        if found: break
        
    # Return as cv2 rect: ((cx, cy), (w, h), angle)
    w = float(right - left)
    h = float(bottom - top)
    cx = left + w / 2.0
    cy = top + h / 2.0
    
    
    return ((cx, cy), (w, h), 0.0)

def map_all_sift_features(global_transforms, features_dict, T_shift, canvas_w, canvas_h):
    """
    Transforms all SIFT keypoints from individual image spaces into the final canvas space.
    First removes the 5% spatial outliers based on Euclidean distance from absolute canvas center.
    Then filters out the 5% weakest outliers based on response (assuming normal distribution).
    """
    print("Mapping all SIFT features to canvas space...")
    all_mapped_pts = []
    all_responses = []
    
    for img_name, G in global_transforms.items():
        if img_name not in features_dict:
            continue
            
        keypoints = features_dict[img_name]['keypoints']
        if not keypoints:
            continue
            
        # Reshape for perspectiveTransform: (N, 1, 2)
        pts = np.float32([kp.pt for kp in keypoints]).reshape(-1, 1, 2)
        responses = [kp.response for kp in keypoints]
        
        F = T_shift @ G
        mapped = cv2.perspectiveTransform(pts, F)
        
        all_mapped_pts.extend(mapped)
        all_responses.extend(responses)

    if not all_mapped_pts:
        return None

    # all_mapped_pts = np.array(all_mapped_pts)
    # all_responses = np.array(all_responses)
    
    # # Filter 1: Spatial Outliers (5% furthest from canvas center)
    # center = np.array([canvas_w / 2.0, canvas_h / 2.0])
    # distances = np.linalg.norm(all_mapped_pts.reshape(-1, 2) - center, axis=1)
    # mean_dist = np.mean(distances)
    # std_dist = np.std(distances)
    # dist_threshold = mean_dist + stats.norm.ppf(1-0.00000001) * std_dist
    
    # spatial_mask = distances <= dist_threshold
    # all_mapped_pts = all_mapped_pts[spatial_mask]
    # all_responses = all_responses[spatial_mask]
    # print(f" -> Filtered out {np.sum(~spatial_mask)} spatial outliers (threshold: {dist_threshold:.2f}px)")
    
    # # Filter 2: Weakest outliers assuming normal distribution (z < -1.645)
    # mean_resp = np.mean(all_responses)
    # std_resp = np.std(all_responses)
    # threshold = mean_resp + stats.norm.ppf(0.0000001) * std_resp
    
    # binary_mask = all_responses >= threshold
    # filtered_pts = all_mapped_pts[binary_mask]
    
    # print(f" -> Filtered out {np.sum(~binary_mask)} weak outliers (threshold: {threshold:.2f})")

    filtered_pts = np.array(all_mapped_pts)
    
    return filtered_pts

def fine_tune_corners_harris(canvas, valid_mask, corners, idx):
    print(" -> Fine-tuning 3D corners via CLAHE + Harris Corner Detection...")
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    
    # Mask out the invalid regions to prevent false corner detection
    gray = cv2.bitwise_and(gray, gray, mask=valid_mask)

    
    # Apply CLAHE to boost contrast of faded corners
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    
    refined_harris = np.copy(corners).astype(np.float32)
    window = 50
    
    # Termination criteria for sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    
    for i in range(4):
        C = corners[i]
        x, y = int(C[0]), int(C[1])
        x1 = max(0, x - window)
        x2 = min(canvas.shape[1], x + window)
        y1 = max(0, y - window)
        y2 = min(canvas.shape[0], y + window)
        
        patch = clahe_img[y1:y2, x1:x2]
        if patch.size > 0:
            dst = cv2.cornerHarris(patch, 2, 3, 0.04)
            _, _, _, max_loc = cv2.minMaxLoc(dst)
            
            # Sub-pixel refinement
            corner_pt = np.array([[[np.float32(max_loc[0]), np.float32(max_loc[1])]]])
            cv2.cornerSubPix(patch, corner_pt, (5, 5), (-1, -1), criteria)
            
            sub_x = corner_pt[0][0][0]
            sub_y = corner_pt[0][0][1]
            
            refined_harris[i] = [x1 + sub_x, y1 + sub_y]
                
    return refined_harris

def find_quadrilateral_corners(canvas, valid_mask, rough_rect, idx, mapped_features=None, expansion_factor=1.05):
    ((cx, cy), (w, h), angle) = rough_rect
    exp_w = w * expansion_factor
    exp_h = h * expansion_factor
    exp_rect = ((cx, cy), (exp_w, exp_h), angle)
    
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    
    # Crop to the expanded rectangle to save immense computational power
    box = cv2.boxPoints(exp_rect)
    x_min, y_min, w_roi, h_roi = cv2.boundingRect(np.int32(box))
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(canvas.shape[1], x_min + w_roi)
    y_max = min(canvas.shape[0], y_min + h_roi)
    
    roi = gray[y_min:y_max, x_min:x_max]
    
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(roi)
    blur = cv2.GaussianBlur(gray_clahe, (5, 5), 0)
    
    v = np.median(blur)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged_roi = cv2.Canny(blur, lower, upper)
    
    edged = np.zeros_like(gray)
    edged[y_min:y_max, x_min:x_max] = edged_roi
    
    # Create mask for the outer 20% ring (exclude inner 80%)
    ring_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(ring_mask, [np.int32(box)], 0, 255, -1)
    
    inner_w = exp_w * 0.8
    inner_h = exp_h * 0.8
    inner_rect = ((cx, cy), (inner_w, inner_h), angle)
    cv2.drawContours(ring_mask, [np.int32(cv2.boxPoints(inner_rect))], 0, 0, -1)
    
    # Exclude both the inner 80% AND the explicitly invalid regions (the SIFT tile boundary)
    final_mask = cv2.bitwise_and(ring_mask, valid_mask)
    masked_edges = cv2.bitwise_and(edged, edged, mask=final_mask)
    
    lines_p = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)
    
    if lines_p is None:
        print(" -> Warning: Hough transform found no lines, falling back to minAreaRect.")
        return cv2.boxPoints(rough_rect)
        
    endpoints = []
    for line in lines_p:
        x1, y1, x2, y2 = line[0]
        endpoints.append([x1, y1])
        endpoints.append([x2, y2])
        
    endpoints = np.float32(endpoints)
    hull = cv2.convexHull(endpoints)
    
    corners_out = None

    # Try different epsilons to force 4 points
    for eps_factor in np.linspace(0.01, 0.2, 50):
        epsilon = eps_factor * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4:
            corners_out = approx.reshape(4, 2)
            break
            
    if corners_out is None:
        print(" -> Warning: Could not find exactly 4 corners from Hough hull, falling back to minAreaRect of hull.")
        rect_hull = cv2.minAreaRect(endpoints)
        corners_out = cv2.boxPoints(rect_hull)

    return corners_out

def orient_and_crop(image, mapped_features, idx, tile_polygons=None):
    print("\\nOrienting and cropping the final composite...")

    valid_mask = create_canvas_valid_mask(image, tile_polygons)
    rect = find_rough_bounds_shannon_scan(image, valid_mask, idx, threshold=5.25)
    
    if rect is None:
        print(" -> Warning: Entropy bounds failed. Returning uncropped image.")
        return image
    print(" -> Refining bounding box via Canny + Hough lines into Quadrilateral...")
    corners = find_quadrilateral_corners(image, valid_mask, rect, idx, mapped_features, expansion_factor=1.05)

    
    def order_points(pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left
        rect[2] = pts[np.argmax(s)] # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right
        rect[3] = pts[np.argmax(diff)] # Bottom-left
        return rect
        
    ordered_corners = order_points(corners)
    ordered_corners = fine_tune_corners_harris(image, valid_mask, ordered_corners, idx)

    def nudge_corners_inward(corners, nudge_px):
        center = np.mean(corners, axis=0)
        nudged = []
        for pt in corners:
            vec = center - pt
            dist = np.linalg.norm(vec)
            if dist > 0:
                vec_norm = vec / dist
                nudged.append(pt + vec_norm * nudge_px)
            else:
                nudged.append(pt)
        return np.array(nudged, dtype=np.float32)

    ordered_corners = nudge_corners_inward(ordered_corners, 45.0)

    (tl, tr, br, bl) = ordered_corners
    
    
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    if maxWidth == 0 or maxHeight == 0:
        print(" -> Warning: Computed bounds are 0. Returning uncropped image.")
        return image
    
    dst = np.array([
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight]], dtype="float32")
        
    M_persp = cv2.getPerspectiveTransform(ordered_corners, dst)
    print(" -> Flattening 3D perspective to rectilinear bounds...")
    # Use INTER_LINEAR instead of INTER_LANCZOS4 to prevent any ringing artifacts, and rely on constant border.
    flattened_img = cv2.warpPerspective(image, M_persp, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    cropped = flattened_img
    print(f" -> Found exact rectilinear bounds: {maxWidth}x{maxHeight}")
    
    return cropped, M_persp, (0, 0)