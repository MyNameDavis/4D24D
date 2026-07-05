import os
import cv2
import numpy as np
import collections
from scipy.optimize import minimize
from scipy import stats

def load_clean_image(path, flat_field_img=None):
    img = cv2.imread(path)
    if img is None:
        return None
        
    if flat_field_img is not None:
        ff_resized = cv2.resize(flat_field_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        ff = ff_resized.astype(np.float32) / 255.0
        ff[ff == 0] = 1.0 
        img = (img.astype(np.float32) / ff).clip(0, 255).astype(np.uint8)
        
    return img

def extract_sift_features(image_paths, flat_field_img=None, downscale_factor=1.0, max_features=None):
    if max_features is not None:
        sift = cv2.SIFT_create(nfeatures=max_features)
    else:
        sift = cv2.SIFT_create()
    features = {}
    total_success_pct = 0.0

    print(f"Extracting SIFT features from {len(image_paths)} images...")
    for path in image_paths:
        filename = os.path.basename(path)
        img_bgr = load_clean_image(path, flat_field_img)
        if img_bgr is None:
            continue
            
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
        if downscale_factor < 1.0:
            h, w = img.shape
            img = cv2.resize(img, (int(w * downscale_factor), int(h * downscale_factor)), interpolation=cv2.INTER_AREA)

        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        if downscale_factor < 1.0:
            for kp in keypoints:
                kp.pt = (kp.pt[0] / downscale_factor, kp.pt[1] / downscale_factor)
                
        features[filename] = {
            'path': path,
            'keypoints': keypoints,
            'descriptors': descriptors
        }
        
        target_features = (img_bgr.shape[0] * img_bgr.shape[1]) / 2000.0
        num_features = len(keypoints)
        success_pct = min(100.0, (num_features / target_features) * 100.0) if target_features > 0 else 0.0
        total_success_pct += success_pct
        
    if features:
        overall_success = total_success_pct / len(features)
        print(f" -> Feature extraction complete. Overall detection success score: {overall_success:.1f}%")
        
    return features

def compute_matches(features_dict, pairs, min_inliers=40, pbar=None):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    valid_connections = []
    for file1, file2 in pairs:
        if pbar is not None:
            pbar.update(1)
            
        kp1, des1 = features_dict[file1]['keypoints'], features_dict[file1]['descriptors']
        kp2, des2 = features_dict[file2]['keypoints'], features_dict[file2]['descriptors']

        if des1 is None or len(des1) < 2 or des2 is None or len(des2) < 2:
            continue

        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for match_group in matches:
            if len(match_group) == 2:
                m, n = match_group
                if m.distance < 0.70 * n.distance:
                    good_matches.append(m)

        candidate_count = len(good_matches)
        if candidate_count >= min_inliers:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                # DEBUG VISUALIZATION: SIFT MATCHES
                debug_dir = "./output/debug"
                os.makedirs(debug_dir, exist_ok=True)
                # img1_path = features_dict[file1]['path']
                # img2_path = features_dict[file2]['path']
                # if os.path.exists(img1_path) and os.path.exists(img2_path):
                #     img1 = cv2.imread(img1_path)
                #     img2 = cv2.imread(img2_path)
                #     match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, matchesMask=mask.ravel().tolist(), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                #     cv2.imwrite(os.path.join(debug_dir, f"match_{os.path.basename(file1)}_{os.path.basename(file2)}.jpg"), match_img)

                inlier_count = np.sum(mask)
                if inlier_count >= min_inliers:
                    inliers = src_pts[mask.ravel() == 1].reshape(-1, 2)
                    cov = np.cov(inliers.T)
                    eigenvalues, _ = np.linalg.eig(cov)
                    
                    max_eig = max(eigenvalues)
                    ratio = min(eigenvalues) / max_eig if max_eig > 0 else 0.0
                    
                    if ratio < 0.05:
                        continue 
                    
                    strength = (inlier_count / candidate_count) * 100
                    valid_connections.append({
                        'img1': file1,
                        'img2': file2,
                        'inliers': inlier_count,
                        'candidates': candidate_count,
                        'strength': strength,
                        'matches': good_matches,
                        'mask': mask.ravel().tolist(),
                        'transform': M
                    })
    return valid_connections

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
        
    # Calculate histogram
    hist = cv2.calcHist([gray_slice], [0], None, [256], [0, 256])
    
    # Normalize histogram to get probabilities
    hist = hist.ravel() / hist.sum()
    
    # Filter out zero probabilities to avoid log2(0)
    p = hist[hist > 0]
    
    # Calculate Shannon Entropy
    entropy = -np.sum(p * np.log2(p))
    return entropy

def generate_2d_entropy_heatmap(image, idx, block_size=8):
    print(" -> Generating 2D Entropy Heatmap...")
    H, W = image.shape[:2]
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    out_h = H // block_size + (1 if H % block_size else 0)
    out_w = W // block_size + (1 if W % block_size else 0)
    
    entropy_map = np.zeros((out_h, out_w), dtype=np.float32)
    
    for y in range(out_h):
        for x in range(out_w):
            y_start = y * block_size
            x_start = x * block_size
            block = gray[y_start:min(y_start+block_size, H), x_start:min(x_start+block_size, W)]
            if block.size > 0:
                entropy_map[y, x] = calculate_1d_entropy(block)
            
    # Normalize entropy (max for 8-bit is 8.0)
    entropy_vis = np.clip(entropy_map * (255.0 / 8.0), 0, 255).astype(np.uint8)
    
    entropy_vis_full = cv2.resize(entropy_vis, (W, H), interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.applyColorMap(entropy_vis_full, cv2.COLORMAP_JET)
    
    # Save side-by-side with original
    combined = np.hstack((image, heatmap))
    
    debug_dir = "./output/debug"
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, f"entropy_heatmap_{idx:02d}.jpg"), combined)
    
    return entropy_map

def find_rough_bounds_with_heatmap(entropy_map, block_size=8, threshold=5.5):
    print(f" -> Fitting optimized rectangle via entropy threshold (>{threshold})...")
    y_indices, x_indices = np.where(entropy_map >= threshold)
    
    if len(y_indices) == 0:
        print(" -> Warning: No blocks exceeded the entropy threshold. Returning None.")
        return None
        
    min_x = np.min(x_indices) * block_size
    max_x = np.max(x_indices) * block_size + block_size
    min_y = np.min(y_indices) * block_size
    max_y = np.max(y_indices) * block_size + block_size
    
    # Return as an unrotated cv2 box format for consistency
    rect = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0), (float(max_x - min_x), float(max_y - min_y)), 0.0
    return rect



def orient_and_crop(image, mapped_features, idx):
    print("\nOrienting and cropping the final composite...")

    # Generate the heatmap for debugging and bounds finding
    entropy_map = generate_2d_entropy_heatmap(image, idx, block_size=8)
    
    rect = find_rough_bounds_with_heatmap(entropy_map, block_size=8, threshold=4.5)
    
    if rect is None:
        print(" -> Warning: Entropy bounds failed. Returning uncropped image.")
        return image
    print(" -> Refining bounding box via Canny + Hough lines into Quadrilateral...")
    corners = find_quadrilateral_corners(image, rect, idx, mapped_features, expansion_factor=1.2)
    
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
    ordered_corners = fine_tune_corners_harris(image, ordered_corners, idx)
    (tl, tr, br, bl) = ordered_corners
    
    cropping_vis_path = f"./output/debug/cropping_{idx:02d}.jpg"
    import os
    if os.path.exists(cropping_vis_path):
        cropping_vis = cv2.imread(cropping_vis_path)
        cv2.polylines(cropping_vis, [np.int32(ordered_corners)], True, (0, 255, 0), 4) # Final quad in green
        for pt in ordered_corners:
            cv2.circle(cropping_vis, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), -1)
        cv2.imwrite(cropping_vis_path, cropping_vis)
    
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
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
        
    M_persp = cv2.getPerspectiveTransform(ordered_corners, dst)
    print(" -> Flattening 3D perspective to rectilinear bounds...")
    flattened_img = cv2.warpPerspective(image, M_persp, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    cropped = flattened_img
    print(f" -> Found exact rectilinear bounds: {maxWidth}x{maxHeight}")
    
    return cropped, M_persp, (0, 0)

def compute_exposure_gains(connections, features_dict, global_transforms, anchor, flat_field_img):
    print("\nCalculating Global Transformations and Exposure Compensation...")
    
    graph = collections.defaultdict(list)
    for conn in connections:
        u = conn['img1']
        v = conn['img2']
        M = conn['transform']
        
        M_3x3 = M
        M_inv = np.linalg.pinv(M_3x3)
        
        graph[u].append((v, M_inv))
        graph[v].append((u, M_3x3))

    global_gains = {anchor: 1.0}
    queue = collections.deque([anchor])

    while queue:
        curr = queue.popleft()
        G_curr = global_transforms[curr]
        gain_curr = global_gains[curr]

        for neighbor, factor in graph[curr]:
            if neighbor not in global_transforms:
                G_neighbor = G_curr @ factor
                global_transforms[neighbor] = G_neighbor
                
                img_curr = load_clean_image(features_dict[curr]['path'], flat_field_img)
                img_neigh = load_clean_image(features_dict[neighbor]['path'], flat_field_img)
                
                scale = 0.1
                small_curr = cv2.resize(img_curr, (0,0), fx=scale, fy=scale)
                small_neigh = cv2.resize(img_neigh, (0,0), fx=scale, fy=scale)
                
                S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
                S_inv = np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]])
                factor_small = S @ factor @ S_inv
                
                h, w = small_curr.shape[:2]
                warped_neigh = cv2.warpPerspective(small_neigh, factor_small, (w, h))
                
                mask_curr = (small_curr.sum(axis=2) > 0)
                mask_neigh = (warped_neigh.sum(axis=2) > 0)
                overlap = mask_curr & mask_neigh
                
                if np.any(overlap):
                    mean_curr = small_curr[overlap].mean()
                    mean_neigh = warped_neigh[overlap].mean()
                    rel_gain = mean_curr / mean_neigh if mean_neigh > 0 else 1.0
                else:
                    rel_gain = 1.0
                
                global_gains[neighbor] = gain_curr * rel_gain
                queue.append(neighbor)
                
    return global_gains

def compute_canvas_bounds(global_transforms, features_dict):
    print("Calculating final canvas dimensions...")
    corners_list = []
    for img_name, G in global_transforms.items():
        img_path = features_dict[img_name]['path']
        h, w = cv2.imread(img_path).shape[:2]
        
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, G)
        corners_list.append(transformed_corners)

    all_corners = np.concatenate(corners_list, axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    print(f" -> Master Canvas Size: {canvas_w}x{canvas_h}")

    T_shift = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    return canvas_w, canvas_h, T_shift

def blend_images_onto_canvas(global_transforms, global_gains, features_dict, flat_field_img, canvas_w, canvas_h, T_shift, idx):
    print(f"Warping, gain-compensating, and feathering {len(global_transforms)} images (Using Gaussian Blend)...")
    canvas_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    canvas_count = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

    # DEBUG VISUALIZATION: CANVAS COMPOSITION
    debug_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for img_name, G in global_transforms.items():
        img = load_clean_image(features_dict[img_name]['path'], flat_field_img)
        
        gain = global_gains[img_name]
        img = (img.astype(np.float32) * gain).clip(0, 255).astype(np.uint8)
        
        h_img, w_img = img.shape[:2]
        
        F = T_shift @ G
        warped = cv2.warpPerspective(img, F, (canvas_w, canvas_h), flags=cv2.INTER_LANCZOS4)
        
        base_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        crop_margin = 50
        cv2.rectangle(base_mask, (crop_margin, crop_margin), 
                     (max(crop_margin+1, w_img - crop_margin), 
                      max(crop_margin+1, h_img - crop_margin)), 255, -1)
        
        dist_map = cv2.distanceTransform(base_mask, cv2.DIST_L2, 3)
        
        sigma = 100.0
        gaussian_weight = 1.0 - np.exp(- (dist_map ** 2) / (2 * sigma ** 2))
        
        warped_weight = cv2.warpPerspective(gaussian_weight, F, (canvas_w, canvas_h))
        warped_weight = np.expand_dims(warped_weight, axis=2).astype(np.float32)
        
        canvas_sum += warped.astype(np.float32) * warped_weight
        canvas_count += warped_weight

        # Draw box on debug canvas
        corners = np.float32([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, F)
        cv2.polylines(debug_canvas, [np.int32(transformed_corners)], True, (0, 255, 0), 5)
        # Add a text label
        cx = int(np.mean(transformed_corners[:, 0, 0]))
        cy = int(np.mean(transformed_corners[:, 0, 1]))
        cv2.putText(debug_canvas, os.path.basename(img_name), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

    debug_dir = "./output/debug"
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, f"canvas_composition_{idx:02d}.jpg"), debug_canvas)

    canvas_count[canvas_count == 0] = 1.0 
    
    # Extract the polygons from the debug canvas for the seams visualizer
    # Actually we can just recompute or return them from a list
    tile_polygons = []
    for img_name, G in global_transforms.items():
        img_w, img_h = 0, 0
        if img_name in features_dict:
            img = load_clean_image(features_dict[img_name]['path'], flat_field_img)
            img_h, img_w = img.shape[:2]
        corners = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]).reshape(-1, 1, 2)
        F = T_shift @ G
        transformed_corners = cv2.perspectiveTransform(corners, F)
        tile_polygons.append(transformed_corners)

    return (canvas_sum / canvas_count).astype(np.uint8), tile_polygons

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
    
    # valid_mask = all_responses >= threshold
    # filtered_pts = all_mapped_pts[valid_mask]
    
    # print(f" -> Filtered out {np.sum(~valid_mask)} weak outliers (threshold: {threshold:.2f})")

    filtered_pts = np.array(all_mapped_pts)
    
    return filtered_pts

def fine_tune_corners_harris(canvas, corners, idx):
    print(" -> Fine-tuning 3D corners via CLAHE + Harris Corner Detection...")
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to boost contrast of faded corners
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    
    refined_harris = np.copy(corners)
    window = 50
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
            refined_harris[i] = [x1 + max_loc[0], y1 + max_loc[1]]
                
    return refined_harris

def find_quadrilateral_corners(canvas, rough_rect, idx, mapped_features=None, expansion_factor=1.2):
    """
    Expands the rough bounding box, creates a mask, and uses Canny + Hough
    to find the true physical edges of the image as a quadrilateral.
    """
    ((cx, cy), (w, h), angle) = rough_rect
    exp_w = w * expansion_factor
    exp_h = h * expansion_factor
    exp_rect = ((cx, cy), (exp_w, exp_h), angle)
    
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    
    blur = cv2.GaussianBlur(gray_clahe, (5, 5), 0)
    
    # Automatic Canny edge detection based on median
    v = np.median(blur)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(blur, lower, upper)
    
    # Create mask for the expanded rectangle
    mask = np.zeros(gray.shape, dtype=np.uint8)
    box = cv2.boxPoints(exp_rect)
    box = np.int32(box)
    cv2.drawContours(mask, [box], 0, 255, -1)
    
    # Apply mask
    masked_edges = cv2.bitwise_and(edged, edged, mask=mask)
    
    # Hough Lines
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
    
    # DEBUG VISUALIZATION: CROP BOUNDARIES
    debug_crop = canvas.copy()
    
    if mapped_features is not None:
        for pt in mapped_features.reshape(-1, 2):
            cv2.circle(debug_crop, (int(pt[0]), int(pt[1])), 4, (255, 0, 255), -1) # SIFT features in magenta
            
    cv2.polylines(debug_crop, [np.int32(cv2.boxPoints(rough_rect))], True, (255, 0, 0), 2) # Rough rect in blue
    cv2.polylines(debug_crop, [np.int32(cv2.boxPoints(exp_rect))], True, (0, 255, 0), 2) # Expanded rect in green
    for line in lines_p:
        x1, y1, x2, y2 = line[0]
        cv2.line(debug_crop, (x1, y1), (x2, y2), (0, 255, 255), 2) # Hough lines in yellow
    
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

    cv2.polylines(debug_crop, [np.int32(corners_out)], True, (0, 0, 255), 4) # Final quad in red
    for pt in corners_out:
        cv2.circle(debug_crop, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)

    debug_dir = "./output/debug"
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, f"cropping_{idx:02d}.jpg"), debug_crop)

    return corners_out

def stitch_mosaic(comp, features_dict, connections, output_dir, idx, flat_field_img=None):
    if not connections:
        print("No connections to stitch.")
        return

    best_match = sorted(connections, key=lambda x: x['inliers'], reverse=True)[0]
    anchor = best_match['img1']
    
    global_transforms = {anchor: np.eye(3, dtype=np.float32)}

    global_gains = compute_exposure_gains(connections, features_dict, global_transforms, anchor, flat_field_img)
    canvas_w, canvas_h, T_shift = compute_canvas_bounds(global_transforms, features_dict)
    final_canvas, tile_polygons = blend_images_onto_canvas(global_transforms, global_gains, features_dict, flat_field_img, canvas_w, canvas_h, T_shift, idx)
    mapped_features = map_all_sift_features(global_transforms, features_dict, T_shift, canvas_w, canvas_h)
    
    cropped_canvas_data = orient_and_crop(final_canvas, mapped_features, idx)
    if cropped_canvas_data is None:
        print(" -> Warning: orient_and_crop returned None.")
        return
        
    if isinstance(cropped_canvas_data, tuple):
        cropped_canvas, M_persp, (rx_min, ry_min) = cropped_canvas_data
    else:
        cropped_canvas = cropped_canvas_data
        M_persp = None
        rx_min, ry_min = 0, 0

    print("\nSaving final outputs...")
    equations = []
    if cropped_canvas is not None and cropped_canvas.size > 0:
        full_crop_path = os.path.join(output_dir, f"mosaic_{idx:02d}.tif")
        uncorrected_path = os.path.join(output_dir, f"mosaic_{idx:02d}_uncorrected.tif")
        # Save uncorrected first (will be overwritten if we did local correction, but we defer it now)
        cv2.imwrite(uncorrected_path, cropped_canvas)
        cv2.imwrite(full_crop_path, cropped_canvas)
        print(f" -> High-resolution lossless crop saved to {full_crop_path}")
        
        try:
            from point_cloud_aspect_ratio import extract_aspect_ratio_equations
            print(f" -> Analyzing film grain and SIFT point cloud to extract aspect ratio equations for mosaic {idx}...")
            c_h, c_w = cropped_canvas.shape[:2]
            equations = extract_aspect_ratio_equations(features_dict, connections, global_transforms, T_shift, M_persp, idx, c_w, c_h)
            
            import json
            
            # The equations MUST be written to disk to pass them out of the multiprocessing pool back to the batch processor
            os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)
            with open(os.path.join(output_dir, "debug", f"equations_{idx:02d}.json"), "w") as f:
                json.dump(equations, f, indent=4)
                
            ENABLE_DEBUG_VISUALIZATIONS = False
        except ImportError as e:
            print(f" -> Warning: point_cloud_aspect_ratio module not found ({e}). Skipping equation extraction.")
        except Exception as e:
            print(f" -> Warning: Equation extraction failed: {e}")
            
        if M_persp is not None:
            # Generate the seams visualization
            seams_vis = cropped_canvas.copy()
            # Create a transparent overlay for polygons
            overlay = np.zeros_like(seams_vis)
            
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), 
                (255, 255, 0), (255, 0, 255), (0, 255, 255),
                (255, 128, 0), (128, 0, 255), (0, 255, 128)
            ]
            
            for i, poly in enumerate(tile_polygons):
                # Transform the polygon from Master Canvas space to Final Flattened space
                flat_poly = cv2.perspectiveTransform(poly, M_persp)
                # Shift by the rx_min, ry_min entropy crop
                flat_poly[:, 0, 0] -= rx_min
                flat_poly[:, 0, 1] -= ry_min
                
                color = colors[i % len(colors)]
                cv2.polylines(seams_vis, [np.int32(flat_poly)], True, color, 2)
                cv2.fillPoly(overlay, [np.int32(flat_poly)], color)
            
            # Blend overlay with 20% opacity
            seams_vis = cv2.addWeighted(seams_vis, 1.0, overlay, 0.2, 0)
            
            if ENABLE_DEBUG_VISUALIZATIONS:
                debug_dir = "./output/debug"
                seams_path = os.path.join(debug_dir, f"seams_{idx:02d}_uncorrected.jpg")
                cv2.imwrite(seams_path, seams_vis)
                print(f" -> Uncorrected assembly seam visualization saved to {seams_path}")
            
    else:
        print(" -> Warning: Cropped canvas is empty, skipping output.")
        
    return equations