import os
import cv2
import numpy as np
import collections
from tqdm import tqdm
from scipy.optimize import minimize

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

            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

            if M is not None:
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

def fine_tune_bounds_with_entropy(image, rx_min, rx_max, ry_min, ry_max, threshold=7, max_steps=5):
    print(f" -> Fine-tuning boundaries via Shannon Entropy (threshold={threshold})...")
    H, W = image.shape[:2]
    
    # Helper to safely slice and calculate entropy
    def get_h_entropy(y, x_min, x_max):
        if y < 0 or y >= H or x_min >= x_max: return 0.0
        slice_1d = image[y:y+1, x_min:x_max]
        return calculate_1d_entropy(slice_1d)
        
    def get_v_entropy(x, y_min, y_max):
        if x < 0 or x >= W or y_min >= y_max: return 0.0
        slice_1d = image[y_min:y_max, x:x+1]
        return calculate_1d_entropy(slice_1d)

    # 1. Top Face (ry_min)
    steps = 0
    while steps < max_steps:
        ent = get_h_entropy(ry_min, rx_min, rx_max)
        if ent < threshold:
            ry_min += 1 # Shrink
        else:
            # High entropy, try a grow
            ent_grow = get_h_entropy(ry_min - 1, rx_min, rx_max)
            if ent_grow >= threshold and ry_min > 0:
                ry_min -= 1 # Grow
            else:
                break # Grow is low entropy, we are done
        steps += 1

    # 2. Bottom Face (ry_max)
    steps = 0
    while steps < max_steps:
        ent = get_h_entropy(ry_max - 1, rx_min, rx_max)
        if ent < threshold:
            ry_max -= 1 # Shrink
        else:
            ent_grow = get_h_entropy(ry_max, rx_min, rx_max)
            if ent_grow >= threshold and ry_max < H:
                ry_max += 1 # Grow
            else:
                break
        steps += 1

    # 3. Left Face (rx_min)
    steps = 0
    while steps < max_steps:
        ent = get_v_entropy(rx_min, ry_min, ry_max)
        if ent < threshold:
            rx_min += 1 # Shrink
        else:
            ent_grow = get_v_entropy(rx_min - 1, ry_min, ry_max)
            if ent_grow >= threshold and rx_min > 0:
                rx_min -= 1 # Grow
            else:
                break
        steps += 1

    # 4. Right Face (rx_max)
    steps = 0
    while steps < max_steps:
        ent = get_v_entropy(rx_max - 1, ry_min, ry_max)
        if ent < threshold:
            rx_max -= 1 # Shrink
        else:
            ent_grow = get_v_entropy(rx_max, ry_min, ry_max)
            if ent_grow >= threshold and rx_max < W:
                rx_max += 1 # Grow
            else:
                break
        steps += 1

    return rx_min, rx_max, ry_min, ry_max

def orient_and_crop(image, mapped_features):
    print("\nOrienting and cropping the final composite...")

    if mapped_features is None or len(mapped_features) == 0:
        print(" -> Warning: No mapped features found. Returning uncropped image.")
        return image

    H, W = image.shape[:2]
    
    print(" -> Finding extreme boundary points...")
    extreme_pts = find_extreme_points(mapped_features, iterations=5)
    
    print(" -> Fitting optimized rectangle via least squares...")
    rect = fit_rectangle_least_squares(extreme_pts, W / 2.0, H / 2.0)
    
    if rect is None:
        print(" -> Warning: Least squares fit failed. Returning uncropped image.")
        return image
        
    print(" -> Refining bounding box via Canny + Hough lines...")
    final_rect = refine_box_with_hough(image, rect, expansion_factor=1.2)
    
    (cx, cy), (w, h), angle = final_rect
    
    if w < h:
        angle += 90
        w, h = h, w
        
    print(f" -> Image leveled (rotated by {angle:.2f} degrees)")

    M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated_img = cv2.warpAffine(image, M_rot, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    rx_min = max(0, int(cx - w / 2))
    ry_min = max(0, int(cy - h / 2))
    rx_max = min(W, int(cx + w / 2))
    ry_max = min(H, int(cy + h / 2))

    rx_min, rx_max, ry_min, ry_max = fine_tune_bounds_with_entropy(
        rotated_img, rx_min, rx_max, ry_min, ry_max, threshold=7, max_steps=5
    )

    cropped = rotated_img[ry_min:ry_max, rx_min:rx_max]
    print(f" -> Found exact rectilinear bounds: {rx_max-rx_min}x{ry_max-ry_min}")
    
    return cropped

def compute_exposure_gains(connections, features_dict, global_transforms, anchor, flat_field_img):
    print("\nCalculating Global Transformations and Exposure Compensation...")
    
    graph = collections.defaultdict(list)
    for conn in connections:
        u = conn['img1']
        v = conn['img2']
        M = conn['transform']
        
        M_3x3 = np.vstack((M, [0, 0, 1]))
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
                warped_neigh = cv2.warpAffine(small_neigh, factor_small[:2, :], (w, h))
                
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

def blend_images_onto_canvas(global_transforms, global_gains, features_dict, flat_field_img, canvas_w, canvas_h, T_shift):
    print(f"Warping, gain-compensating, and feathering {len(global_transforms)} images (Using Gaussian Blend)...")
    canvas_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    canvas_count = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

    for img_name, G in tqdm(global_transforms.items(), desc="Blending Images", leave=False):
        img = load_clean_image(features_dict[img_name]['path'], flat_field_img)
        
        gain = global_gains[img_name]
        img = (img.astype(np.float32) * gain).clip(0, 255).astype(np.uint8)
        
        h_img, w_img = img.shape[:2]
        
        F = T_shift @ G
        F_2x3 = F[:2, :] 
        warped = cv2.warpAffine(img, F_2x3, (canvas_w, canvas_h), flags=cv2.INTER_LANCZOS4)
        
        base_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.rectangle(base_mask, (1, 1), (w_img - 2, h_img - 2), 255, -1)
        
        dist_map = cv2.distanceTransform(base_mask, cv2.DIST_L2, 3)
        
        sigma = 100.0
        gaussian_weight = 1.0 - np.exp(- (dist_map ** 2) / (2 * sigma ** 2))
        
        warped_weight = cv2.warpAffine(gaussian_weight, F_2x3, (canvas_w, canvas_h))
        warped_weight = np.expand_dims(warped_weight, axis=2).astype(np.float32)
        
        canvas_sum += warped.astype(np.float32) * warped_weight
        canvas_count += warped_weight

    canvas_count[canvas_count == 0] = 1.0 
    return (canvas_sum / canvas_count).astype(np.uint8)

def map_all_sift_features(global_transforms, features_dict, T_shift):
    """
    Transforms all SIFT keypoints from individual image spaces into the final canvas space
    and filters out the 5% weakest outliers based on response (assuming normal distribution).
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

    all_mapped_pts = np.array(all_mapped_pts)
    all_responses = np.array(all_responses)
    
    # Filter out 5% weakest outliers assuming normal distribution (z < -1.645)
    mean_resp = np.mean(all_responses)
    std_resp = np.std(all_responses)
    threshold = mean_resp - 1.645 * std_resp
    
    valid_mask = all_responses >= threshold
    filtered_pts = all_mapped_pts[valid_mask]
    
    print(f" -> Filtered out {len(all_responses) - len(filtered_pts)} weak outliers (threshold: {threshold:.2f})")
    
    return filtered_pts

def find_extreme_points(pts, iterations=10):
    """
    Iteratively finds the centroid of the cluster, then finds the farthest points 
    in the +/- X and Y directions, removes them, and repeats.
    Returns the accumulated extreme boundary points.
    """
    if pts is None or len(pts) == 0:
        return []
        
    pts_2d = pts.reshape(-1, 2)
    remaining_pts = pts_2d.copy()
    extreme_points = []
    
    for _ in range(iterations):
        if len(remaining_pts) < 4:
            break
            
        # Find centroid
        centroid = np.mean(remaining_pts, axis=0)
        
        # Calculate distances along axes
        x_dist = remaining_pts[:, 0] - centroid[0]
        y_dist = remaining_pts[:, 1] - centroid[1]
        
        # Find indices of furthest points
        idx_max_x = np.argmax(x_dist)
        idx_min_x = np.argmin(x_dist)
        idx_max_y = np.argmax(y_dist)
        idx_min_y = np.argmin(y_dist)
        
        # Collect unique extreme points for this iteration
        iter_extremes = set([idx_max_x, idx_min_x, idx_max_y, idx_min_y])
        
        for idx in iter_extremes:
            extreme_points.append(remaining_pts[idx])
            
        # Remove these points from remaining set
        mask = np.ones(len(remaining_pts), dtype=bool)
        for idx in iter_extremes:
            mask[idx] = False
        remaining_pts = remaining_pts[mask]
        
    return np.array(extreme_points).reshape(-1, 1, 2)

def fit_rectangle_least_squares(pts, cx, cy, interior_weight=0.1):
    """
    Fits a rectangle to a set of extreme points using a non-linear least squares solver,
    fixing the center of the rectangle to (cx, cy).
    Uses an asymmetric loss function to de-weight points that fall inside the box.
    """
    if pts is None or len(pts) < 4:
        return None
        
    pts_2d = pts.reshape(-1, 2)
    
    # Get a reasonable initial guess for w, h, and theta using minAreaRect
    init_rect = cv2.minAreaRect(np.ascontiguousarray(pts_2d, dtype=np.float32))
    init_w, init_h = init_rect[1]
    init_theta = init_rect[2] * np.pi / 180.0
    
    def loss_func(params):
        w, h, theta = params
        
        # Translate points to origin (fixed center)
        x = pts_2d[:, 0] - cx
        y = pts_2d[:, 1] - cy
        
        # Rotate points by -theta to axis-align them
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        
        x_rot = x * cos_t - y * sin_t
        y_rot = x * sin_t + y * cos_t
        
        # Calculate signed distance to the edges
        # Positive means outside the box, negative means inside
        dx_signed = np.abs(x_rot) - w / 2.0
        dy_signed = np.abs(y_rot) - h / 2.0
        
        # Signed Distance Function (SDF) approximation
        sdf = np.maximum(dx_signed, dy_signed)
        
        # Asymmetric weighting: 
        # Strong penalty (1.0) for points outside the box (sdf > 0)
        # Weak penalty (interior_weight) for points inside the box (sdf <= 0)
        weights = np.where(sdf > 0, 1.0, interior_weight)
        
        # Return weighted sum of squared distances
        return np.sum(weights * (sdf ** 2))
        
    initial_guess = [init_w, init_h, init_theta]
    bounds = ((1.0, None), (1.0, None), (None, None)) # Width and height must be positive
    
    result = minimize(loss_func, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    if result.success:
        opt_w, opt_h, opt_theta = result.x
        opt_theta_deg = opt_theta * 180.0 / np.pi
        return ((cx, cy), (opt_w, opt_h), opt_theta_deg)
    else:
        print(" -> Optimization failed, falling back to initial guess.")
        return ((cx, cy), (init_w, init_h), init_rect[2])

def refine_box_with_hough(canvas, rough_rect, expansion_factor=1.2):
    """
    Expands the rough bounding box, creates a mask, and uses Canny + Hough
    to find the true physical edges of the image.
    """
    ((cx, cy), (w, h), angle) = rough_rect
    exp_w = w * expansion_factor
    exp_h = h * expansion_factor
    exp_rect = ((cx, cy), (exp_w, exp_h), angle)
    
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # This massively boosts local contrast, making faint edges (like a bright sky against a bright background)
    # pop out for the edge detector to see.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)
    
    if lines is not None and len(lines) > 0:
        endpoints = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            endpoints.append((x1, y1))
            endpoints.append((x2, y2))
            
        endpoints = np.float32(endpoints)
        final_rect = cv2.minAreaRect(endpoints)
        return final_rect
    else:
        print(" -> Warning: Hough transform found no lines, falling back to expanded rough box.")
        return exp_rect

def stitch_mosaic(comp, features_dict, connections, output_dir, idx, flat_field_img=None):
    if not connections:
        print("No connections to stitch.")
        return

    best_match = sorted(connections, key=lambda x: x['inliers'], reverse=True)[0]
    anchor = best_match['img1']
    
    global_transforms = {anchor: np.eye(3, dtype=np.float32)}

    global_gains = compute_exposure_gains(connections, features_dict, global_transforms, anchor, flat_field_img)
    canvas_w, canvas_h, T_shift = compute_canvas_bounds(global_transforms, features_dict)
    final_canvas = blend_images_onto_canvas(global_transforms, global_gains, features_dict, flat_field_img, canvas_w, canvas_h, T_shift)
    mapped_features = map_all_sift_features(global_transforms, features_dict, T_shift)
    cropped_canvas = orient_and_crop(final_canvas, mapped_features)

    print("\nSaving final outputs...")
    if cropped_canvas is not None and cropped_canvas.size > 0:
        full_crop_path = os.path.join(output_dir, f"mosaic_{idx:02d}.tif")
        cv2.imwrite(full_crop_path, cropped_canvas)
        print(f" -> High-resolution lossless crop saved to {full_crop_path}")
    else:
        print(" -> Warning: Cropped canvas is empty, skipping output.")