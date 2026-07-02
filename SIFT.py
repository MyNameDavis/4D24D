import os
import cv2
import numpy as np
import collections
from tqdm import tqdm

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

def orient_and_crop(image, mapped_inliers):
    print("\nOrienting and cropping the final composite...")

    if mapped_inliers is None or len(mapped_inliers) == 0:
        print(" -> Warning: No mapped inliers found. Returning uncropped image.")
        return image

    H, W = image.shape[:2]
    
    inlier_pts = np.array(mapped_inliers)
    ix_min, iy_min = np.percentile(inlier_pts, 5, axis=0).ravel()
    ix_max, iy_max = np.percentile(inlier_pts, 95, axis=0).ravel()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(" -> Warning: No contours found. Returning uncropped image.")
        return image

    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        if (x <= ix_min + 50) and (x + w >= ix_max - 50) and \
           (y <= iy_min + 50) and (y + h >= iy_max - 50):
            valid_contours.append(cnt)
            
    if not valid_contours:
        print(" -> Warning: No contours enclosed the SIFT inliers. Returning uncropped image.")
        return image
        
    def bbox_area(cnt):
        _, _, w, h = cv2.boundingRect(cnt)
        return w * h
        
    best_cnt = min(valid_contours, key=bbox_area)
    
    rect = cv2.minAreaRect(best_cnt)
    (cx, cy), (w, h), angle = rect
    
    w -= 8
    h -= 8
    
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

def map_sift_inliers(connections, global_transforms, features_dict, T_shift):
    print("Mapping SIFT inliers to canvas space...")
    all_mapped_inliers = []
    for conn in connections:
        img1 = conn['img1']
        if img1 not in global_transforms: continue
        
        kp1 = features_dict[img1]['keypoints']
        matches = conn['matches']
        mask = conn['mask']
        
        pts = []
        for i, m in enumerate(matches):
            if mask[i]:
                pts.append(kp1[m.queryIdx].pt)
                
        if pts:
            pts = np.float32(pts).reshape(-1, 1, 2)
            F = T_shift @ global_transforms[img1]
            mapped = cv2.perspectiveTransform(pts, F)
            all_mapped_inliers.append(mapped)

    if all_mapped_inliers:
        return np.concatenate(all_mapped_inliers, axis=0)
    return None

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
    mapped_inliers = map_sift_inliers(connections, global_transforms, features_dict, T_shift)
    cropped_canvas = orient_and_crop(final_canvas, mapped_inliers)

    print("\nSaving final outputs...")
    if cropped_canvas is not None and cropped_canvas.size > 0:
        full_crop_path = os.path.join(output_dir, f"mosaic_{idx:02d}.tif")
        cv2.imwrite(full_crop_path, cropped_canvas)
        print(f" -> High-resolution lossless crop saved to {full_crop_path}")
    else:
        print(" -> Warning: Cropped canvas is empty, skipping output.")