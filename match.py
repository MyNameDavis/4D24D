import cv2
import numpy as np
import collections
import json
with open("param.json") as _f:
    PARAMS = json.load(_f)
from SIFT import load_clean_image

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

def cluster_film_scenes(features_dict, min_inliers, window_size=5):
    filenames = sorted(list(features_dict.keys()))
    N = len(filenames)
    
    print(f"\n--- Phase 1: Local Sliding Window Matching ---")
    W = window_size
    local_pairs = set()
    for i in range(N):
        for j in range(i + 1, min(i + 1 + W, N)):
            local_pairs.add((filenames[i], filenames[j]))
            
    print(f"Evaluating {len(local_pairs)} local sequential pairs...")
    local_connections = compute_matches(features_dict, local_pairs, min_inliers)
    
    graph = collections.defaultdict(list)
    for conn in local_connections:
        graph[conn['img1']].append(conn['img2'])
        graph[conn['img2']].append(conn['img1'])
        
    components = []
    visited = set()
    for img in filenames:
        if img not in visited:
            comp = set()
            q = collections.deque([img])
            while q:
                curr = q.popleft()
                if curr not in visited:
                    visited.add(curr)
                    comp.add(curr)
                    for neighbor in graph[curr]:
                        if neighbor not in visited:
                            q.append(neighbor)
            components.append(list(comp))
            
    print(f" -> Found {len(components)} initial components before outlier recovery.")
    
    print(f"\n--- Phase 2: Outlier Recovery (Component Merging) ---")
    cross_pairs = set()
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            comp_A = components[i]
            comp_B = components[j]
            
            for a in comp_A:
                for b in comp_B:
                    pair = tuple(sorted([a, b]))
                    if pair not in local_pairs:
                        cross_pairs.add(pair)
                        
    print(f"Evaluating {len(cross_pairs)} cross-component pairs to find outliers...")
    outlier_connections = compute_matches(features_dict, cross_pairs, min_inliers)
    
    all_connections = local_connections + outlier_connections
    
    final_graph = collections.defaultdict(list)
    for conn in all_connections:
        final_graph[conn['img1']].append(conn['img2'])
        final_graph[conn['img2']].append(conn['img1'])
        
    final_components = []
    visited = set()
    for img in filenames:
        if img not in visited:
            comp = set()
            q = collections.deque([img])
            while q:
                curr = q.popleft()
                if curr not in visited:
                    visited.add(curr)
                    comp.add(curr)
                    for neighbor in final_graph[curr]:
                        if neighbor not in visited:
                            q.append(neighbor)
            final_components.append(list(comp))

    print(f" -> Matching complete! Consolidated into {len(final_components)} distinct mosaics.")
    return final_components, all_connections

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
                
                import gc
                del img_curr
                del img_neigh
                gc.collect()
                
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


    for img_name, G in global_transforms.items():
        img = load_clean_image(features_dict[img_name]['path'], flat_field_img)
        
        gain = global_gains[img_name]
        img = (img.astype(np.float32) * gain).clip(0, 255).astype(np.uint8)
        
        h_img, w_img = img.shape[:2]
        
        F = T_shift @ G
        warped = cv2.warpPerspective(img, F, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)
        
        base_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        crop_margin = PARAMS['CROP_MARGIN']
        cv2.rectangle(base_mask, (crop_margin, crop_margin), 
                     (max(crop_margin+1, w_img - crop_margin), 
                      max(crop_margin+1, h_img - crop_margin)), 255, -1)
        
        dist_map = cv2.distanceTransform(base_mask, cv2.DIST_L2, 3)
        
        sigma = PARAMS['GAUSSIAN_SIGMA']
        gaussian_weight = (1.0 - np.exp(- (dist_map ** 2) / (2 * sigma ** 2))).astype(np.float32)
        
        warped_weight = cv2.warpPerspective(gaussian_weight, F, (canvas_w, canvas_h))
        warped_weight = np.expand_dims(warped_weight, axis=2)
        
        warped_f32 = warped.astype(np.float32)
        warped_f32 *= warped_weight
        canvas_sum += warped_f32
        canvas_count += warped_weight

        
        # Free memory to prevent macOS swapping!
        del img, warped, base_mask, dist_map, gaussian_weight, warped_weight, warped_f32
        import gc
        gc.collect()


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