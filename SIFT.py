import os
import cv2
import glob
import itertools
import collections
import numpy as np

def load_clean_image(path, flat_field_img=None):
    """
    Loads an image and immediately applies Flat Field Correction if available.
    This divides out the static vignetting and backlight falloff before any math happens.
    """
    img = cv2.imread(path)
    if img is None:
        return None
        
    if flat_field_img is not None:
        # Convert flat field to a 0.0 - 1.0 float map
        ff = flat_field_img.astype(np.float32) / 255.0
        ff[ff == 0] = 1.0 # Prevent division by zero
        
        # Divide the image by the flat field map to normalize illumination
        img = (img.astype(np.float32) / ff).clip(0, 255).astype(np.uint8)
        
    return img

def extract_sift_features(image_paths, flat_field_img=None, downscale_factor=0.5):
    """
    Loads images, optionally downscales them for speed, and computes SIFT features.
    Downscaling during the discovery phase saves massive amounts of RAM and time.
    """
    sift = cv2.SIFT_create()
    features = {}
    total_success_pct = 0.0

    print(f"Extracting SIFT features from {len(image_paths)} images...")
    for path in image_paths:
        filename = os.path.basename(path)
        
        # Load and dynamically apply flat-field correction
        img_bgr = load_clean_image(path, flat_field_img)
        if img_bgr is None:
            continue
            
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
        # Downscale for faster feature extraction
        if downscale_factor < 1.0:
            h, w = img.shape
            img = cv2.resize(img, (int(w * downscale_factor), int(h * downscale_factor)), interpolation=cv2.INTER_AREA)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        # Scale keypoint coordinates back to original image size
        if downscale_factor < 1.0:
            for kp in keypoints:
                kp.pt = (kp.pt[0] / downscale_factor, kp.pt[1] / downscale_factor)
                
        features[filename] = {
            'path': path,
            'keypoints': keypoints,
            'descriptors': descriptors
        }
        
        # Calculate feature detection success metric
        target_features = (img_bgr.shape[0] * img_bgr.shape[1]) / 2000.0
        num_features = len(keypoints)
        success_pct = min(100.0, (num_features / target_features) * 100.0) if target_features > 0 else 0.0
        total_success_pct += success_pct
        
    if features:
        overall_success = total_success_pct / len(features)
        print(f" -> Feature extraction complete. Overall detection success score: {overall_success:.1f}%")
        
    return features

def match_features(features_dict, min_inliers=10):
    """
    Compares all image pairs (O(N^2)) to find overlapping tiles using FLANN and RANSAC.
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    filenames = list(features_dict.keys())
    pairs = list(itertools.combinations(filenames, 2))
    
    valid_connections = []

    print(f"\nComparing {len(pairs)} possible pairs to find overlaps...")

    for file1, file2 in pairs:
        kp1, des1 = features_dict[file1]['keypoints'], features_dict[file1]['descriptors']
        kp2, des2 = features_dict[file2]['keypoints'], features_dict[file2]['descriptors']

        if des1 is None or len(des1) < 2 or des2 is None or len(des2) < 2:
            continue

        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for match_group in matches:
            if len(match_group) == 2:
                m, n = match_group
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        candidate_count = len(good_matches)
        if candidate_count >= min_inliers:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

            if M is not None:
                inlier_count = np.sum(mask)
                if inlier_count >= min_inliers:
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

    if valid_connections:
        best_strength_per_tile = collections.defaultdict(float)
        
        for conn in valid_connections:
            img1 = conn['img1']
            img2 = conn['img2']
            strength = conn['strength']
            
            if strength > best_strength_per_tile[img1]: best_strength_per_tile[img1] = strength
            if strength > best_strength_per_tile[img2]: best_strength_per_tile[img2] = strength
                
        overall_match_score = sum(best_strength_per_tile.values()) / len(best_strength_per_tile)
        print(f" -> Matching complete. Cumulative overall match strength: {overall_match_score:.1f}% (across {len(best_strength_per_tile)} connected tiles)")
    else:
        print(" -> Matching complete. No valid overlaps found.")

    return valid_connections

def orient_and_crop(image, target_ratio=1.5):
    """
    Finds the minimum bounding rect of the stitched canvas to level the rotation,
    then finds the TRUE largest inscribed target_ratio rectangle (any position,
    not just centered), ignoring thin interior seam/blend artifacts that aren't
    actually missing image data.
    """
    print("\nOrienting and cropping the final composite...")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    (cx, cy), (w, h), angle = rect

    if w < h:
        angle += 90

    # IMPORTANT: build a SOLID filled mask from just the outer contour shape,
    # rather than using the raw thresholded mask directly. The raw mask has
    # thin interior holes/seams from feather-blending and Lanczos warping
    # between tiles - these are stitching artifacts, not real gaps in the
    # photographed content. Requiring a candidate crop rectangle to avoid
    # every one of these hairline seams was collapsing the search down to
    # tiny clean patches and throwing away huge amounts of good image data.
    # Filling the outer silhouette keeps the search honest about the REAL
    # boundary (the jagged mosaic edge) while ignoring interior noise.
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Use Lanczos-4 for the leveling rotation to prevent detail loss
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LANCZOS4)
    rotated_mask = cv2.warpAffine(clean_mask, M, (image.shape[1], image.shape[0]))
    # Re-binarize after rotation (interpolation can introduce gray edge pixels)
    _, rotated_mask = cv2.threshold(rotated_mask, 127, 255, cv2.THRESH_BINARY)

    print(f" -> Image leveled (rotated by {angle:.2f} degrees)")
    print(" -> Searching for the true largest inscribed rectangle (exact, all positions)...")

    H, W = rotated_mask.shape
    binary = (rotated_mask > 0).astype(np.uint8)
    # cv2.integral needs uint8/float32/float64 input; request a 32-bit signed
    # integer accumulator via sdepth so large-canvas sums don't overflow.
    integral = cv2.integral(binary, sdepth=cv2.CV_32S)  # shape (H+1, W+1)

    def find_matches(rw, rh):
        """
        Vectorized check: for a rectangle of size (rw, rh), compute the filled-
        pixel sum for EVERY possible top-left origin at once using the integral
        image, then return the (y, x) origins where the rectangle is 100% filled.
        """
        if rw < 1 or rh < 1 or rw > W or rh > H:
            return None
        br = integral[rh:H + 1, rw:W + 1]
        tr = integral[0:H - rh + 1, rw:W + 1]
        bl = integral[rh:H + 1, 0:W - rw + 1]
        tl = integral[0:H - rh + 1, 0:W - rw + 1]
        sums = br - tr - bl + tl
        target = rw * rh
        matches = np.argwhere(sums == target)  # rows of (y, x) top-left origins
        return matches if matches.size > 0 else None

    # Binary search the maximum width (height derived from target_ratio) for
    # which AT LEAST ONE fully-filled placement exists anywhere in the canvas.
    low, high = 1, W
    best_w, best_h, best_matches = 0, 0, None

    while low <= high:
        mid_w = (low + high) // 2
        mid_h = int(mid_w / target_ratio)

        matches = find_matches(mid_w, mid_h)
        if matches is not None:
            best_w, best_h, best_matches = mid_w, mid_h, matches
            low = mid_w + 1
        else:
            high = mid_w - 1

    if best_matches is not None:
        # Multiple valid positions can tie at the max size - prefer the one
        # whose center sits closest to the mask centroid, for a natural-
        # looking crop rather than one jammed into a corner.
        centers_y = best_matches[:, 0] + best_h / 2.0
        centers_x = best_matches[:, 1] + best_w / 2.0
        dist2 = (centers_x - cx) ** 2 + (centers_y - cy) ** 2
        idx = np.argmin(dist2)
        by, bx = best_matches[idx]

        cropped_img = rotated_img[by:by + best_h, bx:bx + best_w]
        print(f" -> Found largest inscribed rect: {best_w}x{best_h} at ({bx},{by}), ratio {target_ratio}:1")
        return cropped_img
    else:
        print(" -> Warning: Inscribed crop failed. Returning uncropped rotated image.")
        return rotated_img

def overlay_all_images(features_dict, connections, output_dir, flat_field_img=None):
    """
    Uses a Graph Spanning Tree approach to chain the pairwise transforms together.
    Also calculates Gain Compensation multipliers based on overlapping regions 
    to normalize global exposure flicker before feather blending.
    """
    if not connections:
        print("No connections to stitch.")
        return

    print("\nCalculating Global Transformations and Exposure Compensation...")
    
    graph = collections.defaultdict(list)
    for conn in connections:
        u = conn['img1']
        v = conn['img2']
        M = conn['transform'] # maps u coordinates to v coordinates
        
        M_3x3 = np.vstack((M, [0, 0, 1]))
        M_inv = np.linalg.pinv(M_3x3)
        
        graph[u].append((v, M_inv))
        graph[v].append((u, M_3x3))

    best_match = sorted(connections, key=lambda x: x['inliers'], reverse=True)[0]
    anchor = best_match['img1']

    # We track both the global geometry matrix (G) and the global exposure gain
    global_transforms = {anchor: np.eye(3, dtype=np.float32)}
    global_gains = {anchor: 1.0}
    
    queue = collections.deque([anchor])

    # Breadth-First Search to map every image back to the Anchor
    while queue:
        curr = queue.popleft()
        G_curr = global_transforms[curr]
        gain_curr = global_gains[curr]

        for neighbor, factor in graph[curr]:
            if neighbor not in global_transforms:
                # 1. Geometry mapping
                G_neighbor = G_curr @ factor
                global_transforms[neighbor] = G_neighbor
                
                # 2. Exposure mapping (Gain Compensation)
                # Load images (with static vignetting removed)
                img_curr = load_clean_image(features_dict[curr]['path'], flat_field_img)
                img_neigh = load_clean_image(features_dict[neighbor]['path'], flat_field_img)
                
                # Downscale heavily to calculate overlap brightness very quickly
                scale = 0.1
                small_curr = cv2.resize(img_curr, (0,0), fx=scale, fy=scale)
                small_neigh = cv2.resize(img_neigh, (0,0), fx=scale, fy=scale)
                
                # Adjust the relative mapping matrix (factor) to match the thumbnail scale
                S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
                S_inv = np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]])
                factor_small = S @ factor @ S_inv
                
                # Warp the neighbor thumbnail into the current thumbnail's space
                h, w = small_curr.shape[:2]
                warped_neigh = cv2.warpAffine(small_neigh, factor_small[:2, :], (w, h))
                
                # Find the intersection where both images have valid pixel data
                mask_curr = (small_curr.sum(axis=2) > 0)
                mask_neigh = (warped_neigh.sum(axis=2) > 0)
                overlap = mask_curr & mask_neigh
                
                if np.any(overlap):
                    # Compare the brightness of the exact same physical spot
                    mean_curr = small_curr[overlap].mean()
                    mean_neigh = warped_neigh[overlap].mean()
                    rel_gain = mean_curr / mean_neigh if mean_neigh > 0 else 1.0
                else:
                    rel_gain = 1.0
                
                # Multiply the relative gain up the chain
                global_gains[neighbor] = gain_curr * rel_gain
                queue.append(neighbor)

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

    canvas_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    canvas_count = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

    print(f"Warping, gain-compensating, and feathering {len(global_transforms)} images (Using Lanczos-4)...")
    for img_name, G in global_transforms.items():
        # Load the image with Flat Field correction already applied
        img = load_clean_image(features_dict[img_name]['path'], flat_field_img)
        
        # Apply the mathematically derived Exposure Gain for this specific image
        # This completely normalizes global brightness across the entire grid
        gain = global_gains[img_name]
        img = (img.astype(np.float32) * gain).clip(0, 255).astype(np.uint8)
        
        h_img, w_img = img.shape[:2]
        
        F = T_shift @ G
        F_2x3 = F[:2, :] 

        # USE INTER_LANCZOS4 TO PRESERVE HIGH FREQUENCY GRAIN DETAIL DURING ROTATION/SUB-PIXEL SHIFTS
        warped = cv2.warpAffine(img, F_2x3, (canvas_w, canvas_h), flags=cv2.INTER_LANCZOS4)
        
        # Feathering logic
        base_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.rectangle(base_mask, (1, 1), (w_img - 2, h_img - 2), 255, -1)
        
        dist_map = cv2.distanceTransform(base_mask, cv2.DIST_L2, 3)
        dist_map = cv2.normalize(dist_map, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        warped_weight = cv2.warpAffine(dist_map, F_2x3, (canvas_w, canvas_h))
        warped_weight = np.expand_dims(warped_weight, axis=2)
        
        canvas_sum += warped.astype(np.float32) * warped_weight
        canvas_count += warped_weight

    canvas_count[canvas_count == 0] = 1.0 
    final_canvas = (canvas_sum / canvas_count).astype(np.uint8)

    cropped_canvas = orient_and_crop(final_canvas, target_ratio=1.5)

    print("\nSaving final outputs...")
    # 8. Save FULL RESOLUTION, Lossless TIFFs to completely preserve grain
    full_raw_path = os.path.join(output_dir, "sift_full_overlay_raw_highres.tif")
    cv2.imwrite(full_raw_path, final_canvas)
    
    if cropped_canvas is not None and cropped_canvas.size > 0:
        full_crop_path = os.path.join(output_dir, "sift_final_cropped_highres.tif")
        cv2.imwrite(full_crop_path, cropped_canvas)
        print(f" -> High-resolution lossless crop saved to {full_crop_path}")

    # 9. Also save downscaled JPEG previews for easy viewing
    max_dim = 2500.0
    scale = min(1.0, max_dim / max(canvas_w, canvas_h))
    
    preview_w = int(canvas_w * scale)
    preview_h = int(canvas_h * scale)
    preview_img = cv2.resize(final_canvas, (preview_w, preview_h), interpolation=cv2.INTER_AREA)

    preview_raw_path = os.path.join(output_dir, "sift_full_overlay_raw_preview.jpg")
    cv2.imwrite(preview_raw_path, preview_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    
    if cropped_canvas is not None and cropped_canvas.size > 0:
        crop_h, crop_w = cropped_canvas.shape[:2]
        crop_scale = min(1.0, max_dim / max(crop_w, crop_h))
        crop_preview = cv2.resize(cropped_canvas, (int(crop_w * crop_scale), int(crop_h * crop_scale)), interpolation=cv2.INTER_AREA)
        
        preview_final_path = os.path.join(output_dir, "sift_final_cropped_preview.jpg")
        cv2.imwrite(preview_final_path, crop_preview, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        print(f" -> Downscaled JPEG preview saved to {preview_final_path}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    INPUT_DIR = "./image_seg_dataset"
    OUTPUT_DIR = "./"
    DOWNSCALE_FACTOR = 0.5 
    MIN_INLIERS = 10 
    # ---------------------

    # 1. Search for a flat field reference first (to avoid stitching it!)
    ff_path = os.path.join(INPUT_DIR, "flat_field_reference.jpg")
    flat_field_img = None
    if os.path.exists(ff_path):
        print("Flat Field Reference detected! Applying illumination normalization...")
        flat_field_img = cv2.imread(ff_path)

    # 2. Gather only the actual film tiles
    search_path = os.path.join(INPUT_DIR, "*.jpg")
    image_paths = sorted([p for p in glob.glob(search_path) if "flat_field_reference" not in p])

    if len(image_paths) < 2:
        print(f"Need at least 2 images to match. Found {len(image_paths)} in {INPUT_DIR}.")
    else:
        features_dict = extract_sift_features(image_paths, flat_field_img, downscale_factor=DOWNSCALE_FACTOR)
        connections = match_features(features_dict, min_inliers=MIN_INLIERS)
        overlay_all_images(features_dict, connections, OUTPUT_DIR, flat_field_img)