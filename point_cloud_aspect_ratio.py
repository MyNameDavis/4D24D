import cv2
import numpy as np
import glob
import os
import itertools
from scipy.optimize import least_squares
from collections import defaultdict

def get_gaussian_bandpass(h, w, low_freq=10, high_freq=70):
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mid_freq = (low_freq + high_freq) / 2.0
    sigma = (high_freq - low_freq) / 2.0
    if sigma == 0: sigma = 1e-5
    return np.exp(-0.5 * ((dist - mid_freq) / sigma)**2)

def compute_spatial_covariance(patch, low_freq=10, high_freq=70):
    """Computes the spatial covariance matrix of the film grain in a patch."""
    h, w = patch.shape
    patch_float = patch.astype(np.float32)
    window = np.outer(np.hanning(h), np.hanning(w))
    patch_w = (patch_float - np.mean(patch_float)) * window
    
    f = np.fft.fftshift(np.fft.fft2(patch_w))
    mag = np.abs(f)
    
    mask = get_gaussian_bandpass(h, w, low_freq, high_freq)
    mag_filtered = mag * mask
    
    # Compute moments of the frequency spectrum
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    Y = Y - cy
    X = X - cx
    
    weight = mag_filtered
    total_weight = np.sum(weight)
    if total_weight == 0:
        return np.eye(2)
        
    m20 = np.sum(weight * X**2) / total_weight
    m02 = np.sum(weight * Y**2) / total_weight
    m11 = np.sum(weight * X * Y) / total_weight
    
    # Frequency covariance matrix
    Sigma_f = np.array([[m20, m11], [m11, m02]])
    
    # Spatial covariance is proportional to the inverse of frequency covariance
    try:
        Sigma_s = np.linalg.inv(Sigma_f)
    except np.linalg.LinAlgError:
        Sigma_s = np.eye(2)
        
    # Normalize spatial covariance so det = 1 (to prevent scale explosion)
    Sigma_s = Sigma_s / np.sqrt(np.linalg.det(Sigma_s) + 1e-8)
    return Sigma_s

def jacobian_of_homography(H, x, y):
    """Calculates the 2x2 Jacobian matrix of a Homography H at point (x, y)."""
    # H = [[h11, h12, h13], [h21, h22, h23], [h31, h32, 1]]
    # u = (h11 x + h12 y + h13) / w
    # v = (h21 x + h22 y + h23) / w
    # w = h31 x + h32 y + 1
    num_u = H[0, 0]*x + H[0, 1]*y + H[0, 2]
    num_v = H[1, 0]*x + H[1, 1]*y + H[1, 2]
    den = H[2, 0]*x + H[2, 1]*y + 1.0
    
    if den == 0: den = 1e-8
    den2 = den**2
    
    du_dx = (H[0, 0]*den - num_u*H[2, 0]) / den2
    du_dy = (H[0, 1]*den - num_u*H[2, 1]) / den2
    dv_dx = (H[1, 0]*den - num_v*H[2, 0]) / den2
    dv_dy = (H[1, 1]*den - num_v*H[2, 1]) / den2
    
    return np.array([[du_dx, du_dy], [dv_dx, dv_dy]])

def homography_residual(h_params, points, covs_spatial):
    """Residual function for Homography optimization."""
    # h_params: 8 parameters [h11, h12, h13, h21, h22, h23, h31, h32]
    H = np.array([
        [h_params[0], h_params[1], h_params[2]],
        [h_params[3], h_params[4], h_params[5]],
        [h_params[6], h_params[7], 1.0]
    ])
    
    residuals = []
    for (x, y), Sigma_s in zip(points, covs_spatial):
        J = jacobian_of_homography(H, x, y)
        # We want the transformed covariance J * Sigma_s * J^T to be proportional to Identity
        # meaning the grain becomes completely isotropic (circular)
        mapped_cov = J @ Sigma_s @ J.T
        
        # Normalize to det = 1 to ignore scale
        det = np.linalg.det(mapped_cov)
        if det <= 0:
            mapped_cov_norm = mapped_cov
            residuals.extend([1e6, 1e6, 1e6]) # Heavily penalize invalid jacobians
            continue
        
        mapped_cov_norm = mapped_cov / np.sqrt(det)
        
        # For an identity matrix, off-diagonal is 0, and diagonals are equal (1 and 1)
        res_diag = mapped_cov_norm[0, 0] - mapped_cov_norm[1, 1]
        res_off = mapped_cov_norm[0, 1]
        
        residuals.extend([res_diag, res_off * 2.0]) # Multiply off-diagonal by 2 for weight
        
    return np.array(residuals)

def estimate_segment_unwarp(img_path, patch_size=256):
    """Estimates the 3D perspective Homography that unwarps the film grain of a segment."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    H_img, W_img = img.shape
    
    # Extract a 3x3 grid of patches
    points = []
    covs = []
    
    step_y = (H_img - patch_size) // 2
    step_x = (W_img - patch_size) // 2
    
    for i in range(3):
        for j in range(3):
            y = i * step_y
            x = j * step_x
            patch = img[y:y+patch_size, x:x+patch_size]
            if patch.shape != (patch_size, patch_size): continue
            
            Sigma_s = compute_spatial_covariance(patch)
            cx, cy = x + patch_size//2, y + patch_size//2
            points.append((cx, cy))
            covs.append(Sigma_s)
            
    if len(points) < 4:
        return np.eye(3)
        
    # Initial guess: Identity matrix
    h0 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    
    res = least_squares(homography_residual, h0, args=(points, covs), method='lm')
    h_opt = res.x
    
    H_unwarp = np.array([
        [h_opt[0], h_opt[1], h_opt[2]],
        [h_opt[3], h_opt[4], h_opt[5]],
        [h_opt[6], h_opt[7], 1.0]
    ])
    
    return H_unwarp

def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(nfeatures=5000)
    kps, des = sift.detectAndCompute(img, None)
    return kps, des, img.shape

def extract_aspect_ratio_equations(features_dict, connections, global_transforms, T_shift, M_persp, mosaic_id, canvas_w, canvas_h):
    import random
    from collections import defaultdict
    
    A_raw = canvas_w / float(canvas_h)
    
    print(f"\n[Point-Cloud Solver] Calculating local 3D FFT unwarps for mosaic {mosaic_id} (Raw Aspect: {A_raw:.3f})...")
    unwarps = {}
    for name, data in features_dict.items():
        # features_dict[name]['path'] gives the file path
        unwarps[name] = estimate_segment_unwarp(data['path'])
        
    print(f"[Point-Cloud Solver] Building SIFT tracks for mosaic {mosaic_id}...")
    parent = {}
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            
    for name in features_dict:
        for i in range(len(features_dict[name]['keypoints'])):
            parent[(name, i)] = (name, i)
            
    for conn in connections:
        n1, n2 = conn['img1'], conn['img2']
        for m in conn['matches']:
            union((n1, m.queryIdx), (n2, m.trainIdx))
            
    tracks = defaultdict(list)
    for node in parent:
        tracks[find(node)].append(node)
        
    valid_tracks = [t for t in tracks.values() if len(t) >= 2]
    print(f" -> Found {len(valid_tracks)} valid SIFT tracks across multiple segments.")
    
    if len(valid_tracks) == 0:
        print(" -> Error: No shared SIFT tracks found! Returning empty equations.")
        return []
    
    segment_to_tracks = defaultdict(list)
    for tid, track in enumerate(valid_tracks):
        for node in track:
            segment_to_tracks[node[0]].append((tid, node[1]))
            
    print(f"[Point-Cloud Solver] Extracting Least Squares equations for mosaic {mosaic_id}...")
    
    equations = []
    
    segment_names = sorted(list(global_transforms.keys()))
    
    for name in segment_names:
        seg_tracks = segment_to_tracks[name]
        
        # We want to use the pairs with the absolute maximum physical distance to minimize SIFT sub-pixel quantization error
        num_candidates = min(5000, len(seg_tracks)//2)
        if num_candidates == 0:
            continue
            
        candidate_idx = random.sample(range(len(seg_tracks)), num_candidates * 2)
        H_local = unwarps[name]
        F_global = M_persp @ T_shift @ global_transforms[name]
        kps = features_dict[name]['keypoints']
        
        pair_distances = []
        
        for i in range(num_candidates):
            tid_A, idx_A = seg_tracks[candidate_idx[2*i]]
            tid_B, idx_B = seg_tracks[candidate_idx[2*i+1]]
            
            ptA = np.array([kps[idx_A].pt[0], kps[idx_A].pt[1], 1.0])
            ptB = np.array([kps[idx_B].pt[0], kps[idx_B].pt[1], 1.0])
            
            globA = F_global @ ptA
            globA /= globA[2]
            globB = F_global @ ptB
            globB /= globB[2]
            
            dX = globA[0] - globB[0]
            dY = globA[1] - globB[1]
            dist = np.hypot(dX, dY)
            
            pair_distances.append({
                'dist': dist,
                'ptA': ptA,
                'ptB': ptB,
                'dX': dX,
                'dY': dY
            })
            
        # Sort by physical distance descending, take top 400 longest pairs
        pair_distances.sort(key=lambda x: x['dist'], reverse=True)
        top_pairs = pair_distances[:400]
        
        for pair in top_pairs:
            ptA = pair['ptA']
            ptB = pair['ptB']
            dX = pair['dX']
            dY = pair['dY']
            
            # Distance in the local 256x256 FFT space (dIso)
            locA = H_local @ ptA
            locA /= locA[2]
            locB = H_local @ ptB
            locB /= locB[2]
            dIso = np.linalg.norm(locA[:2] - locB[:2])
            
            # Geometric noise filter: Just in case even the top 400 pairs are too close
            if pair['dist'] < 300:
                continue
            
            equations.append({
                'coeff_R2': dX**2,
                'coeff_beta': -dIso**2,
                'rhs': -(dY**2) * (A_raw**2),
                'segment_key': f"m{mosaic_id}_{name}"
            })
            
    return equations
