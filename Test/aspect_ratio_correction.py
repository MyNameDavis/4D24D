import cv2
import numpy as np
import collections

def calculate_entropy(img_gray):
    """Calculates the Shannon Entropy of a grayscale image patch."""
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    p = hist[hist > 0]
    return -np.sum(p * np.log2(p))

def get_gaussian_bandpass(h, w, low_freq=10, high_freq=70):
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    mid_freq = (low_freq + high_freq) / 2.0
    sigma = (high_freq - low_freq) / 2.0
    if sigma == 0: sigma = 1e-5
    mask = np.exp(-0.5 * ((dist_from_center - mid_freq) / sigma)**2)
    return mask

def analyze_patch(patch):
    h, w = patch.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    patch_windowed = (patch.astype(np.float32) - np.mean(patch)) * window
    
    fshift = np.fft.fftshift(np.fft.fft2(patch_windowed))
    mag = np.log(np.abs(fshift) + 1)
    
    # Apply soft Gaussian bandpass filter
    bandpass = get_gaussian_bandpass(h, w, low_freq=10, high_freq=70)
    mag_filtered = mag * bandpass
    
    mag_norm = cv2.normalize(mag_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    ratios = []
    for pct in [75, 80, 85, 90]:
        thresh_val = np.percentile(mag_norm[mag_norm > 0], pct)
        _, thresh = cv2.threshold(mag_norm, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if len(largest) >= 5:
                (ecx, ecy), (ew, eh), angle = cv2.fitEllipse(largest)
                if ew > 0 and eh > 0:
                    if 45 < angle < 135:
                        freq_x, freq_y = eh, ew
                    else:
                        freq_x, freq_y = ew, eh
                    ratios.append(freq_y / freq_x)
                    
    if ratios:
        return np.median(ratios)
    return None

def estimate_aspect_ratio_correction(img_path, patch_size=256, debug_output=False, num_patches=5):
    """
    Analyzes the film grain of an image to determine if it has been stretched.
    Returns a stretch factor (e.g., 1.2) that you should DIVIDE the image's WIDTH by 
    to restore the physically correct aspect ratio.
    """
    img = cv2.imread(img_path)
    if img is None: 
        raise ValueError(f"Could not load image: {img_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    
    margin = patch_size // 2
    step = patch_size // 2
    
    patches_info = []
    
    # 1. Slide across the interior and collect entropy for all valid patches
    for y in range(margin, H - patch_size - margin, step):
        for x in range(margin, W - patch_size - margin, step):
            patch = gray[y:y+patch_size, x:x+patch_size]
            var = np.var(patch)
            if var > 1.0:
                ent = calculate_entropy(patch)
                patches_info.append({
                    'coords': (x, y),
                    'patch': patch,
                    'entropy': ent
                })
                
    if not patches_info:
        print("Warning: Could not find any suitable film grain patches.")
        return 1.0
        
    # Sort by entropy (lowest first) and take the top N
    patches_info.sort(key=lambda item: item['entropy'])
    top_patches = patches_info[:num_patches]
    
    patch_ratios = []
    for p_info in top_patches:
        ratio = analyze_patch(p_info['patch'])
        if ratio is not None:
            patch_ratios.append(ratio)
            
    if debug_output:
        vis = img.copy()
        for p_info in top_patches:
            cx, cy = p_info['coords']
            cv2.rectangle(vis, (cx, cy), (cx+patch_size, cy+patch_size), (0, 255, 0), 2)
        cv2.imwrite("debug_patch_locations.jpg", vis)
                    
    if patch_ratios:
        # Taking the median across all selected patches to average out local distortions
        final_correction = np.median(patch_ratios)
        print(f" -> Sampled {len(patch_ratios)} patches. Median ratio: {final_correction:.3f}")
        return final_correction
    
    return 1.0

if __name__ == "__main__":
    # Quick Test Block
    print("Testing Aspect Ratio Stretch Detection...")
    correction = estimate_aspect_ratio_correction("output/mosaic_04.tif", debug_output=True, num_patches=5)
    print(f"To restore true proportions, divide the image width by: {correction:.3f}x")
