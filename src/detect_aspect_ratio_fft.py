import cv2
import numpy as np

def calculate_entropy(img_gray):
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    p = hist[hist > 0]
    return -np.sum(p * np.log2(p))

def estimate_stretch_from_grain(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    H, W = gray.shape
    patch_size = 256
    margin = patch_size // 2
    step = patch_size // 2
    
    best_patch = None
    min_entropy = float('inf')
    
    # 1. Find the best flat patch (low entropy but has some variance)
    for y in range(margin, H - patch_size - margin, step):
        for x in range(margin, W - patch_size - margin, step):
            patch = gray[y:y+patch_size, x:x+patch_size]
            ent = calculate_entropy(patch)
            if 3.0 < ent < min_entropy and np.var(patch) > 1.0:
                min_entropy = ent
                best_patch = patch
                
    if best_patch is None: return None
    
    # 2. Compute 2D FFT
    h, w = best_patch.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    patch_windowed = (best_patch.astype(np.float32) - np.mean(best_patch)) * window
    
    fshift = np.fft.fftshift(np.fft.fft2(patch_windowed))
    mag = np.log(np.abs(fshift) + 1)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Mask out DC/low freq
    cv2.circle(mag_norm, (w//2, h//2), 15, 0, -1)
    
    # 3. Robust Ellipse Fitting over multiple thresholds
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
                    # Align width/height to X/Y axes based on angle
                    if 45 < angle < 135:
                        freq_x, freq_y = eh, ew
                    else:
                        freq_x, freq_y = ew, eh
                    ratios.append(freq_y / freq_x)
                    
    if ratios:
        return np.median(ratios)
    return None

def main():
    print("Testing robust grain estimator...")
    base_file = 'output/mosaic_04.tif'
    gray = cv2.cvtColor(cv2.imread(base_file), cv2.COLOR_BGR2GRAY)
    
    for stretch in [1.0, 1.25, 1.5, 2.0, 0.75]:
        stretched = cv2.resize(gray, (int(gray.shape[1] * stretch), gray.shape[0]))
        cv2.imwrite("temp_stretch.tif", stretched)
        detected = estimate_stretch_from_grain("temp_stretch.tif")
        print(f"Applied stretch: {stretch:4.2f}x | Detected stretch: {detected:4.2f}x")

if __name__ == "__main__":
    main()
