import cv2
import numpy as np
import json
import os
from image_io import read_image

with open("param.json") as _f:
    PARAMS = json.load(_f)

def load_clean_image(path, flat_field_img=None):
    img = read_image(path)
    if img is None:
        return None
        
    if flat_field_img is not None:
        ff_resized = cv2.resize(flat_field_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        ff = ff_resized.astype(np.float32) / 255.0
        ff[ff == 0] = 1.0 
        img = (img.astype(np.float32) / ff).clip(0, 255).astype(np.uint8)
        
    return img

def extract_sift_features(image_paths, flat_field_img=None, downscale_factor=1.0, max_features=0):
    sift = cv2.SIFT_create(nfeatures=max_features) # when nfeatures = 0, no limit is applied
    features = {}
    total_success_pct = 0.0

    print(f"Extracting SIFT features from {len(image_paths)} images...")
    for path in image_paths:
        filename = os.path.basename(path)
        img_bgr = load_clean_image(path, flat_field_img)
        if img_bgr is None:
            continue
            
        # Convert to grayscale for SIFT
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

        target_features = max_features or 1000
        
        num_features = len(keypoints)
        success_pct = min(100.0, (num_features / target_features) * 100.0)
        total_success_pct += success_pct
        
    if features:
        overall_success = total_success_pct / len(features)
        print(f" -> Feature extraction complete. Overall detection success score: {overall_success:.1f}%")
        
    return features
