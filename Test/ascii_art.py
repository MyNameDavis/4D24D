import cv2
import numpy as np
import sys

img = cv2.imread('output/debug/entropy_heatmap_00.jpg')
if img is None: sys.exit(1)

# we only care about the heatmap part (right half)
h, w = img.shape[:2]
heatmap = img[:, w//2:]

# resize to 80x40 for ascii
small = cv2.resize(heatmap, (100, 40))

# convert to grayscale for intensity
gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

chars = " .:-=+*#%@"
ascii_str = ""
for y in range(small.shape[0]):
    for x in range(small.shape[1]):
        val = gray[y, x]
        idx = int((val / 255.0) * (len(chars) - 1))
        ascii_str += chars[idx]
    ascii_str += "\n"

print(ascii_str)
