import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import importlib
import app
importlib.reload(app)

import cv2
import numpy as np

from app import (auto_detect_corners, build_transform, detect_foot_on_paper,
                 warp_contour, measure_foot, heel_outline_is_visible)

# 固定15%截面：该合成轮廓在脚跟前15%位置宽50像素。
synthetic = np.zeros((101, 101), np.uint8)
cv2.fillPoly(synthetic, [np.array([
    [40, 0], [60, 0], [80, 80], [75, 85],
    [50, 100], [25, 85], [20, 80]
], np.int32)], 255)
synthetic_contours, _ = cv2.findContours(
    synthetic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
_, _, synthetic_heel, _, _, _ = measure_foot(
    synthetic_contours[0], 1.0, 100)
assert abs(synthetic_heel - 50.0) <= 1.0, synthetic_heel
assert heel_outline_is_visible(
    synthetic_contours[0].reshape(-1, 2).astype(np.float32), 100)

occluded = np.zeros((101, 101), np.uint8)
cv2.fillPoly(occluded, [np.array([
    [40, 0], [60, 0], [80, 80], [75, 85],
    [70, 100], [30, 100], [25, 85], [20, 80]
], np.int32)], 255)
occluded_contours, _ = cv2.findContours(
    occluded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
assert not heel_outline_is_visible(
    occluded_contours[0].reshape(-1, 2).astype(np.float32), 100)

photos = [
    (r'D:\foot_measurement\8e0a6ffc2aed2a1863ded5746e894ce5.jpg', 'flat1'),
    (r'D:\foot_measurement\f9c54f1f7f8e2e7f70975d2ee0c679ba.jpg', 'flat2'),
    (r'D:\foot_measurement\fe23cfec4010fe8c8e394c86ddbe44fb.jpg', 'flat3'),
    (r'C:\Users\Admin\OneDrive\Desktop\111\1\5a233e2e6850965c31a0de21881f6483.jpg', 'wood1'),
    (r'C:\Users\Admin\OneDrive\Desktop\111\1\6dd6b7bd63d155a094105e46639b778e.jpg', 'wood2'),
    (r'C:\Users\Admin\OneDrive\Desktop\111\1\b88d57466239988acc59c30fea07d185.jpg', 'wood3'),
]

failures = []
for fn, name in photos:
    img = cv2.imread(fn)
    if img is None:
        failures.append(name + ': cannot read'); continue
    corners = auto_detect_corners(img)
    if corners is None:
        failures.append(name + ': no paper'); continue
    M, mm_per_px, out_size = build_transform(corners)
    foot = detect_foot_on_paper(img, corners)
    if foot is None:
        failures.append(name + ': no foot'); continue
    foot_w = warp_contour(foot, M, out_size)
    fl, bw, hw, ax, h, t = measure_foot(
        foot_w, mm_per_px, out_size[1] - 1)
    aligned = foot_w.reshape(-1, 2)[:, 1].max() >= out_size[1] - 6
    print(name + ': len=' + str(round(fl,1)) + ' ball=' + str(round(bw,1)) +
          ' heel=' + str(round(hw,1)) + ' baseline=' + str(aligned))
    sys.stdout.flush()

if failures:
    raise AssertionError('; '.join(failures))
