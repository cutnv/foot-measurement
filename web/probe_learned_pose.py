import sys
from pathlib import Path

import cv2
import numpy as np

import app


def read(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


group = Path(sys.argv[1])
top = int(sys.argv[2])
paths = sorted(p for p in group.iterdir()
               if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
order = [top] + [index for index in range(3) if index != top]
images = app._normalize_multiview_images([read(paths[index]) for index in order])
mode = sys.argv[3] if len(sys.argv) > 3 else ''
if mode.startswith('selected') or mode.startswith('mixed'):
    selected = app._select_adaptive_multiview_paper_corners(
        images, images, (images[0].shape[1], images[0].shape[0]),
        strict_corners=[app.detect_partial_paper_corners(image, 30)
                        for image in images])
    corners = selected[0] if selected is not None else [None] * 3
else:
    corners = [app._learned_paper_candidate(image) for image in images]
if mode.startswith('mixed'):
    learned = [app._learned_paper_candidate(image) for image in images]
    corners = [corners[0], learned[1], learned[2]]
print('files', [paths[index].name for index in order])
print('corners', [item is not None for item in corners])
if any(item is None for item in corners):
    raise SystemExit(1)
calibrated = app._calibrate_multiview(
    corners, (images[0].shape[1], images[0].shape[0]))
centers = calibrated[5]
top_x = centers[0][0]
offsets = [float(center[0] - top_x) for center in centers[1:]]
angles = [float(np.degrees(np.arctan2(
    abs(offset), max(abs(center[2]), 1))))
    for offset, center in zip(offsets, centers[1:])]
print('error', round(float(calibrated[0]), 3))
print('centers', [[round(float(value), 2) for value in center]
                  for center in centers])
print('offsets', [round(value, 2) for value in offsets])
print('angles', [round(value, 2) for value in angles])
print('opposite', offsets[0] * offsets[1] < 0)
object_points = np.float32([
    [0, 0, 0], [app.A4_WIDTH_MM, 0, 0],
    [app.A4_WIDTH_MM, app.A4_HEIGHT_MM, 0],
    [0, app.A4_HEIGHT_MM, 0]])
for index, item in enumerate(corners):
    solved = cv2.solvePnPGeneric(
        object_points, np.asarray(item, np.float32), calibrated[1][index],
        np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
    poses = []
    for rotation, translation in zip(solved[1], solved[2]):
        matrix, _ = cv2.Rodrigues(rotation)
        center = (-matrix.T @ translation).ravel()
        projected, _ = cv2.projectPoints(
            object_points, rotation, translation,
            calibrated[1][index], np.zeros(5))
        error = np.sqrt(np.mean(np.sum(
            (projected.reshape(-1, 2) - item) ** 2, axis=1)))
        poses.append(([round(float(value), 1) for value in center],
                      round(float(error), 2)))
    print('ippe', index, poses)
for left in (False, True):
    for right in (False, True):
        flags = (False, left, right)
        trial = [np.asarray(item, np.float32)[[1, 0, 3, 2]]
                 if flag else item
                 for item, flag in zip(corners, flags)]
        pose = app._calibrate_multiview(
            trial, (images[0].shape[1], images[0].shape[0]))
        trial_centers = pose[5]
        trial_top = trial_centers[0][0]
        trial_offsets = [float(center[0] - trial_top)
                         for center in trial_centers[1:]]
        print('trial', flags, round(float(pose[0]), 3),
              [round(value, 1) for value in trial_offsets])
print('mirror', app._mirrored_multiview_corners(
    images, corners, (images[0].shape[1], images[0].shape[0]), True)
    is not None)

if mode.endswith('hulls'):
    masks = [app._multiview_foot_mask(image, item)
             for image, item in zip(images, corners)]
    transform = cv2.getPerspectiveTransform(
        np.asarray(corners[0], np.float32),
        np.float32([[0, 0], [app.A4_WIDTH_MM, 0],
                    [app.A4_WIDTH_MM, app.A4_HEIGHT_MM],
                    [0, app.A4_HEIGHT_MM]]))
    top_mask = cv2.warpPerspective(
        masks[0], transform,
        (int(app.A4_WIDTH_MM) + 1, int(app.A4_HEIGHT_MM) + 1))
    variants = ((0, 1, 2, 3), (1, 0, 3, 2),
                (2, 3, 0, 1), (3, 2, 1, 0))
    for side in (1, 2):
        for variant in variants:
            trial = [corners[0], np.asarray(corners[side])[list(variant)]]
            pose = app._calibrate_multiview(
                trial, (images[0].shape[1], images[0].shape[0]))
            center = pose[5][1]
            angle = np.degrees(np.arctan2(
                abs(float(center[0]) - app.A4_WIDTH_MM / 2),
                max(abs(float(center[2])), 1)))
            hull = app._visual_hull_footprint(
                [masks[0], masks[side]], pose[1], pose[2], pose[3], pose[4],
                minimum_side_views=1)
            footprint, groundprint, _, height_map = hull
            top_angle = np.degrees(np.arctan2(
                abs(float(pose[5][0][1]) - app.A4_HEIGHT_MM / 2),
                max(abs(float(pose[5][0][2])), 1)))
            measured = app._measure_footprint(
                footprint, groundprint, height_map, top_mask,
                False, True, top_angle, mirrored_geometry=True)
            intersection = cv2.countNonZero(cv2.bitwise_and(
                footprint, top_mask))
            union = cv2.countNonZero(cv2.bitwise_or(footprint, top_mask))
            print('hull', side, variant, 'error', round(float(pose[0]), 2),
                  'center', [round(float(value), 1) for value in center],
                  'angle', round(float(angle), 1),
                  'agreement', round(intersection / max(union, 1), 3),
                  'measure', None if measured is None else
                  [round(float(value), 1) for value in measured[:3]])
