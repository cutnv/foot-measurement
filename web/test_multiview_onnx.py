import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

import app
import test_multiview as cases


ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get(
    "FOOT_MODEL", ROOT / "models" / "foot_lraspp_384.onnx"))
NET = cv2.dnn.readNetFromONNX(str(MODEL))
WIDTH, HEIGHT = 384, 512
LOGIT_THRESHOLD = math.log(0.35 / 0.65)


def model_mask(image, corners, **_kwargs):
    resized = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.float32([0.485, 0.456, 0.406])) / \
          np.float32([0.229, 0.224, 0.225])
    NET.setInput(rgb.transpose(2, 0, 1)[None])
    mask = NET.forward()[0, 0] >= LOGIT_THRESHOLD
    mask = cv2.resize(mask.astype(np.uint8),
                      (image.shape[1], image.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    size = max(3, int(round(min(image.shape[:2]) * 0.004))) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((size, size), np.uint8))
    paper = np.zeros_like(mask)
    cv2.fillConvexPoly(paper, np.asarray(corners, np.int32), 1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        overlap = int(np.count_nonzero((labels == label) & (paper > 0)))
        if overlap < image.shape[0] * image.shape[1] * 0.01:
            continue
        score = overlap + area * 0.1
        if best is None or score > best[0]:
            best = score, label
    if best is None:
        return None
    return ((labels == best[1]).astype(np.uint8) * 255)


def run_case(folder, names, truth):
    images = [cv2.imread(os.path.join(folder, name)) for name in names]
    result, error = app.process_multiview_images(images)
    if error:
        return {"error": error}
    values = {key: result[key] for key in truth}
    return {
        "measured_mm": values,
        "absolute_error_mm": {
            key: round(abs(values[key] - truth[key]), 1) for key in truth
        },
        "calibration_error_px": result["calibration_error_px"],
        "side_angles": result["side_angles"],
    }


def save_preview(folder, names):
    rows = []
    for name in names:
        image = cv2.imread(os.path.join(folder, name))
        corners = app.detect_partial_paper_corners(image, 30)
        current = app._multiview_foot_mask(image, corners)
        learned = model_mask(image, corners)
        scale = 360 / image.shape[1]
        size = (360, int(round(image.shape[0] * scale)))
        original = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        panels = [original]
        for mask, color in ((current, (0, 255, 0)),
                            (learned, (0, 0, 255))):
            resized_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            panel = original.copy()
            panel[resized_mask > 0] = (
                panel[resized_mask > 0] * 0.45 + np.asarray(color) * 0.55)
            panels.append(panel.astype(np.uint8))
        rows.append(np.hstack(panels))
    cv2.imwrite(str(ROOT / os.environ.get(
        "FOOT_PREVIEW", "foot3d_onnx_real_masks_preview.jpg")),
                np.vstack(rows))


def main():
    groups = [
        ("6", cases.FOLDER, cases.NAMES,
         {"foot_length": 265, "ball_width": 96, "heel_width": 56}),
        ("7", cases.LATEST_FOLDER, cases.LATEST_NAMES,
         {"foot_length": 248, "ball_width": 93, "heel_width": 62}),
        ("8", cases.PARTIAL_FOLDER, cases.PARTIAL_NAMES,
         {"foot_length": 281, "ball_width": 93, "heel_width": 58}),
    ]
    original = app._multiview_foot_mask
    report = {"current": {}, "model_only": {}}
    for name, folder, names, truth in groups:
        report["current"][name] = run_case(folder, names, truth)
    app._multiview_foot_mask = model_mask
    try:
        for name, folder, names, truth in groups:
            report["model_only"][name] = run_case(folder, names, truth)
    finally:
        app._multiview_foot_mask = original
    save_preview(cases.FOLDER, cases.NAMES)
    output = ROOT / os.environ.get(
        "FOOT_REPORT", "foot3d_onnx_real_photos_report.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
