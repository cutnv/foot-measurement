import json
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np

import app
from evaluate_smartdoc_paper import polygon_iou


ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get(
    "PAPER_SEGMENTER_MODEL", ROOT / "models" / "paper_lraspp_v1.onnx"))
REPORT = Path(os.environ.get(
    "PAPER_SEGMENTER_REPORT", ROOT / "paper_lraspp_v1_report.json"))
MANIFEST = ROOT / "datasets" / "paper_segmentation_v1.json"
GROUPS = Path(r"C:\Users\Admin\OneDrive\Desktop\111\group")
OUTPUT = ROOT / "diagnostics" / "paper_segmenter_v1"
SIZE = (384, 512)


class PaperSegmenter:
    def __init__(self):
        self.net = cv2.dnn.readNetFromONNX(str(MODEL))
        self.threshold = json.loads(REPORT.read_text("utf-8"))["threshold"]

    def mask(self, image):
        resized = cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.float32([0.485, 0.456, 0.406])) / np.float32(
            [0.229, 0.224, 0.225])
        self.net.setInput(rgb.transpose(2, 0, 1)[None])
        logits = self.net.forward()[0, 0]
        cutoff = math.log(self.threshold / (1 - self.threshold))
        mask = (logits >= cutoff).astype(np.uint8) * 255
        return cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    def quad(self, image):
        mask = self.mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask
        image_area = image.shape[0] * image.shape[1]
        choices = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * 0.015:
                continue
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)
            candidates = []
            for ratio in np.linspace(0.005, 0.08, 76):
                polygon = cv2.approxPolyDP(hull, ratio * perimeter, True)
                if len(polygon) == 4:
                    candidates.append(polygon[:, 0].astype(np.float32))
            quad = (max(candidates, key=lambda value: abs(cv2.contourArea(value)))
                    if candidates else cv2.boxPoints(cv2.minAreaRect(hull)))
            quad = np.asarray(app._order_box(quad), np.float32)
            quad_area = max(abs(cv2.contourArea(quad)), 1.0)
            edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
            if edges.max() / max(edges.min(), 1.0) > 3.5:
                continue
            evidence = app._paper_quad_evidence(image, quad)
            rank = app._paper_candidate_rank_score(image, quad)
            fill = min(area / quad_area, 1.0)
            choices.append((evidence + rank * 2.0 + fill, quad))
        if not choices:
            return None, mask
        return max(choices, key=lambda item: item[0])[1], mask


def read_image(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def public_test(segmenter):
    records = [item for item in json.loads(MANIFEST.read_text("utf-8"))
               if item["split"] == "test"]
    learned, classic = [], []
    for item in records:
        image = read_image(Path(item["image"]))
        truth = np.asarray(item["polygon"], np.float32)
        predicted, _ = segmenter.quad(image)
        baseline = app._detect_reliable_paper_corners(image)
        learned.append(polygon_iou(predicted, truth) if predicted is not None else 0.0)
        classic.append(polygon_iou(baseline, truth) if baseline is not None else 0.0)
    return {
        "images": len(records),
        "learned_mean_iou": round(float(np.mean(learned)), 4),
        "learned_iou_80": int(np.sum(np.asarray(learned) >= 0.8)),
        "classic_mean_iou": round(float(np.mean(classic)), 4),
        "classic_iou_80": int(np.sum(np.asarray(classic) >= 0.8)),
    }


def holdout(segmenter):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for folder in sorted(GROUPS.iterdir()):
        match = re.match(r"\s*(\d+)", folder.name)
        if match is None or not 21 <= int(match.group(1)) <= 32:
            continue
        group = int(match.group(1))
        for index, path in enumerate(sorted(
                value for value in folder.iterdir()
                if value.suffix.lower() in {".jpg", ".jpeg", ".png"}), 1):
            image = read_image(path)
            predicted, mask = segmenter.quad(image)
            cv2.imwrite(str(OUTPUT / f"{group}_{index}_mask.png"), mask)
            baseline = app._detect_reliable_paper_corners(image)
            canvas = image.copy()
            if baseline is not None:
                cv2.polylines(canvas, [np.asarray(baseline, np.int32)], True,
                              (0, 0, 255), max(2, image.shape[0] // 400))
            if predicted is not None:
                cv2.polylines(canvas, [predicted.astype(np.int32)], True,
                              (0, 255, 0), max(2, image.shape[0] // 400))
            scale = min(420 / image.shape[1], 520 / image.shape[0])
            preview = cv2.resize(canvas, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)
            cv2.putText(preview, f"G{group}-{index}", (8, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imwrite(str(OUTPUT / f"{group}_{index}.jpg"), preview)
            agreement = None
            if predicted is not None and baseline is not None:
                agreement = float(np.mean(np.linalg.norm(
                    predicted - np.asarray(baseline), axis=1)))
            evidence = (app._paper_quad_evidence(image, predicted)
                        if predicted is not None else None)
            records.append({"group": group, "image": index,
                            "agreement_px": None if agreement is None else round(agreement, 2),
                            "learned_evidence": None if evidence is None else round(float(evidence), 3)})
    return records


def main():
    segmenter = PaperSegmenter()
    result = {"public_test": public_test(segmenter),
              "holdout": holdout(segmenter)}
    path = OUTPUT / "report.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result["public_test"], ensure_ascii=False))
    print(str(path))


if __name__ == "__main__":
    main()
