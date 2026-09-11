import json
import time
from pathlib import Path

import cv2
import numpy as np

import app
from evaluate_smartdoc_paper import (draw_preview, polygon_iou, read_image,
                                     summarize)


ROOT = Path(r"D:\foot_measurement\datasets\document_scanner_kaggle\dataset")
OUTPUT = Path(r"D:\foot_measurement\document_scanner_paper_report.json")


def main():
    rows = []
    previews = []
    for annotation_path in sorted((ROOT / "annotations").glob("*/*.json")):
        split = annotation_path.parent.name
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not annotation.get("has_document"):
            continue
        image_path = ROOT / "raw" / split / annotation["filename"]
        image = read_image(image_path)
        ground_truth = np.asarray(annotation["corners"], np.float32)

        started = time.perf_counter()
        prediction = app._detect_reliable_paper_corners(image)
        strict_triggered = prediction is not None
        method = "strict"
        if prediction is None:
            prediction = app.detect_partial_paper_corners(image, 48)
            method = "relaxed"
        if prediction is None:
            prediction = app._recover_paper_from_three_edges(image)
            method = "three_edges"
        elapsed_ms = (time.perf_counter() - started) * 1000

        prediction = (None if prediction is None else
                      np.asarray(prediction, np.float32))
        iou = 0.0 if prediction is None else polygon_iou(
            prediction, ground_truth)
        corner_error = (None if prediction is None else float(np.mean(
            np.linalg.norm(prediction - ground_truth, axis=1))))
        row = {
            "split": split,
            "image": annotation["filename"],
            "strict_triggered": strict_triggered,
            "triggered": prediction is not None,
            "method": method if prediction is not None else None,
            "iou": round(iou, 4),
            "corner_error_px": (None if corner_error is None else
                                round(corner_error, 2)),
            "elapsed_ms": round(elapsed_ms, 2),
        }
        rows.append(row)
        if prediction is None or iou < 0.80:
            previews.append((iou, draw_preview(
                image, ground_truth, prediction,
                f"IoU {iou:.3f} {method}")))
        print(f"{len(rows)}/140 strict={strict_triggered} "
              f"method={row['method']} iou={iou:.3f}", flush=True)

    report = {
        "source": "Kaggle tiendq/dataset-document-scanner",
        "overall": summarize(rows),
        "strict_triggered": sum(row["strict_triggered"] for row in rows),
        "strict_trigger_rate": round(
            sum(row["strict_triggered"] for row in rows) / len(rows), 4),
        "method_counts": {
            method: sum(row["method"] == method for row in rows)
            for method in ("strict", "relaxed", "three_edges", None)
        },
        "samples": rows,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    preview_dir = OUTPUT.with_suffix("")
    preview_dir.mkdir(exist_ok=True)
    previews.sort(key=lambda item: item[0])
    for index, (_, preview) in enumerate(previews[:30], 1):
        cv2.imwrite(str(preview_dir / f"failure_{index:02d}.jpg"), preview)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
