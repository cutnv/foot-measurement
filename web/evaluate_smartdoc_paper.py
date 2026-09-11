import argparse
import csv
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import app


def read_image(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def polygon_iou(first, second):
    first = np.asarray(first, np.float32)
    second = np.asarray(second, np.float32)
    area_first = abs(cv2.contourArea(first))
    area_second = abs(cv2.contourArea(second))
    intersection, _ = cv2.intersectConvexConvex(first, second)
    return float(intersection / max(area_first + area_second - intersection, 1.0))


def percentile(values, level):
    return None if not values else round(float(np.percentile(values, level)), 4)


def summarize(rows):
    times = [row["elapsed_ms"] for row in rows]
    ious = [row["iou"] for row in rows]
    triggered = [row for row in rows if row["triggered"]]
    return {
        "count": len(rows),
        "triggered": len(triggered),
        "trigger_rate": round(len(triggered) / max(len(rows), 1), 4),
        "iou_ge_0_80": sum(row["iou"] >= 0.80 for row in rows),
        "iou_ge_0_80_rate": round(
            sum(row["iou"] >= 0.80 for row in rows) / max(len(rows), 1), 4),
        "wrong_trigger_iou_lt_0_50": sum(
            row["triggered"] and row["iou"] < 0.50 for row in rows),
        "iou_median_triggered": percentile(
            [row["iou"] for row in triggered], 50),
        "iou_p10_triggered": percentile(
            [row["iou"] for row in triggered], 10),
        "corner_error_px_median": percentile(
            [row["corner_error_px"] for row in triggered], 50),
        "elapsed_ms_median": percentile(times, 50),
        "elapsed_ms_p95": percentile(times, 95),
        "iou_mean_all_failures_as_zero": round(float(np.mean(ious)), 4),
    }


def draw_preview(image, ground_truth, prediction, label):
    preview = image.copy()
    cv2.polylines(preview, [ground_truth.astype(np.int32)], True,
                  (255, 255, 255), 5, cv2.LINE_AA)
    if prediction is not None:
        cv2.polylines(preview, [prediction.astype(np.int32)], True,
                      (0, 255, 0), 5, cv2.LINE_AA)
    cv2.putText(preview, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3, cv2.LINE_AA)
    return preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--per-sequence", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metadata = args.dataset / "metadata.csv.gz"
    with gzip.open(metadata, "rt", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))

    sequences = defaultdict(list)
    for row in all_rows:
        sequences[(row["bg_name"], row["model_name"])].append(row)

    selected = []
    for rows in sequences.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
        indexes = np.linspace(0, len(rows) - 1, args.per_sequence + 2)[1:-1]
        selected.extend(rows[int(round(index))] for index in indexes)

    results = []
    previews = []
    for index, row in enumerate(selected, 1):
        path = args.dataset / row["image_path"]
        image = read_image(path)
        ground_truth = np.float32([
            [row["tl_x"], row["tl_y"]],
            [row["tr_x"], row["tr_y"]],
            [row["br_x"], row["br_y"]],
            [row["bl_x"], row["bl_y"]],
        ])
        started = time.perf_counter()
        prediction = app.detect_partial_paper_corners(image, 30)
        elapsed_ms = (time.perf_counter() - started) * 1000
        prediction = (None if prediction is None else
                      np.asarray(prediction, np.float32))
        iou = 0.0 if prediction is None else polygon_iou(
            prediction, ground_truth)
        corner_error = (None if prediction is None else float(np.mean(
            np.linalg.norm(prediction - ground_truth, axis=1))))
        item = {
            "image_path": row["image_path"],
            "background": row["bg_name"],
            "model_type": row["modeltype_name"],
            "triggered": prediction is not None,
            "iou": round(iou, 4),
            "corner_error_px": (None if corner_error is None else
                                round(corner_error, 2)),
            "elapsed_ms": round(elapsed_ms, 2),
        }
        results.append(item)
        if prediction is None or iou < 0.80:
            previews.append((iou, draw_preview(
                image, ground_truth, prediction,
                f"IoU {iou:.3f} {row['bg_name']}")))
        print(f"{index}/{len(selected)} trigger={prediction is not None} "
              f"iou={iou:.3f}", flush=True)

    by_background = {}
    for background in sorted({row["background"] for row in results}):
        by_background[background] = summarize(
            [row for row in results if row["background"] == background])
    by_model_type = {}
    for model_type in sorted({row["model_type"] for row in results}):
        by_model_type[model_type] = summarize(
            [row for row in results if row["model_type"] == model_type])

    output = args.output or args.dataset.parent / "smartdoc_paper_report.json"
    report = {
        "source": "SmartDoc 2015 Challenge 1 v2.0.0",
        "sampling": {
            "sequences": len(sequences),
            "per_sequence": args.per_sequence,
            "sample_count": len(results),
        },
        "overall": summarize(results),
        "by_background": by_background,
        "by_model_type": by_model_type,
        "samples": results,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")

    previews.sort(key=lambda item: item[0])
    preview_dir = output.with_suffix("")
    preview_dir.mkdir(parents=True, exist_ok=True)
    for index, (_, preview) in enumerate(previews[:30], 1):
        cv2.imwrite(str(preview_dir / f"failure_{index:02d}.jpg"), preview)
    print(json.dumps(report["overall"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
