import argparse
import json
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np

import app


def read_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def group_number(path):
    match = re.match(r"(\d+)", path.name)
    return int(match.group(1)) if match else math.inf


def top_view_score(image):
    corners = app.detect_partial_paper_corners(image, 30)
    if corners is None:
        return math.inf
    points = np.asarray(corners, np.float32)
    edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    if np.min(edges) < 1:
        return math.inf
    opposite = abs(math.log(edges[0] / edges[2])) + abs(
        math.log(edges[1] / edges[3]))
    cosines = []
    for index in range(4):
        first = points[index - 1] - points[index]
        second = points[(index + 1) % 4] - points[index]
        cosines.append(abs(float(np.dot(first, second))) /
                       (np.linalg.norm(first) * np.linalg.norm(second)))
    return opposite + 2.0 * float(np.mean(cosines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--top-indexes", default="",
        help="可选，按文件名字典序指定正上方图，例如 1:0,2:0,4:2")
    parser.add_argument(
        "--groups", default="",
        help="可选，仅测指定组，例如 1,2,3")
    parser.add_argument(
        "--order-manifest", action="append", type=Path, default=[],
        help="可重复指定已有报告；按input_order使用已确认的三视图顺序")
    args = parser.parse_args()
    selected_groups = {int(value) for value in args.groups.split(",") if value}
    top_indexes = {}
    for pair in filter(None, args.top_indexes.split(",")):
        group, index = pair.split(":", 1)
        top_indexes[int(group)] = int(index)
    input_orders = {}
    for manifest in args.order_manifest:
        for item in json.loads(manifest.read_text(encoding="utf-8")):
            names = item.get("input_order")
            if isinstance(names, list) and len(names) == 3:
                input_orders[int(item["group"])] = names

    report = []
    for folder in sorted((item for item in args.folder.iterdir()
                          if item.is_dir()), key=group_number):
        if selected_groups and group_number(folder) not in selected_groups:
            continue
        paths = sorted(path for path in folder.iterdir()
                       if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        item = {"group": group_number(folder), "folder": folder.name}
        if len(paths) != 3:
            item["error"] = f"需要3张照片，实际{len(paths)}张"
            report.append(item)
            continue
        images = [read_image(path) for path in paths]
        scores = [top_view_score(image) if image is not None else math.inf
                  for image in images]
        group = group_number(folder)
        confirmed = input_orders.get(group)
        if confirmed:
            indexes = {path.name: index for index, path in enumerate(paths)}
            if any(name not in indexes for name in confirmed):
                item["error"] = "已确认顺序中的照片不存在"
                report.append(item)
                continue
            order = [indexes[name] for name in confirmed]
        else:
            top = top_indexes.get(group, int(np.argmin(scores)))
            order = [top] + [index for index in range(3) if index != top]
        started = time.perf_counter()
        try:
            result, error = app.process_multiview_images(
                [images[index] for index in order])
        except Exception as exc:
            # 批量回归不能因单组异常中断后续样本；接口本身仍应返回失败。
            result, error = None, f'{type(exc).__name__}: {exc}'
        elapsed_seconds = time.perf_counter() - started
        item.update({
            "input_order": [paths[index].name for index in order],
            "top_scores": [round(score, 4) if np.isfinite(score) else None
                           for score in scores],
            "result": result,
            "error": error,
            "elapsed_seconds": round(elapsed_seconds, 3),
        })
        report.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    output = args.output or args.folder / "measurement_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")


if __name__ == "__main__":
    main()
