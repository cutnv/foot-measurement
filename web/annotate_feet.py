import argparse
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file

import app as measurement


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path(r"C:\Users\Admin\OneDrive\Desktop\111\group")
DEFAULT_MANIFEST = ROOT / "known_oriented_feature_report.json"
KEYPOINT_LABELS = (
    "longest_toe", "heel_back", "ball_inner", "ball_outer",
    "heel_inner", "heel_outer",
)
ROLES = {"top", "side_1", "side_2"}


def _read_image(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def _initial_mask(path):
    image = _read_image(path)
    if image is None:
        return None
    corners = measurement.detect_partial_paper_corners(image, 30)
    if corners is None:
        mask = measurement._learned_foot_mask(image)
    else:
        mask = measurement._multiview_foot_mask(
            image, corners, consensus=True, paper_margin_ratio=0.06)
        if mask is None:
            mask = measurement._multiview_foot_mask(
                image, corners, paper_margin_ratio=0.06)
        if mask is None:
            mask = measurement._learned_foot_mask(image, corners)
    return (np.zeros(image.shape[:2], np.uint8)
            if mask is None else mask)


def _load_items(source, manifest, limit):
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    items = []
    for row in rows:
        folder = source / row["folder"]
        for role, name in zip(("top", "side_1", "side_2"),
                              row["input_order"]):
            path = folder / name
            if path.is_file():
                items.append({
                    "group": row["group"], "role": role,
                    "name": name, "path": path,
                })
            if len(items) >= limit:
                return items
    return items


def create_app(source=DEFAULT_SOURCE, output=None, manifest=DEFAULT_MANIFEST,
               limit=30):
    source = Path(source).resolve()
    output = Path(output or ROOT / "datasets" / "foot_annotations_v1").resolve()
    manifest = Path(manifest).resolve()
    items = _load_items(source, manifest, limit)
    if not items:
        raise RuntimeError("没有找到待标注照片")

    masks = output / "masks"
    records = output / "records"
    masks.mkdir(parents=True, exist_ok=True)
    records.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(24)
    initial_cache = {}
    server = Flask(__name__)

    def item_at(index):
        if not 0 <= index < len(items):
            abort(404)
        item = items[index]
        key = f"g{item['group']:02d}_{item['role']}_{Path(item['name']).stem}"
        return item, key

    def record_for(index):
        item, key = item_at(index)
        path = records / f"{key}.json"
        data = json.loads(path.read_text("utf-8")) if path.exists() else {}
        return item, key, path, data

    @server.before_request
    def local_only():
        if request.remote_addr not in {"127.0.0.1", "::1"}:
            abort(403)

    @server.get("/")
    def index():
        return render_template("annotate_feet.html", token=token,
                               keypoint_labels=KEYPOINT_LABELS)

    @server.get("/api/items")
    def list_items():
        payload = []
        for index, item in enumerate(items):
            _, key = item_at(index)
            record_path = records / f"{key}.json"
            data = (json.loads(record_path.read_text("utf-8"))
                    if record_path.exists() else {})
            payload.append({
                "index": index, "group": item["group"],
                "role": data.get("role", item["role"]),
                "suggested_role": item["role"], "name": item["name"],
                "reviewed": bool(data),
            })
        return jsonify(payload)

    @server.get("/api/items/<int:index>")
    def get_item(index):
        item, _, _, data = record_for(index)
        image = _read_image(item["path"])
        if image is None:
            abort(404)
        return jsonify({
            "index": index, "group": item["group"],
            "role": data.get("role", item["role"]),
            "suggested_role": item["role"],
            "name": item["name"], "width": image.shape[1],
            "height": image.shape[0], "points": data.get("points", {}),
            "reviewed": bool(data),
        })

    @server.get("/images/<int:index>")
    def image(index):
        item, _ = item_at(index)
        return send_file(item["path"])

    @server.get("/masks/<int:index>.png")
    def mask(index):
        item, key = item_at(index)
        saved = masks / f"{key}.png"
        if saved.exists() and request.args.get("initial") != "1":
            return send_file(saved, mimetype="image/png")
        if index not in initial_cache:
            initial_cache[index] = _initial_mask(item["path"])
        value = initial_cache[index]
        if value is None:
            abort(404)
        ok, encoded = cv2.imencode(".png", value)
        if not ok:
            abort(500)
        return encoded.tobytes(), 200, {"Content-Type": "image/png"}

    @server.post("/api/items/<int:index>")
    def save_item(index):
        if not secrets.compare_digest(
                request.headers.get("X-Annotation-Token", ""), token):
            abort(403)
        item, key, record_path, _ = record_for(index)
        image = _read_image(item["path"])
        upload = request.files.get("mask")
        raw = np.frombuffer(upload.read(), np.uint8) if upload else None
        value = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE) if raw is not None else None
        if value is None or value.shape != image.shape[:2]:
            return jsonify({"error": "轮廓图尺寸与原图不一致"}), 400
        value = ((value >= 128).astype(np.uint8) * 255)
        role = request.form.get("role", "")
        if role not in ROLES:
            return jsonify({"error": "照片姿态无效"}), 400
        points = json.loads(request.form.get("points", "{}"))
        if not isinstance(points, dict) or set(points) - set(KEYPOINT_LABELS):
            return jsonify({"error": "关键点名称无效"}), 400
        for point in points.values():
            if (not isinstance(point, list) or len(point) != 2 or
                    not 0 <= point[0] < image.shape[1] or
                    not 0 <= point[1] < image.shape[0]):
                return jsonify({"error": "关键点坐标无效"}), 400
        if role != "top":
            points = {}
        mask_path = masks / f"{key}.png"
        if not cv2.imwrite(str(mask_path), value):
            abort(500)
        record = {
            "group": item["group"], "role": role,
            "suggested_role": item["role"],
            "source": str(item["path"].relative_to(source)),
            "mask": str(mask_path.relative_to(output)), "points": points,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        record_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), "utf-8")
        return jsonify({"saved": True})

    return server


def main():
    parser = argparse.ArgumentParser(description="真实足部轮廓标注工具")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--port", type=int, default=15001)
    args = parser.parse_args()
    server = create_app(args.source, args.output, args.manifest, args.limit)
    server.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
