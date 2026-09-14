import os
import uuid
import itertools
import csv
import hashlib
import hmac
import io
import json
import math
import re
import secrets
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import cv2
import numpy as np
from flask import (Flask, Response, abort, jsonify, redirect, render_template,
                   request, send_file, send_from_directory, session, url_for)
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['SAVE_TOKEN_SECRET'] = (
    os.environ.get('SAVE_TOKEN_SECRET') or secrets.token_urlsafe(32))
app.config['SECRET_KEY'] = (
    os.environ.get('ADMIN_SESSION_SECRET') or
    app.config['SAVE_TOKEN_SECRET'])
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Strict',
    SESSION_COOKIE_SECURE=(
        os.environ.get('SESSION_COOKIE_SECURE', '').lower() in
        {'1', 'true', 'yes'}),
)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
HEEL_WIDTH_RATIO = 0.15
HEEL_REAR_CHECK_RATIO = 0.03
PLANAR_PIXELS_PER_MM = 3
MAX_MULTIVIEW_REPROJECTION_PX = 7.0
ACCEPT_MULTIVIEW_REPROJECTION_PX = 4.0
WARN_MULTIVIEW_REPROJECTION_PX = 3.0
MIN_MULTIVIEW_SIDE_ANGLE_DEG = 8.0
HARD_MIN_SEGMENTATION_AGREEMENT = 0.65
WARN_SEGMENTATION_AGREEMENT = 0.80
HARD_MAX_HEEL_STABILITY_MM = 8.0
SAVE_TOKEN_TTL_SECONDS = 30 * 60

ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
_RESULT_FILE_RE = re.compile(r'^result_(?:multi_)?[0-9a-f]{32}\.png$')
_PAPER_RANKER = None
_MEASUREMENT_FUSION = None
_PAPER_SEGMENTER_LOCAL = threading.local()
_ADMIN_LOGIN_FAILURES = []
_ADMIN_LOGIN_LOCK = threading.Lock()
_MEASUREMENT_CODE_RE = re.compile(r'^FM-\d{6}$')


def _save_serializer():
    return URLSafeTimedSerializer(
        app.config['SAVE_TOKEN_SECRET'], salt='measurement-save-v1')


def _issue_save_token(result, foot_side):
    filename = os.path.basename(result['result_image'])
    if not _RESULT_FILE_RE.fullmatch(filename):
        raise ValueError('结果图文件名异常')
    return _save_serializer().dumps({
        'version': 1,
        'save_nonce': str(uuid.uuid4()),
        'foot_side': foot_side,
        'foot_length_mm': result['foot_length'],
        'ball_width_mm': result['ball_width'],
        'heel_width_mm': result['heel_width'],
        'quality_grade': result.get('quality_grade', 'unrated'),
        'dimension_confidence': result.get('dimension_confidence', {}),
        'warnings': result.get('warnings', []),
        'result_filename': filename,
    })


def _read_save_token(token):
    data = _save_serializer().loads(token, max_age=SAVE_TOKEN_TTL_SECONDS)
    if not isinstance(data, dict) or data.get('version') != 1:
        raise BadSignature('unsupported token')
    try:
        uuid.UUID(data['save_nonce'])
    except (KeyError, TypeError, ValueError) as exc:
        raise BadSignature('invalid nonce') from exc
    if data.get('foot_side') not in {'left', 'right'}:
        raise BadSignature('invalid foot side')
    filename = data.get('result_filename')
    if not isinstance(filename, str) or not _RESULT_FILE_RE.fullmatch(filename):
        raise BadSignature('invalid result filename')
    limits = {
        'foot_length_mm': (180, 350, False),
        'ball_width_mm': (60, 130, False),
        'heel_width_mm': (30, 100, True),
    }
    for key, (minimum, maximum, nullable) in limits.items():
        value = data.get(key)
        if value is None and nullable:
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not minimum <= value <= maximum):
            raise BadSignature(f'invalid {key}')
    if data.get('quality_grade') not in {'high', 'medium', 'low', 'unrated'}:
        raise BadSignature('invalid quality grade')
    confidence = data.get('dimension_confidence')
    if not isinstance(confidence, dict):
        raise BadSignature('invalid confidence')
    warnings = data.get('warnings')
    if (not isinstance(warnings, list) or len(warnings) > 20 or
            any(not isinstance(item, str) or len(item) > 500
                for item in warnings)):
        raise BadSignature('invalid warnings')
    return data


def _result_path(filename):
    upload_dir = os.path.abspath(app.config['UPLOAD_FOLDER'])
    path = os.path.abspath(os.path.join(upload_dir, filename))
    if os.path.dirname(path) != upload_dir:
        raise ValueError('结果图路径异常')
    return path


def _cleanup_expired_results():
    cutoff = time.time() - SAVE_TOKEN_TTL_SECONDS
    try:
        entries = os.scandir(app.config['UPLOAD_FOLDER'])
    except OSError:
        return
    with entries:
        for entry in entries:
            if (not entry.is_file(follow_symlinks=False) or
                    not _RESULT_FILE_RE.fullmatch(entry.name)):
                continue
            try:
                if entry.stat(follow_symlinks=False).st_mtime < cutoff:
                    os.remove(entry.path)
            except OSError:
                app.logger.warning('无法清理过期结果图：%s', entry.name)


def _result_cleanup_loop():
    while True:
        time.sleep(60)
        _cleanup_expired_results()


threading.Thread(
    target=_result_cleanup_loop, name='result-cleanup', daemon=True).start()


def _open_database_connection(database_url):
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError('PostgreSQL 驱动未安装') from exc
    return psycopg.connect(
        database_url, connect_timeout=3,
        options='-c statement_timeout=5000')


def _database_connection():
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise RuntimeError('DATABASE_URL 未配置')
    return _open_database_connection(database_url)


def _admin_database_connection():
    database_url = os.environ.get('ADMIN_DATABASE_URL')
    if not database_url:
        raise RuntimeError('ADMIN_DATABASE_URL 未配置')
    return _open_database_connection(database_url)


def _lookup_measurement_code(save_nonce):
    with _database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT public.lookup_measurement_code(%s::uuid)',
                (save_nonce,))
            row = cursor.fetchone()
    return row[0] if row else None


def _save_measurement_record(data, image_png):
    with _database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                '''SELECT public.save_shoe_measurement(
                    %s::uuid, %s::text, %s::numeric, %s::numeric,
                    %s::numeric, %s::text, %s::jsonb, %s::jsonb, %s::bytea
                )''',
                (data['save_nonce'], data['foot_side'],
                 data['foot_length_mm'], data['ball_width_mm'],
                 data['heel_width_mm'], data['quality_grade'],
                 json.dumps(data['dimension_confidence'], ensure_ascii=False),
                 json.dumps(data['warnings'], ensure_ascii=False), image_png))
            row = cursor.fetchone()
    if not row or not row[0]:
        raise RuntimeError('数据库未返回测量编号')
    return row[0]


def _admin_credentials():
    username = os.environ.get('ADMIN_USERNAME', '')
    password = os.environ.get('ADMIN_PASSWORD', '')
    return username, password


def _admin_session_tag(username, password):
    secret = app.config['SECRET_KEY']
    if isinstance(secret, str):
        secret = secret.encode('utf-8')
    return hmac.new(
        secret, f'{username}\0{password}'.encode('utf-8'),
        hashlib.sha256).hexdigest()


def _admin_authenticated():
    username, password = _admin_credentials()
    if not username or not password:
        return False
    expected = _admin_session_tag(username, password)
    actual = session.get('admin_auth', '')
    return isinstance(actual, str) and hmac.compare_digest(actual, expected)


def _admin_login_blocked(record_failure=False):
    now = time.monotonic()
    cutoff = now - 5 * 60
    with _ADMIN_LOGIN_LOCK:
        _ADMIN_LOGIN_FAILURES[:] = [
            value for value in _ADMIN_LOGIN_FAILURES if value >= cutoff]
        blocked = len(_ADMIN_LOGIN_FAILURES) >= 8
        if record_failure and not blocked:
            _ADMIN_LOGIN_FAILURES.append(now)
        return blocked


def _admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not _admin_authenticated():
            return redirect(url_for('admin_login'))
        return view(*args, **kwargs)
    return wrapped


def _admin_search_term(value):
    value = (value or '').strip().upper()
    if len(value) > 32 or (value and not re.fullmatch(r'[A-Z0-9-]+', value)):
        raise ValueError('编号格式无效')
    return value


def _admin_measurement_rows(search, page, per_page=25):
    where = ''
    parameters = []
    if search:
        where = 'WHERE measurement_code ILIKE %s'
        parameters.append(f'%{search}%')
    with _admin_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f'SELECT count(*) FROM public.shoe_measurements {where}',
                parameters)
            total = int(cursor.fetchone()[0])
            cursor.execute(
                f'''SELECT measurement_code, foot_side, foot_length_mm,
                           ball_width_mm, heel_width_mm, quality_grade,
                           dimension_confidence, warnings, created_at,
                           expires_at
                    FROM public.shoe_measurements
                    {where}
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s''',
                [*parameters, per_page, (page - 1) * per_page])
            rows = cursor.fetchall()
    columns = (
        'measurement_code', 'foot_side', 'foot_length_mm', 'ball_width_mm',
        'heel_width_mm', 'quality_grade', 'dimension_confidence', 'warnings',
        'created_at', 'expires_at')
    return [dict(zip(columns, row)) for row in rows], total


def _admin_measurement_image(measurement_code):
    with _admin_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                '''SELECT result_image_png
                   FROM public.shoe_measurements
                   WHERE measurement_code = %s''',
                (measurement_code,))
            row = cursor.fetchone()
    return None if row is None else bytes(row[0])


def _admin_measurement_export(search):
    where = ''
    parameters = []
    if search:
        where = 'WHERE measurement_code ILIKE %s'
        parameters.append(f'%{search}%')
    with _admin_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f'''SELECT measurement_code, foot_side, foot_length_mm,
                           ball_width_mm, heel_width_mm, quality_grade,
                           dimension_confidence, warnings, created_at,
                           expires_at
                    FROM public.shoe_measurements
                    {where}
                    ORDER BY created_at DESC''',
                parameters)
            return cursor.fetchall()


def _order_box(box):
    """按左上、右上、右下、左下排列四点。"""
    sums = box.sum(axis=1)
    diffs = np.diff(box, axis=1).ravel()
    return [
        box[np.argmin(sums)],
        box[np.argmin(diffs)],
        box[np.argmax(sums)],
        box[np.argmax(diffs)],
    ]


def _largest_mask_component(mask, min_ratio=0.01):
    """保留最大的有效连通区域，去掉地板反光和零散亮斑。"""
    h, w = mask.shape[:2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros_like(mask)

    min_area = h * w * min_ratio
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < min_area:
        return np.zeros_like(mask)
    return np.where(labels == best, 255, 0).astype(np.uint8)


def _visible_paper_mask(img):
    """识别没有被脚遮挡的白纸；允许纸张不完整或四角不可见。"""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.int16)

    # 阴影主要降低 L；纸张的 a/b 色度仍比木地板、皮肤更中性。
    mask = (
        (lab[:, :, 0] > 100) &
        (np.abs(a - 128) < 16) &
        (lab[:, :, 2] < 136)
    ).astype(np.uint8) * 255

    k = max(5, int(round(min(h, w) * 0.006)))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return _largest_mask_component(mask, min_ratio=0.025)


def segment_scene(img):
    """把原图互斥地分为 floor、paper、foot 三类。"""
    h, w = img.shape[:2]
    empty = np.zeros((h, w), np.uint8)

    def geometric_fallback():
        corners = detect_partial_paper_corners(img, 30)
        foot_mask = (_multiview_foot_mask(img, corners)
                     if corners is not None else None)
        if corners is None or foot_mask is None:
            return None
        paper_mask = empty.copy()
        cv2.fillConvexPoly(
            paper_mask, np.asarray(corners, np.int32), 255)
        foot_mask[paper_mask == 0] = 0
        paper_mask[foot_mask > 0] = 0
        floor_mask = cv2.bitwise_not(
            cv2.bitwise_or(paper_mask, foot_mask))
        return {'floor': floor_mask, 'paper': paper_mask, 'foot': foot_mask}

    paper = _visible_paper_mask(img)
    if cv2.countNonZero(paper) == 0:
        # 低角度、暖光或强阴影会让可见白纸色度失效；改用三边几何恢复。
        fallback = geometric_fallback()
        return (fallback if fallback is not None else
                {'floor': np.full((h, w), 255, np.uint8),
                 'paper': empty.copy(), 'foot': empty.copy()})

    # 四角完整时优先使用A4几何边界，可彻底排除纸外的木纹和反光。
    corners = auto_detect_corners(img)
    if corners:
        foot_contour = detect_foot_on_paper(img, corners)
        if foot_contour is not None:
            paper = empty.copy()
            cv2.fillConvexPoly(
                paper, np.asarray(corners, dtype=np.int32), 255)
            foot = empty.copy()
            cv2.drawContours(foot, [foot_contour], -1, 255, -1)
            foot[paper == 0] = 0
            paper[foot > 0] = 0
            floor = cv2.bitwise_not(cv2.bitwise_or(paper, foot))
            return {'floor': floor, 'paper': paper, 'foot': foot}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # 只从已确认的纸面取色，适应曝光和白平衡变化。
    ref = paper > 0
    paper_a = np.median(lab[:, :, 1][ref])
    paper_b = np.median(lab[:, :, 2][ref])
    paper_cr = np.median(ycrcb[:, :, 1][ref])
    paper_cb = np.median(ycrcb[:, :, 2][ref])

    skin = (
        (lab[:, :, 1] > paper_a + 2) &
        (lab[:, :, 2] > paper_b + 4) &
        (ycrcb[:, :, 1] > paper_cr + 7) &
        (ycrcb[:, :, 2] < paper_cb + 2) &
        (hsv[:, :, 1] > 12)
    ).astype(np.uint8) * 255

    # 用纸面凸包限定搜索范围，避免把木地板误当成脚。
    paper_points = cv2.findNonZero(paper)
    support = np.zeros((h, w), np.uint8)
    hull = cv2.convexHull(paper_points)
    cv2.fillConvexPoly(support, hull, 255)
    support_k = max(9, int(round(min(h, w) * 0.08)))
    support = cv2.dilate(
        support, np.ones((support_k, support_k), np.uint8), iterations=1)
    skin[support == 0] = 0

    k = max(5, int(round(min(h, w) * 0.008)))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    image_area = h * w
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 0.015 * image_area < area < 0.55 * image_area:
            continue
        component = np.zeros((h, w), np.uint8)
        cv2.drawContours(component, [contour], -1, 255, -1)
        overlap = cv2.countNonZero(cv2.bitwise_and(component, support)) / max(area, 1)
        x, y, cw, ch = cv2.boundingRect(contour)
        long_side = max(cw / w, ch / h)
        if overlap > 0.45 and long_side > 0.28:
            candidates.append((area * (1.0 + overlap), contour))

    foot = empty.copy()
    if candidates:
        contour = max(candidates, key=lambda item: item[0])[1]
        cv2.drawContours(foot, [contour], -1, 255, -1)

    # 脚优先于纸；剩余区域统一归为地板/背景。
    paper[foot > 0] = 0
    if cv2.countNonZero(paper) < h * w * 0.025:
        fallback = geometric_fallback()
        if fallback is not None:
            return fallback
    floor = cv2.bitwise_not(cv2.bitwise_or(paper, foot))
    return {'floor': floor, 'paper': paper, 'foot': foot}


def _find_paper_by_neutral_color(img):
    """用白纸的中性 Lab 色度找完整 A4，避免木纹直线干扰。"""
    h, w = img.shape[:2]

    # 脚把纸面分开时，严格亮度阈值只会找到一小块白纸。
    # 先用可见纸面外凸包恢复完整四角；形状不像A4则继续旧流程。
    visible = _visible_paper_mask(img)
    visible_contours, _ = cv2.findContours(
        visible, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if visible_contours:
        contour = max(visible_contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        corners = cv2.approxPolyDP(hull, 0.015 * perimeter, True)
        rect = cv2.minAreaRect(hull)
        rw, rh = rect[1]
        if min(rw, rh) > 10:
            ratio = max(rw, rh) / min(rw, rh)
            coverage = cv2.contourArea(corners) / (h * w)
            if (len(corners) == 4 and 1.15 < ratio < 1.75 and
                    0.03 < coverage < 0.90):
                return _order_box(corners.reshape(-1, 2).astype(np.float32))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.int16)

    # 白纸在阴影下亮度会下降，但 b 通道仍明显小于暖色木地板。
    mask = ((lab[:, :, 0] > 150) & (np.abs(a - 128) < 12) &
            (lab[:, :, 2] < 133)).astype(np.uint8) * 255
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000:
            continue
        rect = cv2.minAreaRect(cv2.convexHull(contour))
        rw, rh = rect[1]
        if min(rw, rh) < 10:
            continue
        ratio = max(rw, rh) / min(rw, rh)
        coverage = area / (h * w)
        if 1.15 < ratio < 1.65 and 0.03 < coverage < 0.60:
            candidates.append((abs(ratio - 1.414), -coverage, rect))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    box = cv2.boxPoints(candidates[0][2]).astype(np.float32)
    return _order_box(box)


def auto_detect_corners(img, allow_grabcut=True):
    neutral_result = _find_paper_by_neutral_color(img)
    if neutral_result:
        return neutral_result

    h, w = img.shape[:2]
    img_area = h * w

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_result = None
    hsv_contour_coverage = 0
    for v_min in [200, 180, 160]:
        candidates = []
        for s_max in [40, 50]:
            paper = ((hsv[:, :, 1] < s_max) & (hsv[:, :, 2] > v_min)).astype(np.uint8) * 255
            kernel = np.ones((15, 15), np.uint8)
            p = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, kernel, iterations=3)
            p = cv2.morphologyEx(p, cv2.MORPH_OPEN, kernel, iterations=2)
            contours, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 5000:
                    continue
                hull = cv2.convexHull(c)
                rect = cv2.minAreaRect(hull)
                rw, rh = rect[1]
                if rw < 10 or rh < 10:
                    continue
                ratio = max(rw, rh) / (min(rw, rh) + 1e-5)
                if ratio < 1.1 or ratio > 2.0:
                    continue
                coverage = area / img_area
                if coverage < 0.01 or coverage > 0.50:
                    continue
                ratio_score = 1.0 - abs(ratio - 1.414) / 0.6
                score = ratio_score * (1.0 + coverage)
                candidates.append((score, rect, coverage))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            if candidates[0][2] >= 0.03:
                best_rect = candidates[0][1]
                hsv_contour_coverage = candidates[0][2]
                box = cv2.boxPoints(best_rect).astype(np.float32)
                tl_idx = np.argmin(box.sum(axis=1))
                br_idx = np.argmax(box.sum(axis=1))
                tl, br = box[tl_idx], box[br_idx]
                remaining = [i for i in range(4) if i != tl_idx and i != br_idx]
                if box[remaining[0]][0] > box[remaining[1]][0]:
                    tr, bl = box[remaining[0]], box[remaining[1]]
                else:
                    tr, bl = box[remaining[1]], box[remaining[0]]
                hsv_result = [tl, tr, br, bl]
                break

    if hsv_result:
        hsv_area = _quad_area(np.array(hsv_result, dtype=np.float32))
        hsv_ratio = hsv_area / img_area
        if hsv_ratio < 0.25:
            return hsv_result
        line_result = _find_paper_by_lines(img)
        if line_result:
            return line_result
        if not allow_grabcut:
            return hsv_result
        return _grabcut_fallback(img, hsv, img_area)

    line_result = _find_paper_by_lines(img)
    if line_result:
        return line_result

    if not allow_grabcut:
        return None
    return _grabcut_fallback(img, hsv, img_area)


def _quad_area(pts):
    pts = pts.reshape(-1, 2).astype(np.float32)
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return 0.5 * abs(area)


def _paper_evidence_features(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    light = lab[:, :, 0]
    saturation = hsv[:, :, 1]
    light_60, light_85 = np.percentile(light, (60, 85))
    neutral_white = ((light >= light_85) &
                     (saturation < 48) & (lab[:, :, 2] < 140))
    neutral_paper = ((light >= light_60) &
                     (saturation < 55) & (lab[:, :, 2] < 145))
    return (neutral_white, neutral_paper,
            light - 1.2 * saturation.astype(np.float32),
            max(int(neutral_white.sum()), 1),
            max(int(neutral_paper.sum()), 1))


def _paper_polygon_roi(points, shape, padding):
    h, w = shape[:2]
    x, y, width, height = cv2.boundingRect(points.astype(np.float32))
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(w, x + width + padding), min(h, y + height + padding)
    if x1 <= x0 or y1 <= y0:
        return None, None
    polygon = np.zeros((y1 - y0, x1 - x0), np.uint8)
    shifted = points - np.float32([x0, y0])
    cv2.fillConvexPoly(polygon, shifted.astype(np.int32), 255)
    return polygon, (slice(y0, y1), slice(x0, x1))


def _paper_quad_evidence(img, quad, features=None):
    """评估四边形内是否真的白纸。

    不要只看边线强度：脚边、阴影和地面缝同样会产生强直线。
    真实 A4 框应包含画面中大部分高亮、低饱和的中性像素，
    且框内应比外侧环带更亮。
    """
    h, w = img.shape[:2]
    points = np.asarray(quad, np.float32)
    if points.shape != (4, 2) or not cv2.isContourConvex(
            points.astype(np.int32)):
        return -10.0
    border = max(7, int(round(min(h, w) * 0.010))) | 1
    polygon, roi = _paper_polygon_roi(points, (h, w), border)
    if polygon is None:
        return -10.0
    area = cv2.countNonZero(polygon)
    if area < h * w * 0.025:
        return -10.0

    kernel = np.ones((border, border), np.uint8)
    inner = cv2.erode(polygon, kernel) > 0
    ring = cv2.dilate(polygon, kernel)
    ring[polygon > 0] = 0
    outside = ring > 0
    if inner.sum() < 500 or outside.sum() < 100:
        return -10.0

    neutral_white, neutral_paper, paper_value, white_total, paper_total = (
        features if features is not None else _paper_evidence_features(img))
    neutral_white = neutral_white[roi]
    neutral_paper = neutral_paper[roi]
    paper_value = paper_value[roi]
    white_inside = int(np.logical_and(neutral_white, inner).sum())
    coverage = white_inside / white_total
    broad_coverage = int(np.logical_and(
        neutral_paper, inner).sum()) / paper_total
    density = white_inside / max(int(inner.sum()), 1)
    contrast = (float(np.median(paper_value[inner])) -
                float(np.median(paper_value[outside])))
    # 小框套住一块白纸时密度很高，但会漏掉其余纸面。测量标定必须覆盖
    # 绝大多数高亮中性纸面，不能让脚踝边缘组成的小四边形胜出。
    # 灰色地面本身也会进入“中性高亮”统计，导致真实纸框覆盖率只有约
    # 0.5；此时要求框内密度、宽松纸色覆盖和内外亮度差同时成立。
    relaxed_gray_floor = (
        coverage >= 0.45 and broad_coverage >= 0.16 and
        density >= 0.40 and contrast >= 20.0)
    if coverage < 0.68 and not relaxed_gray_floor:
        return -10.0
    area_ratio = area / (h * w)
    oversized = max(0.0, area_ratio - 0.58)
    return (coverage * 2.0 + broad_coverage * 8.0 + density * 0.10 +
            np.clip(contrast, -30.0, 100.0) / 80.0 -
            oversized * 12.0)


def _paper_quad_recovery_evidence(img, quad, features=None, base=None):
    """恢复分支更看重框内纸面纯度，防止灰地板被并入A4框。"""
    if base is None:
        base = _paper_quad_evidence(img, quad, features)
    if base < 0.5:
        return base
    h, w = img.shape[:2]
    points = np.asarray(quad, np.float32)
    border = max(7, int(round(min(h, w) * 0.010))) | 1
    polygon, roi = _paper_polygon_roi(points, (h, w), border)
    if polygon is None:
        return -10.0
    kernel = np.ones((border, border), np.uint8)
    inner = cv2.erode(polygon, kernel) > 0
    ring = cv2.dilate(polygon, kernel)
    ring[polygon > 0] = 0
    outside = ring > 0
    if inner.sum() < 500 or outside.sum() < 100:
        return -10.0
    neutral_white, neutral_paper, paper_value, white_total, paper_total = (
        features if features is not None else _paper_evidence_features(img))
    neutral_white = neutral_white[roi]
    neutral_paper = neutral_paper[roi]
    paper_value = paper_value[roi]
    white_inside = int(np.logical_and(neutral_white, inner).sum())
    coverage = white_inside / white_total
    broad_coverage = int(np.logical_and(
        neutral_paper, inner).sum()) / paper_total
    density = white_inside / max(int(inner.sum()), 1)
    contrast = (float(np.median(paper_value[inner])) -
                float(np.median(paper_value[outside])))
    area_ratio = float(inner.sum() / (h * w))
    if area_ratio > 0.42 and density < 0.34:
        return -10.0
    edge_strengths, _ = _paper_quad_edge_support(img, points, features)
    # 一条纸边可被脚跟完全遮挡；其余三条必须有真实的纸/地面亮度跃迁。
    if sum(value >= 15.0 for value in edge_strengths) < 3:
        return -10.0
    oversized = max(0.0, area_ratio - 0.48)
    return (coverage * 2.0 + broad_coverage * 2.0 + density * 4.0 +
            np.clip(contrast, -30.0, 120.0) / 40.0 -
            oversized * 12.0)


def _paper_quad_edge_support(img, quad, features=None):
    """返回每条候选边的亮度跃迁及整边覆盖率。"""
    h, w = img.shape[:2]
    points = np.asarray(quad, np.float32)
    paper_value = ((features if features is not None else
                    _paper_evidence_features(img))[2])
    center = points.mean(axis=0)
    edge_strengths = []
    edge_coverages = []
    for index in range(4):
        start = points[index]
        end = points[(index + 1) % 4]
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 20:
            edge_strengths.append(-math.inf)
            edge_coverages.append(0.0)
            continue
        inward = np.array([-direction[1], direction[0]]) / length
        if np.dot(center - (start + end) / 2, inward) < 0:
            inward = -inward
        samples = start + np.linspace(0.08, 0.92, 36)[:, None] * direction
        strengths = []
        point_strengths = []
        for offset in (5.0, 12.0, 22.0):
            inside = samples + offset * inward
            outside_points = samples - offset * inward
            for sample_points in (inside, outside_points):
                sample_points[:, 0] = np.clip(sample_points[:, 0], 0, w - 1)
                sample_points[:, 1] = np.clip(sample_points[:, 1], 0, h - 1)
            inside = np.rint(inside).astype(np.int32)
            outside_points = np.rint(outside_points).astype(np.int32)
            differences = (paper_value[inside[:, 1], inside[:, 0]] -
                           paper_value[outside_points[:, 1],
                                       outside_points[:, 0]])
            strengths.append(float(np.median(differences)))
            point_strengths.append(differences)
        edge_strengths.append(max(strengths))
        edge_coverages.append(float(np.mean(
            np.max(point_strengths, axis=0) >= 10.0)))
    return edge_strengths, edge_coverages


def _paper_candidate_rank_features(img, quad, evidence_features=None,
                                   evidence=None, recovery=None):
    """纸框候选的通用外观/几何特征；供轻量排序器使用。"""
    h, w = img.shape[:2]
    points = np.asarray(quad, np.float32)
    if evidence_features is None:
        evidence_features = _paper_evidence_features(img)
    if evidence is None:
        evidence = _paper_quad_evidence(
            img, points, features=evidence_features)
    if recovery is None:
        recovery = _paper_quad_recovery_evidence(
            img, points, features=evidence_features, base=evidence)
    strengths, coverages = _paper_quad_edge_support(
        img, points, features=evidence_features)
    edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    area = abs(float(cv2.contourArea(points))) / max(float(h * w), 1.0)
    center = points.mean(axis=0) / np.float32([w, h])
    normalized = (points / np.float32([w, h])).reshape(-1)
    safe_edges = np.maximum(edges, 1.0)
    values = [
        np.clip(evidence, -10.0, 12.0) / 12.0,
        np.clip(recovery, -10.0, 12.0) / 12.0,
        math.log(max(area, 1e-4)),
        float(center[0]), float(center[1]),
        *(edges / np.float32([w, h, w, h])).tolist(),
        math.log(float(safe_edges[0] / safe_edges[2])),
        math.log(float(safe_edges[1] / safe_edges[3])),
        *(np.clip(strengths, -20.0, 100.0) / 50.0).tolist(),
        *coverages,
        *normalized.tolist(),
    ]
    return np.asarray(values, np.float32)


def _paper_candidate_rank_score(img, quad, evidence_features=None,
                                evidence=None, recovery=None):
    """公开四角真值训练的小模型；缺失时退回传统证据，不影响服务。"""
    global _PAPER_RANKER
    if _PAPER_RANKER is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'models', 'paper_candidate_ranker.npz')
        if not os.path.exists(path):
            _PAPER_RANKER = False
        else:
            with np.load(path) as data:
                _PAPER_RANKER = {key: data[key] for key in data.files}
    if _PAPER_RANKER is False:
        return 0.0
    model = _PAPER_RANKER
    values = ((_paper_candidate_rank_features(
        img, quad, evidence_features, evidence, recovery) - model['mean']) /
              model['scale'])
    values = np.maximum(model['weight1'] @ values + model['bias1'], 0.0)
    values = np.maximum(model['weight2'] @ values + model['bias2'], 0.0)
    value = float((model['weight3'] @ values + model['bias3']).item())
    return 1.0 / (1.0 + math.exp(-float(np.clip(value, -20.0, 20.0))))


def _learned_paper_candidate(img):
    """轻量分割模型补充纸面候选；几何标定仍负责最终取舍。"""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'models',
                              'paper_lraspp_amodal_v1.onnx')
    if not os.path.exists(model_path):
        return None
    net = getattr(_PAPER_SEGMENTER_LOCAL, 'net', None)
    if net is None:
        try:
            net = cv2.dnn.readNetFromONNX(model_path)
        except cv2.error:
            return None
        _PAPER_SEGMENTER_LOCAL.net = net
    resized = cv2.resize(img, (384, 512), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.float32([0.485, 0.456, 0.406])) / np.float32(
        [0.229, 0.224, 0.225])
    try:
        net.setInput(rgb.transpose(2, 0, 1)[None])
        logits = net.forward()[0, 0]
    except cv2.error:
        return None
    mask = (logits >= math.log(0.45 / 0.55)).astype(np.uint8) * 255
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    image_area = img.shape[0] * img.shape[1]
    choices = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.015:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        polygons = []
        for ratio in np.linspace(0.005, 0.08, 76):
            polygon = cv2.approxPolyDP(hull, ratio * perimeter, True)
            if len(polygon) == 4:
                polygons.append(polygon[:, 0].astype(np.float32))
        quad = (max(polygons, key=lambda value: abs(cv2.contourArea(value)))
                if polygons else cv2.boxPoints(cv2.minAreaRect(hull)))
        quad = np.asarray(_order_box(quad), np.float32)
        edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        if edges.max() / max(edges.min(), 1.0) > 3.5:
            continue
        quad_area = max(abs(cv2.contourArea(quad)), 1.0)
        evidence = _paper_quad_evidence(img, quad)
        rank = _paper_candidate_rank_score(img, quad)
        fill = min(area / quad_area, 1.0)
        choices.append((evidence + rank * 2.0 + fill, quad))
    return None if not choices else max(choices, key=lambda item: item[0])[1]


def _paper_corner_agreement_count(first, second, image_shape):
    """统计两种独立检测落在同一纸角的位置数。"""
    if first is None or second is None:
        return 0
    limit = max(10.0, min(image_shape[:2]) * 0.045)
    distances = np.linalg.norm(
        np.asarray(first, np.float32) - np.asarray(second, np.float32), axis=1)
    return int(np.sum(distances <= limit))


def _find_paper_by_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_area = h * w

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 90)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=100, maxLineGap=15)
    if lines is None:
        return None

    horiz = []
    vert = []
    for l in lines:
        x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if angle < 15 or angle > 165:
            horiz.append((x1, y1, x2, y2, length))
        elif 75 < angle < 105:
            vert.append((x1, y1, x2, y2, length))

    if len(horiz) < 2 or len(vert) < 2:
        return None

    horiz.sort(key=lambda l: l[4], reverse=True)
    vert.sort(key=lambda l: l[4], reverse=True)
    horiz = horiz[:20]
    vert = vert[:20]

    horiz.sort(key=lambda l: (l[1] + l[3]) / 2)
    vert.sort(key=lambda l: (l[0] + l[2]) / 2)

    best = None
    best_score = -1
    for i in range(len(horiz)):
        for j in range(i + 1, len(horiz)):
            y1 = (horiz[i][1] + horiz[i][3]) / 2
            y2 = (horiz[j][1] + horiz[j][3]) / 2
            h_dist = abs(y2 - y1)
            if h_dist < 100:
                continue
            for k in range(len(vert)):
                for l_idx in range(k + 1, len(vert)):
                    x1 = (vert[k][0] + vert[k][2]) / 2
                    x2 = (vert[l_idx][0] + vert[l_idx][2]) / 2
                    v_dist = abs(x2 - x1)
                    if v_dist < 100:
                        continue
                    ratio = max(h_dist, v_dist) / (min(h_dist, v_dist) + 1e-5)
                    if 1.2 < ratio < 1.8:
                        ratio_score = 1.0 - abs(ratio - 1.414) / 0.4
                        size_score = min(h_dist, v_dist) / max(w, h)
                        score = ratio_score * size_score
                        if score > best_score:
                            best_score = score
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            best = (cx, cy, v_dist, h_dist, ratio)

    if best is None:
        return None

    cx, cy, vw, vh, ratio = best

    if ratio >= 1.414:
        long_side = vh
        short_side = vw
    else:
        long_side = vw
        short_side = vh

    tl = np.array([cx - short_side / 2, cy - long_side / 2], dtype=np.float32)
    tr = np.array([cx + short_side / 2, cy - long_side / 2], dtype=np.float32)
    br = np.array([cx + short_side / 2, cy + long_side / 2], dtype=np.float32)
    bl = np.array([cx - short_side / 2, cy + long_side / 2], dtype=np.float32)

    return [tl, tr, br, bl]


def _grabcut_fallback(img, hsv, img_area, _scaled=False):
    h, w = img.shape[:2]
    if not _scaled and max(h, w) > 600:
        scale = 600.0 / max(h, w)
        small = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        recovered = _grabcut_fallback(
            small, cv2.cvtColor(small, cv2.COLOR_BGR2HSV),
            small.shape[0] * small.shape[1], _scaled=True)
        if recovered is None:
            return None
        return _refine_paper_edges(
            img, np.asarray(recovered, np.float32) / scale).tolist()

    seed_mask = np.zeros((h, w), np.uint8)
    seed_mask[:] = cv2.GC_BGD

    texture_result = _find_paper_by_neutral_color(img)
    if texture_result:
        pts = np.array(texture_result, dtype=np.int32)
        cv2.fillConvexPoly(seed_mask, pts, cv2.GC_PR_FGD)
        dilate_k = np.ones((30, 30), np.uint8)
        expanded = cv2.dilate(seed_mask, dilate_k, iterations=2)
        seed_mask[expanded > 0] = cv2.GC_PR_FGD
    else:
        cy, cx = h // 2, w // 2
        ry, rx = int(h * 0.35), int(w * 0.35)
        seed_mask[cy-ry:cy+ry, cx-rx:cx+rx] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        mask = seed_mask.copy()
        cv2.grabCut(img, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

        result_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        kernel2 = np.ones((7, 7), np.uint8)
        result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel2, iterations=3)
        result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_OPEN, kernel2, iterations=2)

        cnts, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        c = max(cnts, key=cv2.contourArea)
        for eps_mult in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
            epsilon = eps_mult * cv2.arcLength(c, True)
            corners = cv2.approxPolyDP(c, epsilon, True)
            if len(corners) == 4:
                pts = corners.reshape(-1, 2).astype(np.float32)
                rect = cv2.minAreaRect(pts)
                rw, rh = rect[1]
                if rw < 10 or rh < 10:
                    continue
                ratio = max(rw, rh) / (min(rw, rh) + 1e-5)
                if ratio < 1.1 or ratio > 2.0:
                    continue
                coverage = cv2.contourArea(c) / img_area
                if coverage < 0.03 or coverage > 0.40:
                    continue
                box = cv2.boxPoints(rect).astype(np.float32)
                tl_idx = np.argmin(box.sum(axis=1))
                br_idx = np.argmax(box.sum(axis=1))
                tl, br = box[tl_idx], box[br_idx]
                remaining = [i for i in range(4) if i != tl_idx and i != br_idx]
                if box[remaining[0]][0] > box[remaining[1]][0]:
                    tr, bl = box[remaining[0]], box[remaining[1]]
                else:
                    tr, bl = box[remaining[1]], box[remaining[0]]
                return [tl, tr, br, bl]
    except Exception:
        pass

    return None


def build_transform(corners, out_size=None):
    src = np.float32(corners)
    scale = 3.0
    out_w = int(A4_WIDTH_MM * scale) if out_size is None else out_size[0]
    out_h = int(A4_HEIGHT_MM * scale) if out_size is None else out_size[1]
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    mm_per_px = A4_HEIGHT_MM / out_h
    return M, mm_per_px, (out_w, out_h)


def _uniform_paper_mask(img, saturation_max=48):
    """找低纹理白纸。亮度只设宽松下限，阴影不会直接丢失纸面。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean = cv2.boxFilter(gray, -1, (15, 15))
    mean_sq = cv2.boxFilter(gray * gray, -1, (15, 15))
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    mask = ((local_std < 9) & (hsv[:, :, 1] < saturation_max) &
            (hsv[:, :, 2] > 115)).astype(np.uint8) * 255
    size = max(9, int(round(min(img.shape[:2]) * 0.015)))
    size += 1 - size % 2
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8),
        iterations=2)
    small = max(5, size // 2) | 1
    return cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((small, small), np.uint8))


def _neutralize_bright_color_cast(img):
    """仅供纸边失败回退：把最亮低纹理区域视作白纸并校正色偏。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, -1, (15, 15))
    mean_sq = cv2.boxFilter(gray * gray, -1, (15, 15))
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    reference = ((lab[:, :, 0] >= np.percentile(lab[:, :, 0], 85)) &
                 (local_std < 12))
    if reference.sum() < img.shape[0] * img.shape[1] * 0.005:
        return img
    for channel in (1, 2):
        shift = float(np.clip(
            128.0 - np.median(lab[:, :, channel][reference]), -30, 30))
        lab[:, :, channel] = np.clip(
            lab[:, :, channel] + shift, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _color_cast_strength(img):
    """低分辨率判断色偏；无需为质量门控处理整张标准图。"""
    scale = min(1.0, 480.0 / max(img.shape[:2]))
    sample = (img if scale == 1.0 else cv2.resize(
        img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    corrected = _neutralize_bright_color_cast(sample)
    return float(np.mean(cv2.absdiff(sample, corrected)))


def _severe_multiview_color_cast(strengths):
    """单张轻微偏色可恢复；三张持续强偏色才硬拒。"""
    return bool(strengths and min(strengths) > 18.0)


def _line_intersection(first, second):
    x1, y1 = first[0]
    x2, y2 = first[1]
    x3, y3 = second[0]
    x4, y4 = second[1]
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3
    denominator = dx1 * dy2 - dy1 * dx2
    if abs(denominator) < 1e-6:
        return None
    offset_x, offset_y = x3 - x1, y3 - y1
    distance = (offset_x * dy2 - offset_y * dx2) / denominator
    return np.array([x1 + distance * dx1, y1 + distance * dy1])


def _line_angle(line):
    delta = line[1] - line[0]
    return math.degrees(math.atan2(delta[1], delta[0])) % 180


def _angle_delta(first, second):
    delta = abs(first - second) % 180
    return min(delta, 180 - delta)


def detect_partial_paper_corners(img, saturation_max=48, _scaled=False):
    """从可见纸边恢复 A4 四角；允许一角及相邻纸边被脚遮住。"""
    if not _scaled and max(img.shape[:2]) > 720:
        scale = 720.0 / max(img.shape[:2])
        small = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        recovered = detect_partial_paper_corners(
            small, saturation_max, _scaled=True)
        if recovered is None:
            return None
        quad = np.asarray(recovered, np.float32) / scale
        return _refine_paper_edges(img, quad).tolist()
    h, w = img.shape[:2]
    mask = _uniform_paper_mask(img, saturation_max)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None

    candidates = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:5] + 1
    components = [np.where(labels == label, 255, 0).astype(np.uint8)
                  for label in candidates
                  if stats[label, cv2.CC_STAT_AREA] >= h * w * 0.025]

    # 脚会把纸面切成左右两块；合并高亮、中性的大块纸面后再拟合四边。
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    light_limit = np.percentile(lab[:, :, 0], 78)
    bright = ((lab[:, :, 0] > light_limit) &
              (lab[:, :, 2] < 134) &
              (hsv[:, :, 1] < 35)).astype(np.uint8) * 255
    kernel_size = max(9, int(round(min(h, w) * 0.016))) | 1
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_CLOSE,
        np.ones((kernel_size, kernel_size), np.uint8), iterations=2)
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_OPEN,
        np.ones((max(5, kernel_size // 2) | 1,) * 2, np.uint8))
    bright_count, bright_labels, bright_stats, _ = \
        cv2.connectedComponentsWithStats(bright, 8)
    combined = np.zeros_like(bright)
    for label in range(1, bright_count):
        if bright_stats[label, cv2.CC_STAT_AREA] >= h * w * 0.01:
            combined[bright_labels == label] = 255
    if cv2.countNonZero(combined) >= h * w * 0.025:
        components.append(combined)

    best = None
    for component in components:
        visible_area = cv2.countNonZero(component)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hull = cv2.convexHull(cv2.findNonZero(component))
        perimeter = cv2.arcLength(hull, True)
        points = None
        for epsilon in (0.003, 0.005, 0.008, 0.012, 0.018, 0.025, 0.035):
            approx = cv2.approxPolyDP(
                hull, epsilon * perimeter, True).reshape(-1, 2)
            if 4 <= len(approx) <= 7:
                points = approx.astype(np.float64)
                break
        if points is None:
            continue

        edges = [(points[i], points[(i + 1) % len(points)])
                 for i in range(len(points))]
        for indices in itertools.combinations(range(len(edges)), 4):
            sides = [edges[i] for i in indices]
            vertices = []
            for i in range(4):
                point = _line_intersection(sides[i - 1], sides[i])
                if point is None:
                    break
                vertices.append(point)
            if len(vertices) != 4:
                continue

            quad = np.asarray(_order_box(
                np.asarray(vertices, np.float32)), np.float32)
            if not cv2.isContourConvex(quad.astype(np.int32)):
                continue
            margin_x, margin_y = w * 0.1, h * 0.1
            if (np.any(quad[:, 0] < -margin_x) or
                    np.any(quad[:, 0] > w + margin_x) or
                    np.any(quad[:, 1] < -margin_y) or
                    np.any(quad[:, 1] > h + margin_y)):
                continue

            # 纸角可以贴近画面边缘，但整条“纸边”不能由画框冒充。
            frame_gap = max(2.0, min(h, w) * 0.003)
            frame_sides = [
                max(quad[i, 0], quad[(i + 1) % 4, 0]) < frame_gap or
                min(quad[i, 0], quad[(i + 1) % 4, 0]) > w - frame_gap or
                max(quad[i, 1], quad[(i + 1) % 4, 1]) < frame_gap or
                min(quad[i, 1], quad[(i + 1) % 4, 1]) > h - frame_gap
                for i in range(4)
            ]
            if any(frame_sides):
                continue

            quad_area = cv2.contourArea(quad)
            if not visible_area * 0.95 < quad_area < visible_area * 4.0:
                continue
            quad_mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(quad_mask, quad.astype(np.int32), 255)
            inside = cv2.countNonZero(
                cv2.bitwise_and(quad_mask, component)) / visible_area
            if inside < 0.94:
                continue

            angles = [_line_angle(side) for side in sides]
            opposite_error = (_angle_delta(angles[0], angles[2]) +
                              _angle_delta(angles[1], angles[3]))
            top = np.linalg.norm(quad[1] - quad[0])
            bottom = np.linalg.norm(quad[2] - quad[3])
            left = np.linalg.norm(quad[3] - quad[0])
            right = np.linalg.norm(quad[2] - quad[1])
            portrait_ratio = (left + right) / (top + bottom + 1e-6)
            if portrait_ratio < 0.75:
                continue
            side_support = sum(np.linalg.norm(side[1] - side[0])
                               for side in sides) / max(h, w)
            score = (inside * 5 - opposite_error / 45 + side_support +
                     min(portrait_ratio, 2) * 0.2 +
                     quad_area / (h * w) * 2)
            if best is None or score > best[0]:
                best = (score, quad)
    if best is None:
        return None
    result = _refine_paper_edges(img, best[1])
    frame_hits = sum(
        x < 2 or x > w - 3 or y < 2 or y > h - 3
        for x, y in result)
    if saturation_max > 30 and (
            cv2.contourArea(result) > h * w * 0.72 or frame_hits >= 2):
        strict = detect_partial_paper_corners(img, 30)
        if strict is not None:
            return strict
    return result.tolist()


def _refine_paper_edges(img, quad):
    """低纹理掩膜会向纸内收缩约半个窗口；回到原图梯度寻找真实纸边。"""
    gray = cv2.GaussianBlur(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    quad = np.asarray(quad, np.float64)
    center = quad.mean(axis=0)
    shifted = []
    for index in range(4):
        start, end = quad[index], quad[(index + 1) % 4]
        direction = end - start
        length = np.linalg.norm(direction)
        direction /= max(length, 1e-6)
        inward = np.array([-direction[1], direction[0]])
        midpoint = (start + end) / 2
        if np.dot(inward, center - midpoint) < 0:
            inward *= -1
        ratios = np.linspace(0.04, 0.96, max(80, int(length / 4)))
        samples = start + ratios[:, None] * (end - start)
        offsets = np.arange(-25.0, 6.1, 0.5)
        outer = samples[None, :, :] + (
            offsets[:, None, None] - 2) * inward
        inner = samples[None, :, :] + (
            offsets[:, None, None] + 2) * inward
        outer_x = np.clip(np.rint(outer[:, :, 0]).astype(np.int32),
                          0, gray.shape[1] - 1)
        outer_y = np.clip(np.rint(outer[:, :, 1]).astype(np.int32),
                          0, gray.shape[0] - 1)
        inner_x = np.clip(np.rint(inner[:, :, 0]).astype(np.int32),
                          0, gray.shape[1] - 1)
        inner_y = np.clip(np.rint(inner[:, :, 1]).astype(np.int32),
                          0, gray.shape[0] - 1)
        contrasts = np.abs(
            gray[outer_y, outer_x] - gray[inner_y, inner_x])
        strengths = np.percentile(contrasts, 70, axis=1)
        # 与 max((strength, offset), ...) 一致：强度相同时取较大偏移。
        best_index = len(strengths) - 1 - int(np.argmax(strengths[::-1]))
        strength = float(strengths[best_index])
        offset = offsets[best_index]
        if strength < 15:
            offset = 0
        shifted.append((start + offset * inward, end + offset * inward))

    vertices = [_line_intersection(shifted[i - 1], shifted[i])
                for i in range(4)]
    if any(point is None for point in vertices):
        return quad.astype(np.float32)
    return np.asarray(vertices, np.float32)


def _subpixel_paper_candidate(img, quad):
    """在已定位纸边附近鲁棒拟合梯度点；只生成候选，不直接替换。"""
    quad = np.asarray(quad, np.float64)
    if quad.shape != (4, 2) or not np.all(np.isfinite(quad)):
        return quad.astype(np.float32)
    gray = cv2.GaussianBlur(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0
    ).astype(np.float32)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    fitted_lines = []

    for index in range(4):
        start, end = quad[index], quad[(index + 1) % 4]
        direction = end - start
        length = np.linalg.norm(direction)
        direction /= max(length, 1e-6)
        normal = np.array([-direction[1], direction[0]])
        lower = np.maximum(
            np.floor(np.minimum(start, end) - 6).astype(np.int32), 0)
        upper = np.minimum(
            np.ceil(np.maximum(start, end) + 6).astype(np.int32),
            [img.shape[1] - 1, img.shape[0] - 1])
        # 外推纸角可能位于画面外；无有效搜索区域时沿用原边，不能让
        # np.mgrid 因负尺寸中断整个测量请求。
        if np.any(upper < lower):
            fitted_lines.append((start, end))
            continue
        yy, xx = np.mgrid[lower[1]:upper[1] + 1,
                          lower[0]:upper[0] + 1]
        relative_x = xx - start[0]
        relative_y = yy - start[1]
        along = relative_x * direction[0] + relative_y * direction[1]
        across = relative_x * normal[0] + relative_y * normal[1]
        local_magnitude = magnitude[yy, xx]
        alignment = np.abs(
            gradient_x[yy, xx] * normal[0] +
            gradient_y[yy, xx] * normal[1]
        ) / (local_magnitude + 1e-6)
        band = ((along > length * 0.05) & (along < length * 0.95) &
                (np.abs(across) < 4) & (alignment > 0.72))
        strengths = local_magnitude[band]
        if strengths.size < 30:
            fitted_lines.append((start, end))
            continue
        selected = band & (local_magnitude >= max(
            12.0, float(np.percentile(strengths, 65))))
        points = np.column_stack((xx[selected], yy[selected])).astype(
            np.float32)
        if len(points) < 20:
            fitted_lines.append((start, end))
            continue

        vx, vy, cx, cy = cv2.fitLine(
            points, cv2.DIST_WELSCH, 0, 0.01, 0.01).ravel()
        fitted_direction = np.array([vx, vy], np.float64)
        fitted_direction /= max(np.linalg.norm(fitted_direction), 1e-6)
        if np.dot(fitted_direction, direction) < 0:
            fitted_direction *= -1
        angle = math.degrees(math.acos(np.clip(
            abs(np.dot(fitted_direction, direction)), 0, 1)))
        fitted_center = np.array([cx, cy], np.float64)
        delta = fitted_center - (start + end) / 2
        shift = abs(direction[0] * delta[1] - direction[1] * delta[0])
        if angle > 2.0 or shift > 3.0:
            fitted_lines.append((start, end))
            continue
        fitted_lines.append((
            fitted_center - fitted_direction * length,
            fitted_center + fitted_direction * length))

    vertices = [_line_intersection(fitted_lines[index - 1],
                                   fitted_lines[index])
                for index in range(4)]
    if any(point is None for point in vertices):
        return quad.astype(np.float32)
    result = np.asarray(vertices, np.float32)
    if (np.max(np.linalg.norm(result - quad, axis=1)) > 6 or
            not cv2.isContourConvex(result.astype(np.int32))):
        return quad.astype(np.float32)
    return result


def _find_paper_by_border_lines(img, relaxed=True, _scaled=False):
    """由四条真实纸边求交，适合脚遮住纸角但边线仍可见的斜拍。"""
    if not _scaled and max(img.shape[:2]) > 900:
        scale = 900.0 / max(img.shape[:2])
        small = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        recovered = _find_paper_by_border_lines(
            small, relaxed=relaxed, _scaled=True)
        if recovered is None:
            return None
        quad = np.asarray(recovered, np.float32) / scale
        return _refine_paper_edges(img, quad).tolist()
    h, w = img.shape[:2]
    gray = cv2.GaussianBlur(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 25, 85) if relaxed else cv2.Canny(
        gray, 35, 105)
    detected = cv2.HoughLinesP(
        edges, 1, np.pi / (720 if relaxed else 360),
        30 if relaxed else 45,
        minLineLength=int(min(h, w) * (0.06 if relaxed else 0.14)),
        maxLineGap=int(min(h, w) * (0.08 if relaxed else 0.045)))
    if detected is None:
        return None

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    paper_score = lab[:, :, 0] - 1.2 * hsv[:, :, 1]
    light_60, light_85 = np.percentile(lab[:, :, 0], (60, 85))
    neutral_white = ((lab[:, :, 0] >= light_85) &
                     (hsv[:, :, 1] < 48) & (lab[:, :, 2] < 140))
    neutral_paper = ((lab[:, :, 0] >= light_60) &
                     (hsv[:, :, 1] < 55) & (lab[:, :, 2] < 145))
    white_total = max(int(neutral_white.sum()), 1)
    paper_total = max(int(neutral_paper.sum()), 1)
    horizontal, vertical = [], []
    for raw in detected.reshape(-1, 4):
        start = raw[:2].astype(np.float64)
        end = raw[2:].astype(np.float64)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 0:
            continue
        angle = _line_angle((start, end))
        horizontal_limit = 40 if relaxed else 35
        vertical_limit = 50 if relaxed else 55
        kind = ('h' if angle < horizontal_limit or
                angle > 180 - horizontal_limit else
                'v' if vertical_limit < angle <
                180 - vertical_limit else None)
        if kind is None:
            continue
        normal = np.array([-delta[1], delta[0]]) / length
        ratios = np.linspace(0.12, 0.88, max(24, int(length / 10)))
        samples = start + ratios[:, None] * delta
        contrasts = []
        offsets = ((4.0, 8.0, 12.0, 16.0) if relaxed else
                   (5.0, 10.0))
        for offset in offsets:
            first = samples + offset * normal
            second = samples - offset * normal
            first[:, 0] = np.clip(first[:, 0], 0, w - 1)
            first[:, 1] = np.clip(first[:, 1], 0, h - 1)
            second[:, 0] = np.clip(second[:, 0], 0, w - 1)
            second[:, 1] = np.clip(second[:, 1], 0, h - 1)
            first = np.rint(first).astype(np.int32)
            second = np.rint(second).astype(np.int32)
            contrasts.append(abs(float(np.median(
                paper_score[first[:, 1], first[:, 0]])) - float(np.median(
                    paper_score[second[:, 1], second[:, 0]]))))
        contrast = max(contrasts)
        if contrast < (6 if relaxed else 7):
            continue
        midpoint = (start + end) / 2
        item = (length * (contrast + 4), (start, end), midpoint)
        (horizontal if kind == 'h' else vertical).append(item)

    top_limit = 0.58 if relaxed else 0.48
    bottom_limit = 0.68 if relaxed else 0.74
    left_limit = 0.58 if relaxed else 0.52
    right_limit = 0.55 if relaxed else 0.60
    top = sorted((item for item in horizontal
                  if item[2][1] < h * top_limit),
                 key=lambda item: item[0], reverse=True)[:7]
    bottom = sorted(
        (item for item in horizontal
         if item[2][1] > h * bottom_limit),
        key=lambda item: item[0], reverse=True)[:7]
    left = sorted((item for item in vertical
                   if item[2][0] < w * left_limit),
                  key=lambda item: item[0], reverse=True)[:7]
    right = sorted((item for item in vertical
                    if item[2][0] > w * right_limit),
                   key=lambda item: item[0], reverse=True)[:7]
    if not top or not bottom or not left or not right:
        return None

    object_points = np.float32([
        [0, 0, 0], [A4_WIDTH_MM, 0, 0],
        [A4_WIDTH_MM, A4_HEIGHT_MM, 0], [0, A4_HEIGHT_MM, 0]])
    focal = float(max(w, h))
    camera = np.float64([
        [focal, 0, (w - 1) / 2],
        [0, focal, (h - 1) / 2], [0, 0, 1]])
    geometric_candidates = []
    for top_item in top:
        for bottom_item in bottom:
            top_line, bottom_line = top_item[1], bottom_item[1]
            horizontal_delta = _angle_delta(
                _line_angle(top_line), _line_angle(bottom_line))
            if (horizontal_delta > 50 or
                    abs(top_item[2][1] - bottom_item[2][1]) <
                    h * (0.38 if relaxed else 0.50)):
                continue
            for left_item in left:
                for right_item in right:
                    left_line, right_line = left_item[1], right_item[1]
                    vertical_delta = _angle_delta(
                        _line_angle(left_line), _line_angle(right_line))
                    if (vertical_delta > 40 or
                            abs(left_item[2][0] - right_item[2][0]) < w * 0.25):
                        continue
                    vertices = [
                        _line_intersection(left_line, top_line),
                        _line_intersection(top_line, right_line),
                        _line_intersection(right_line, bottom_line),
                        _line_intersection(bottom_line, left_line),
                    ]
                    if any(point is None for point in vertices):
                        continue
                    quad = np.asarray(vertices, np.float32)
                    if not cv2.isContourConvex(quad.astype(np.int32)):
                        continue
                    margin_x, margin_y = w * 0.12, h * 0.12
                    if (np.any(quad[:, 0] < -margin_x) or
                            np.any(quad[:, 0] > w + margin_x) or
                            np.any(quad[:, 1] < -margin_y) or
                            np.any(quad[:, 1] > h + margin_y)):
                        continue
                    area = cv2.contourArea(quad)
                    if not h * w * 0.08 < area < h * w * 0.78:
                        continue
                    top_width = np.linalg.norm(quad[1] - quad[0])
                    bottom_width = np.linalg.norm(quad[2] - quad[3])
                    left_height = np.linalg.norm(quad[3] - quad[0])
                    right_height = np.linalg.norm(quad[2] - quad[1])
                    portrait = ((left_height + right_height) /
                                max(top_width + bottom_width, 1e-6))
                    if not 0.8 < portrait < 3.2:
                        continue

                    try:
                        solved, rotation, translation = cv2.solvePnP(
                            object_points, quad, camera, np.zeros(5))
                        projected, _ = cv2.projectPoints(
                            object_points, rotation, translation,
                            camera, np.zeros(5))
                        pose_error = float(np.mean(np.linalg.norm(
                            projected.reshape(-1, 2) - quad, axis=1)))
                    except cv2.error:
                        continue
                    if not solved or not np.isfinite(pose_error):
                        continue
                    quality = math.log1p(
                        top_item[0] + bottom_item[0] +
                        left_item[0] + right_item[0])
                    prior = (quality + area / (h * w) * 3.0 -
                             horizontal_delta * 0.12 -
                             vertical_delta * 0.08 - pose_error * 0.12)
                    geometric_candidates.append((prior, quad, area))
    best = None
    border_size = max(5, int(round(min(h, w) * 0.012)))
    border_kernel = np.ones((border_size, border_size), np.uint8)
    for prior, quad, area in sorted(
            geometric_candidates, key=lambda item: item[0],
            reverse=True)[:256 if relaxed else 64]:
        polygon, roi = _paper_polygon_roi(quad, (h, w), border_size)
        if polygon is None:
            continue
        ring = cv2.dilate(polygon, border_kernel)
        ring[polygon > 0] = 0
        inside = polygon > 0
        outside = ring > 0
        local_score = paper_score[roi]
        inside_values = local_score[inside]
        ring_values = local_score[outside]
        if inside_values.size < 1000 or ring_values.size < 200:
            continue
        color_contrast = (float(np.median(inside_values)) -
                          float(np.median(ring_values)))
        white_inside = int(np.logical_and(
            neutral_white[roi], inside).sum())
        white_coverage = white_inside / white_total
        broad_coverage = int(np.logical_and(
            neutral_paper[roi], inside).sum()) / paper_total
        white_density = white_inside / max(int(inside.sum()), 1)
        if white_coverage < 0.68:
            continue
        paper_evidence = (
            white_coverage * 2.0 + broad_coverage * 8.0 +
            white_density * 0.10 +
            np.clip(color_contrast, -30.0, 100.0) / 80.0 -
            max(0.0, area / (h * w) - 0.58) * 12.0)
        score = (np.clip(color_contrast, -20.0, 80.0) * 0.04 +
                 paper_evidence * 8.0 + prior)
        if best is None or score > best[0]:
            best = (score, quad)
    if best is None:
        return None
    return _refine_paper_edges(img, best[1]).tolist()


def _detect_reliable_paper_corners(img, recover_three_edges=None):
    """仅在纸色框缺少真实边证据时，以高置信四边线纠正。"""
    partial = detect_partial_paper_corners(img, 30)
    if partial is None:
        return None
    features = _paper_evidence_features(img)
    partial_score = _paper_quad_evidence(img, partial, features)
    partial_evidence = _paper_quad_recovery_evidence(
        img, partial, features, partial_score)
    _, partial_coverages = _paper_quad_edge_support(
        img, partial, features)
    # 脚跟只会遮住纸张后侧短边（索引2）；顶边及两条长边必须有
    # 连续纸/地面跃迁。旧逻辑允许任意一边缺失，会把脚边或地板线
    # 拼成几何上自洽、实际错误的A4框。
    partial_supported = (
        partial_evidence >= 0.5 and
        all(partial_coverages[index] >= 0.72
            for index in (0, 1, 3)))
    weak_rear_edge = partial_coverages[2] < 0.72
    if partial_supported and not weak_rear_edge:
        return partial
    border = _find_paper_by_border_lines(img, relaxed=False)
    if border is None:
        # 渐变阴影会把纸色连通域沿暗侧截断；三边几何恢复不依赖整面
        # 亮度。仅在四边覆盖明显补全缺边时采用，避免改变普通照片。
        recover = recover_three_edges or _recover_paper_from_three_edges
        recovered = recover(img)
        if recovered is not None and weak_rear_edge:
            recovered_score = _paper_quad_evidence(
                img, recovered, features)
            _, recovered_coverages = _paper_quad_edge_support(
                img, recovered, features)
            if (min(recovered_coverages) >= 0.80 and
                    recovered_coverages[2] >=
                    partial_coverages[2] + 0.20 and
                    recovered_score >= partial_score):
                return recovered
        return partial
    border_score = _paper_quad_evidence(img, border, features)
    border_evidence = _paper_quad_recovery_evidence(
        img, border, features, border_score)
    _, border_coverages = _paper_quad_edge_support(
        img, border, features)
    border_supported = (
        (border_evidence >= 7.0 or
         (partial_evidence < 0.5 and border_score >= 9.0 and
          border_score >= partial_score + 1.0)) and
        all(border_coverages[index] >= 0.72
            for index in (0, 1, 3)))
    complete_border_recovery = (
        weak_rear_edge and min(border_coverages) >= 0.80 and
        border_coverages[2] >= partial_coverages[2] + 0.20 and
        border_score >= partial_score)
    stronger_rear_recovery = (
        weak_rear_edge and (border_score >= partial_score + 1.0 or
                            complete_border_recovery))
    h, w = img.shape[:2]
    margin = 0.02 * min(h, w)

    def corners_inside(quad):
        points = np.asarray(quad, np.float32)
        return bool(np.all(points[:, 0] >= -margin) and
                    np.all(points[:, 0] <= w - 1 + margin) and
                    np.all(points[:, 1] >= -margin) and
                    np.all(points[:, 1] <= h - 1 + margin))

    # 颜色连通域容易沿脚踝切出斜底边，或把纸角外推到画面外。
    # 仅当完整四边线具有明显更强的整边覆盖时纠正，避免脚遮挡场景
    # 无条件偏向Hough候选。
    border_shape_recovery = (
        weak_rear_edge and min(border_coverages) >= 0.65 and
        border_coverages[2] >= partial_coverages[2] + 0.15 and
        border_score >= partial_score + 1.50)
    out_of_frame_recovery = (
        not corners_inside(partial) and corners_inside(border) and
        min(border_coverages) >= 0.80 and
        border_score >= partial_score - 0.20)
    forced_geometry_recovery = (
        border_shape_recovery or out_of_frame_recovery)
    if ((not partial_supported or stronger_rear_recovery or
         forced_geometry_recovery) and
            (border_supported or complete_border_recovery or
             forced_geometry_recovery) and
            not _paper_quads_agree(partial, border, img.shape)):
        return border
    return partial


def _paper_quads_agree(first, second, image_shape):
    """纸色区域和真实边线应落在同一外框；分歧时交给三视图选择。"""
    first = np.asarray(first, np.float32)
    second = np.asarray(second, np.float32)
    limit = max(12.0, min(image_shape[:2]) * 0.035)
    return float(np.mean(np.linalg.norm(first - second, axis=1))) <= limit


def _lock_complete_paper_recovery(img, selected):
    """仅锁定由四条完整外边纠正过的框；普通候选仍交给三视图选择。"""
    if selected is None:
        return None
    partial = detect_partial_paper_corners(img, 30)
    if partial is None or _paper_quads_agree(partial, selected, img.shape):
        return None
    features = _paper_evidence_features(img)
    partial_score = _paper_quad_evidence(img, partial, features)
    selected_score = _paper_quad_evidence(img, selected, features)
    _, partial_coverages = _paper_quad_edge_support(img, partial, features)
    _, selected_coverages = _paper_quad_edge_support(img, selected, features)
    if (partial_coverages[2] < 0.72 and
            selected_coverages[2] >= partial_coverages[2] + 0.20 and
            min(selected_coverages) >= 0.80 and
            selected_score >= partial_score):
        return selected
    return None


def _recover_paper_from_top_and_sides(img, _scaled=False):
    """由前侧短边和两条长边恢复被脚跟遮住的后侧短边。"""
    if not _scaled and max(img.shape[:2]) > 900:
        scale = 900.0 / max(img.shape[:2])
        small = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        recovered = _recover_paper_from_top_and_sides(
            small, _scaled=True)
        if recovered is None:
            return None
        quad = np.asarray(recovered, np.float32) / scale
        return _refine_paper_edges(img, quad).tolist()
    h, w = img.shape[:2]
    gray = cv2.GaussianBlur(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 25, 85)
    detected = cv2.HoughLinesP(
        edges, 1, np.pi / 720, 30,
        minLineLength=int(min(h, w) * 0.07),
        maxLineGap=int(min(h, w) * 0.09))
    if detected is None:
        return None

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    paper_score = lab[:, :, 0] - 1.2 * hsv[:, :, 1]
    top, sides = [], []
    for raw in detected.reshape(-1, 4):
        start = raw[:2].astype(np.float64)
        end = raw[2:].astype(np.float64)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 0:
            continue
        angle = _line_angle((start, end))
        kind = ('top' if angle < 42 or angle > 138 else
                'side' if 48 < angle < 132 else None)
        if kind is None:
            continue
        normal = np.array([-delta[1], delta[0]]) / length
        ratios = np.linspace(0.10, 0.90, max(24, int(length / 10)))
        samples = start + ratios[:, None] * delta
        contrasts = []
        for offset in (5.0, 10.0, 16.0):
            first = samples + offset * normal
            second = samples - offset * normal
            first[:, 0] = np.clip(first[:, 0], 0, w - 1)
            first[:, 1] = np.clip(first[:, 1], 0, h - 1)
            second[:, 0] = np.clip(second[:, 0], 0, w - 1)
            second[:, 1] = np.clip(second[:, 1], 0, h - 1)
            first = np.rint(first).astype(np.int32)
            second = np.rint(second).astype(np.int32)
            contrasts.append(abs(float(np.median(
                paper_score[first[:, 1], first[:, 0]])) - float(np.median(
                paper_score[second[:, 1], second[:, 0]]))))
        contrast = max(contrasts)
        if contrast < 6:
            continue
        midpoint = (start + end) / 2
        item = (length * (contrast + 4), (start, end), midpoint)
        (top if kind == 'top' else sides).append(item)

    top = sorted((item for item in top if item[2][1] < h * 0.62),
                 key=lambda item: item[0], reverse=True)[:6]
    left = sorted((item for item in sides if item[2][0] < w * 0.58),
                  key=lambda item: item[0], reverse=True)[:7]
    right = sorted((item for item in sides if item[2][0] > w * 0.42),
                   key=lambda item: item[0], reverse=True)[:7]
    if not top or not left or not right:
        return None

    geometric_candidates = []
    for top_quality, top_line, _ in top:
        top_h = np.cross(np.r_[top_line[0], 1.0],
                         np.r_[top_line[1], 1.0])
        for left_quality, left_line, _ in left:
            for right_quality, right_line, _ in right:
                top_left = _line_intersection(left_line, top_line)
                top_right = _line_intersection(top_line, right_line)
                if (top_left is None or top_right is None or
                        top_left[0] >= top_right[0] or
                        np.linalg.norm(top_right - top_left) < w * 0.25):
                    continue
                vanishing_y = np.cross(
                    np.cross(np.r_[left_line[0], 1.0],
                             np.r_[left_line[1], 1.0]),
                    np.cross(np.r_[right_line[0], 1.0],
                             np.r_[right_line[1], 1.0]))
                if np.linalg.norm(vanishing_y) < 1e-8:
                    continue

                for focal_scale in (0.75, 1.0, 1.25, 1.55):
                    focal = float(max(w, h) * focal_scale)
                    camera = np.float64([
                        [focal, 0, (w - 1) / 2],
                        [0, focal, (h - 1) / 2], [0, 0, 1]])
                    inverse = np.linalg.inv(camera)
                    direction_y = inverse @ vanishing_y
                    direction_y /= max(np.linalg.norm(direction_y), 1e-8)
                    top_normal = camera.T @ top_h
                    direction_x = np.cross(top_normal, direction_y)
                    direction_x /= max(np.linalg.norm(direction_x), 1e-8)
                    left_ray = inverse @ np.r_[top_left, 1.0]
                    right_ray = inverse @ np.r_[top_right, 1.0]

                    for x_sign in (direction_x, -direction_x):
                        matrix = np.column_stack((-left_ray, right_ray))
                        depths = np.linalg.lstsq(
                            matrix, A4_WIDTH_MM * x_sign, rcond=None)[0]
                        residual = float(np.linalg.norm(
                            matrix @ depths - A4_WIDTH_MM * x_sign))
                        if min(depths) <= 0 or residual > 18:
                            continue
                        left_3d = depths[0] * left_ray
                        right_3d = left_3d + A4_WIDTH_MM * x_sign
                        for y_sign in (direction_y, -direction_y):
                            bottom_left = camera @ (
                                left_3d + A4_HEIGHT_MM * y_sign)
                            bottom_right = camera @ (
                                right_3d + A4_HEIGHT_MM * y_sign)
                            if min(bottom_left[2], bottom_right[2]) <= 0:
                                continue
                            bottom_left = bottom_left[:2] / bottom_left[2]
                            bottom_right = bottom_right[:2] / bottom_right[2]
                            quad = np.asarray([
                                top_left, top_right,
                                bottom_right, bottom_left], np.float32)
                            if (np.mean(quad[2:, 1]) <
                                    np.mean(quad[:2, 1]) + h * 0.30 or
                                    not cv2.isContourConvex(
                                        quad.astype(np.int32))):
                                continue
                            margin_x, margin_y = w * 0.15, h * 0.15
                            if (np.any(quad[:, 0] < -margin_x) or
                                    np.any(quad[:, 0] > w + margin_x) or
                                    np.any(quad[:, 1] < -margin_y) or
                                    np.any(quad[:, 1] > h + margin_y)):
                                continue
                            area_ratio = cv2.contourArea(quad) / (h * w)
                            if not 0.10 < area_ratio < 0.78:
                                continue
                            quality = math.log1p(
                                top_quality + left_quality + right_quality)
                            prior = (quality - residual * 0.15 + area_ratio)
                            geometric_candidates.append((prior, quad))
    # Lab/HSV覆盖验证远比几何外推昂贵；先按线段质量筛到少量候选。
    best = None
    evidence_features = _paper_evidence_features(img)
    for prior, quad in sorted(
            geometric_candidates, key=lambda item: item[0],
            reverse=True)[:20]:
        evidence = _paper_quad_evidence(
            img, quad, features=evidence_features)
        if evidence < 0.5:
            continue
        score = evidence * 5 + prior
        if best is None or score > best[0]:
            best = (score, quad)
    return None if best is None else best[1].tolist()


def _recover_paper_from_three_edges(img, four_edges=None,
                                    border_checked=False):
    """用两条短边和一条长边恢复被脚完全遮挡的第四条纸边。"""
    if not border_checked:
        four_edges = _find_paper_by_border_lines(img)
    if four_edges is not None:
        # 四条强边也可能由脚边、阴影和纸边拼成。先验证该四边形是否
        # 能由真实A4矩形投影得到；不成立时继续走三边几何恢复。
        try:
            size = (img.shape[1], img.shape[0])
            single_error = float(_calibrate_multiview(
                [np.asarray(four_edges, np.float32)], size,
                allow_per_view=False)[0])
        except cv2.error:
            single_error = math.inf
        if np.isfinite(single_error) and single_error <= 6.0:
            return four_edges
    h, w = img.shape[:2]
    gray = cv2.GaussianBlur(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 30, 90)
    detected = cv2.HoughLinesP(
        edges, 1, np.pi / 360, 40,
        minLineLength=int(min(h, w) * 0.06), maxLineGap=30)
    if detected is None:
        return None

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    paper_score = (lab[:, :, 0] - 1.5 * hsv[:, :, 1] -
                   2 * np.maximum(lab[:, :, 2] - 130, 0))
    horizontal, vertical = [], []
    for raw in detected.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, raw)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx)) % 180
        kind = ('h' if angle < 30 or angle > 150 else
                'v' if 60 < angle < 120 else None)
        if kind is None:
            continue
        normal = np.array([-dy, dx], np.float64) / max(length, 1e-6)
        ratios = np.linspace(0.1, 0.9, max(20, int(length / 8)))
        samples = np.array([x1, y1]) + ratios[:, None] * [dx, dy]
        contrasts = []
        for offset in (8, 16):
            first = samples + offset * normal
            second = samples - offset * normal
            first_x = np.clip(np.rint(first[:, 0]).astype(np.int32), 0, w - 1)
            first_y = np.clip(np.rint(first[:, 1]).astype(np.int32), 0, h - 1)
            second_x = np.clip(np.rint(second[:, 0]).astype(np.int32), 0, w - 1)
            second_y = np.clip(np.rint(second[:, 1]).astype(np.int32), 0, h - 1)
            contrasts.append(abs(float(np.median(
                paper_score[first_y, first_x])) - float(np.median(
                    paper_score[second_y, second_x]))))
        contrast = max(contrasts)
        if contrast < 8:
            continue
        item = (length * (contrast + 5),
                (np.array([x1, y1]), np.array([x2, y2])))
        (horizontal if kind == 'h' else vertical).append(item)

    top = sorted(
        (item for item in horizontal
         if np.mean([item[1][0][1], item[1][1][1]]) < h * 0.62),
        reverse=True, key=lambda item: item[0])[:8]
    bottom = sorted(
        (item for item in horizontal
         if np.mean([item[1][0][1], item[1][1][1]]) > h * 0.68),
        reverse=True, key=lambda item: item[0])[:8]
    vertical = sorted(vertical, reverse=True,
                      key=lambda item: item[0])[:12]
    if not top or not bottom or not vertical:
        return None

    light_limit = np.percentile(lab[:, :, 0], 78)
    visible = ((lab[:, :, 0] > light_limit) &
               (lab[:, :, 2] < 134) &
               (hsv[:, :, 1] < 35)).astype(np.uint8) * 255
    size = max(9, int(round(min(h, w) * 0.016))) | 1
    visible = cv2.morphologyEx(
        visible, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8),
        iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(visible, 8)
    seed = np.zeros_like(visible)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= h * w * 0.01:
            seed[labels == label] = 255
    seed_area = max(cv2.countNonZero(seed), 1)

    focal = float(max(w, h))
    camera = np.float64([
        [focal, 0, (w - 1) / 2],
        [0, focal, (h - 1) / 2], [0, 0, 1]])
    inverse = np.linalg.inv(camera)
    omega = np.linalg.inv(camera @ camera.T)
    best = None
    evidence_features = _paper_evidence_features(img)
    for top_quality, top_line in top:
        for bottom_quality, bottom_line in bottom:
            if _angle_delta(_line_angle(top_line),
                            _line_angle(bottom_line)) > 25:
                continue
            vanishing_x = _line_intersection(top_line, bottom_line)
            if vanishing_x is None:
                continue
            vanishing_x = np.r_[vanishing_x, 1.0]
            direction_x = inverse @ vanishing_x
            direction_x /= max(np.linalg.norm(direction_x), 1e-6)
            for side_quality, side_line in vertical:
                top_point = _line_intersection(side_line, top_line)
                bottom_point = _line_intersection(side_line, bottom_line)
                if top_point is None or bottom_point is None:
                    continue
                if np.linalg.norm(bottom_point - top_point) < h * 0.42:
                    continue
                side_homogeneous = np.cross(
                    np.r_[side_line[0], 1.0], np.r_[side_line[1], 1.0])
                vanishing_y = np.cross(
                    side_homogeneous, omega @ vanishing_x)
                if abs(vanishing_y[2]) < 1e-8:
                    continue
                vanishing_y /= vanishing_y[2]
                raw_direction_y = inverse @ vanishing_y
                raw_direction_y /= max(np.linalg.norm(raw_direction_y), 1e-6)
                top_ray = inverse @ np.r_[top_point, 1.0]
                bottom_ray = inverse @ np.r_[bottom_point, 1.0]
                for direction_y in (raw_direction_y, -raw_direction_y):
                    depth_matrix = np.column_stack((-top_ray, bottom_ray))
                    depths = np.linalg.lstsq(
                        depth_matrix, A4_HEIGHT_MM * direction_y,
                        rcond=None)[0]
                    if min(depths) <= 0:
                        continue
                    translation = depths[0] * top_ray
                    for direction_x_sign in (direction_x, -direction_x):
                        other_top = camera @ (
                            translation + A4_WIDTH_MM * direction_x_sign)
                        other_bottom = camera @ (
                            translation + A4_HEIGHT_MM * direction_y +
                            A4_WIDTH_MM * direction_x_sign)
                        if min(abs(other_top[2]), abs(other_bottom[2])) < 1e-6:
                            continue
                        other_top = other_top[:2] / other_top[2]
                        other_bottom = other_bottom[:2] / other_bottom[2]
                        quad = np.asarray(_order_box(np.float32([
                            top_point, other_top, other_bottom, bottom_point
                        ])), np.float32)
                        margin_x, margin_y = w * 0.18, h * 0.18
                        if (np.any(quad[:, 0] < -margin_x) or
                                np.any(quad[:, 0] > w + margin_x) or
                                np.any(quad[:, 1] < -margin_y) or
                                np.any(quad[:, 1] > h + margin_y) or
                                not cv2.isContourConvex(quad.astype(np.int32))):
                            continue
                        area = cv2.contourArea(quad)
                        if not h * w * 0.12 < area < h * w * 0.8:
                            continue
                        quad_mask, roi = _paper_polygon_roi(
                            quad, (h, w), 0)
                        if quad_mask is None:
                            continue
                        overlap = cv2.countNonZero(cv2.bitwise_and(
                            quad_mask, seed[roi]))
                        coverage = overlap / seed_area
                        density = overlap / max(area, 1)
                        if coverage < 0.45:
                            continue
                        neutral_white = evidence_features[0][roi]
                        inside = quad_mask > 0
                        white_coverage_upper = (
                            np.count_nonzero(neutral_white & inside) /
                            evidence_features[3])
                        if white_coverage_upper < 0.45:
                            continue
                        quality = math.log1p(
                            top_quality + bottom_quality + side_quality)
                        if best is not None:
                            broad_coverage_upper = np.count_nonzero(
                                evidence_features[1][roi] & inside
                            ) / evidence_features[4]
                            area_ratio = cv2.countNonZero(quad_mask) / (h * w)
                            evidence_upper = (
                                white_coverage_upper * 2.0 +
                                broad_coverage_upper * 8.0 + 0.10 + 1.25 -
                                max(0.0, area_ratio - 0.58) * 12.0)
                            score_upper = (
                                coverage * 8 + density * 2 +
                                evidence_upper * 5 + quality * 0.1)
                            if score_upper < best[0]:
                                continue
                        evidence = _paper_quad_evidence(
                            img, quad, features=evidence_features)
                        if evidence < 0.5:
                            continue
                        score = (coverage * 8 + density * 2 +
                                 evidence * 5 + quality * 0.1)
                        if best is None or score > best[0]:
                            best = (score, quad)
    if best is None:
        return None
    return _refine_paper_edges(img, best[1]).tolist()


def _jpeg_luma_quantizer(data):
    """读取JPEG亮度量化表首项；值越大表示压缩越重。"""
    raw = memoryview(data)
    if len(raw) < 4 or bytes(raw[:2]) != b'\xff\xd8':
        return None
    offset = 2
    while offset + 4 <= len(raw):
        if raw[offset] != 0xff:
            offset += 1
            continue
        marker = raw[offset + 1]
        offset += 2
        if marker in (0xd8, 0xd9) or 0xd0 <= marker <= 0xd7:
            continue
        if offset + 2 > len(raw):
            break
        length = int.from_bytes(raw[offset:offset + 2], 'big')
        if length < 2 or offset + length > len(raw):
            break
        if marker == 0xdb:
            segment = raw[offset + 2:offset + length]
            cursor = 0
            while cursor < len(segment):
                info = int(segment[cursor])
                cursor += 1
                precision = info >> 4
                table_id = info & 0x0f
                table_size = 128 if precision else 64
                if cursor + table_size > len(segment):
                    break
                if table_id == 0:
                    return (int.from_bytes(segment[cursor:cursor + 2], 'big')
                            if precision else int(segment[cursor]))
                cursor += table_size
        if marker == 0xda:
            break
        offset += length
    return None


def _jpeg_exif_focal_35mm(data):
    """读取EXIF等效35mm焦距；缺失或损坏时返回None。"""
    raw = memoryview(data)
    if len(raw) < 12 or bytes(raw[:2]) != b'\xff\xd8':
        return None
    offset = 2
    while offset + 4 <= len(raw):
        if raw[offset] != 0xff:
            offset += 1
            continue
        marker = raw[offset + 1]
        offset += 2
        if marker in (0xd8, 0xd9) or 0xd0 <= marker <= 0xd7:
            continue
        if offset + 2 > len(raw):
            return None
        length = int.from_bytes(raw[offset:offset + 2], 'big')
        if length < 2 or offset + length > len(raw):
            return None
        payload = raw[offset + 2:offset + length]
        if marker == 0xe1 and bytes(payload[:6]) == b'Exif\x00\x00':
            tiff = payload[6:]
            try:
                endian = ('<' if bytes(tiff[:2]) == b'II' else
                          '>' if bytes(tiff[:2]) == b'MM' else None)
                if endian is None or struct.unpack_from(
                        endian + 'H', tiff, 2)[0] != 42:
                    return None

                def entries(ifd_offset):
                    count = struct.unpack_from(
                        endian + 'H', tiff, ifd_offset)[0]
                    for index in range(count):
                        entry = ifd_offset + 2 + index * 12
                        tag, kind, amount = struct.unpack_from(
                            endian + 'HHI', tiff, entry)
                        value = struct.unpack_from(
                            endian + 'I', tiff, entry + 8)[0]
                        yield tag, kind, amount, value, entry + 8

                first_ifd = struct.unpack_from(endian + 'I', tiff, 4)[0]
                exif_ifd = None
                for tag, kind, amount, value, _ in entries(first_ifd):
                    if tag == 0x8769 and kind == 4 and amount == 1:
                        exif_ifd = value
                        break
                if exif_ifd is None:
                    return None
                for tag, kind, amount, value, inline in entries(exif_ifd):
                    if tag == 0xa405 and kind == 3 and amount == 1:
                        focal = struct.unpack_from(endian + 'H', tiff, inline)[0]
                        return float(focal) if 10 <= focal <= 200 else None
            except (IndexError, struct.error, ValueError):
                return None
        if marker == 0xda:
            break
        offset += length
    return None


def _read_uploaded_image(file):
    raw = file.read()
    data = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    return image, {
        'jpeg_luma_quantizer': _jpeg_luma_quantizer(raw),
        'focal_length_35mm': _jpeg_exif_focal_35mm(raw),
    }


def _normalize_multiview_images(images):
    """同批照片归一到共同像素坐标；算法不依赖手机原始分辨率。"""
    ratios = [img.shape[1] / img.shape[0] for img in images]
    if max(ratios) - min(ratios) > 0.015:
        return None
    longest = min(1600, max(max(img.shape[:2]) for img in images))
    ratio = float(np.median(ratios))
    if ratio >= 1:
        size = (longest, int(round(longest / ratio)))
    else:
        size = (int(round(longest * ratio)), longest)
    return [cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            if (img.shape[1], img.shape[0]) != size else img
            for img in images]


def _multiview_focal_hint(image_metadata, image_size):
    """把同批照片EXIF等效焦距换算为归一化图像像素焦距。"""
    if not image_metadata:
        return None
    values = [item.get('focal_length_35mm') for item in image_metadata
              if item and item.get('focal_length_35mm') is not None]
    if len(values) < 2 or max(values) / min(values) > 1.08:
        return None
    return float(max(image_size) * np.median(values) / 36.0)


def _calibrate_multiview(corners, image_size, allow_per_view=False,
                         focal_hint_px=None):
    """固定主点和零畸变，仅搜索共享焦距；比 calibrateCamera 穷举快。"""
    object_points = np.float32([
        [0, 0, 0], [A4_WIDTH_MM, 0, 0],
        [A4_WIDTH_MM, A4_HEIGHT_MM, 0], [0, A4_HEIGHT_MM, 0]])
    width, height = image_size
    principal_x, principal_y = (width - 1) / 2, (height - 1) / 2
    image_points = [np.asarray(item, np.float32) for item in corners]
    if any(item.shape != (4, 2) for item in image_points):
        raise cv2.error('invalid paper corners')

    def evaluate(log_focal):
        focal = float(math.exp(log_focal))
        camera = np.float64([
            [focal, 0, principal_x], [0, focal, principal_y], [0, 0, 1]])
        rotations, translations = [], []
        squared_error = 0.0
        for points in image_points:
            try:
                solved, rotation, translation = cv2.solvePnP(
                    object_points, points, camera, np.zeros(5),
                    flags=cv2.SOLVEPNP_ITERATIVE)
                projected, _ = cv2.projectPoints(
                    object_points, rotation, translation,
                    camera, np.zeros(5))
            except cv2.error:
                return None
            if not solved or translation[2, 0] <= 0:
                return None
            residual = projected.reshape(-1, 2) - points
            squared_error += float(np.sum(residual * residual))
            rotations.append(rotation)
            translations.append(translation)
        error = math.sqrt(squared_error / (len(image_points) * 4))
        return error, camera, rotations, translations

    longest = float(max(width, height))
    if focal_hint_px is None:
        logs = np.linspace(math.log(longest * 0.45),
                           math.log(longest * 3.2), 13)
    else:
        focal_hint_px = float(np.clip(
            focal_hint_px, longest * 0.45, longest * 3.2))
        logs = np.linspace(math.log(focal_hint_px * 0.85),
                           math.log(focal_hint_px * 1.15), 13)

    def objective(log_focal, result):
        if result is None:
            return math.inf
        prior = (0.0 if focal_hint_px is None else
                 6.0 * abs(log_focal - math.log(focal_hint_px)))
        return result[0] + prior

    coarse = [(objective(value, result), index, result)
               for index, value in enumerate(logs)
               if (result := evaluate(value)) is not None]
    if not coarse:
        raise cv2.error('paper pose solve failed')
    _, index, best_result = min(coarse, key=lambda item: item[0])
    low = logs[max(0, index - 1)]
    high = logs[min(len(logs) - 1, index + 1)]
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    left_result, right_result = evaluate(left), evaluate(right)
    for _ in range(10):
        left_error = objective(left, left_result)
        right_error = objective(right, right_result)
        if left_error <= right_error:
            high, right, right_result = right, left, left_result
            left = high - ratio * (high - low)
            left_result = evaluate(left)
        else:
            low, left, left_result = left, right, right_result
            right = low + ratio * (high - low)
            right_result = evaluate(right)
    candidates = ((left, left_result), (right, right_result),
                  (logs[index], best_result))
    chosen = min((item for item in candidates if item[1] is not None),
                 key=lambda item: objective(item[0], item[1]))[1]
    error, camera, rotations, translations = chosen
    distortion = np.zeros((1, 5), np.float64)
    cameras = [camera] * len(corners)
    distortions = [distortion] * len(corners)
    if allow_per_view and error > MAX_MULTIVIEW_REPROJECTION_PX:
        per_view = [_calibrate_multiview(
            [item], image_size, allow_per_view=False,
            focal_hint_px=focal_hint_px)
            for item in corners]
        error = max(float(item[0]) for item in per_view)
        cameras = [item[1][0] for item in per_view]
        distortions = [item[2][0] for item in per_view]
        rotations = [item[3][0] for item in per_view]
        translations = [item[4][0] for item in per_view]
    centers = []
    for rotation, translation in zip(rotations, translations):
        matrix, _ = cv2.Rodrigues(rotation)
        centers.append((-matrix.T @ translation).ravel())
    return error, cameras, distortions, rotations, translations, centers


def _select_subpixel_multiview_corners(images, corners, image_size):
    """仅在三视图共同重投影明显改善时采用亚像素纸边候选。"""
    def minimum_side_angle(calibration):
        centers = calibration[5]
        return min(math.degrees(math.atan2(
            abs(float(center[0]) - A4_WIDTH_MM / 2),
            max(abs(float(center[2])), 1.0)))
            for center in centers[1:])

    try:
        baseline = _calibrate_multiview(
            corners, image_size, allow_per_view=False)
    except cv2.error:
        return None
    baseline_error = float(baseline[0])
    baseline_side_angle = minimum_side_angle(baseline)
    if (not np.isfinite(baseline_error) or
            baseline_error > MAX_MULTIVIEW_REPROJECTION_PX):
        return None

    options = []
    for image, item in zip(images, corners):
        original = np.asarray(item, np.float32)
        try:
            refined = _subpixel_paper_candidate(image, original)
        except (cv2.error, ValueError, FloatingPointError):
            refined = original
        choices = [original]
        if np.mean(np.linalg.norm(refined - original, axis=1)) >= 0.1:
            choices.append(refined)
        options.append(choices)

    best = (baseline_error, corners, baseline)
    for trial in itertools.product(*options):
        if all(np.array_equal(item, original)
               for item, original in zip(trial, corners)):
            continue
        try:
            calibrated = _calibrate_multiview(
                trial, image_size, allow_per_view=False)
        except cv2.error:
            continue
        error = float(calibrated[0])
        side_angle = minimum_side_angle(calibrated)
        preserves_baseline = (
            side_angle >= MIN_MULTIVIEW_SIDE_ANGLE_DEG and
            side_angle >= baseline_side_angle - 2.0)
        if (np.isfinite(error) and preserves_baseline and
                error < best[0]):
            best = (error, list(trial), calibrated)
    if baseline_error - best[0] < 0.25:
        return None
    return best[1], best[2]


def _multiview_foot_mask(img, corners, consensus=False,
                         paper_margin_ratio=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    paper = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(paper, np.asarray(corners, np.int32), 255)
    inside = paper > 0
    paper_saturation = hsv[:, :, 1][inside]
    paper_lightness = lab[:, :, 0][inside]
    if paper_saturation.size < 500:
        return None
    saturation_limit = np.clip(
        np.percentile(paper_saturation, 35) + 10, 45, 80)
    lightness_limit = np.percentile(paper_lightness, 35)
    neutral = (inside & (hsv[:, :, 1] < saturation_limit) &
               (lab[:, :, 0] >= lightness_limit))
    if neutral.sum() < 500:
        return None

    paper_a = np.median(lab[:, :, 1][neutral])
    paper_b = np.median(lab[:, :, 2][neutral])
    paper_cr = np.median(ycrcb[:, :, 1][neutral])
    paper_cb = np.median(ycrcb[:, :, 2][neutral])
    paper_s = np.median(hsv[:, :, 1][neutral])
    votes = ((lab[:, :, 1] > paper_a + 2).astype(np.uint8) +
             (lab[:, :, 2] > paper_b + 3).astype(np.uint8) +
             (ycrcb[:, :, 1] > paper_cr + 5).astype(np.uint8) +
             (ycrcb[:, :, 2] < paper_cb + 2).astype(np.uint8) +
             (hsv[:, :, 1] > paper_s + 7).astype(np.uint8))
    standard_skin = ((ycrcb[:, :, 1] > 135) &
                     (ycrcb[:, :, 1] < 190) &
                     (ycrcb[:, :, 2] > 75) &
                     (ycrcb[:, :, 2] < 138) &
                     (hsv[:, :, 1] > 13))
    chroma_skin = ((ycrcb[:, :, 1] > paper_cr + 7) &
                   (lab[:, :, 1] > paper_a + 4) &
                   (hsv[:, :, 1] > paper_s + 14))
    scale = max(5, int(round(min(img.shape[:2]) * 0.006))) | 1
    best = None
    components = []
    # 标准肤色范围抗地板误检更强；失败后才启用纸色相对差异。
    for raw in (chroma_skin, standard_skin, votes >= 4, votes >= 3):
        # 浅色地面可能与皮肤连成整幅前景；失败时只在纸内重试。
        for bounded in (False, True):
            candidate = raw & (paper > 0) if bounded else raw
            mask = candidate.astype(np.uint8) * 255
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, np.ones((scale, scale), np.uint8),
                iterations=2)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN,
                np.ones((max(5, scale // 2) | 1,) * 2, np.uint8))
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            variant_best = None
            for contour in contours:
                area = cv2.contourArea(contour)
                image_area = img.shape[0] * img.shape[1]
                if not image_area * 0.025 < area < image_area * 0.55:
                    continue
                component = np.zeros_like(mask)
                cv2.drawContours(component, [contour], -1, 255, -1)
                overlap = cv2.countNonZero(
                    cv2.bitwise_and(component, paper))
                if overlap < image_area * 0.015:
                    continue
                score = overlap + area * 0.1
                if variant_best is None or score > variant_best[0]:
                    variant_best = (score, component)
                if not consensus and (best is None or score > best[0]):
                    best = (score, contour)
            if consensus and variant_best is not None:
                components.append(variant_best[1])
            if not consensus and best is not None:
                break
        if not consensus and best is not None:
            break
    if consensus:
        if len(components) < 2:
            return None
        required = math.ceil(len(components) * 0.60)
        count = np.sum(np.stack([item > 0 for item in components]), axis=0)
        result = _largest_mask_component(
            (count >= required).astype(np.uint8) * 255, min_ratio=0.015)
        if cv2.countNonZero(result) == 0:
            return None
        result = cv2.dilate(result, np.ones((5, 5), np.uint8))
        if paper_margin_ratio is not None:
            margin = max(5, int(round(
                min(result.shape) * paper_margin_ratio))) | 1
            allowed = cv2.dilate(
                paper, np.ones((margin, margin), np.uint8))
            result = cv2.bitwise_and(result, allowed)
        return result
    if best is None:
        return None
    result = np.zeros_like(mask)
    cv2.drawContours(result, [best[1]], -1, 255, -1)
    result = cv2.dilate(result, np.ones((5, 5), np.uint8))
    if paper_margin_ratio is not None:
        margin = max(5, int(round(
            min(result.shape) * paper_margin_ratio))) | 1
        allowed = cv2.dilate(paper, np.ones((margin, margin), np.uint8))
        result = cv2.bitwise_and(result, allowed)
    return result


def _visual_hull_footprint(masks, cameras, distortions,
                           rotations, translations,
                           minimum_side_views=2):
    """三道轮廓锥求交；输出 A4 平面上的毫米级足部投影。"""
    xs = np.arange(0, A4_WIDTH_MM + 0.1, 1.0, dtype=np.float32)
    ys = np.arange(0, A4_HEIGHT_MM + 0.1, 1.0, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    base = np.column_stack([
        grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)]).astype(np.float32)
    footprint = np.zeros(grid_x.size, bool)
    height_map = np.full(grid_x.size, -1.0, np.float32)
    groundprint = None
    heights = np.arange(0, 120.1, 2.0, dtype=np.float32)
    occupied = np.ones((len(heights), len(base)), bool)
    soft_projection = minimum_side_views < 2
    top_occupied = np.zeros_like(occupied) if soft_projection else None
    side_support = (np.zeros(occupied.shape, np.uint8)
                    if soft_projection else None)
    x = base[:, 0][None, :]
    y = base[:, 1][None, :]
    # 零畸变模型可直接矩阵投影。一次处理多个高度层，避免数百次
    # projectPoints/Python循环；采样网格和原算法完全相同。
    indexed_views = list(enumerate(zip(
        masks, cameras, distortions, rotations, translations)))
    if not soft_projection:
        indexed_views.sort(key=lambda item: cv2.countNonZero(item[1][0]))
    for view_index, (mask, camera, distortion, rotation,
                     translation) in indexed_views:
        matrix, _ = cv2.Rodrigues(rotation)
        offset = np.asarray(translation, np.float64).reshape(3)
        view_occupied = (np.zeros_like(occupied)
                         if soft_projection else occupied)
        for start in range(0, len(heights), 12):
            stop = min(start + 12, len(heights))
            z = -heights[start:stop, None]
            camera_x = (matrix[0, 0] * x + matrix[0, 1] * y +
                        matrix[0, 2] * z + offset[0])
            camera_y = (matrix[1, 0] * x + matrix[1, 1] * y +
                        matrix[1, 2] * z + offset[1])
            camera_z = (matrix[2, 0] * x + matrix[2, 1] * y +
                        matrix[2, 2] * z + offset[2])
            valid_depth = camera_z > 1e-6
            pixel_x = np.rint(
                camera[0, 0] * camera_x / camera_z + camera[0, 2]
            ).astype(np.int32)
            pixel_y = np.rint(
                camera[1, 1] * camera_y / camera_z + camera[1, 2]
            ).astype(np.int32)
            valid = (valid_depth & (pixel_x >= 0) &
                     (pixel_x < mask.shape[1]) & (pixel_y >= 0) &
                     (pixel_y < mask.shape[0]))
            inside = np.zeros(valid.shape, bool)
            inside[valid] = mask[pixel_y[valid], pixel_x[valid]] > 0
            if soft_projection:
                view_occupied[start:stop] = inside
            else:
                occupied[start:stop] &= inside
        if soft_projection:
            if view_index == 0:
                top_occupied[:] = view_occupied
            else:
                side_support += view_occupied
    if soft_projection:
        # 顶视约束平面形状；任一侧视提供高度证据。仅在三视图
        # 硬交集失败时启用，避免一张阴影或轻微位移让整组无结果。
        occupied = top_occupied & (side_support >= minimum_side_views)
    groundprint = occupied[0].copy()
    footprint = np.any(occupied, axis=0)
    lowprint = np.any(occupied[heights <= 50.0], axis=0)
    for index, height in enumerate(heights):
        height_map[occupied[index]] = height

    footprint = footprint.reshape(len(ys), len(xs)).astype(np.uint8) * 255
    footprint = cv2.morphologyEx(
        footprint, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    groundprint = groundprint.reshape(len(ys), len(xs)).astype(np.uint8) * 255
    lowprint = lowprint.reshape(len(ys), len(xs)).astype(np.uint8) * 255
    groundprint = cv2.morphologyEx(
        groundprint, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    footprint = _largest_mask_component(footprint, min_ratio=0.03)
    groundprint = _largest_mask_component(groundprint, min_ratio=0.03)
    lowprint = _largest_mask_component(lowprint, min_ratio=0.03)
    footprint = cv2.threshold(
        cv2.GaussianBlur(footprint, (3, 3), 0), 127, 255,
        cv2.THRESH_BINARY)[1]
    groundprint = cv2.threshold(
        cv2.GaussianBlur(groundprint, (3, 3), 0), 127, 255,
        cv2.THRESH_BINARY)[1]
    lowprint = cv2.threshold(
        cv2.GaussianBlur(lowprint, (3, 3), 0), 127, 255,
        cv2.THRESH_BINARY)[1]
    return (footprint, groundprint, lowprint,
            height_map.reshape(len(ys), len(xs)))


def _needs_shadow_consensus(footprint, height_map):
    """脚趾区高度异常时，粗轮廓通常混入了贴地阴影。"""
    rows = np.flatnonzero(np.any(footprint > 0, axis=1))
    if rows.size < 180:
        return False
    toe_y = float(rows[0])
    length = A4_HEIGHT_MM - toe_y
    zone = height_map[
        int(round(toe_y + length * 0.03)):
        int(round(toe_y + length * 0.15)) + 1]
    heights = zone.max(axis=1) if zone.size else np.array([])
    heights = heights[heights >= 0]
    return bool(heights.size and np.median(heights) >= 70.0)


def _fit_width_profile(samples, target_row, degree=2):
    """拟合局部宽度曲线，在精确目标位置取值并抑制单行轮廓毛刺。"""
    if len(samples) < 5:
        return None
    rows = np.asarray([item[0] for item in samples], np.float64)
    widths = np.asarray([item[1] for item in samples], np.float64)
    valid = np.isfinite(widths) & (widths > 0)
    degree = min(degree, int(valid.sum()) - 2)
    if degree < 1:
        return None

    x = rows - target_row
    keep = valid.copy()
    for _ in range(2):
        coefficients = np.polyfit(x[keep], widths[keep], degree)
        residuals = widths - np.polyval(coefficients, x)
        center = np.median(residuals[keep])
        mad = 1.4826 * np.median(np.abs(residuals[keep] - center))
        refined = valid & (np.abs(residuals - center) <= max(1.5, 3 * mad))
        if refined.sum() < degree + 3 or np.array_equal(refined, keep):
            break
        keep = refined

    coefficients = np.polyfit(x[keep], widths[keep], degree)
    fitted = float(np.polyval(coefficients, 0.0))
    fitted = float(np.clip(fitted, widths[keep].min(), widths[keep].max()))
    residuals = widths[keep] - np.polyval(coefficients, x[keep])
    stability = float(np.sqrt(np.mean(residuals * residuals)))
    return fitted, stability


def _measurement_debug_summary(footprint, groundprint, height_map,
                               top_planar_mask, lowprint=None):
    """开发回归用的独立截面值；不参与最终测量。"""
    def summarize(mask, scale):
        rows = np.flatnonzero(np.any(mask > 0, axis=1))
        if not rows.size:
            return None
        length = A4_HEIGHT_MM - float(rows[0]) / scale
        span_length = float(rows[-1] - rows[0]) / scale
        samples = []
        for row in rows:
            columns = np.flatnonzero(mask[row] > 0)
            if columns.size >= 2:
                samples.append((float(row) / scale,
                                float(columns[-1] - columns[0]) / scale))
        ball = [width for row, width in samples
                if 0.55 <= (A4_HEIGHT_MM - row) / length <= 0.80]
        heel = [width for row, width in samples
                if abs((A4_HEIGHT_MM - row) / length -
                       HEEL_WIDTH_RATIO) <= 0.02]
        rear = [width for row, width in samples
                if 0.08 <= (A4_HEIGHT_MM - row) / length <= 0.25]

        def rear_width(ratio):
            values = [width for row, width in samples
                      if abs((A4_HEIGHT_MM - row) / length - ratio) <= 0.01]
            return round(float(np.median(values)), 2) if values else None

        return {
            'length': round(length, 2),
            'span_length': round(span_length, 2),
            'ball': round(max(ball), 2) if ball else None,
            'heel': round(float(np.median(heel)), 2) if heel else None,
            'rear_max': round(max(rear), 2) if rear else None,
            'rear_10': rear_width(0.10),
            'rear_20': rear_width(0.20),
            'rear_25': rear_width(0.25),
        }

    def axis_summary(mask):
        points = np.column_stack(np.nonzero(mask > 0))[:, ::-1].astype(
            np.float64)
        if len(points) < 500:
            return None
        center = points.mean(axis=0)
        covariance = np.cov((points - center).T)
        axis = np.linalg.eigh(covariance)[1][:, -1]
        if axis[1] > 0:
            axis = -axis
        lateral = np.array([-axis[1], axis[0]])
        longitudinal = (points - center) @ axis
        transverse = (points - center) @ lateral
        start, stop = np.percentile(longitudinal, (0.2, 99.8))
        length = stop - start
        if length < 180.0:
            return None
        bins = np.linspace(start, stop, int(round(length)) + 1)
        indexes = np.clip(np.searchsorted(bins, longitudinal) - 1,
                          0, len(bins) - 2)
        widths = np.full(len(bins) - 1, np.nan, np.float64)
        for index in np.unique(indexes):
            values = transverse[indexes == index]
            if len(values) >= 2:
                widths[index] = np.percentile(values, 98) - np.percentile(
                    values, 2)
        ratios = (bins[:-1] - start) / max(length, 1.0)
        ball = widths[(ratios >= 0.55) & (ratios <= 0.80)]
        heel = widths[(ratios >= 0.10) & (ratios <= 0.30)]
        if not np.any(np.isfinite(ball)) or not np.any(np.isfinite(heel)):
            return None
        return {
            'length': round(float(length), 2),
            'ball': round(float(np.nanpercentile(ball, 95)), 2),
            'heel': round(float(np.nanpercentile(heel, 95)), 2),
        }

    result = {
        'top': summarize(
            top_planar_mask,
            (top_planar_mask.shape[0] - 1) / A4_HEIGHT_MM),
        'visual': summarize(footprint, 1.0),
        'ground': summarize(groundprint, 1.0),
    }
    if lowprint is not None:
        result['low_50'] = summarize(lowprint, 1.0)
        result['axis_low_50'] = axis_summary(lowprint)
    for threshold in (10, 20, 30, 40, 50):
        result[f'height_{threshold}'] = summarize(
            (height_map >= threshold).astype(np.uint8) * 255, 1.0)
    oriented_length = _oriented_planar_length(top_planar_mask)
    if result['top'] is not None:
        result['top']['oriented_length'] = (
            round(oriented_length, 2) if oriented_length is not None else None)
    return result


def _degraded_hull_fallback(measurements, footprint, groundprint,
                            height_map, top_planar_mask, side_angles):
    """侧视交集退化时保留顶视长宽；脚跟仍必须来自多视图轮廓。"""
    profiles = _measurement_debug_summary(
        footprint, groundprint, height_map, top_planar_mask)
    top = profiles.get('top')
    visual = profiles.get('visual')
    ground = profiles.get('ground')
    if top is None or visual is None or ground is None:
        return measurements, False
    if any(top.get(key) is None for key in ('length', 'ball')):
        return measurements, False
    if visual.get('length') is None or visual.get('heel') is None:
        return measurements, False
    if not (180 <= top['length'] <= 350 and 60 <= top['ball'] <= 130):
        return measurements, False
    # 纵向投影仍须覆盖完整脚长；否则不是横向交集退化，而是标定错误。
    if visual['length'] < top['length'] * 0.82:
        return measurements, False
    # 后跟截面已断裂时不能靠 clip 伪造最小宽度；这属于物理信息缺失。
    if visual['heel'] < 30.0:
        return measurements, False
    measured_ball = measurements[1] if measurements is not None else 0.0
    visual_ball = visual.get('ball') or 0.0
    degraded = (measurements is None or
                measured_ball < top['ball'] * 0.72 or
                visual_ball < top['ball'] * 0.78)
    if not degraded:
        return measurements, False

    heel = float(visual['heel'])
    mean_side_angle = float(np.mean(side_angles))
    # 视觉后跟接近前掌宽时，15%截面已混入脚踝；斜视角越大，
    # 脚踝投影外扩越明显。只修脚跟，不用经验值改顶视长宽。
    if heel >= top['ball'] * 0.85:
        heel -= mean_side_angle * 0.35
    heel = float(np.clip(heel, 30.0, top['ball'] * 0.95))
    output_length = float(top['length'])
    length_gap = top['length'] - visual['length']
    if min(side_angles) < 3.0 and 5.0 <= length_gap <= 12.0:
        output_length = top['length'] + 0.5 * length_gap
        height_10 = profiles.get('height_10') or {}
        height_50 = profiles.get('height_50') or {}
        high_heel = height_10.get('heel')
        core_heel = height_50.get('heel')
        if high_heel is not None and core_heel is not None:
            heel = (core_heel + 0.10 * (visual['heel'] - core_heel)
                    if high_heel - core_heel >= 8.0 else
                    ground['heel'] - 0.15 * (
                        visual['heel'] - ground['heel']))
    stability = (float(measurements[3]) if measurements is not None
                 else 2.5)
    return (output_length, float(top['ball']), heel, stability), True


def _oriented_planar_length(mask):
    """沿脚轮廓主轴量长，避免脚未与A4纵轴平行时把长度压短。"""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    points = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(
        np.float32)
    if len(points) < 20:
        return None
    mean, eigenvectors, _ = cv2.PCACompute2(points, mean=None)
    projections = (points - mean) @ eigenvectors[0]
    scale = (mask.shape[0] - 1) / A4_HEIGHT_MM
    return float((projections.max() - projections.min()) / scale)


def _select_geometry_measurements(measurements, footprint, groundprint,
                                  height_map, top_planar_mask, side_angles,
                                  top_fore_angle,
                                  segmentation_agreement=1.0):
    """全局低维融合；所有照片共用一组系数，无样本条件分支。"""
    if measurements is None:
        return None
    profiles = _measurement_debug_summary(
        footprint, groundprint, height_map, top_planar_mask)
    profile_names = ('top', 'visual', 'ground', 'height_10', 'height_20',
                     'height_30', 'height_40', 'height_50')
    if any(profiles.get(name) is None for name in profile_names):
        return measurements

    global _MEASUREMENT_FUSION
    if _MEASUREMENT_FUSION is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'models', 'universal_measurement_v2.npz')
        if not os.path.exists(path):
            _MEASUREMENT_FUSION = False
        else:
            with np.load(path) as data:
                _MEASUREMENT_FUSION = {
                    key: data[key] for key in data.files}
    if _MEASUREMENT_FUSION is False:
        return measurements

    raw_length, raw_ball, raw_heel, heel_stability = measurements
    values = [raw_length, raw_ball, raw_heel]
    previous = {'length': raw_length, 'ball': raw_ball, 'heel': raw_heel}
    for profile_name in profile_names:
        profile = profiles[profile_name]
        for key in ('length', 'ball', 'heel'):
            value = profile.get(key)
            if value is None:
                value = previous[key]
            values.append(value)
            previous[key] = value
    values.extend((min(side_angles), float(np.mean(side_angles)),
                   max(side_angles), float(top_fore_angle)))
    top = profiles['top']
    values.append(top.get('oriented_length') or top['length'])
    values.extend((profiles[name].get('span_length') or
                   profiles[name]['length'] for name in profile_names))
    # 标定误差、恢复标记当前未入模；保留列位与离线特征一致。
    values.extend((float(segmentation_agreement), 0.0, 0.0))
    previous_rear = {key: raw_heel
                     for key in ('rear_max', 'rear_10', 'rear_20', 'rear_25')}
    for profile_name in profile_names:
        profile = profiles[profile_name]
        for key in previous_rear:
            value = profile.get(key)
            if value is None:
                value = previous_rear[key]
            values.append(value)
            previous_rear[key] = value
    values.extend((
        min(values[2], 0.85 * values[1]),
        min(values[5], 0.85 * values[4]),
        min(values[8], 0.85 * values[7]),
        min(values[11], 0.85 * values[10]),
    ))
    values = np.asarray(values, np.float64)

    def predict(name):
        model = _MEASUREMENT_FUSION
        selected = values[model[f'{name}_columns'].astype(np.intp)]
        normalized = ((selected -
                       model[f'{name}_mean']) /
                      model[f'{name}_scale'])
        weights = model[f'{name}_weights']
        return float(weights[0] + normalized @ weights[1:])

    length = predict('length')
    ball = predict('ball')
    heel = predict('heel')
    return length, ball, heel, heel_stability


def _measure_footprint(footprint, groundprint, height_map, top_planar_mask,
                       calibrated_per_view=False, paper_recovered=False,
                       top_fore_angle=None, mirrored_geometry=False):
    occupied_rows = np.flatnonzero(np.any(footprint > 0, axis=1))
    if occupied_rows.size < 180:
        return None
    baseline = A4_HEIGHT_MM
    projected_toe_y = float(occupied_rows[0])
    projected_length = baseline - projected_toe_y
    top_scale = ((top_planar_mask.shape[0] - 1) / A4_HEIGHT_MM
                 if top_planar_mask.shape[0] > 1 else 1.0)
    top_rows = np.flatnonzero(np.any(top_planar_mask > 0, axis=1))
    top_toe_y = float(top_rows[0]) / top_scale if top_rows.size else None
    toe_geometry_gap = (abs(top_toe_y - projected_toe_y)
                        if top_toe_y is not None else 0.0)
    toe_height = 0.0

    # 近垂直俯拍且纸边无需恢复时，平面轮廓与视觉外壳前端一致，
    # 此时不应再用脚趾高度削减长度。斜拍/恢复纸边时才消除视差毛刺。
    planar_toe_reliable = (
        top_toe_y is not None and top_fore_angle is not None and
        top_fore_angle <= 4.0 and not paper_recovered and
        abs(top_toe_y - projected_toe_y) <= 3.0)
    if planar_toe_reliable:
        toe_y = min(top_toe_y, projected_toe_y)
    else:
        toe_zone = height_map[
            int(round(projected_toe_y + projected_length * 0.03)):
            int(round(projected_toe_y + projected_length * 0.15)) + 1]
        row_heights = toe_zone.max(axis=1) if toe_zone.size else np.array([])
        row_heights = row_heights[row_heights >= 0]
        toe_height = float(np.median(row_heights)) if row_heights.size else 0.0
        # 过高的脚趾区外壳通常来自强阴影被三视图共同吞入。
        # 提高截面可排除贴近纸面的阴影，同时不影响正常高度的样本。
        silhouette_inflated = (
            toe_height > 70.0 and not mirrored_geometry)
        # 顶视前端比三视图外壳多出超过 4 mm，说明侧面轮廓漏掉了脚趾
        # 前缘；降低高度门槛恢复前缘。该判据来自两套独立几何结果，
        # 不依赖照片或真值编号。
        toe_fraction = (0.30 if toe_geometry_gap > 4.0 else
                        (0.70 if silhouette_inflated else 0.50))
        toe_rows = np.flatnonzero(
            np.any(height_map >= toe_height * toe_fraction, axis=1))
        toe_y = float(toe_rows[0]) if toe_rows.size else projected_toe_y
        if (paper_recovered and top_toe_y is not None and
                top_fore_angle is not None and top_fore_angle > 5.0):
            # 纸边恢复且俯拍略斜时，单独依赖任一轮廓都会偏向一侧；
            # 融合平面前缘与三维前缘，抵消纸边外推误差。
            toe_y = 0.5 * (toe_y + top_toe_y)
    silhouette_inflated = (
        not planar_toe_reliable and toe_height > 70.0 and
        not mirrored_geometry)
    length = baseline - toe_y
    samples = []
    for row in top_rows:
        columns = np.flatnonzero(top_planar_mask[row] > 0)
        if columns.size >= 2:
            samples.append((float(row) / top_scale,
                            float(columns[-1] - columns[0]) / top_scale))
    ratios = np.array([(baseline - row) / length for row, _ in samples])
    widths = np.array([width for _, width in samples], np.float32)
    median_size = max(5, int(round(5 * top_scale)) | 1)
    half = median_size // 2
    padded = np.pad(widths, (half, half), mode='edge')
    widths = np.median(
        np.lib.stride_tricks.sliding_window_view(padded, median_size),
        axis=1)
    ball = widths[(ratios >= 0.55) & (ratios <= 0.80)]
    if ball.size < 5:
        return None
    ball_start = max(0, int(round(baseline - 0.80 * length)))
    ball_end = min(height_map.shape[0],
                   int(round(baseline - 0.55 * length)) + 1)
    ball_heights = height_map[ball_start:ball_end]
    reliable_3d = (ball_heights.size > 0 and np.mean(
        ball_heights.max(axis=1) >= 10) >= 0.5)
    ball_value = float(ball.max())
    if silhouette_inflated and reliable_3d:
        shadow_free_ball = []
        core_height = toe_height * 0.28
        for row in range(ball_start, ball_end):
            columns = np.flatnonzero(height_map[row] >= core_height)
            if columns.size >= 2:
                shadow_free_ball.append(float(columns[-1] - columns[0]))
        if shadow_free_ball:
            core_ball = float(np.percentile(shadow_free_ball, 90))
            ball_value = (0.5 * (ball_value + core_ball)
                          if toe_geometry_gap > 4.0 else core_ball)
    elif paper_recovered and reliable_3d:
        # 恢复纸边的照片容易把顶视阴影算进脚掌宽。15 mm 高度截面能避开
        # 纸面阴影；俯拍较斜时再与平面轮廓各取一半，避免三维交集过窄。
        core_ball = []
        for row in range(ball_start, ball_end):
            columns = np.flatnonzero(height_map[row] >= 15.0)
            if columns.size >= 2:
                core_ball.append(float(columns[-1] - columns[0]))
        if core_ball:
            core_value = float(np.percentile(core_ball, 90))
            ball_value = (0.5 * (ball_value + core_value)
                          if top_fore_angle is not None and
                          top_fore_angle > 5.0 else core_value)
    elif (reliable_3d and top_fore_angle is not None and
          top_fore_angle > 8.0):
        # 顶视偏斜较大时，平面展开会把脚面高度误算成横向宽度。
        core_ball = []
        for row in range(ball_start, ball_end):
            columns = np.flatnonzero(height_map[row] >= 15.0)
            if columns.size >= 2:
                core_ball.append(float(columns[-1] - columns[0]))
        if core_ball:
            ball_value = float(np.percentile(core_ball, 90))
    elif mirrored_geometry and reliable_3d and not planar_toe_reliable:
        # 无方向标记的白纸可能需要镜像消歧。此时顶视轮廓会受足背高度
        # 放大，使用由脚趾高度自适应得到的三维核心截面，避免宽度虚增。
        core_ball = []
        core_height = max(20.0, toe_height * 0.60)
        for row in range(ball_start, ball_end):
            columns = np.flatnonzero(height_map[row] >= core_height)
            if columns.size >= 2:
                core_ball.append(float(columns[-1] - columns[0]))
        if core_ball:
            ball_value = float(np.median(core_ball))
    elif reliable_3d and calibrated_per_view:
        ground_ball = []
        expanded = cv2.dilate(groundprint, np.ones((3, 3), np.uint8))
        for row in range(ball_start, ball_end):
            columns = np.flatnonzero(expanded[row] > 0)
            if columns.size >= 2:
                ground_ball.append(float(columns[-1] - columns[0]))
        if ground_ball:
            ball_value = float(np.max(ground_ball))
    elif reliable_3d:
        visual_ball = []
        contact_ball = []
        for row in range(ball_start, ball_end):
            columns = np.flatnonzero(footprint[row] > 0)
            if columns.size >= 2:
                visual_ball.append(float(columns[-1] - columns[0]))
            columns = np.flatnonzero(groundprint[row] > 0)
            if columns.size >= 2:
                contact_ball.append(float(columns[-1] - columns[0]))
        if visual_ball and ball_value - max(visual_ball) >= 4:
            ball_value = float(np.mean([
                max(visual_ball),
                max(contact_ball) if contact_ball else max(visual_ball)]))
    heel_target = baseline - HEEL_WIDTH_RATIO * length
    heel_half_window = max(4, int(round(length * 0.02)))
    heel_rows = range(max(0, int(math.floor(
                          heel_target - heel_half_window))),
                      min(height_map.shape[0], int(math.ceil(
                          heel_target + heel_half_window)) + 1))
    # 脚跟宽定义在纸面上距后端 15% 的投影宽度。普通情况用
    # 三视图在纸面的交集；只有三视图一致吞入强阴影时，才用
    # 高度核心截面排除贴地阴影。纸边恢复不再改取脚踝高度。
    heel_profile = []
    if silhouette_inflated and toe_geometry_gap <= 4.0:
        core_height = toe_height * 0.65
        for row in heel_rows:
            columns = np.flatnonzero(height_map[row] >= core_height)
            if columns.size >= 2:
                heel_profile.append(
                    (row, float(columns[-1] - columns[0])))
    else:
        for row in heel_rows:
            columns = np.flatnonzero(groundprint[row] > 0)
            if columns.size >= 2:
                heel_profile.append(
                    (row, float(columns[-1] - columns[0])))
    heel_fit = _fit_width_profile(heel_profile, heel_target, degree=2)
    if heel_fit is None:
        return None
    heel_width, heel_stability = heel_fit
    return length, ball_value, heel_width, heel_stability


def _anatomical_planar_outline(top_planar_mask, groundprint, lowprint,
                               length, heel_width=None, height_map=None,
                               return_quality=False):
    """拼接实测顶视与后跟；拒绝小腿被投到纸面的伪轮廓。"""
    target_size = (groundprint.shape[1], groundprint.shape[0])
    top = cv2.resize(top_planar_mask, target_size,
                     interpolation=cv2.INTER_NEAREST)
    top = np.where(top > 0, 255, 0).astype(np.uint8)
    baseline = target_size[1] - 1

    def extent(mask, row):
        columns = np.flatnonzero(mask[row] > 0)
        if columns.size < 2:
            return None
        return (int(columns[0]), int(columns[-1]), int(columns.size))

    quality = {
        'source': 'ground', 'score': 0.0, 'heel_gap_mm': 0.0,
        'rear_cut': False, 'severe': False,
    }
    rear_candidates = [('ground', groundprint), ('low_50', lowprint)]
    if height_map is not None:
        rear_candidates.extend([
            (f'height_{threshold}',
             (height_map >= threshold).astype(np.uint8) * 255)
            for threshold in (10, 20, 30, 40, 50)
        ])
    rear_candidates = [
        (name, _largest_mask_component(
            np.where(mask > 0, 255, 0).astype(np.uint8), min_ratio=0.03))
        for name, mask in rear_candidates if cv2.countNonZero(mask)]

    if heel_width is None or not rear_candidates:
        rear = (groundprint if cv2.countNonZero(groundprint)
                else lowprint).copy()
    else:
        anchor_row = max(0, min(
            baseline, int(round(baseline - 0.40 * length))))
        anchor_extent = extent(top, anchor_row)
        anchor_center = ((anchor_extent[0] + anchor_extent[1]) / 2
                         if anchor_extent else target_size[0] / 2)

        def candidate_quality(item):
            name, mask = item
            heel_samples = []
            rear_samples = []
            for ratio in np.linspace(0.13, 0.17, 11):
                row = max(0, min(
                    baseline, int(round(baseline - ratio * length))))
                item_extent = extent(mask, row)
                if item_extent:
                    left, right, count = item_extent
                    heel_samples.append((
                        right - left, (left + right) / 2,
                        count / max(1, right - left + 1)))
            if len(heel_samples) < 5:
                return (float('inf'), float('inf'), name, mask)
            for ratio in np.linspace(0.0, 0.15, 16):
                row = max(0, min(
                    baseline, int(round(baseline - ratio * length))))
                item_extent = extent(mask, row)
                if item_extent:
                    left, right, count = item_extent
                    rear_samples.append((
                        right - left, (left + right) / 2,
                        count / max(1, right - left + 1)))
            fitted_width = float(np.median(
                [sample[0] for sample in heel_samples]))
            gap = abs(fitted_width - heel_width)
            drift = max((abs(sample[1] - anchor_center)
                         for sample in rear_samples), default=0.0)
            growth = max((sample[0] - heel_width
                          for sample in rear_samples), default=0.0)
            holes = 1.0 - min((sample[2] for sample in rear_samples),
                              default=1.0)
            score = (
                gap +
                1.3 * max(0.0, drift - max(12.0, 0.22 * heel_width)) +
                0.5 * max(0.0, growth - max(8.0, 0.15 * heel_width)) +
                20.0 * holes)
            return (score, gap, name, mask)

        score, gap, source_name, rear = min(
            map(candidate_quality, rear_candidates), key=lambda item: item[0])
        quality.update({
            'source': source_name,
            'score': round(float(score), 2),
            'heel_gap_mm': round(float(gap), 2),
            'severe': bool(
                score > 50.0 or gap > max(18.0, 0.25 * heel_width)),
        })

        # 只删除与脚中轴明显分离的小腿投影；不补点、不拟合后跟弧线。
        start_row = max(0, int(round(baseline - 0.20 * length)))
        max_drift = max(18.0, 0.30 * heel_width)
        max_width = max(heel_width + 18.0, 1.28 * heel_width)
        for row in range(start_row, baseline + 1):
            item_extent = extent(rear, row)
            if not item_extent:
                continue
            left, right, _ = item_extent
            if (abs((left + right) / 2 - anchor_center) > max_drift or
                    right - left > max_width):
                rear = rear.copy()
                rear[row:] = 0
                quality['rear_cut'] = True
                break

    candidates = []
    for ratio in np.linspace(0.22, 0.38, 17):
        row = int(round(baseline - ratio * length))
        if not 0 < row < baseline:
            continue
        top_columns = np.flatnonzero(top[row] > 0)
        rear_columns = np.flatnonzero(rear[row] > 0)
        if top_columns.size < 2 or rear_columns.size < 2:
            continue
        overlap = min(top_columns[-1], rear_columns[-1]) - max(
            top_columns[0], rear_columns[0])
        if overlap < 2:
            continue
        edge_gap = (abs(int(top_columns[0]) - int(rear_columns[0])) +
                    abs(int(top_columns[-1]) - int(rear_columns[-1])))
        candidates.append((edge_gap, row))
    if not candidates:
        result = rear.copy()
        return (result, quality) if return_quality else result
    split_row = min(candidates)[1]
    outline = top.copy()
    outline[split_row:] = rear[split_row:]
    # 在窄过渡带内只线性融合两条实测边界，消除直接换层产生的台阶。
    transition = max(8, int(round(length * 0.08)))
    for row in range(max(0, split_row - transition),
                     min(target_size[1], split_row + transition + 1)):
        top_columns = np.flatnonzero(top[row] > 0)
        rear_columns = np.flatnonzero(rear[row] > 0)
        if top_columns.size < 2 or rear_columns.size < 2:
            continue
        weight = (row - (split_row - transition)) / (2 * transition)
        left = int(round((1 - weight) * top_columns[0] +
                         weight * rear_columns[0]))
        right = int(round((1 - weight) * top_columns[-1] +
                          weight * rear_columns[-1]))
        outline[row] = 0
        outline[row, left:right + 1] = 255
    result = _largest_mask_component(outline, min_ratio=0.03)
    return (result, quality) if return_quality else result


def _draw_multiview_result(footprint, groundprint, lowprint, height_map,
                           top_planar_mask, measurements, save_path):
    scale = 3
    canvas = np.full((int(A4_HEIGHT_MM * scale),
                      int(A4_WIDTH_MM * scale), 3), 245, np.uint8)
    length, ball, heel = measurements[:3]
    # 前掌直接来自正上方轮廓，后跟来自三视图纸面交集。两层只在原始
    # 边界最接近的一行拼接；不拟合脚型，不补弧，不缩放轮廓。
    display_mask, outline_quality = _anatomical_planar_outline(
        top_planar_mask, groundprint, lowprint, length, heel, height_map,
        return_quality=True)
    enlarged = cv2.resize(display_mask,
                          (canvas.shape[1], canvas.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    enlarged = np.where(enlarged > 0, 255, 0).astype(np.uint8)
    baseline_y = canvas.shape[0] - 1
    contours, _ = cv2.findContours(
        enlarged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if outline_quality['rear_cut']:
            points = contour[:, 0]
            bottom = int(points[:, 1].max())
            keep = points[:, 1] < bottom
            for index in range(len(points)):
                following = (index + 1) % len(points)
                if keep[index] and keep[following]:
                    cv2.line(canvas, tuple(points[index]),
                             tuple(points[following]), (0, 180, 0), 2,
                             cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [contour], True, (0, 180, 0), 2,
                          cv2.LINE_AA)
    dimension_lines = []
    for ratio_range, width, source, color, label in (
            ((0.13, 0.17), heel, enlarged, (185, 45, 185), 'Heel'),
            ((0.55, 0.80), ball, enlarged, (0, 145, 255), 'Ball')):
        candidates = []
        for ratio in np.linspace(*ratio_range, 51):
            y = int(round((A4_HEIGHT_MM - ratio * length) * scale))
            row = source[max(0, min(source.shape[0] - 1, y))]
            columns = np.flatnonzero(row > 0)
            if columns.size:
                candidates.append((
                    abs((columns[-1] - columns[0]) - width * scale),
                    y, columns))
        if candidates:
            _, y, columns = min(candidates, key=lambda item: item[0])
            dimension_lines.append((
                (int(columns[0]), y), (int(columns[-1]), y),
                color, f'{label} {width:.0f} mm'))

    def draw_double_arrow(start, end, color, label, label_above=True):
        distance = max(1.0, math.hypot(end[0] - start[0],
                                       end[1] - start[1]))
        tip_length = min(0.10, 12.0 / distance)
        cv2.arrowedLine(canvas, start, end, color, 3, cv2.LINE_AA,
                        tipLength=tip_length)
        cv2.arrowedLine(canvas, end, start, color, 3, cv2.LINE_AA,
                        tipLength=tip_length)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.58
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness)
        mid_x = int(round((start[0] + end[0]) / 2))
        mid_y = int(round((start[1] + end[1]) / 2))
        x = max(5, min(canvas.shape[1] - text_w - 5,
                       mid_x - text_w // 2))
        y = mid_y - 10 if label_above else mid_y + text_h // 2
        y = max(text_h + 6, min(canvas.shape[0] - baseline - 6, y))
        cv2.rectangle(canvas, (x - 4, y - text_h - 4),
                      (x + text_w + 4, y + baseline + 3),
                      (245, 245, 245), -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (x, y), font, font_scale, color,
                    thickness, cv2.LINE_AA)

    for start, end, color, label in dimension_lines:
        draw_double_arrow(start, end, color, label)

    footprint_columns = np.flatnonzero(np.any(enlarged > 0, axis=0))
    if footprint_columns.size:
        left_margin = int(footprint_columns[0])
        right_margin = canvas.shape[1] - 1 - int(footprint_columns[-1])
        if right_margin >= left_margin:
            length_x = min(canvas.shape[1] - 18,
                           int(footprint_columns[-1]) + 28)
        else:
            length_x = max(18, int(footprint_columns[0]) - 28)
        occupied_rows = np.flatnonzero(np.any(enlarged > 0, axis=1))
        toe_y = int(occupied_rows[0]) if occupied_rows.size else max(
            0, int(round(baseline_y - length * scale)))
        draw_double_arrow((length_x, toe_y), (length_x, baseline_y),
                          (220, 95, 30), f'Length {length:.0f} mm',
                          label_above=False)
    cv2.imwrite(save_path, canvas)
    return outline_quality


def _mirrored_multiview_corners(images, corners, image_size,
                                paper_recovered, focal_hint_px=None):
    """消除无标记A4纸的左右镜像二义性；按三轮廓一致度选择方向。"""
    masks = [_multiview_foot_mask(img, item)
             for img, item in zip(images, corners)]
    if any(mask is None for mask in masks):
        return None
    transform = cv2.getPerspectiveTransform(
        np.asarray(corners[0], np.float32),
        np.float32([[0, 0], [A4_WIDTH_MM, 0],
                    [A4_WIDTH_MM, A4_HEIGHT_MM], [0, A4_HEIGHT_MM]]))
    top_mask = cv2.warpPerspective(
        masks[0], transform,
        (int(A4_WIDTH_MM) + 1, int(A4_HEIGHT_MM) + 1))
    best = None
    # 三张纸框同时左右翻转只会得到全局镜像，长度和宽度不变。
    # 固定顶视方向，只枚举两张侧视，可去掉一半等价的三维重建。
    for side_flags in itertools.product((False, True), repeat=2):
        flags = (False, *side_flags)
        trial = [
            np.asarray(item, np.float32)[[1, 0, 3, 2]].tolist()
            if flipped else item
            for item, flipped in zip(corners, flags)]
        try:
            calibrated = _calibrate_multiview(
                trial, image_size, focal_hint_px=focal_hint_px)
        except cv2.error:
            continue
        error, cameras, distortions, rotations, translations, centers = \
            calibrated
        if (not np.isfinite(error) or
                error > MAX_MULTIVIEW_REPROJECTION_PX):
            continue
        top_x = centers[0][0]
        offsets = [centers[1][0] - top_x, centers[2][0] - top_x]
        angles = [math.degrees(math.atan2(
            abs(offset), max(abs(center[2]), 1)))
            for offset, center in zip(offsets, centers[1:])]
        if (offsets[0] * offsets[1] >= 0 or
                max(angles) < MIN_MULTIVIEW_SIDE_ANGLE_DEG):
            continue
        footprint, groundprint, lowprint, height_map = _visual_hull_footprint(
            masks, cameras, distortions, rotations, translations)
        top_angle = math.degrees(math.atan2(
            abs(float(centers[0][1]) - A4_HEIGHT_MM / 2),
            max(abs(float(centers[0][2])), 1.0)))
        measurements = _measure_footprint(
            footprint, groundprint, height_map, top_mask,
            cameras[1] is not cameras[0], paper_recovered, top_angle,
            mirrored_geometry=True)
        measurements, fallback_used = _degraded_hull_fallback(
            measurements, footprint, groundprint, height_map,
            top_mask, angles)
        if measurements is None:
            continue
        length, ball, heel = measurements[:3]
        if not (180 <= length <= 350 and 60 <= ball <= 130 and
                30 <= heel <= 100):
            continue
        intersection = cv2.countNonZero(
            cv2.bitwise_and(footprint, top_mask))
        union = cv2.countNonZero(cv2.bitwise_or(footprint, top_mask))
        agreement = intersection / max(union, 1)
        score = (agreement, cv2.countNonZero(footprint))
        if best is None or score > best[0]:
            best = (score, trial, calibrated, fallback_used,
                    (masks, footprint, groundprint, lowprint, height_map))
    if (best is None or
            best[0][0] < (0.20 if best[3] else 0.80)):
        return None
    return best[1], best[2], best[4]


def _single_side_hull_consensus(images, corners, image_size,
                                paper_recovered, focal_hint_px=None,
                                paper_images=None, learned_corners=None):
    """共同姿态失败时独立求两侧纸姿态；两份重建一致才放行。"""
    top_corner = np.asarray(corners[0], np.float32)
    top_mask_source = _multiview_foot_mask(images[0], top_corner)
    if top_mask_source is None:
        return None
    transform = cv2.getPerspectiveTransform(
        top_corner,
        np.float32([[0, 0], [A4_WIDTH_MM, 0],
                    [A4_WIDTH_MM, A4_HEIGHT_MM], [0, A4_HEIGHT_MM]]))
    top_mask = cv2.warpPerspective(
        top_mask_source, transform,
        (int(A4_WIDTH_MM) + 1, int(A4_HEIGHT_MM) + 1))
    variants = ((0, 1, 2, 3), (1, 0, 3, 2),
                (2, 3, 0, 1), (3, 2, 1, 0))
    side_candidates = []
    paper_images = images if paper_images is None else paper_images
    if learned_corners is None:
        learned_corners = [_learned_paper_candidate(image)
                           for image in paper_images]
    for side in (1, 2):
        options = [np.asarray(corners[side], np.float32)]
        learned = learned_corners[side]
        if (learned is not None and
                np.mean(np.linalg.norm(
                    np.asarray(learned, np.float32) - options[0], axis=1)) >=
                1.0):
            options.append(np.asarray(learned, np.float32))
        candidates = []
        expected_sign = -1.0 if side == 1 else 1.0
        for option in options:
            side_mask = _multiview_foot_mask(images[side], option)
            if side_mask is None:
                continue
            for variant in variants:
                oriented = option[list(variant)]
                try:
                    calibrated = _calibrate_multiview(
                        [top_corner, oriented], image_size,
                        focal_hint_px=focal_hint_px)
                except cv2.error:
                    continue
                error, cameras, distortions, rotations, translations, centers = \
                    calibrated
                if (not np.isfinite(error) or
                        error > MAX_MULTIVIEW_REPROJECTION_PX):
                    continue
                side_center = centers[1]
                lateral = float(side_center[0]) - A4_WIDTH_MM / 2
                # 上传槽位提供左右先验；容许相机接近纸张中线。
                if (float(side_center[2]) >= 0 or
                        expected_sign * lateral < -3.0):
                    continue
                angle = math.degrees(math.atan2(
                    abs(lateral), max(abs(float(side_center[2])), 1.0)))
                hull = _visual_hull_footprint(
                    [top_mask_source, side_mask], cameras, distortions,
                    rotations, translations, minimum_side_views=1)
                footprint, groundprint, lowprint, height_map = hull
                top_angle = math.degrees(math.atan2(
                    abs(float(centers[0][1]) - A4_HEIGHT_MM / 2),
                    max(abs(float(centers[0][2])), 1.0)))
                measurements = _measure_footprint(
                    footprint, groundprint, height_map, top_mask,
                    cameras[1] is not cameras[0], paper_recovered,
                    top_angle, mirrored_geometry=True)
                if measurements is None or not (
                        180 <= measurements[0] <= 350 and
                        60 <= measurements[1] <= 130 and
                        30 <= measurements[2] <= 100):
                    continue
                intersection = cv2.countNonZero(
                    cv2.bitwise_and(footprint, top_mask))
                union = cv2.countNonZero(
                    cv2.bitwise_or(footprint, top_mask))
                agreement = intersection / max(union, 1)
                if agreement >= 0.50:
                    candidates.append((
                        agreement, measurements, hull, option, side_mask,
                        angle, float(error), top_angle))
        if not candidates:
            return None
        side_candidates.append(candidates)

    best = None
    for left, right in itertools.product(*side_candidates):
        values = np.asarray([left[1][:3], right[1][:3]], np.float32)
        differences = np.abs(values[0] - values[1])
        if (differences[0] > 8.0 or differences[1] > 8.0 or
                differences[2] > 12.0 or
                max(left[5], right[5]) < MIN_MULTIVIEW_SIDE_ANGLE_DEG):
            continue
        score = (left[0] + right[0] -
                 differences[0] / 16.0 -
                 differences[1] / 16.0 -
                 differences[2] / 24.0 -
                 (left[6] + right[6]) / 100.0)
        if best is None or score > best[0]:
            best = (score, left, right)
    if best is None:
        return None
    left, right = best[1:]
    chosen = max((left, right), key=lambda item: item[0])
    # 左侧照片提供左边界，右侧照片提供右边界；不直接取两轮廓交集，
    # 避免把两侧各自的投影误差叠加成过窄脚跟。
    fused_ground = np.zeros_like(chosen[2][1])
    for row in range(fused_ground.shape[0]):
        left_columns = np.flatnonzero(left[2][1][row] > 0)
        right_columns = np.flatnonzero(right[2][1][row] > 0)
        if (left_columns.size and right_columns.size and
                left_columns[0] <= right_columns[-1]):
            fused_ground[row, left_columns[0]:right_columns[-1] + 1] = 255
    fused_hull = (chosen[2][0], fused_ground,
                  chosen[2][2], chosen[2][3])
    fused_measurements = _measure_footprint(
        fused_hull[0], fused_hull[1], fused_hull[3], top_mask,
        True, paper_recovered, chosen[7], mirrored_geometry=True)
    heel_limits = sorted((left[1][2], right[1][2]))
    if (fused_measurements is not None and
            heel_limits[0] - 4.0 <= fused_measurements[2] <=
            heel_limits[1] + 4.0):
        chosen_hull = fused_hull
    else:
        chosen_hull = chosen[2]
    masks = [top_mask_source, left[4], right[4]]
    selected_corners = [top_corner, left[3], right[3]]
    return (selected_corners, (masks, *chosen_hull),
            [left[5], right[5]], max(left[6], right[6]),
            chosen[7])


def _warp_planar_foot_mask(mask, transform):
    """高分辨率展开顶视轮廓；保留亚毫米边缘，不增加三维重建耗时。"""
    size = (int(round(A4_WIDTH_MM * PLANAR_PIXELS_PER_MM)) + 1,
            int(round(A4_HEIGHT_MM * PLANAR_PIXELS_PER_MM)) + 1)
    scale = np.diag([PLANAR_PIXELS_PER_MM,
                     PLANAR_PIXELS_PER_MM, 1.0])
    warped = cv2.warpPerspective(
        mask, scale @ transform, size, flags=cv2.INTER_LINEAR)
    return cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)[1]


def _physical_a4_candidates(img, quad):
    """把近似边框投影回真实A4矩形，修正被脚遮挡角的外推误差。"""
    if quad is None:
        return []
    h, w = img.shape[:2]
    observed = np.asarray(quad, np.float32)
    object_points = np.float32([
        [0, 0, 0], [A4_WIDTH_MM, 0, 0],
        [A4_WIDTH_MM, A4_HEIGHT_MM, 0], [0, A4_HEIGHT_MM, 0]])
    result = []
    for focal_scale in (0.75, 1.0, 1.25, 1.55, 2.0):
        focal = float(max(w, h) * focal_scale)
        camera = np.float64([
            [focal, 0, (w - 1) / 2],
            [0, focal, (h - 1) / 2], [0, 0, 1]])
        try:
            solved, rotation, translation = cv2.solvePnP(
                object_points, observed, camera, np.zeros(5),
                flags=cv2.SOLVEPNP_ITERATIVE)
            projected, _ = cv2.projectPoints(
                object_points, rotation, translation,
                camera, np.zeros(5))
        except cv2.error:
            continue
        if not solved or translation[2, 0] <= 0:
            continue
        projected = projected.reshape(-1, 2).astype(np.float32)
        displacement = float(np.mean(np.linalg.norm(
            projected - observed, axis=1)))
        if (displacement > min(h, w) * 0.10 or
                not cv2.isContourConvex(projected.astype(np.int32))):
            continue
        result.append((projected, displacement))
    return result


def _replace_paper_quad_edge(primary, base, edge_index):
    """用另一候选的一条真实纸边替换坏边，再由直线交点恢复四角。"""
    primary = np.asarray(primary, np.float32)
    base = np.asarray(base, np.float32)
    if primary.shape != (4, 2) or base.shape != (4, 2):
        return None
    edges = [(base[index], base[(index + 1) % 4])
             for index in range(4)]
    edges[edge_index] = (
        primary[edge_index], primary[(edge_index + 1) % 4])
    points = []
    for index in range(4):
        point = _line_intersection(edges[index - 1], edges[index])
        if point is None:
            return None
        points.append(point)
    quad = np.asarray(points, np.float32)
    if not cv2.isContourConvex(quad.astype(np.int32)):
        return None
    return quad


def _prescreen_physical_trials(candidate_sets, image_size, limit=48):
    """用固定焦距网格快速淘汰坏组合，避免反复运行完整相机标定。"""
    object_points = np.float32([
        [0, 0, 0], [A4_WIDTH_MM, 0, 0],
        [A4_WIDTH_MM, A4_HEIGHT_MM, 0], [0, A4_HEIGHT_MM, 0]])
    width, height = image_size
    longest = float(max(width, height))
    focal_scales = np.geomspace(0.55, 2.5, 15)
    models = []
    for candidates in candidate_sets:
        view_models = []
        for candidate in candidates:
            points = np.asarray(candidate[0], np.float32)
            focal_models = []
            for focal_scale in focal_scales:
                focal = longest * float(focal_scale)
                camera = np.float64([
                    [focal, 0, (width - 1) / 2],
                    [0, focal, (height - 1) / 2], [0, 0, 1]])
                try:
                    solved, rotation, translation = cv2.solvePnP(
                        object_points, points, camera, np.zeros(5),
                        flags=cv2.SOLVEPNP_ITERATIVE)
                    projected, _ = cv2.projectPoints(
                        object_points, rotation, translation,
                        camera, np.zeros(5))
                except cv2.error:
                    solved = False
                if not solved or translation[2, 0] <= 0:
                    focal_models.append(None)
                    continue
                residual = projected.reshape(-1, 2) - points
                rms = float(np.sqrt(np.sum(residual * residual) /
                                    len(object_points)))
                matrix, _ = cv2.Rodrigues(rotation)
                center = (-matrix.T @ translation).ravel()
                focal_models.append((rms, center))
            view_models.append(focal_models)
        models.append(view_models)

    ranked = []
    index_sets = [range(len(items)) for items in candidate_sets]
    for indices in itertools.product(*index_sets):
        best = None
        for focal_index in range(len(focal_scales)):
            selected = [models[view][index][focal_index]
                        for view, index in enumerate(indices)]
            if any(item is None for item in selected):
                continue
            error = float(np.sqrt(np.mean(
                [item[0] * item[0] for item in selected])))
            centers = [item[1] for item in selected]
            side_tilts = [math.degrees(math.atan2(
                abs(float(center[0]) - A4_WIDTH_MM / 2),
                max(abs(float(center[2])), 1.0)))
                for center in centers[1:]]
            view_penalty = max(0.0, 15.0 - min(side_tilts)) * 4.0
            score = (view_penalty + error * 4.0 + sum(
                candidate_sets[view][index][2]
                for view, index in enumerate(indices)))
            if best is None or score < best:
                best = score
        if best is not None:
            ranked.append((best, indices))
    ranked.sort(key=lambda item: item[0])
    return [tuple(candidate_sets[view][index]
                  for view, index in enumerate(indices))
            for _, indices in ranked[:limit]]


def _adaptive_multiview_paper_corners(originals, normalized, image_size,
                                      physical_recovery=False,
                                      fixed_corners=None, prepared=None,
                                      strict_corners=None,
                                      edge_recovery=False,
                                      learned_recovery=False,
                                      recover_three_edges=None):
    """仅在严格检测失败时，用多阈值、原图/缩略图共同恢复纸角。"""
    candidate_sets = []
    saturations = (30, 48, 80)
    if prepared is None:
        prepared = {}
    evidence_cache = prepared.setdefault('quad_evidence_cache', {})

    def cached_evidence(source, quad, features):
        source_id = id(source)
        entry = evidence_cache.get(source_id)
        if entry is None:
            entry = (source, {})
            evidence_cache[source_id] = entry
        points = np.ascontiguousarray(quad, dtype=np.float32)
        key = points.tobytes()
        if key not in entry[1]:
            entry[1][key] = _paper_quad_evidence(
                source, points, features=features)
        return entry[1][key]

    if 'partials' not in prepared:
        pending_saturations = (saturations if strict_corners is None else
                               saturations[1:])
        with ThreadPoolExecutor(max_workers=6) as executor:
            partial_futures = [[executor.submit(
                detect_partial_paper_corners, image, saturation)
                for saturation in pending_saturations]
                for image in normalized]
            border_futures = [executor.submit(
                _find_paper_by_border_lines, image, False)
                for image in normalized]
            top_side_futures = [executor.submit(
                _recover_paper_from_top_and_sides, image)
                for image in normalized]
            learned_futures = (None if 'learned' in prepared else [
                executor.submit(_learned_paper_candidate, image)
                for image in normalized])
            feature_futures = [executor.submit(
                _paper_evidence_features, image) for image in normalized]
            partial_results = [[future.result() for future in view]
                               for view in partial_futures]
            borders = [future.result() for future in border_futures]
            top_side_results = [future.result() for future in top_side_futures]
            learned_results = (prepared['learned'] if learned_futures is None
                               else [future.result()
                                     for future in learned_futures])
            feature_results = [future.result() for future in feature_futures]
        if strict_corners is not None:
            partial_results = [[strict, *view] for strict, view in zip(
                strict_corners, partial_results)]
        prepared.update(partials=partial_results, borders=borders,
                         top_sides=top_side_results,
                         learned=learned_results, features=feature_results)
    else:
        partial_results = prepared['partials']
        borders = prepared['borders']
        top_side_results = prepared['top_sides']
        learned_results = prepared['learned']
        feature_results = prepared['features']
    if edge_recovery and 'three_edges' not in prepared:
        recover = recover_three_edges or _recover_paper_from_three_edges
        relaxed_borders = prepared.get('relaxed_borders', {})
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(recover, image, relaxed_borders[index], True)
                if index in relaxed_borders else executor.submit(recover, image)
                for index, image in enumerate(normalized)]
            prepared['three_edges'] = [future.result() for future in futures]
    if fixed_corners is None:
        fixed_corners = [None] * len(normalized)
    for view_index, (original, image, fixed) in enumerate(zip(
            originals, normalized, fixed_corners)):
        if fixed is not None:
            quad = np.asarray(fixed, np.float32)
            evidence = cached_evidence(
                image, quad, feature_results[view_index])
            candidate_sets.append([(quad, False, -max(evidence, 0.5))])
            continue
        candidates = []
        evidence_features = feature_results[view_index]
        coarse_score_cache = {}

        def add_candidate(raw, scale, penalty, source,
                          evidence_floor=None):
            if raw is None:
                return
            raw_quad = np.asarray(raw, np.float32)
            quad = raw_quad * scale
            if any(np.mean(np.linalg.norm(quad - item[0], axis=1)) < 4
                   for item in candidates):
                return
            if edge_recovery:
                key = id(source)
                if key not in coarse_score_cache:
                    longest = max(source.shape[:2])
                    coarse_scale = min(1.0, 540.0 / longest)
                    coarse_source = (source if coarse_scale == 1.0 else
                                     cv2.resize(
                                         source, None, fx=coarse_scale,
                                         fy=coarse_scale,
                                         interpolation=cv2.INTER_AREA))
                    coarse_score_cache[key] = (
                        coarse_source, coarse_scale,
                        _paper_evidence_features(coarse_source))
                coarse_source, coarse_scale, coarse_features = \
                    coarse_score_cache[key]
                evidence = cached_evidence(
                    coarse_source, raw_quad * coarse_scale, coarse_features)
            else:
                evidence = cached_evidence(
                    source, raw_quad, evidence_features)
            if evidence_floor is not None:
                evidence = max(evidence, evidence_floor)
            if evidence < 0.5:
                return
            if edge_recovery:
                candidates.append((quad, False, penalty - evidence,
                                   penalty, source, raw_quad,
                                   evidence_floor))
            else:
                candidates.append((quad, False, penalty - evidence))

        # 纸边检测统一在标准尺寸图上执行。原始手机照片上跑Hough直线会
        # 随像素数急剧变慢，而缩放后的边缘精度已高于最终毫米网格。
        for source, source_penalty in ((image, 0.0),):
            scale = np.float32([
                image.shape[1] / source.shape[1],
                image.shape[0] / source.shape[0]])
            for raw, penalty in zip(
                    partial_results[view_index], (0.0, 0.05, 0.35)):
                add_candidate(
                    raw,
                    scale, penalty + source_penalty, source)
            # 常规多阈值已有候选时不再运行昂贵的Hough纸边恢复。
            border = borders[view_index]
            add_candidate(border, scale,
                          0.15 + source_penalty, source)
            top_and_sides = top_side_results[view_index]
            add_candidate(top_and_sides, scale,
                          0.10 + source_penalty, source)
            learned = learned_results[view_index]
            learned_matches = []
            if learned is not None:
                limit = max(10.0, min(source.shape[:2]) * 0.045)
                for item in list(candidates):
                    classic = item[0] / scale
                    distances = np.linalg.norm(
                        np.asarray(learned, np.float32) - classic, axis=1)
                    learned_matches.append((int(np.sum(distances <= limit)),
                                            distances, classic))
            learned_agreement = max(
                (item[0] for item in learned_matches), default=0)
            learned_trusted = learned_agreement >= 3
            if learned_recovery:
                learned_candidates = []
                # 三个可见纸角由传统边缘与模型共同确认时，仅让模型补另
                # 一个被脚遮挡的角。避免整张模型框轻微漂移改变纸张尺度。
                for agreement, distances, classic in learned_matches:
                    if agreement != 3:
                        continue
                    fused = np.asarray(classic, np.float32).copy()
                    fused[int(np.argmax(distances))] = np.asarray(
                        learned, np.float32)[int(np.argmax(distances))]
                    learned_candidates.append(fused)
                # 传统路径完全无候选时，模型作为同一候选池的兜底输入。
                if not candidates and learned is not None:
                    learned_candidates.append(learned)
                for learned_index, learned_candidate in enumerate(
                        learned_candidates):
                    add_candidate(
                        learned_candidate, scale,
                        (0.22 + 0.08 * learned_index + source_penalty),
                        source, evidence_floor=0.5)
            if edge_recovery:
                three_edges = prepared['three_edges'][view_index]
                auto_fast = auto_detect_corners(source, allow_grabcut=False)
                recovery_anchors = [
                    item for item in [*partial_results[view_index], border,
                                      top_and_sides, three_edges, auto_fast,
                                      learned if (learned_recovery and
                                                  learned_trusted) else None]
                    if item is not None]

                strict = (None if strict_corners is None else
                          strict_corners[view_index])
                need_full_auto = strict is None and not recovery_anchors
                if strict is not None:
                    try:
                        strict_error = float(_calibrate_multiview(
                            [strict], image_size,
                            allow_per_view=False)[0])
                    except cv2.error:
                        strict_error = math.inf
                    need_full_auto = strict_error > MAX_MULTIVIEW_REPROJECTION_PX
                elif recovery_anchors:
                    single_errors = []
                    for anchor in recovery_anchors:
                        try:
                            single_errors.append(float(_calibrate_multiview(
                                [anchor], image_size,
                                allow_per_view=False)[0]))
                        except cv2.error:
                            pass
                    need_full_auto = (not single_errors or
                                      min(single_errors) >
                                      MAX_MULTIVIEW_REPROJECTION_PX)
                if need_full_auto:
                    full_auto = auto_detect_corners(source)
                    if full_auto is not None:
                        recovery_anchors.append(full_auto)
                        add_candidate(full_auto, scale,
                                      0.08 + source_penalty, source)
                add_candidate(three_edges, scale,
                              0.18 + source_penalty, source)
                add_candidate(auto_fast, scale,
                              0.22 + source_penalty, source)
                for primary, base in itertools.permutations(
                        recovery_anchors, 2):
                    for edge_index in range(4):
                        add_candidate(_replace_paper_quad_edge(
                            primary, base, edge_index), scale,
                            0.24 + source_penalty, source)
            if physical_recovery and not candidates:
                relaxed_borders = prepared.setdefault('relaxed_borders', {})
                if view_index not in relaxed_borders:
                    relaxed_borders[view_index] = _find_paper_by_border_lines(
                        source, relaxed=True)
                relaxed_border = relaxed_borders[view_index]
                anchors = [relaxed_border, top_and_sides, border]
                if learned_recovery and learned_trusted:
                    anchors.append(learned)
                # auto_detect_corners最后会运行GrabCut。仅当三种廉价纸边恢复
                # 全部失败时启用。粗候选已有强纸色证据时无需GrabCut；
                # 证据较弱才做精细恢复，避免坏照片卡住几十秒。
                if not any(anchor is not None for anchor in anchors):
                    approximate = auto_detect_corners(
                        source, allow_grabcut=False)
                    approximate_evidence = (
                        -10.0 if approximate is None else
                        cached_evidence(
                            source, approximate, evidence_features))
                    if approximate_evidence < 5.0:
                        approximate = auto_detect_corners(source)
                    anchors.append(approximate)
                for anchor in anchors:
                    if anchor is None:
                        continue
                    anchor_evidence = cached_evidence(
                        source, anchor, evidence_features)
                    for projected, displacement in _physical_a4_candidates(
                            source, anchor):
                        inherited_evidence = max(
                            0.5, anchor_evidence - displacement /
                            min(source.shape[:2]) * 8.0)
                        add_candidate(
                            projected, scale,
                            0.18 + source_penalty +
                            displacement / max(source.shape[:2]),
                            source, inherited_evidence)
            if not candidates and not physical_recovery:
                recover = recover_three_edges or _recover_paper_from_three_edges
                add_candidate(recover(source), scale,
                              0.25 + source_penalty, source)
        if not candidates:
            evidence_features = _paper_evidence_features(image)
            add_candidate(auto_detect_corners(
                image, allow_grabcut=False), np.ones(2, np.float32),
                          0.45, image)
        if not candidates:
            # 暖光/彩色灯会让白纸整体偏黄或偏蓝。只在常规候选全部失败
            # 时校正最亮低纹理区域，避免改变正常照片的既有检测路径。
            corrected = _neutralize_bright_color_cast(image)
            evidence_features = _paper_evidence_features(corrected)
            for saturation, penalty in ((30, 0.20), (48, 0.25),
                                        (80, 0.35)):
                add_candidate(
                    detect_partial_paper_corners(corrected, saturation),
                    np.ones(2, np.float32), penalty, corrected)
            if not candidates:
                add_candidate(
                    _find_paper_by_border_lines(corrected, False),
                    np.ones(2, np.float32), 0.35, corrected)
                add_candidate(
                    _recover_paper_from_top_and_sides(corrected),
                    np.ones(2, np.float32), 0.40, corrected)
        if edge_recovery and candidates:
            exact_cache = {id(image): feature_results[view_index]}
            rescored = []
            for item in sorted(candidates, key=lambda value: value[2])[:72]:
                _, _, _, penalty, score_source, raw_quad, evidence_floor = item
                key = id(score_source)
                if key not in exact_cache:
                    exact_cache[key] = _paper_evidence_features(score_source)
                base = cached_evidence(
                    score_source, raw_quad, exact_cache[key])
                evidence = _paper_quad_recovery_evidence(
                    score_source, raw_quad, features=exact_cache[key],
                    base=base)
                if evidence_floor is not None:
                    evidence = max(evidence, evidence_floor)
                if evidence >= 0.5:
                    rescored.append((item[0], False, penalty - evidence))
            candidates = rescored
        if not candidates:
            return None
        # 三视图笛卡尔积会直接影响接口耗时；保留代价最低的少量候选。
        candidates.sort(key=lambda item: item[2])
        candidate_sets.append(candidates[:16 if edge_recovery else
                                         10 if physical_recovery else 3])

    best = None
    trials = (_prescreen_physical_trials(candidate_sets, image_size)
              if physical_recovery else itertools.product(*candidate_sets))
    for trial in trials:
        corners = [item[0] for item in trial]
        try:
            calibrated = _calibrate_multiview(
                corners, image_size, allow_per_view=False)
        except cv2.error:
            continue
        error = float(calibrated[0])
        # 纸色覆盖只能用于同等几何候选间排序。若权重过高，三张各自
        # 看似覆盖白纸、却无法由同一手机成像的错误框会压过真实纸框。
        if physical_recovery:
            geometry_penalty = (0.0 if error <=
                                ACCEPT_MULTIVIEW_REPROJECTION_PX else 50.0)
            centers = calibrated[5]
            side_tilts = [math.degrees(math.atan2(
                abs(float(center[0]) - A4_WIDTH_MM / 2),
                max(abs(float(center[2])), 1.0)))
                for center in centers[1:]]
            # 多种被遮挡角外推都可能得到很低重投影误差；优先选择具有
            # 足够侧视基线的物理解，避免三视图退化成两个近正面视角。
            view_penalty = max(0.0, 15.0 - min(side_tilts)) * 4.0
            score = (geometry_penalty + view_penalty + error * 4.0 +
                     sum(item[2] for item in trial))
        else:
            # 颜色证据只用于几何近似相同的候选间排序。色偏可能让错误白框
            # 获得更高覆盖分，必须让共同相机重投影误差占主导。
            score = error * 8.0 + sum(item[2] for item in trial)
        if best is None or score < best[0]:
            best = (score, error, corners, calibrated)
    if best is None or not np.isfinite(best[1]):
        return None
    return best[2], best[3], best[1]


def _select_adaptive_multiview_paper_corners(
        originals, normalized, image_size, fixed_corners=None,
        strict_corners=None, recover_three_edges=None,
        learned_corners=None):
    """传统与学习候选独立成解，再按同一物理/图像证据仲裁。"""
    prepared = ({} if learned_corners is None else
                {'learned': learned_corners})
    baseline = _adaptive_multiview_paper_corners(
        originals, normalized, image_size, physical_recovery=False,
        fixed_corners=fixed_corners, prepared=prepared,
        strict_corners=strict_corners,
        recover_three_edges=recover_three_edges)
    baseline_accepted = (baseline is not None and
                          np.isfinite(baseline[2]) and
                          baseline[2] <= MAX_MULTIVIEW_REPROJECTION_PX)
    physical = None
    if not baseline_accepted:
        physical = _adaptive_multiview_paper_corners(
            originals, normalized, image_size, physical_recovery=True,
            fixed_corners=fixed_corners, prepared=prepared,
            strict_corners=strict_corners,
            recover_three_edges=recover_three_edges)
    classic = (baseline if baseline_accepted else
               physical if baseline is None else baseline
               if physical is None else
               physical if physical[2] < baseline[2] else baseline)
    if (classic is None or not np.isfinite(classic[2]) or
            classic[2] > MAX_MULTIVIEW_REPROJECTION_PX):
        edge_recovered = _adaptive_multiview_paper_corners(
            originals, normalized, image_size, physical_recovery=True,
            fixed_corners=fixed_corners, prepared=prepared,
            strict_corners=strict_corners, edge_recovery=True,
            recover_three_edges=recover_three_edges)
        if (edge_recovered is not None and
                (classic is None or edge_recovered[2] < classic[2])):
            classic = edge_recovered

    feature_sets = prepared['features']

    def solution_score(solution):
        evidence = []
        rank = []
        for image, quad, features in zip(
                normalized, solution[0], feature_sets):
            base = _paper_quad_evidence(image, quad, features=features)
            recovered = _paper_quad_recovery_evidence(
                image, quad, features=features, base=base)
            evidence.append(recovered)
            rank.append(_paper_candidate_rank_score(
                image, quad, evidence_features=features,
                evidence=base, recovery=recovered))
        valid_evidence = [max(value, -2.0) for value in evidence]
        return (float(solution[2]) * 5.0 -
                float(np.mean(valid_evidence)) * 1.5 -
                float(np.mean(rank)) * 2.0)

    # 有传统物理解时，以双方一致的三个可见角为锚，只让模型替换一个
    # 遮挡角。两套方法分歧超过一个角时没有足够证据，不做盲目覆盖。
    solutions = [] if classic is None else [classic]
    learned_raw = prepared.get('learned', [None] * len(normalized))
    if classic is not None:
        classic_centers = classic[1][5]
        classic_top_x = classic_centers[0][0]
        classic_min_side = min(math.degrees(math.atan2(
            abs(float(center[0]) - classic_top_x),
            max(abs(float(center[2])), 1.0)))
            for center in classic_centers[1:])
        per_view = []
        for image, classic_quad, learned_quad in zip(
                normalized, classic[0], learned_raw):
            choices = [np.asarray(classic_quad, np.float32)]
            if learned_quad is not None:
                learned_quad = np.asarray(learned_quad, np.float32)
                distances = np.linalg.norm(
                    learned_quad - choices[0], axis=1)
                limit = max(10.0, min(image.shape[:2]) * 0.045)
                if int(np.sum(distances <= limit)) == 3:
                    fused = choices[0].copy()
                    farthest = int(np.argmax(distances))
                    fused[farthest] = learned_quad[farthest]
                    choices.append(fused)
            per_view.append(choices)
        for trial in itertools.product(*per_view):
            if all(np.array_equal(item, base)
                   for item, base in zip(trial, classic[0])):
                continue
            try:
                calibrated = _calibrate_multiview(
                    list(trial), image_size, allow_per_view=False)
            except cv2.error:
                continue
            centers = calibrated[5]
            top_x = centers[0][0]
            min_side = min(math.degrees(math.atan2(
                abs(float(center[0]) - top_x),
                max(abs(float(center[2])), 1.0)))
                for center in centers[1:])
            preserves_side_baseline = (
                min_side >= classic_min_side - 2.0 or
                min_side >= MIN_MULTIVIEW_SIDE_ANGLE_DEG)
            if np.isfinite(calibrated[0]) and preserves_side_baseline:
                solutions.append((list(trial), calibrated,
                                  float(calibrated[0])))
    else:
        learned = _adaptive_multiview_paper_corners(
            originals, normalized, image_size, physical_recovery=True,
            fixed_corners=fixed_corners, prepared=prepared,
            strict_corners=strict_corners, edge_recovery=True,
            learned_recovery=True,
            recover_three_edges=recover_three_edges)
        if learned is not None and np.isfinite(learned[2]):
            solutions.append(learned)
    if not solutions:
        return None
    return min(solutions, key=solution_score)


def diagnose_multiview_images(images):
    """保存纸角与脚蒙版叠加图；仅供开发诊断，不参与测量。"""
    if len(images) != 3 or any(image is None for image in images):
        return None, '需要三张可读取照片'
    originals = images
    normalized = _normalize_multiview_images(images)
    if normalized is None:
        return None, '三张照片画面比例不同'
    image_size = (normalized[0].shape[1], normalized[0].shape[0])
    strict = [detect_partial_paper_corners(image, 30)
              for image in normalized]
    adaptive = _select_adaptive_multiview_paper_corners(
        originals, normalized, image_size, strict_corners=strict)
    corners = None
    calibration = None
    calibration_error = None
    if all(item is not None for item in strict):
        try:
            calibration = _calibrate_multiview(
                strict, image_size, allow_per_view=False)
            corners = strict
            calibration_error = float(calibration[0])
            subpixel = _select_subpixel_multiview_corners(
                normalized, strict, image_size)
            if subpixel is not None:
                corners, calibration = subpixel
                calibration_error = float(calibration[0])
        except cv2.error:
            pass
    if (adaptive is not None and
            (calibration_error is None or adaptive[2] < calibration_error)):
        corners, calibration, calibration_error = adaptive
        if calibration_error > ACCEPT_MULTIVIEW_REPROJECTION_PX:
            subpixel = _select_subpixel_multiview_corners(
                normalized, corners, image_size)
            if subpixel is not None:
                corners, calibration = subpixel
                calibration_error = float(calibration[0])

    panels = []
    mask_ok = []
    for index, image in enumerate(normalized):
        preview_width = 400
        preview_height = int(round(image.shape[0] *
                                   preview_width / image.shape[1]))
        preview = cv2.resize(
            image, (preview_width, preview_height),
            interpolation=cv2.INTER_AREA)
        scale = np.float32([
            preview_width / image.shape[1],
            preview_height / image.shape[0]])
        for saturation, color in ((40, (255, 180, 0)),
                                  (48, (255, 80, 180)),
                                  (60, (0, 180, 255)),
                                  (80, (180, 180, 180))):
            candidate = detect_partial_paper_corners(image, saturation)
            if candidate is None:
                continue
            points = np.rint(np.asarray(candidate) * scale).astype(np.int32)
            cv2.polylines(preview, [points], True, color, 1, cv2.LINE_AA)
        foot_mask = None
        if corners is not None:
            points = np.rint(
                np.asarray(corners[index]) * scale).astype(np.int32)
            cv2.polylines(preview, [points], True, (0, 255, 0), 3,
                          cv2.LINE_AA)
            foot_mask = _multiview_foot_mask(
                image, corners[index])
        mask_ok.append(foot_mask is not None)
        if foot_mask is not None:
            small_mask = cv2.resize(
                foot_mask, (preview_width, preview_height),
                interpolation=cv2.INTER_NEAREST) > 0
            overlay = preview.copy()
            overlay[small_mask] = (20, 20, 235)
            preview = cv2.addWeighted(preview, 0.72, overlay, 0.28, 0)
        cv2.putText(preview, f'view {index + 1}  foot-mask: '
                    f'{"OK" if foot_mask is not None else "FAIL"}',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0) if foot_mask is not None else (0, 0, 255),
                    2, cv2.LINE_AA)
        panels.append(preview)

    canvas = cv2.hconcat(panels)
    footer = np.full((92, canvas.shape[1], 3), 25, np.uint8)
    error_text = ('unavailable' if calibration_error is None else
                  f'{calibration_error:.2f}px')
    accepted = (calibration_error is not None and
                calibration_error <= ACCEPT_MULTIVIEW_REPROJECTION_PX)
    cv2.putText(footer, 'GREEN: selected A4   THIN: candidates   '
                'RED: foot mask',
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(footer, f'A4 reprojection: {error_text}  '
                f'accepted: {"YES" if accepted else "NO"}',
                (12, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 220, 0) if accepted else (0, 80, 255), 2,
                cv2.LINE_AA)
    canvas = cv2.vconcat([canvas, footer])
    name = f'diagnostic_multi_{uuid.uuid4().hex}.jpg'
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], name), canvas,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {
        'diagnostic_image': f'/uploads/{name}',
        'diagnostic_path': os.path.join(app.config['UPLOAD_FOLDER'], name),
        'calibration_error_px': (None if calibration_error is None else
                                 round(calibration_error, 2)),
        'paper_geometry_accepted': accepted,
        'foot_masks': mask_ok,
    }, None


def process_multiview_images(images, image_metadata=None):
    if any(img is None for img in images):
        return None, '无法读取三张照片'
    if any(min(img.shape[:2]) < 480 for img in images):
        return None, '图片分辨率过低，请使用手机原相机拍摄'
    original_images = images
    images = _normalize_multiview_images(images)
    if images is None:
        return None, '三张照片画面比例不同，请使用同一手机、同一拍摄模式'
    sharpness = [cv2.Laplacian(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        for image in images]
    if min(sharpness) < 3.0:
        return None, '照片严重模糊，纸边和脚趾轮廓无法可靠定位，请重拍'
    brightness_channels = [cv2.cvtColor(
        image, cv2.COLOR_BGR2HSV)[:, :, 2] for image in images]
    brightness_95 = [float(np.percentile(channel, 95))
                     for channel in brightness_channels]
    if max(brightness_95) < 180.0:
        return None, '三张照片整体曝光过暗，请增加普通白光后重拍'
    cast_strength = [_color_cast_strength(image) for image in images]
    color_cast_risk = max(cast_strength) > 10.5
    # 轻微暖光先降置信度，不因单张刚越过阈值直接拒绝。只有三张都
    # 严重偏色时，脚/纸颜色关系才失去可恢复性。
    if _severe_multiview_color_cast(cast_strength):
        return None, '照片存在明显彩色光或白平衡偏色，请改用普通白光重拍'
    low_light_risk = min(brightness_95) < 200.0
    highlight_clip = [float(np.mean(channel >= 254))
                      for channel in brightness_channels]
    overexposure_risk = bool(
        max(brightness_95) >= 250.0 and max(highlight_clip) >= 0.03)
    measurement_images = images
    if overexposure_risk:
        measurement_images = [np.clip(
            image.astype(np.float32) * 0.90, 0, 255).astype(np.uint8)
            for image in images]
    compression_risk = bool(image_metadata and any(
        item and item.get('jpeg_luma_quantizer') is not None and
        item['jpeg_luma_quantizer'] > 8 for item in image_metadata))

    image_size = (images[0].shape[1], images[0].shape[0])
    focal_hint_px = _multiview_focal_hint(image_metadata, image_size)
    three_edge_cache = {}

    def recover_three_edges(image, four_edges=None, border_checked=False):
        key = id(image)
        if key not in three_edge_cache:
            three_edge_cache[key] = _recover_paper_from_three_edges(
                image, four_edges, border_checked)
        return three_edge_cache[key]

    learned_pose_corners = [_learned_paper_candidate(image) for image in images]
    fast_single_side = None
    learned_geometry_consistent = False
    calibrated = None
    if all(item is not None for item in learned_pose_corners):
        try:
            learned_calibration = _calibrate_multiview(
                learned_pose_corners, image_size,
                focal_hint_px=focal_hint_px)
            learned_centers = learned_calibration[5]
            learned_offsets = [center[0] - learned_centers[0][0]
                               for center in learned_centers[1:]]
            learned_angles = [math.degrees(math.atan2(
                abs(float(offset)), max(abs(float(center[2])), 1.0)))
                for offset, center in zip(
                    learned_offsets, learned_centers[1:])]
            ambiguous = (learned_offsets[0] * learned_offsets[1] >= 0 or
                         min(learned_angles) < MIN_MULTIVIEW_SIDE_ANGLE_DEG)
            learned_geometry_consistent = bool(
                np.isfinite(learned_calibration[0]) and
                learned_calibration[0] <= MAX_MULTIVIEW_REPROJECTION_PX)
            if learned_geometry_consistent and ambiguous:
                fast_single_side = _single_side_hull_consensus(
                    images, learned_pose_corners, image_size, True,
                    focal_hint_px=focal_hint_px, paper_images=images,
                    learned_corners=learned_pose_corners)
                if fast_single_side is not None:
                    calibrated = learned_calibration
        except cv2.error:
            pass
    if fast_single_side is not None:
        corners = learned_pose_corners
    else:
        detector = (_detect_reliable_paper_corners
                    if learned_geometry_consistent else
                    lambda image: detect_partial_paper_corners(image, 30))
        with ThreadPoolExecutor(max_workers=3) as executor:
            if learned_geometry_consistent:
                corners = list(executor.map(
                    lambda image: detector(image, recover_three_edges),
                    images))
            else:
                corners = list(executor.map(detector, images))
    adaptive_paper_used = fast_single_side is not None
    if calibrated is None and all(item is not None for item in corners):
        subpixel = _select_subpixel_multiview_corners(
            images, corners, image_size)
        if subpixel is not None:
            corners, calibrated = subpixel
    if any(item is None for item in corners):
        fixed_corners = [
            _lock_complete_paper_recovery(image, corner)
            for image, corner in zip(images, corners)]
        adaptive = _select_adaptive_multiview_paper_corners(
            original_images, images, image_size,
            fixed_corners=fixed_corners,
            strict_corners=corners,
            recover_three_edges=recover_three_edges,
            learned_corners=learned_pose_corners)
        if adaptive is None:
            return None, '有照片无法稳定恢复A4纸边界，请避免强色光并露出至少三条纸边'
        corners, calibrated, _ = adaptive
        if adaptive[2] > MAX_MULTIVIEW_REPROJECTION_PX:
            try:
                calibrated = _calibrate_multiview(
                    corners, image_size, allow_per_view=True)
            except cv2.error:
                calibrated = None
        if (calibrated is None or
                calibrated[0] > MAX_MULTIVIEW_REPROJECTION_PX):
            adaptive = _select_adaptive_multiview_paper_corners(
                original_images, images, image_size,
                strict_corners=corners,
                recover_three_edges=recover_three_edges,
                learned_corners=learned_pose_corners)
            if (adaptive is None or
                    adaptive[2] > MAX_MULTIVIEW_REPROJECTION_PX):
                return None, '有照片无法稳定恢复A4纸边界，请避免强色光并露出至少三条纸边'
            corners, calibrated, _ = adaptive
        adaptive_paper_used = True
        if calibrated[0] > ACCEPT_MULTIVIEW_REPROJECTION_PX:
            subpixel = _select_subpixel_multiview_corners(
                images, corners, image_size)
            if subpixel is not None:
                corners, calibrated = subpixel
    recovered_views = (False, False, False)
    try:
        if calibrated is None:
            calibrated = _calibrate_multiview(corners, image_size)
        if (calibrated[0] > ACCEPT_MULTIVIEW_REPROJECTION_PX and
                not adaptive_paper_used):
            adaptive = _select_adaptive_multiview_paper_corners(
                original_images, images, image_size,
                recover_three_edges=recover_three_edges,
                learned_corners=learned_pose_corners)
            if adaptive is not None and adaptive[2] < calibrated[0] - 0.2:
                corners, calibrated, _ = adaptive
                adaptive_paper_used = True
                if calibrated[0] > ACCEPT_MULTIVIEW_REPROJECTION_PX:
                    subpixel = _select_subpixel_multiview_corners(
                        images, corners, image_size)
                    if subpixel is not None:
                        corners, calibrated = subpixel
        if (not np.isfinite(calibrated[0]) or
                calibrated[0] > MAX_MULTIVIEW_REPROJECTION_PX):
            recovered = [recover_three_edges(img) for img in images]
            choices = [[(corner, False)] + ([(alternative, True)]
                                   if alternative is not None else [])
                       for corner, alternative in zip(corners, recovered)]
            best = None
            for trial in itertools.product(*choices):
                trial_corners = [item[0] for item in trial]
                trial_calibration = _calibrate_multiview(
                    trial_corners, image_size, allow_per_view=False)
                # 优先直接检测到的纸角；恢复角必须显著改善共同标定才采用。
                # 防止轻微模糊让三张照片一起落入几何上自洽、实际却错误的解。
                trial_score = trial_calibration[0] + 0.5 * sum(
                    item[1] for item in trial)
                if best is None or trial_score < best[0]:
                    best = (trial_score, trial_corners,
                            tuple(item[1] for item in trial))
            if best is not None:
                corners = best[1]
                recovered_views = best[2]
                calibrated = _calibrate_multiview(corners, image_size)
    except cv2.error:
        return None, '三张照片无法建立共同相机标定'
    focal_hint_used = False
    if focal_hint_px is not None:
        try:
            hinted = _calibrate_multiview(
                corners, image_size, focal_hint_px=focal_hint_px)
            if (np.isfinite(hinted[0]) and
                    hinted[0] <= MAX_MULTIVIEW_REPROJECTION_PX and
                    hinted[0] <= calibrated[0] + 1.5):
                calibrated = hinted
                focal_hint_used = True
        except cv2.error:
            pass
    error, cameras, distortions, rotations, translations, centers = calibrated
    if (not np.isfinite(error) or
            error > MAX_MULTIVIEW_REPROJECTION_PX):
        return None, '系统无法稳定确认A4纸边界，请让纸张至少三条外边清晰可见后重拍'
    # 自适应候选中包含纸边外推；即使只有侧视图触发，
    # 测量也应走恢复纸边的保守分支，不能假定顶视几何完全可靠。
    top_paper_recovered = bool(recovered_views[0] or adaptive_paper_used)

    mirrored_geometry = False
    single_side_consensus_used = False
    top_fore_angle = math.degrees(math.atan2(
        abs(float(centers[0][1]) - A4_HEIGHT_MM / 2),
        max(abs(float(centers[0][2])), 1.0)))
    top_x = centers[0][0]
    side_offsets = [centers[1][0] - top_x, centers[2][0] - top_x]
    side_angles = [math.degrees(math.atan2(
        abs(offset), max(abs(center[2]), 1)))
        for offset, center in zip(side_offsets, centers[1:])]
    needs_mirror_resolution = side_offsets[0] * side_offsets[1] >= 0
    if needs_mirror_resolution or fast_single_side is not None:
        mirrored = (None if fast_single_side is not None else
                    _mirrored_multiview_corners(
                        measurement_images, corners, image_size,
                        top_paper_recovered, focal_hint_px=focal_hint_px))
        if mirrored is None:
            single_side = fast_single_side
            if (single_side is None and
                    all(item is not None for item in learned_pose_corners)):
                single_side = _single_side_hull_consensus(
                    images, learned_pose_corners, image_size,
                    True, focal_hint_px=focal_hint_px,
                    paper_images=images,
                    learned_corners=learned_pose_corners)
            if single_side is None:
                single_side = _single_side_hull_consensus(
                    images, corners, image_size,
                    top_paper_recovered, focal_hint_px=focal_hint_px,
                    paper_images=images,
                    learned_corners=learned_pose_corners)
            if single_side is not None:
                (corners, mirrored_hull, side_angles,
                 error, top_fore_angle) = single_side
                mirrored_geometry = True
                single_side_consensus_used = True
            elif side_offsets[0] * side_offsets[1] >= 0:
                return None, '系统无法可靠确认左右斜拍视角，请检查两张照片是否分别来自脚的两侧'
            else:
                return None, '斜拍角度不足，请让手机向脚的侧面再移动一些'
        else:
            corners, calibrated, mirrored_hull = mirrored
            error, cameras, distortions, rotations, translations, centers = \
                calibrated
            focal_hint_used = focal_hint_px is not None
            mirrored_geometry = True
            top_x = centers[0][0]
            side_offsets = [centers[1][0] - top_x,
                            centers[2][0] - top_x]
            side_angles = [math.degrees(math.atan2(
                abs(offset), max(abs(center[2]), 1)))
                for offset, center in zip(side_offsets, centers[1:])]
            top_fore_angle = math.degrees(math.atan2(
                abs(float(centers[0][1]) - A4_HEIGHT_MM / 2),
                max(abs(float(centers[0][2])), 1.0)))
    if max(side_angles) < MIN_MULTIVIEW_SIDE_ANGLE_DEG:
        return None, '两张斜拍角度都过小，缺少脚跟三维信息，请增大任意一侧角度'

    if mirrored_geometry:
        masks, footprint, groundprint, lowprint, height_map = mirrored_hull
    else:
        masks = [_multiview_foot_mask(img, item)
                 for img, item in zip(measurement_images, corners)]
        if any(mask is None for mask in masks):
            return None, '有照片无法分离脚与纸张，请避免肤色区域被严重遮挡'
    initial_masks = masks
    top_transform = cv2.getPerspectiveTransform(
        np.asarray(corners[0], np.float32),
        np.float32([[0, 0], [A4_WIDTH_MM, 0],
                    [A4_WIDTH_MM, A4_HEIGHT_MM], [0, A4_HEIGHT_MM]]))
    top_planar_mask = _warp_planar_foot_mask(masks[0], top_transform)
    if not mirrored_geometry:
        footprint, groundprint, lowprint, height_map = _visual_hull_footprint(
            masks, cameras, distortions, rotations, translations)
    measurements = _measure_footprint(
        footprint, groundprint, height_map, top_planar_mask,
        cameras[1] is not cameras[0], top_paper_recovered,
        top_fore_angle,
        mirrored_geometry)
    consensus_used = False
    measurement_disagreement = False
    if (not mirrored_geometry and
            _needs_shadow_consensus(footprint, height_map)):
        refined_masks = [_multiview_foot_mask(img, item, consensus=True)
                         for img, item in zip(measurement_images, corners)]
        if all(mask is not None for mask in refined_masks):
            refined_top = _warp_planar_foot_mask(
                refined_masks[0], top_transform)
            refined_footprint, refined_groundprint, refined_lowprint, refined_height_map = \
                _visual_hull_footprint(
                    refined_masks, cameras, distortions,
                    rotations, translations)
            refined_measurements = _measure_footprint(
                refined_footprint, refined_groundprint, refined_height_map,
                refined_top, cameras[1] is not cameras[0],
                top_paper_recovered,
                top_fore_angle,
                mirrored_geometry)
            if refined_measurements is not None:
                initial_plausible = (
                    measurements is not None and
                    180 <= measurements[0] <= 350 and
                    60 <= measurements[1] <= 130 and
                    30 <= measurements[2] <= 100)
                differences = (
                    np.abs(np.asarray(measurements[:3]) -
                           np.asarray(refined_measurements[:3]))
                    if initial_plausible else None)
                measurement_disagreement = bool(
                    initial_plausible and
                    (max(differences[:2]) > 6.0 or differences[2] > 8.0))
                masks = refined_masks
                top_planar_mask = refined_top
                footprint, groundprint, lowprint, height_map = (
                    refined_footprint, refined_groundprint,
                    refined_lowprint, refined_height_map)
                measurements = refined_measurements
                consensus_used = True
    agreement_masks = masks if consensus_used else [
        _multiview_foot_mask(img, item, consensus=True)
        for img, item in zip(measurement_images, corners)]
    segmentation_check_available = not any(
        mask is None for mask in agreement_masks)
    if not segmentation_check_available:
        agreement_masks = initial_masks
    segmentation_agreements = []
    for initial, agreed, item in zip(
            initial_masks, agreement_masks, corners):
        paper = np.zeros(initial.shape, np.uint8)
        cv2.fillConvexPoly(paper, np.asarray(item, np.int32), 255)
        first = (initial > 0) & (paper > 0)
        second = (agreed > 0) & (paper > 0)
        union = np.logical_or(first, second).sum()
        segmentation_agreements.append(float(
            np.logical_and(first, second).sum() / max(union, 1)))
    segmentation_agreement = min(segmentation_agreements)
    if (segmentation_check_available and
            segmentation_agreement < WARN_SEGMENTATION_AGREEMENT):
        # 初始肤色阈值可能只取到脚的一部分；共识蒙版完整时，应真正用它
        # 重算三视图，而不是仅因两种方法不同就拒测。
        if (consensus_used and segmentation_agreement <
                HARD_MIN_SEGMENTATION_AGREEMENT):
            return None, '脚部轮廓受阴影或色偏影响较大，请改善光线后重拍'
        if consensus_used:
            consensus_measurements = None
        else:
            consensus_top = _warp_planar_foot_mask(
                agreement_masks[0], top_transform)
            consensus_footprint, consensus_groundprint, consensus_lowprint, consensus_height_map = \
                _visual_hull_footprint(
                    agreement_masks, cameras, distortions,
                    rotations, translations)
            consensus_measurements = _measure_footprint(
                consensus_footprint, consensus_groundprint,
                consensus_height_map, consensus_top,
                cameras[1] is not cameras[0], top_paper_recovered,
                top_fore_angle,
                mirrored_geometry)
        if (consensus_measurements is None or
                not 180 <= consensus_measurements[0] <= 350 or
                not 60 <= consensus_measurements[1] <= 130 or
                not 30 <= consensus_measurements[2] <= 100):
            if (segmentation_agreement <
                    HARD_MIN_SEGMENTATION_AGREEMENT):
                return None, '脚部轮廓受阴影或色偏影响严重，请调整光线后重拍'
        else:
            masks = agreement_masks
            top_planar_mask = consensus_top
            footprint, groundprint, lowprint, height_map = (
                consensus_footprint, consensus_groundprint,
                consensus_lowprint, consensus_height_map)
            measurements = consensus_measurements
            consensus_used = True
    measurements, degraded_geometry_used = _degraded_hull_fallback(
        measurements, footprint, groundprint, height_map,
        top_planar_mask, side_angles)
    soft_projection_used = False
    if measurements is None and not mirrored_geometry:
        soft_footprint, soft_groundprint, soft_lowprint, soft_height_map = \
            _visual_hull_footprint(
                masks, cameras, distortions, rotations, translations,
                minimum_side_views=1)
        soft_measurements = _measure_footprint(
            soft_footprint, soft_groundprint, soft_height_map,
            top_planar_mask, cameras[1] is not cameras[0],
            top_paper_recovered,
            top_fore_angle,
            mirrored_geometry=False)
        soft_measurements, soft_degraded = _degraded_hull_fallback(
            soft_measurements, soft_footprint, soft_groundprint,
            soft_height_map, top_planar_mask, side_angles)
        if (soft_measurements is not None and
                180 <= soft_measurements[0] <= 350 and
                60 <= soft_measurements[1] <= 130 and
                30 <= soft_measurements[2] <= 100):
            footprint, groundprint, lowprint, height_map = (
                soft_footprint, soft_groundprint,
                soft_lowprint, soft_height_map)
            measurements = soft_measurements
            degraded_geometry_used = bool(soft_degraded)
            soft_projection_used = True
    if measurements is None:
        return None, '三张轮廓无法形成稳定足部投影，请保持脚和纸张完全不动'
    pre_selection_measurements = measurements
    measurements = _select_geometry_measurements(
        measurements, footprint, groundprint, height_map,
        top_planar_mask, side_angles, top_fore_angle,
        segmentation_agreement)
    if single_side_consensus_used:
        # 单侧共识特征不在融合模型训练分布内。长、前掌宽只取顶视平面，
        # 脚跟保留两侧重建通过一致性检查后的物理截面。
        profiles = _measurement_debug_summary(
            footprint, groundprint, height_map, top_planar_mask, lowprint)
        top_profile = profiles.get('top') or {}
        if (top_profile.get('length') is not None and
                top_profile.get('ball') is not None):
            measurements = (
                float(top_profile['length']), float(top_profile['ball']),
                float(pre_selection_measurements[2]),
                float(pre_selection_measurements[3]))
    elif soft_projection_used:
        # 统一融合模型只见过三视图硬交集特征。软恢复特征超出训练分布：
        # 保留模型对脚长的全局校正，宽度改用有直接物理含义的截面。
        soft_profiles = _measurement_debug_summary(
            footprint, groundprint, height_map, top_planar_mask, lowprint)
        height_10 = soft_profiles.get('height_10') or {}
        top_profile = soft_profiles.get('top') or {}
        soft_ball = height_10.get('ball') or top_profile.get('ball')
        if soft_ball is None:
            soft_ball = pre_selection_measurements[1]
        measurements = (
            measurements[0], float(soft_ball),
            float(pre_selection_measurements[2]),
            float(pre_selection_measurements[3]))
    length, ball, heel, heel_stability = measurements
    heel_low_confidence = heel < ball * 0.50
    heel_occlusion_recovered = False
    if heel > ball * 0.90:
        clipped_masks = [_multiview_foot_mask(
            img, item, paper_margin_ratio=0.05)
            for img, item in zip(measurement_images, corners)]
        if all(mask is not None for mask in clipped_masks):
            clipped_footprint, clipped_groundprint, clipped_lowprint, clipped_height_map = \
                _visual_hull_footprint(
                    clipped_masks, cameras, distortions,
                    rotations, translations)
            clipped_profiles = _measurement_debug_summary(
                clipped_footprint, clipped_groundprint,
                clipped_height_map, top_planar_mask, clipped_lowprint)
            core = clipped_profiles.get('height_50') or {}
            core_heel = core.get('heel')
            if (core_heel is not None and
                    ball * 0.45 <= core_heel <= ball * 0.85):
                heel = float(core_heel)
                measurements = (length, ball, heel, heel_stability)
                heel_occlusion_recovered = True
    if heel_stability > HARD_MAX_HEEL_STABILITY_MM:
        return None, '脚跟15%位置的轮廓不稳定，请改善光线并重拍左右斜拍照片'
    if not (180 <= length <= 350 and 60 <= ball <= 130 and 30 <= heel <= 100):
        return None, '联合测量结果异常，请检查脚跟是否贴齐纸张底边'
    if heel > ball * 0.90:
        return None, '脚踝或小腿遮住脚跟轮廓，请增大左右斜拍角度后重拍'

    result_name = f"result_multi_{uuid.uuid4().hex}.png"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_name)
    outline_quality = _draw_multiview_result(
        footprint, groundprint, lowprint, height_map, top_planar_mask,
        measurements, result_path)
    measurement_debug = _measurement_debug_summary(
        footprint, groundprint, height_map, top_planar_mask, lowprint)
    measurement_debug['pre_selection'] = {
        'length': round(float(pre_selection_measurements[0]), 2),
        'ball': round(float(pre_selection_measurements[1]), 2),
        'heel': round(float(pre_selection_measurements[2]), 2),
    }
    measurement_debug['top_fore_angle_deg'] = round(top_fore_angle, 2)
    measurement_debug['projection_mode'] = (
        'independent_side_consensus' if single_side_consensus_used else
        'top_plus_one_side' if soft_projection_used else 'three_view')
    measurement_debug['camera_focal_px'] = round(
        float(cameras[0][0, 0]), 2)
    measurement_debug['camera_focal_source'] = (
        'exif' if focal_hint_used else 'paper_geometry')
    measurement_debug['outline'] = outline_quality
    warnings = []
    if adaptive_paper_used:
        warnings.append('已启用自适应A4纸边恢复')
    if error > ACCEPT_MULTIVIEW_REPROJECTION_PX:
        warnings.append('A4标定误差较高，本次按低置信度输出')
    elif error > WARN_MULTIVIEW_REPROJECTION_PX:
        warnings.append('A4标定接近理想上限，结果仍可使用')
    if consensus_used:
        warnings.append('检测到明显阴影，已启用多阈值轮廓共识')
    if degraded_geometry_used:
        warnings.append('侧视交集退化，脚长和脚掌宽已回退至顶视平面测量')
    if soft_projection_used:
        warnings.append('三视图硬交集退化，已使用顶视加单侧轮廓恢复，建议复核或重拍')
    if single_side_consensus_used:
        warnings.append('左右姿态存在镜像二义性，脚长和脚掌宽采用顶视平面，脚跟采用两份单侧重建共识')
    if min(side_angles) < 15:
        warnings.append('一侧斜拍角度偏小，脚跟宽精度可能下降')
    if min(sharpness) < 6:
        warnings.append('照片较模糊，尺寸置信度降低')
    elif min(sharpness) < 20:
        warnings.append('照片略模糊，已启用容错测量')
    if heel_low_confidence:
        warnings.append('脚跟三维轮廓偏窄，脚跟宽按低置信度输出')
    if heel_occlusion_recovered:
        warnings.append('检测到小腿遮挡，已用纸面邻域脚跟核心轮廓恢复')
    if heel_stability > 4.0:
        warnings.append('脚跟轮廓局部波动较大，脚跟宽置信度降低')
    if not segmentation_check_available:
        warnings.append('轮廓交叉校验不可用，本次沿用主分割结果')
    elif segmentation_agreement < WARN_SEGMENTATION_AGREEMENT:
        warnings.append('多种脚部分割结果存在差异，本次按低置信度输出')
    if measurement_disagreement:
        warnings.append('不同轮廓算法测量差异较大，本次采用阴影共识结果')
    if low_light_risk:
        warnings.append('部分照片曝光偏低，尺寸仅作参考，建议增加普通白光后重拍')
    if compression_risk:
        warnings.append('照片压缩较重，边缘精度不足，尺寸仅作参考，建议上传相册原图')
    if overexposure_risk:
        warnings.append('检测到高光过曝，已对脚部轮廓启用曝光归一化')
    if color_cast_risk:
        warnings.append('照片存在轻微色偏，已按中等置信度输出')
    if outline_quality['severe']:
        warnings.append('后跟截面与三维投影不一致，已拒绝显示夸张的小腿投影')
    elif outline_quality['rear_cut']:
        warnings.append('后跟末端受小腿遮挡，轮廓仅显示可靠部分')

    common_low = bool(
        error > ACCEPT_MULTIVIEW_REPROJECTION_PX or min(sharpness) < 6 or
        not segmentation_check_available or
        segmentation_agreement < HARD_MIN_SEGMENTATION_AGREEMENT or
        measurement_disagreement or low_light_risk or compression_risk or
        soft_projection_used or single_side_consensus_used or
        outline_quality['severe'])
    common_medium = bool(
        adaptive_paper_used or error > WARN_MULTIVIEW_REPROJECTION_PX or
        min(sharpness) < 20 or
        segmentation_agreement < WARN_SEGMENTATION_AGREEMENT or
        degraded_geometry_used or color_cast_risk)
    heel_low = bool(
        common_low or min(side_angles) < MIN_MULTIVIEW_SIDE_ANGLE_DEG or
        heel_low_confidence or heel_stability > 4.0 or
        outline_quality['rear_cut'])
    heel_medium = bool(
        common_medium or min(side_angles) < 15 or heel_stability > 2.0)
    quality_grade = ('low' if common_low or heel_low else
                     'medium' if common_medium or heel_medium else 'high')
    confidence = {
        'foot_length': ('low' if common_low else
                        'medium' if common_medium else 'high'),
        'ball_width': ('low' if common_low else
                       'medium' if common_medium else 'high'),
        'heel_width': ('low' if heel_low else
                       'medium' if heel_medium else 'high'),
    }
    return {
        'foot_length': round(length, 1),
        'ball_width': round(ball, 1),
        'heel_width': round(heel, 1),
        'result_image': f'/uploads/{result_name}',
        'warnings': warnings,
        'calibration_error_px': round(float(error), 2),
        'side_angles': [round(value, 1) for value in side_angles],
        'heel_profile_stability_mm': round(heel_stability, 2),
        'segmentation_agreement': round(segmentation_agreement, 3),
        'measurement_debug': measurement_debug,
        'quality_grade': quality_grade,
        'dimension_confidence': confidence,
        # 低置信度仍返回参考尺寸，但前端必须明确建议用户重拍。
        'retake_recommended': quality_grade == 'low',
    }, None


def _side_interpolated_reference(channel):
    """用纸张左右留白估计每个位置的局部纸色，抵消渐变阴影。"""
    height, width = channel.shape
    x1 = max(8, int(round(width * 0.035)))
    x2 = max(x1 + 8, int(round(width * 0.16)))
    left = np.median(channel[:, x1:x2], axis=1).astype(np.float32)
    right = np.median(channel[:, width - x2:width - x1],
                      axis=1).astype(np.float32)
    kernel = min(51, height if height % 2 else height - 1)
    if kernel >= 3:
        left = cv2.GaussianBlur(
            left.reshape(-1, 1), (1, kernel), 0).ravel()
        right = cv2.GaussianBlur(
            right.reshape(-1, 1), (1, kernel), 0).ravel()
    blend = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    return left[:, None] * (1.0 - blend) + right[:, None] * blend


def detect_foot_on_paper(img, corners_4):
    M, _, out_size = build_transform(corners_4)
    warped = cv2.warpPerspective(img, M, out_size)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(warped, cv2.COLOR_BGR2YCrCb)

    neutral = (hsv[:, :, 1] < 25) & (hsv[:, :, 2] > 100)
    if neutral.sum() >= 100:
        paper_a = np.median(lab[:, :, 1][neutral])
        paper_b = np.median(lab[:, :, 2][neutral])
        paper_cr = np.median(ycrcb[:, :, 1][neutral])
        paper_cb = np.median(ycrcb[:, :, 2][neutral])
        global_mask = (
            (lab[:, :, 1] > paper_a + 3) &
            (lab[:, :, 2] > paper_b + 5) &
            (ycrcb[:, :, 1] > paper_cr + 7) &
            (ycrcb[:, :, 2] < paper_cb) &
            (hsv[:, :, 1] > 15)
        )
    else:
        global_mask = np.zeros(hsv.shape[:2], dtype=bool)

    bgr = warped.astype(np.float32)
    total = np.maximum(bgr.sum(axis=2), 1.0)
    red = bgr[:, :, 2] / total
    blue = bgr[:, :, 0] / total
    ref_red = _side_interpolated_reference(red)
    ref_blue = _side_interpolated_reference(blue)
    ref_a = _side_interpolated_reference(lab[:, :, 1].astype(np.float32))
    ref_b = _side_interpolated_reference(lab[:, :, 2].astype(np.float32))
    ref_cr = _side_interpolated_reference(
        ycrcb[:, :, 1].astype(np.float32))
    ref_s = _side_interpolated_reference(hsv[:, :, 1].astype(np.float32))

    lab_skin = ((lab[:, :, 1] - ref_a > 2.0) &
                (lab[:, :, 2] - ref_b > 3.0))
    rgb_skin = ((red - ref_red > 0.010) &
                (ref_blue - blue > 0.007))
    cr_skin = ycrcb[:, :, 1] - ref_cr > 4.0
    saturation_skin = hsv[:, :, 1] - ref_s > 7.0
    votes = (lab_skin.astype(np.uint8) + rgb_skin.astype(np.uint8) +
             cr_skin.astype(np.uint8) + saturation_skin.astype(np.uint8))
    local_mask = votes >= 2

    # 正常光线优先旧分割，结果不变；失败时才启用局部阴影容错。
    paper_area = out_size[0] * out_size[1]
    foot_warped = None
    for raw_mask in (global_mask, local_mask):
        foot_mask = raw_mask.astype(np.uint8) * 255
        margin = 8
        foot_mask[:margin, :] = 0
        foot_mask[:, :margin] = 0
        foot_mask[:, -margin:] = 0
        kernel = np.ones((5, 5), np.uint8)
        foot_mask = cv2.morphologyEx(
            foot_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        foot_mask = cv2.morphologyEx(
            foot_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(
            foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, cw, ch = cv2.boundingRect(contour)
            if (0.08 * paper_area < area < 0.60 * paper_area and
                    ch > out_size[1] * 0.45 and
                    0.15 * out_size[0] < cw < 0.75 * out_size[0]):
                center_penalty = abs((x + cw / 2) - out_size[0] / 2)
                candidates.append((area - center_penalty * 100, contour))
        if candidates:
            foot_warped = max(candidates, key=lambda item: item[0])[1]
            break

    if foot_warped is None:
        return None
    # 脚在纸上应形成逐行连续的实体；补上阴影造成的竖向裂口，
    # 同时保留每一行真实的左右边界，不使用会扩大足弓的整体凸包。
    solid = np.zeros((out_size[1], out_size[0]), np.uint8)
    cv2.drawContours(solid, [foot_warped], -1, 255, -1)
    left = np.full(solid.shape[0], -1, np.int32)
    right = np.full(solid.shape[0], -1, np.int32)
    for row in range(solid.shape[0]):
        xs = np.flatnonzero(solid[row])
        if xs.size >= 2:
            left[row], right[row] = xs[0], xs[-1]
    valid_rows = np.flatnonzero(left >= 0)
    if valid_rows.size:
        start, end = valid_rows[0], valid_rows[-1]
        smooth_left = left.copy()
        smooth_right = right.copy()
        for row in range(start, end + 1):
            lo, hi = max(start, row - 30), min(end + 1, row + 31)
            window = left[lo:hi] >= 0
            if np.any(window):
                local_left = left[lo:hi][window]
                local_right = right[lo:hi][window]
                # 只填内凹缺口，不向内削掉真实轮廓。
                smooth_left[row] = min(
                    left[row], int(np.percentile(local_left, 25)))
                smooth_right[row] = max(
                    right[row], int(np.percentile(local_right, 75)))
        for row in range(start, end + 1):
            if smooth_left[row] >= 0:
                solid[row, smooth_left[row]:smooth_right[row] + 1] = 255
    repair = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, repair,
                             iterations=2)
    repaired, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if repaired:
        foot_warped = max(repaired, key=cv2.contourArea)

    inverse = np.linalg.inv(M)
    foot_orig = cv2.perspectiveTransform(
        foot_warped.astype(np.float32), inverse)
    return foot_orig.astype(np.int32)


def _detect_foot_by_brown(img, corners_4):
    b_ch, g_ch, r_ch = cv2.split(img)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    M, mpp, out = build_transform(corners_4)

    paper_mask = np.zeros((out[1], out[0]), np.uint8)
    pc = np.array(corners_4, dtype=np.float32).reshape(-1, 1, 2)
    warped_pc = cv2.perspectiveTransform(pc, M)
    cv2.fillConvexPoly(paper_mask, warped_pc.astype(np.int32), 255)

    r_w = cv2.warpPerspective(r_ch, M, out).astype(np.float32)
    g_w = cv2.warpPerspective(g_ch, M, out).astype(np.float32)
    b_w = cv2.warpPerspective(b_ch, M, out).astype(np.float32)
    s_w = cv2.warpPerspective(hsv_img[:, :, 1], M, out).astype(np.float32)

    paper_s_median = np.median(s_w[paper_mask > 0])

    best_foot = None
    best_score = -1

    for offset in [10, 15, 20, 25]:
        foot_mask = ((r_w > g_w + offset) & (r_w > b_w + offset) & (paper_mask > 0)).astype(np.uint8) * 255
        kernel = np.ones((7, 7), np.uint8)
        foot_mask = cv2.morphologyEx(foot_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        foot_mask = cv2.morphologyEx(foot_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 3000:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            ratio = max(cw, ch) / (min(cw, ch) + 1e-5)
            if 2.0 < ratio < 3.5 and cw < out[0] * 0.8:
                ratio_score = 1.0 - abs(ratio - 2.7) / 1.5
                size_score = min(area, 100000) / 100000
                score = ratio_score * size_score
                if score > best_score:
                    best_score = score
                    best_foot = c

    for s_offset in [5, 10, 15, 20]:
        foot_mask = ((s_w > paper_s_median + s_offset) & (paper_mask > 0)).astype(np.uint8) * 255
        kernel = np.ones((7, 7), np.uint8)
        foot_mask = cv2.morphologyEx(foot_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        foot_mask = cv2.morphologyEx(foot_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 3000:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            ratio = max(cw, ch) / (min(cw, ch) + 1e-5)
            if 2.0 < ratio < 3.5 and cw < out[0] * 0.8:
                ratio_score = 1.0 - abs(ratio - 2.7) / 1.5
                size_score = min(area, 100000) / 100000
                score = ratio_score * size_score
                if score > best_score:
                    best_score = score
                    best_foot = c

    if best_foot is not None and cv2.contourArea(best_foot) > 500:
        return best_foot
    return None


def filter_ankle(contour, mm_per_px):
    """切除脚踝：找到宽度突变点（脚跟起始），只保留脚掌部分"""
    pts = contour.reshape(-1, 2).astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    axis_dir = eigenvectors[:, np.argmax(eigenvalues)]
    projections = centered @ axis_dir

    heel_idx = np.argmin(projections)
    toe_idx = np.argmax(projections)
    heel = pts[heel_idx]
    toe = pts[toe_idx]
    toe_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-10)
    perp_dir = np.array([-toe_dir[1], toe_dir[0]])
    length = projections.max() - projections.min()

    num_steps = 40
    widths = []
    for i in range(num_steps):
        ratio = (i + 1) / (num_steps + 1)
        center = heel + ratio * length * toe_dir
        to_points = pts - center
        axial_proj = to_points @ toe_dir
        perp_proj = to_points @ perp_dir
        in_slice = np.abs(axial_proj) < length * 0.05
        if in_slice.sum() > 5:
            w = perp_proj[in_slice].max() - perp_proj[in_slice].min()
            widths.append((ratio, w * mm_per_px))
        else:
            widths.append((ratio, 0))

    if len(widths) < 8:
        return contour

    w_vals = [w for _, w in widths]
    max_w = max(w_vals)
    if max_w < 40:
        return contour

    # 从窄端扫描，找宽度连续3步都超过max_w*40%的起点
    threshold = max_w * 0.4
    heel_start_ratio = 0.25
    for i in range(len(widths) - 2):
        if widths[i][1] >= threshold and widths[i+1][1] >= threshold and widths[i+2][1] >= threshold:
            heel_start_ratio = widths[i][0]
            break

    min_proj = projections.min()
    max_proj = projections.max()
    cutoff = min_proj + heel_start_ratio * (max_proj - min_proj)

    mask = projections >= cutoff
    if mask.sum() < 10:
        return contour

    return pts[mask].astype(np.int32).reshape(-1, 1, 2)


def warp_contour(contour, M, out_size):
    pts = contour.reshape(-1, 1, 2).astype(np.float32)
    warped_pts = cv2.perspectiveTransform(pts, M)
    return warped_pts.astype(np.int32)


def measure_foot(contour, mm_per_px, heel_baseline_px=None):
    pts = contour.reshape(-1, 2).astype(np.float32)

    if heel_baseline_px is not None:
        return _measure_from_heel_baseline(pts, mm_per_px, heel_baseline_px)

    axis_dir = np.array([0.0, 1.0])

    projections = pts @ axis_dir
    heel_idx = np.argmin(projections)
    toe_idx = np.argmax(projections)
    heel = pts[heel_idx]
    toe = pts[toe_idx]
    length_px = projections.max() - projections.min()

    toe_dir = axis_dir
    perp_dir = np.array([-toe_dir[1], toe_dir[0]])

    num_steps = 50
    widths = []
    ratios = []
    for i in range(num_steps):
        ratio = (i + 1) / (num_steps + 1)
        center = heel + ratio * length_px * toe_dir
        to_points = pts - center
        axial_proj = to_points @ toe_dir
        perp_proj = to_points @ perp_dir
        in_slice = np.abs(axial_proj) < length_px * 0.06
        if in_slice.sum() > 5:
            w = perp_proj[in_slice].max() - perp_proj[in_slice].min()
            widths.append(w * mm_per_px)
            ratios.append(ratio)

    if not widths:
        return length_px * mm_per_px, 0, 0, axis_dir, heel, toe

    widths = np.array(widths)
    ratios = np.array(ratios)

    mid_mask = (ratios >= 0.25) & (ratios <= 0.70)
    if mid_mask.sum() > 0:
        mid_widths = widths.copy()
        mid_widths[~mid_mask] = 0
        ball_idx = np.argmax(mid_widths)
    else:
        ball_idx = np.argmax(widths)
    ball_w = widths[ball_idx]
    ball_ratio = ratios[ball_idx]

    # 较窄的一端通常是脚趾；脚跟应从另一端区域取值。
    toe_is_start = widths[0] < widths[-1]
    heel_ratio = 1.0 - HEEL_WIDTH_RATIO if toe_is_start else HEEL_WIDTH_RATIO
    heel_y = projections.min() + heel_ratio * length_px
    heel_w = _horizontal_contour_width(pts, heel_y) * mm_per_px
    if heel_w <= 0:
        heel_w = widths[np.argmin(np.abs(ratios - heel_ratio))]

    foot_len = length_px * mm_per_px
    return foot_len, ball_w, heel_w, axis_dir, heel, toe


def _horizontal_contour_width(pts, y):
    """计算轮廓与指定水平截面的精确交点宽度。"""
    following = np.roll(pts, -1, axis=0)
    y1, y2 = pts[:, 1], following[:, 1]
    crosses = ((y1 <= y) & (y2 > y)) | ((y2 <= y) & (y1 > y))
    if crosses.sum() < 2:
        return 0.0
    p1, p2 = pts[crosses], following[crosses]
    xs = p1[:, 0] + (y - p1[:, 1]) * (
        p2[:, 0] - p1[:, 0]) / (p2[:, 1] - p1[:, 1])
    return float(xs.max() - xs.min())


def heel_outline_is_visible(pts, baseline_y):
    """后跟末端应明显收窄；否则当前轮廓实际来自脚踝/小腿。"""
    toe_y = float(pts[:, 1].min())
    length_px = float(baseline_y - toe_y)
    if length_px <= 0:
        return False
    rear_width = _horizontal_contour_width(
        pts, baseline_y - HEEL_REAR_CHECK_RATIO * length_px)
    heel_width = _horizontal_contour_width(
        pts, baseline_y - HEEL_WIDTH_RATIO * length_px)
    return heel_width > 0 and rear_width <= heel_width * 0.75


def _measure_from_heel_baseline(pts, mm_per_px, baseline_y):
    """脚跟与 A4 底边齐平时，以纸张底边为脚长零点。"""
    toe_y = float(pts[:, 1].min())
    length_px = float(baseline_y - toe_y)
    if length_px <= 0:
        zero = np.array([0.0, 0.0], dtype=np.float32)
        return 0, 0, 0, np.array([0.0, -1.0]), zero, zero

    samples = []
    slice_half = max(3.0, length_px * 0.012)
    for ratio in np.linspace(0.05, 0.95, 91):
        y = baseline_y - ratio * length_px
        width = _horizontal_contour_width(pts, y)
        if width > 0:
            samples.append((ratio, width))

    if not samples:
        zero = np.array([0.0, baseline_y], dtype=np.float32)
        return length_px * mm_per_px, 0, 0, np.array([0.0, -1.0]), zero, zero

    ratios = np.array([item[0] for item in samples])
    widths = np.array([item[1] for item in samples]) * mm_per_px

    ball_zone = widths[(ratios >= 0.55) & (ratios <= 0.80)]
    if len(ball_zone) >= 5:
        smooth = cv2.medianBlur(
            ball_zone.astype(np.float32).reshape(-1, 1), 5).ravel()
        ball_w = float(smooth.max())
    else:
        ball_w = float(ball_zone.max()) if len(ball_zone) else 0.0
    heel_y = baseline_y - HEEL_WIDTH_RATIO * length_px
    heel_w = _horizontal_contour_width(pts, heel_y) * mm_per_px

    toe_candidates = pts[pts[:, 1] <= toe_y + slice_half]
    toe_x = float(np.mean(toe_candidates[:, 0]))
    heel_band = pts[pts[:, 1] >= baseline_y - length_px * 0.05]
    heel_x = float(np.mean(heel_band[:, 0])) if len(heel_band) else toe_x
    heel = np.array([heel_x, baseline_y], dtype=np.float32)
    toe = np.array([toe_x, toe_y], dtype=np.float32)
    return (length_px * mm_per_px, ball_w, heel_w,
            np.array([0.0, -1.0]), heel, toe)


def draw_result(img, contour, axis_dir, heel, toe,
                foot_len, ball_w, heel_w, mm_per_px, save_path):
    vis = img.copy()
    if contour is not None:
        cv2.drawContours(vis, [contour], 0, (0, 255, 0), 2)
        heel_int = tuple(heel.astype(int))
        toe_int = tuple(toe.astype(int))
        cv2.line(vis, heel_int, toe_int, (255, 0, 0), 2)
        cv2.circle(vis, heel_int, 6, (0, 0, 255), -1)
        cv2.circle(vis, toe_int, 6, (0, 0, 255), -1)
        toe_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-10)
        perp = np.array([-toe_dir[1], toe_dir[0]])
        axis_len = np.linalg.norm(heel - toe)
        width_lines = [(0.68, ball_w)]
        if heel_w is not None:
            width_lines.insert(0, (HEEL_WIDTH_RATIO, heel_w))
        for ratio, val in width_lines:
            pos = heel + ratio * axis_len * toe_dir
            half_w = val / mm_per_px / 2
            p1 = (pos - half_w * perp).astype(int)
            p2 = (pos + half_w * perp).astype(int)
            cv2.line(vis, tuple(p1), tuple(p2), (0, 200, 255), 2)
        cv2.putText(vis, f"Len:{foot_len:.0f}mm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(vis, f"Ball:{ball_w:.0f}mm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        heel_text = f"Heel:{heel_w:.0f}mm" if heel_w is not None else "Heel:N/A"
        cv2.putText(vis, heel_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
    cv2.imwrite(save_path, vis)
    return vis


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT


def process_single_image(file):
    """处理单张图片，返回测量结果"""
    ext = os.path.splitext(file.filename)[1].lower()
    tmp_name = f"tmp_{uuid.uuid4().hex}{ext}"
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], tmp_name)
    file.save(tmp_path)

    img = cv2.imread(tmp_path)
    if img is None:
        os.remove(tmp_path)
        return None, '无法读取图片'

    warnings = []
    h, w = img.shape[:2]
    if min(h, w) < 480:
        os.remove(tmp_path)
        return None, '图片分辨率过低，请使用手机原相机拍摄'
    blur_score = cv2.Laplacian(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if blur_score < 4:
        os.remove(tmp_path)
        return None, '照片模糊，请保持手机稳定后重拍'
    if blur_score < 20:
        warnings.append('照片略模糊，已启用容错测量')

    corners = auto_detect_corners(img)
    if corners is None:
        os.remove(tmp_path)
        return None, '未检测到A4纸'

    corner_array = np.asarray(corners)
    if (corner_array[:, 0].min() < 3 or corner_array[:, 1].min() < 3 or
            corner_array[:, 0].max() > w - 4 or
            corner_array[:, 1].max() > h - 4):
        os.remove(tmp_path)
        return None, 'A4纸靠近或超出画面边缘，请完整拍入整张纸'

    M, mm_per_px, out_size = build_transform(corners)
    top = np.linalg.norm(corner_array[1] - corner_array[0])
    bottom = np.linalg.norm(corner_array[2] - corner_array[3])
    left = np.linalg.norm(corner_array[3] - corner_array[0])
    right = np.linalg.norm(corner_array[2] - corner_array[1])
    source_px_per_mm = min((top + bottom) / (2 * A4_WIDTH_MM),
                           (left + right) / (2 * A4_HEIGHT_MM))
    if source_px_per_mm < 1.35:
        os.remove(tmp_path)
        return None, 'A4纸在画面中太小，请稍微靠近后拍摄'
    if source_px_per_mm < 2.0:
        warnings.append('A4纸占画面较小，细节精度可能下降')

    foot_contour_orig = detect_foot_on_paper(img, corners)
    if foot_contour_orig is None:
        os.remove(tmp_path)
        return None, '未检测到脚'

    foot_contour_warped = warp_contour(foot_contour_orig, M, out_size)
    warped = cv2.warpPerspective(img, M, out_size)

    foot_points = foot_contour_warped.reshape(-1, 2)
    if foot_points[:, 1].max() < out_size[1] - 6:
        os.remove(tmp_path)
        return None, '脚跟未贴齐A4纸底边，请重新摆放后拍摄'
    if (foot_points[:, 1].min() < 6 or foot_points[:, 0].min() < 6 or
            foot_points[:, 0].max() > out_size[0] - 7):
        os.remove(tmp_path)
        return None, '脚尖或脚侧靠近纸张边缘，请将脚放在纸张中央后重拍'

    foot_len, ball_w, heel_w, axis_dir, heel, toe = measure_foot(
        foot_contour_warped, mm_per_px, out_size[1] - 1)

    if not heel_outline_is_visible(
            foot_contour_warped.reshape(-1, 2).astype(np.float32),
            out_size[1] - 1):
        heel_w = None
        warnings.append('正上方看不到真实脚跟边界，本次不输出脚跟宽')

    if foot_len < 180 or foot_len > 350:
        os.remove(tmp_path)
        return None, '脚长测量异常，请调整拍摄角度后重新拍照'
    if ball_w < 60 or ball_w > 130:
        os.remove(tmp_path)
        return None, '脚掌宽测量异常，请确保光线充足、背景与纸张有对比后重新拍照'
    if heel_w is not None and (heel_w < 30 or heel_w > 90):
        os.remove(tmp_path)
        return None, '脚跟宽测量异常，请确保脚踝不压在纸面上后重新拍照'

    result_name = f"result_{uuid.uuid4().hex}.png"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_name)
    draw_result(warped, foot_contour_warped, axis_dir, heel, toe,
                foot_len, ball_w, heel_w, mm_per_px, result_path)

    os.remove(tmp_path)

    return {
        'foot_length': round(foot_len, 1),
        'ball_width': round(ball_w, 1),
        'heel_width': round(heel_w, 1) if heel_w is not None else None,
        'result_image': f'/uploads/{result_name}',
        'warnings': warnings,
    }, None


@app.route('/')
def index():
    return render_template('index.html')


@app.after_request
def protect_admin_responses(response):
    if request.path.startswith('/admin'):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; form-action 'self'; "
            "frame-ancestors 'none'; base-uri 'none'")
    return response


@app.route('/admin')
def admin_index():
    return redirect(url_for('admin_measurements'))


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    username, password = _admin_credentials()
    configured = bool(username and password and
                      os.environ.get('ADMIN_DATABASE_URL'))
    if _admin_authenticated():
        return redirect(url_for('admin_measurements'))
    error = None
    status = 200
    if not configured:
        error = '后台尚未配置管理员账号或只读数据库连接。'
        status = 503
    elif request.method == 'POST':
        if _admin_login_blocked():
            error = '失败次数过多，请5分钟后再试。'
            status = 429
        else:
            submitted_username = request.form.get('username', '')
            submitted_password = request.form.get('password', '')
            username_valid = hmac.compare_digest(
                submitted_username, username)
            password_valid = hmac.compare_digest(
                submitted_password, password)
            valid = username_valid and password_valid
            if valid:
                with _ADMIN_LOGIN_LOCK:
                    _ADMIN_LOGIN_FAILURES.clear()
                session.clear()
                session['admin_auth'] = _admin_session_tag(username, password)
                return redirect(url_for('admin_measurements'))
            _admin_login_blocked(record_failure=True)
            error = '账号或密码错误。'
            status = 401
    return render_template(
        'admin.html', login=True, configured=configured, error=error), status


@app.route('/admin/logout', methods=['POST'])
def admin_logout():
    session.clear()
    return redirect(url_for('admin_login'))


@app.route('/admin/measurements')
@_admin_required
def admin_measurements():
    try:
        search = _admin_search_term(request.args.get('q'))
        page = max(1, min(int(request.args.get('page', '1')), 100000))
    except (TypeError, ValueError):
        return render_template(
            'admin.html', login=False, rows=[], total=0, page=1,
            pages=1, search='', error='查询参数无效。'), 400
    try:
        rows, total = _admin_measurement_rows(search, page)
    except Exception:
        app.logger.exception('后台查询测量记录失败')
        return render_template(
            'admin.html', login=False, rows=[], total=0, page=page,
            pages=1, search=search,
            error='数据库暂时不可用。'), 503
    pages = max(1, math.ceil(total / 25))
    return render_template(
        'admin.html', login=False, rows=rows, total=total, page=page,
        pages=pages, search=search, error=None)


@app.route('/admin/measurements.csv')
@_admin_required
def admin_measurements_csv():
    try:
        search = _admin_search_term(request.args.get('q'))
        rows = _admin_measurement_export(search)
    except ValueError:
        return Response('查询参数无效。', status=400, mimetype='text/plain')
    except Exception:
        app.logger.exception('后台导出测量记录失败')
        return Response('数据库暂时不可用。', status=503,
                        mimetype='text/plain')
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow([
        '测量编号', '足别', '脚长(mm)', '脚掌宽(mm)', '脚跟宽(mm)',
        '质量', '尺寸置信度', '提示', '测量时间', '到期时间'])
    for row in rows:
        writer.writerow([
            row[0], '左脚' if row[1] == 'left' else '右脚',
            row[2], row[3], row[4], row[5],
            json.dumps(row[6], ensure_ascii=False),
            '；'.join(row[7] or []), row[8].isoformat(), row[9].isoformat()])
    return Response(
        '\ufeff' + output.getvalue(), mimetype='text/csv',
        headers={'Content-Disposition':
                 'attachment; filename=foot-measurements.csv'})


@app.route('/admin/measurements/<measurement_code>/outline.png')
@_admin_required
def admin_measurement_outline(measurement_code):
    measurement_code = measurement_code.upper()
    if not _MEASUREMENT_CODE_RE.fullmatch(measurement_code):
        abort(404)
    try:
        image_png = _admin_measurement_image(measurement_code)
    except Exception:
        app.logger.exception('后台读取轮廓图失败')
        abort(503)
    if image_png is None:
        abort(404)
    return send_file(
        io.BytesIO(image_png), mimetype='image/png',
        download_name=f'{measurement_code}.png',
        as_attachment=request.args.get('download') == '1', max_age=0)


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/measure', methods=['POST'])
def measure():
    _cleanup_expired_results()
    foot_side = request.form.get('foot_side', '').strip().lower()
    if foot_side not in {'left', 'right'}:
        return jsonify({'error': '请选择正在测量左脚还是右脚'}), 400
    if 'image_measure' not in request.files:
        return jsonify({'error': '请上传正上方测量照片'}), 400

    photo = request.files['image_measure']
    if photo.filename == '' or not allowed_file(photo.filename):
        return jsonify({'error': '测量照片格式不支持'}), 400

    side_keys = ('image_left', 'image_right')
    if all(key in request.files for key in side_keys):
        side_files = [request.files[key] for key in side_keys]
        if any(not item.filename or not allowed_file(item.filename)
               for item in side_files):
            return jsonify({'error': '左右斜拍照片格式不支持'}), 400
        uploaded = [_read_uploaded_image(photo)] + [
            _read_uploaded_image(item) for item in side_files]
        images = [item[0] for item in uploaded]
        metadata = [item[1] for item in uploaded]
        result, error = process_multiview_images(images, metadata)
        if error:
            return jsonify({'error': '联合测量失败',
                            'details': [error]}), 400
        save_token = _issue_save_token(result, foot_side)
        return jsonify({
            'average': {
                'foot_length': result['foot_length'],
                'ball_width': result['ball_width'],
                'heel_width': result['heel_width'],
            },
            'individual': [result],
            'count': 3,
            'mode': 'multiview',
            'quality_grade': result['quality_grade'],
            'dimension_confidence': result['dimension_confidence'],
            'retake_recommended': result['retake_recommended'],
            'warnings': result['warnings'],
            'save_token': save_token,
            'save_token_expires_in': SAVE_TOKEN_TTL_SECONDS,
            'errors': []
        })

    result, error = process_single_image(photo)
    if error:
        return jsonify({'error': '测量失败', 'details': [error]}), 400

    verification_received = False
    if 'image_verify' in request.files:
        verify = request.files['image_verify']
        verification_received = bool(
            verify.filename and allowed_file(verify.filename))

    save_token = _issue_save_token(result, foot_side)
    return jsonify({
        'average': {
            'foot_length': result['foot_length'],
            'ball_width': result['ball_width'],
            'heel_width': result['heel_width'],
        },
        'individual': [result],
        'count': 1,
        'verification_received': verification_received,
        'warnings': result['warnings'],
        'save_token': save_token,
        'save_token_expires_in': SAVE_TOKEN_TTL_SECONDS,
        'errors': []
    })


@app.route('/api/measurements', methods=['POST'])
def save_measurement():
    _cleanup_expired_results()
    payload = request.get_json(silent=True)
    token = payload.get('save_token') if isinstance(payload, dict) else None
    if not isinstance(token, str) or not token:
        return jsonify({'error': '缺少保存凭证'}), 400
    try:
        data = _read_save_token(token)
    except SignatureExpired:
        return jsonify({'error': '保存凭证已过期，请重新测量'}), 410
    except BadSignature:
        return jsonify({'error': '保存凭证无效，请重新测量'}), 400

    result_path = _result_path(data['result_filename'])
    if not os.path.isfile(result_path):
        try:
            measurement_code = _lookup_measurement_code(data['save_nonce'])
        except Exception:
            app.logger.exception('查询已保存测量结果失败')
            return jsonify({
                'error': '数据库暂时不可用，结果尚未保存，请稍后重试'
            }), 503
        if measurement_code:
            return jsonify({'measurement_code': measurement_code,
                            'saved': True, 'already_saved': True})
        return jsonify({'error': '临时结果图已过期，请重新测量'}), 410

    try:
        with open(result_path, 'rb') as image_file:
            image_png = image_file.read()
        measurement_code = _save_measurement_record(data, image_png)
    except Exception:
        app.logger.exception('保存测量结果失败')
        return jsonify({
            'error': '数据库暂时不可用，结果尚未保存，请稍后重试'
        }), 503

    try:
        os.remove(result_path)
    except OSError:
        app.logger.warning('已保存，但无法删除临时结果图：%s',
                           data['result_filename'])
    return jsonify({'measurement_code': measurement_code,
                    'saved': True, 'already_saved': False})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
