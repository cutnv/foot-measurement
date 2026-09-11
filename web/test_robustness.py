from pathlib import Path

import cv2
import numpy as np

from app import (auto_detect_corners, build_transform, detect_foot_on_paper,
                 measure_foot, warp_contour)


PHOTOS = [
    Path(r'C:\Users\Admin\OneDrive\Desktop\111\4\0e0160cff8d5f762b62e2da389245b66.jpg'),
    Path(r'C:\Users\Admin\OneDrive\Desktop\111\5\a18ee99783fe38215a54ce3671b1e809.jpg'),
    Path(r'C:\Users\Admin\AppData\Local\Temp\codex-clipboard-427ec144-172c-4ff4-9813-797a129bc43b.jpg'),
]


def measure(image):
    corners = auto_detect_corners(image)
    assert corners is not None, '未识别A4纸'
    transform, mm_per_px, out_size = build_transform(corners)
    contour = detect_foot_on_paper(image, corners)
    assert contour is not None, '未识别脚'
    warped = warp_contour(contour, transform, out_size)
    length, ball, *_ = measure_foot(warped, mm_per_px, out_size[1] - 1)
    return np.array([length, ball])


def shadow(image, strength=0.15):
    height, width = image.shape[:2]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    shade = 1.0 - strength * np.clip((x - 0.35) / 0.45, 0.0, 1.0)
    factors = shade[None, :, None]
    return np.clip(image.astype(np.float32) * factors, 0, 255).astype(np.uint8)


def jpeg(image, quality=80):
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


variants = {
    'shadow': shadow,
    'blur': lambda image: cv2.GaussianBlur(image, (9, 9), 1.6),
    'jpeg80': jpeg,
}

failures = []
for path in PHOTOS:
    image = cv2.imread(str(path))
    if image is None:
        continue
    baseline = measure(image)
    print(path.name, 'baseline', np.round(baseline, 1))
    for name, transform in variants.items():
        try:
            value = measure(transform(image))
            drift = np.abs(value - baseline)
            print(' ', name, np.round(value, 1), 'drift', np.round(drift, 1))
            assert drift[0] <= 2.0 and drift[1] <= 2.0, (
                f'{path.name}/{name} 漂移过大: {drift}')
        except AssertionError as error:
            print(' ', name, 'FAIL', error)
            failures.append(str(error))

assert not failures, '\n'.join(failures)
