import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import app


DIMENSIONS = ('foot_length', 'ball_width', 'heel_width')


def read_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def jpeg72(images):
    result = []
    for image in images:
        ok, data = cv2.imencode(
            '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 72])
        assert ok
        result.append(cv2.imdecode(data, cv2.IMREAD_COLOR))
    return result


def blur(images):
    return [cv2.GaussianBlur(image, (7, 7), 1.2) for image in images]


def scale_light(images, factor):
    return [np.clip(image.astype(np.float32) * factor, 0, 255).astype(
        np.uint8) for image in images]


def warm_light(images):
    factors = np.float32([0.92, 0.98, 1.04])
    return [np.clip(image.astype(np.float32) * factors, 0, 255).astype(
        np.uint8) for image in images]


def gradient_shadow(images):
    result = []
    for index, image in enumerate(images):
        axis = np.linspace(0.0, 1.0, image.shape[1], dtype=np.float32)
        if index % 2:
            axis = axis[::-1]
        shade = 1.0 - 0.18 * np.clip((axis - 0.15) / 0.70, 0.0, 1.0)
        result.append(np.clip(
            image.astype(np.float32) * shade[None, :, None],
            0, 255).astype(np.uint8))
    return result


def downscale(images):
    result = []
    for image in images:
        scale = min(1.0, 1280.0 / max(image.shape[:2]))
        result.append(cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    return result


def vignette(images):
    result = []
    for image in images:
        y = np.linspace(-1.0, 1.0, image.shape[0], dtype=np.float32)
        x = np.linspace(-1.0, 1.0, image.shape[1], dtype=np.float32)
        radius = x[None, :] ** 2 + y[:, None] ** 2
        shade = 1.0 - 0.14 * np.clip(radius / 2.0, 0.0, 1.0)
        result.append(np.clip(
            image.astype(np.float32) * shade[:, :, None],
            0, 255).astype(np.uint8))
    return result


def crop_two_percent(images):
    result = []
    for image in images:
        h, w = image.shape[:2]
        dx, dy = max(1, round(w * 0.02)), max(1, round(h * 0.02))
        cropped = image[dy:h - dy, dx:w - dx]
        result.append(cv2.resize(
            cropped, (w, h), interpolation=cv2.INTER_LINEAR))
    return result


VARIANTS = {
    'jpeg72': jpeg72,
    'blur': blur,
    'dim80': lambda images: scale_light(images, 0.80),
    'bright112': lambda images: scale_light(images, 1.12),
    'warm_light': warm_light,
    'gradient_shadow': gradient_shadow,
    'downscale1280': downscale,
    'vignette': vignette,
    'crop2pct': crop_two_percent,
}


def dimensions(result):
    return np.asarray([result[key] for key in DIMENSIONS], np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('baseline_report', type=Path)
    parser.add_argument('--groups', default='')
    parser.add_argument('--variants', default='')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    selected_groups = {int(value) for value in args.groups.split(',') if value}
    selected_variants = [value for value in args.variants.split(',') if value]
    variants = ({name: VARIANTS[name] for name in selected_variants}
                if selected_variants else VARIANTS)
    baseline = json.loads(args.baseline_report.read_text(encoding='utf-8'))
    records = []
    draw_result = app._draw_multiview_result
    app._draw_multiview_result = lambda *unused: None
    try:
        for item in baseline:
            group = int(item['group'])
            if selected_groups and group not in selected_groups:
                continue
            if item.get('result') is None:
                continue
            source_folder = args.folder / item['folder']
            images = [read_image(source_folder / name)
                      for name in item['input_order']]
            if any(image is None for image in images):
                raise FileNotFoundError(source_folder)
            expected = dimensions(item['result'])
            for name, transform in variants.items():
                started = time.perf_counter()
                metadata = ([{'jpeg_luma_quantizer': 9}] * 3
                            if name == 'jpeg72' else None)
                result, error = app.process_multiview_images(
                    transform(images), metadata)
                record = {
                    'group': group,
                    'variant': name,
                    'elapsed_seconds': round(time.perf_counter() - started, 3),
                    'error': error,
                }
                if result is not None:
                    measured = dimensions(result)
                    drift = np.abs(measured - expected)
                    record.update({
                        'measured_mm': measured.round(1).tolist(),
                        'drift_mm': drift.round(1).tolist(),
                        'within_2mm': bool(np.all(drift <= 2.0)),
                        'quality_grade': result['quality_grade'],
                        'retake_recommended': result['retake_recommended'],
                        'calibration_error_px': result[
                            'calibration_error_px'],
                        'side_angles': result['side_angles'],
                        'heel_profile_stability_mm': result[
                            'heel_profile_stability_mm'],
                        'segmentation_agreement': result[
                            'segmentation_agreement'],
                        'unsafe_output': bool(
                            np.any(drift > 2.0) and
                            not result['retake_recommended']),
                    })
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        app._draw_multiview_result = draw_result

    summary = {
        'total': len(records),
        'rejected': sum(item.get('error') is not None for item in records),
        'within_2mm': sum(item.get('within_2mm', False) for item in records),
        'unsafe_outputs': sum(item.get('unsafe_output', False)
                              for item in records),
    }
    report = {'summary': summary, 'records': records}
    output = args.output or args.baseline_report.with_name(
        'multiview_robustness_report.json')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
