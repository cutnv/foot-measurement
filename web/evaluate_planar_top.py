import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import app


def read_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--top-indexes', required=True)
    parser.add_argument('--groups', default='')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    indexes = {int(group): int(index) for group, index in
               (pair.split(':') for pair in args.top_indexes.split(','))}
    selected = {int(value) for value in args.groups.split(',') if value}
    report = []
    for folder in args.folder.iterdir():
        try:
            group = int(folder.name.split('、', 1)[0])
        except (ValueError, IndexError):
            continue
        if selected and group not in selected:
            continue
        paths = sorted(path for path in folder.iterdir()
                       if path.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if len(paths) != 3 or group not in indexes:
            continue
        images = [read_image(path) for path in paths]
        order = [indexes[group]] + [index for index in range(3)
                                    if index != indexes[group]]
        normalized = app._normalize_multiview_images(
            [images[index] for index in order])
        reliable = [app._detect_reliable_paper_corners(image)
                    for image in normalized]
        image = normalized[0]
        corners = app._detect_reliable_paper_corners(image)
        item = {'group': group, 'image': paths[indexes[group]].name}
        if all(corner is not None for corner in reliable):
            item['edge_coverages'] = [[round(value, 3) for value in
                                       app._paper_quad_edge_support(
                                           image, corner)[1]]
                                      for image, corner in zip(
                                          normalized, reliable)]
            try:
                item['initial_shared_error'] = round(float(
                    app._calibrate_multiview(
                        reliable, (image.shape[1], image.shape[0]))[0]), 3)
            except cv2.error:
                item['initial_shared_error'] = None
        if corners is None:
            item['error'] = 'paper'
        else:
            mask = app._multiview_foot_mask(image, corners)
            if mask is None:
                item['error'] = 'foot'
            else:
                transform = cv2.getPerspectiveTransform(
                    np.asarray(corners, np.float32),
                    np.float32([[0, 0], [app.A4_WIDTH_MM, 0],
                                [app.A4_WIDTH_MM, app.A4_HEIGHT_MM],
                                [0, app.A4_HEIGHT_MM]]))
                planar = app._warp_planar_foot_mask(mask, transform)
                empty = np.zeros(planar.shape, np.uint8)
                profile = app._measurement_debug_summary(
                    empty, empty, empty.astype(np.float32), planar)['top']
                item['paper_corners'] = np.asarray(
                    corners, np.float32).round(2).tolist()
                item['top'] = profile
                item['oriented_length'] = round(
                    app._oriented_planar_length(planar), 2)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        report.append(item)
    report.sort(key=lambda item: item['group'])
    if args.output:
        args.output.write_text(json.dumps(
            report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
