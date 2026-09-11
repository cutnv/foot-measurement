from pathlib import Path
import struct

import cv2
import numpy as np

import app


assert not app._severe_multiview_color_cast([8.4, 10.3, 11.0])
assert not app._severe_multiview_color_cast([5.0, 9.0, 30.0])
assert app._severe_multiview_color_cast([18.1, 20.0, 24.0])


ROOT = Path(r'C:\Users\Admin\OneDrive\Desktop\111\group')
CASES = {21: 2, 22: 2}
POSE_CASES = {29: 1, 30: 1}


def read(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def main():
    tiff = (b'II' + struct.pack('<H', 42) + struct.pack('<I', 8) +
            struct.pack('<H', 1) +
            struct.pack('<HHII', 0x8769, 4, 1, 26) +
            struct.pack('<I', 0) + struct.pack('<H', 1) +
            struct.pack('<HHI', 0xa405, 3, 1) +
            struct.pack('<H', 26) + b'\x00\x00' + struct.pack('<I', 0))
    exif = b'Exif\x00\x00' + tiff
    jpeg = b'\xff\xd8\xff\xe1' + struct.pack('>H', len(exif) + 2) + exif + b'\xff\xd9'
    assert app._jpeg_exif_focal_35mm(jpeg) == 26.0
    assert app._multiview_focal_hint(
        [{'focal_length_35mm': 26.0}] * 3, (1200, 1600)) == 1600 * 26 / 36
    assert app._multiview_focal_hint([
        {'focal_length_35mm': 26.0}, {'focal_length_35mm': 28.0},
        {'focal_length_35mm': 52.0}], (1200, 1600)) is None

    reference = np.float32([[100, 80], [500, 80], [500, 640], [100, 640]])
    assert app._paper_quads_agree(reference, reference + 8, (720, 720))
    assert not app._paper_quads_agree(
        reference, reference + np.float32([80, 0]), (720, 720))
    synthetic = np.full((720, 720, 3), 50, np.uint8)
    cv2.fillConvexPoly(synthetic, reference.astype(np.int32), (235,) * 3)
    _, outer_coverage = app._paper_quad_edge_support(synthetic, reference)
    inner = reference + np.float32([[50, 50], [-50, 50],
                                    [-50, -50], [50, -50]])
    _, inner_coverage = app._paper_quad_edge_support(synthetic, inner)
    assert sum(value >= 0.72 for value in outer_coverage) == 4
    assert sum(value >= 0.72 for value in inner_coverage) == 0
    camera = np.array([[100.0, 0.0, 100.0],
                       [0.0, 100.0, 100.0],
                       [0.0, 0.0, 1.0]])
    rotation = np.zeros((3, 1), np.float64)
    translation = np.array([[0.0], [0.0], [500.0]])
    full = np.full((300, 300), 255, np.uint8)
    empty = np.zeros_like(full)
    geometry = ([camera] * 3, [np.zeros(5)] * 3,
                [rotation] * 3, [translation] * 3)
    hard = app._visual_hull_footprint(
        [full, full, empty], *geometry)
    soft = app._visual_hull_footprint(
        [full, full, empty], *geometry, minimum_side_views=1)
    assert cv2.countNonZero(hard[0]) == 0
    assert cv2.countNonZero(soft[0]) > 0
    top = np.zeros((298, 211), np.uint8)
    rear = np.zeros_like(top)
    top[40:298, 50:160] = 255
    rear[180:298, 65:145] = 255
    outline = app._anatomical_planar_outline(top, rear, rear, 250.0)
    assert outline[100, 50] == 255
    assert outline[260, 50] == 0
    assert outline[260, 65] == 255
    tailed = rear.copy()
    tailed[250:, 130:205] = 255
    outline, quality = app._anatomical_planar_outline(
        top, tailed, tailed, 250.0, 60.0, return_quality=True)
    assert quality['rear_cut']
    assert cv2.countNonZero(outline[260:]) == 0
    for group, top_index in (CASES | POSE_CASES).items():
        paths = sorted((ROOT / str(group)).glob('*.jpg'))
        images = [read(path) for path in paths]
        order = [top_index] + [index for index in range(3)
                               if index != top_index]
        result, error = app.process_multiview_images(
            [images[index] for index in order])
        assert error is None, (group, error)
        assert result is not None, group
        assert result['calibration_error_px'] <= \
            app.MAX_MULTIVIEW_REPROJECTION_PX, group
        if group in POSE_CASES:
            assert result['measurement_debug']['projection_mode'] == \
                'independent_side_consensus', group


if __name__ == '__main__':
    main()
