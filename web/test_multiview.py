import os

import cv2

from app import (_fit_width_profile, detect_partial_paper_corners,
                 process_multiview_images)


FOLDER = r'C:\Users\Admin\OneDrive\Desktop\111\6'
NAMES = [
    '6cf75a2b8c5e5ec59de20954a727bc04.jpg',
    'ca109b1852b3c6a87e83a677458a0f4c.jpg',
    'f9379ec342c9bb641a6eae30bcae9689.jpg',
]
LATEST_FOLDER = r'C:\Users\Admin\OneDrive\Desktop\111\7'
LATEST_NAMES = [
    '0ef799ec8c1827decf63c4b25fb2d585.jpg',
    '121e7b5082d55729a2ffca619ce402b5.jpg',
    '73c35335ff411d81f192634d877a3202.jpg',
]
PARTIAL_FOLDER = r'C:\Users\Admin\OneDrive\Desktop\111\8'
PARTIAL_NAMES = [
    '96b144d1c93700316e717225ab44cbb2.jpg',
    '22d25ef0cd4fefa23980a5a44c8bf301.jpg',
    'be63f5c372e0ebd8c36353fb6fa93a9c.jpg',
]
NEW_FOLDER = r'C:\Users\Admin\OneDrive\Desktop\111\9'
NEW_NAMES = [
    '5a13ce71c02b69e93ec980fd74ab8873.jpg',
    'a38e7466e5ea302c41293e29ba7b8f17.jpg',
    'c0119817409921667158b1b42894f1cc.jpg',
]
POOR_FOLDER = r'C:\Users\Admin\OneDrive\Desktop\111\10'
POOR_NAMES = [
    '6694c9e0c46a4e92b3130b56a5126b35.jpg',
    '26bbfef621e06d0bd4a37c3ce935d134.jpg',
    'a7b51b100cf41fb05ad5e2654ac03b1f.jpg',
]
SHADOW_FOLDER = r'C:\Users\Admin\OneDrive\Desktop\111\11'
SHADOW_NAMES = [
    '89b36c1c885d518382c51951bf60856c.jpg',
    '96ee6595203071fb7b59819d44fa11ab.jpg',
    'b530e7abdc485e38018b4db54008da9a.jpg',
]


def main():
    fitted = _fit_width_profile(
        [(row, 50 + 0.4 * row + (8 if row == 0 else 0))
         for row in range(-5, 6)], 0)
    assert fitted is not None and abs(fitted[0] - 50) < 0.5, fitted

    images = [cv2.imread(os.path.join(FOLDER, name)) for name in NAMES]
    assert all(image is not None for image in images)
    assert all(detect_partial_paper_corners(image) is not None
               for image in images)
    result, error = process_multiview_images(images)
    assert error is None, error
    truth = {'foot_length': 265, 'ball_width': 96, 'heel_width': 56}
    for key, expected in truth.items():
        assert abs(result[key] - expected) <= 2, (key, result[key], expected)
    assert result['calibration_error_px'] < 3.5
    assert min(result['side_angles']) >= 10
    print(result)

    latest = [cv2.imread(os.path.join(LATEST_FOLDER, name))
              for name in LATEST_NAMES]
    latest_result, latest_error = process_multiview_images(latest)
    assert latest_error is None, latest_error
    latest_truth = {'foot_length': 248, 'ball_width': 93, 'heel_width': 62}
    for key, expected in latest_truth.items():
        assert abs(latest_result[key] - expected) <= 2, (
            key, latest_result[key], expected)
    assert latest_result['calibration_error_px'] < 3.5
    assert min(latest_result['side_angles']) >= 10
    print(latest_result)

    partial = [cv2.imread(os.path.join(PARTIAL_FOLDER, name))
               for name in PARTIAL_NAMES]
    partial_result, partial_error = process_multiview_images(partial)
    assert partial_error is None, partial_error
    partial_truth = {'foot_length': 281, 'ball_width': 93,
                     'heel_width': 58}
    for key, expected in partial_truth.items():
        assert abs(partial_result[key] - expected) <= 2, (
            key, partial_result[key], expected)
    assert partial_result['calibration_error_px'] < 3.5
    assert min(partial_result['side_angles']) >= 10
    print(partial_result)

    new_images = [cv2.imread(os.path.join(NEW_FOLDER, name))
                  for name in NEW_NAMES]
    new_result, new_error = process_multiview_images(new_images)
    assert new_error is None, new_error
    new_truth = {'foot_length': 245, 'ball_width': 97, 'heel_width': 55}
    for key, expected in new_truth.items():
        assert abs(new_result[key] - expected) <= 2, (
            key, new_result[key], expected)
    assert new_result['calibration_error_px'] < 3.5
    assert min(new_result['side_angles']) >= 10
    print(new_result)

    poor_images = [cv2.imread(os.path.join(POOR_FOLDER, name))
                   for name in POOR_NAMES]
    poor_result, poor_error = process_multiview_images(poor_images)
    assert poor_error is None, poor_error
    poor_truth = {'foot_length': 270, 'ball_width': 100,
                  'heel_width': 80}
    tolerances = {'foot_length': 2, 'ball_width': 2, 'heel_width': 3}
    for key, expected in poor_truth.items():
        assert abs(poor_result[key] - expected) <= tolerances[key], (
            key, poor_result[key], expected)
    print(poor_result)

    shadow_images = [cv2.imread(os.path.join(SHADOW_FOLDER, name))
                     for name in SHADOW_NAMES]
    shadow_result, shadow_error = process_multiview_images(shadow_images)
    assert shadow_error is None, shadow_error
    shadow_truth = {'foot_length': 260, 'ball_width': 95,
                    'heel_width': 65}
    for key, expected in shadow_truth.items():
        assert abs(shadow_result[key] - expected) <= 2, (
            key, shadow_result[key], expected)
    print(shadow_result)


if __name__ == '__main__':
    main()
