"""Per-photo preflight; does not certify geometry or measurement accuracy."""
import cv2
import numpy as np


def inspect_photo(image, backend):
    problems, warnings = [], []
    result = {'problems': problems, 'warnings': warnings,
              'angle_verified': False, 'paper_detected': False}
    if image is None:
        return dict(result, status='retake', problems=['照片无法解码，请使用 JPG、PNG 或 WebP 原图。'])
    height, width = image.shape[:2]
    result.update(width=width, height=height, ratio=width / height)
    if min(height, width) < 480:
        problems.append('分辨率不足，请使用短边至少 480 像素的原图。')
    scale = min(1.0, 1200 / max(height, width))
    sample = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    result['sharpness'] = round(sharpness, 2)
    if sharpness < 3:
        problems.append('照片严重模糊，请稳定手机并对焦后重拍。')
    elif sharpness < 15:
        warnings.append('清晰度偏低，建议稳定手机并重新对焦。')
    if np.percentile(gray, 95) < 100:
        problems.append('画面严重偏暗，请增加均匀照明。')
    corners = None if problems else backend._detect_reliable_paper_corners(sample)
    if corners is None:
        warnings.append('单张照片尚无法可靠确认纸边，请让顶边和两条长边清晰入镜；仍需三图联合检查。')
    else:
        result['paper_detected'] = True
        evidence = backend._paper_quad_evidence(sample, corners)
        _, coverages = backend._paper_quad_edge_support(sample, corners)
        result['paper_edge_coverage'] = [round(float(value), 3) for value in coverages]
        result['paper_boundary_verified'] = bool(
            evidence >= 0.5 and all(coverages[index] >= 0.72 for index in (0, 1, 3)))
        if not result['paper_boundary_verified']:
            warnings.append('检测到纸框，但顶边或两条长边与实际图像边缘不够一致；请检查纸边、阴影和地面反光，本张仍需联合验证。')
        masks = [backend._multiview_foot_mask(sample, corners, consensus=value) for value in (False, True)]
        if any(mask is None for mask in masks):
            warnings.append('脚部轮廓尚不稳定，请检查脚趾是否完整、脚边是否被阴影覆盖。')
        else:
            paper = np.zeros(sample.shape[:2], np.uint8)
            cv2.fillConvexPoly(paper, np.asarray(corners, np.int32), 1)
            first, second = [(mask > 0) & (paper > 0) for mask in masks]
            agreement = float(np.count_nonzero(first & second) / max(np.count_nonzero(first | second), 1))
            result['segmentation_agreement'] = round(agreement, 3)
            if agreement < backend.WARN_SEGMENTATION_AGREEMENT:
                warnings.append('两种脚部轮廓识别不一致，可能把纸上阴影算进脚部。建议遮挡直射光源或改用均匀光后只重拍本张。')
    if backend._color_cast_strength(sample) > 10.5:
        warnings.append('存在色偏风险，请避免彩色灯和一侧强光。')
    result['status'] = 'retake' if problems else 'warning' if warnings else 'pass'
    return result
