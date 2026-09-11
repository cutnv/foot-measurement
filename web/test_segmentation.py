import sys
from pathlib import Path

import cv2
import numpy as np

from app import segment_scene


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r'C:\Users\Admin\OneDrive\Desktop\111')
output = Path(sys.argv[2]) if len(sys.argv) > 2 else None
if output:
    output.mkdir(parents=True, exist_ok=True)

failed = []
previews = []
for path in sorted(root.glob('*/*.jpg')):
    image = cv2.imread(str(path))
    if image is None:
        failed.append(f'{path}: 无法读取')
        continue

    masks = segment_scene(image)
    total = image.shape[0] * image.shape[1]
    paper_ratio = cv2.countNonZero(masks['paper']) / total
    foot_ratio = cv2.countNonZero(masks['foot']) / total
    overlap = cv2.countNonZero(
        cv2.bitwise_and(masks['paper'], masks['foot']))
    union = cv2.bitwise_or(masks['floor'],
                          cv2.bitwise_or(masks['paper'], masks['foot']))

    ok = (paper_ratio > 0.025 and foot_ratio > 0.015 and
          overlap == 0 and cv2.countNonZero(union) == total)
    print(f'{path.parent.name}/{path.name} '
          f'paper={paper_ratio:.3f} foot={foot_ratio:.3f} '
          f'{"OK" if ok else "FAIL"}')
    if not ok:
        failed.append(str(path))

    if output:
        preview = image.copy()
        color = np.zeros_like(image)
        color[masks['floor'] > 0] = (60, 60, 60)
        color[masks['paper'] > 0] = (255, 100, 0)
        color[masks['foot'] > 0] = (0, 0, 255)
        preview = cv2.addWeighted(preview, 0.55, color, 0.45, 0)
        cv2.imwrite(str(output / f'{path.parent.name}_{path.name}'), preview)
        thumb = cv2.resize(preview, (320, 427))
        cv2.putText(thumb, f'group {path.parent.name}', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        previews.append(thumb)

if output and previews:
    rows = [cv2.hconcat(previews[i:i + 3])
            for i in range(0, len(previews), 3)]
    cv2.imwrite(str(output / 'overview.jpg'), cv2.vconcat(rows))

if failed:
    raise SystemExit('分割失败:\n' + '\n'.join(failed))
