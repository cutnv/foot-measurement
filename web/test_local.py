import cv2
import sys
sys.path.insert(0, '.')
from app import auto_detect_corners, build_transform, detect_foot_on_paper, warp_contour, filter_ankle, measure_foot

photos = [
    '../8e0a6ffc2aed2a1863ded5746e894ce5.jpg',
    '../f9c54f1f7f8e2e7f70975d2ee0c679ba.jpg',
    '../fe23cfec4010fe8c8e394c86ddbe44fb.jpg'
]

for p in photos:
    img = cv2.imread(p)
    name = p.split('/')[-1][:12]
    print(f'--- {name} ---')
    corners = auto_detect_corners(img)
    if corners is None:
        print('  Paper: NOT DETECTED')
        continue
    print('  Paper: OK')
    M, mm_per_px, out_size = build_transform(corners)
    foot = detect_foot_on_paper(img, corners)
    if foot is None:
        print('  Foot: NOT DETECTED')
        continue
    foot_w = warp_contour(foot, M, out_size)
    foot_f = filter_ankle(foot_w, mm_per_px)
    print(f'  Before filter: {len(foot_w)} pts, After: {len(foot_f)} pts')
    fl, bw, hw, _, _, _ = measure_foot(foot_f, mm_per_px)
    print(f'  Length={fl:.1f}mm Ball={bw:.1f}mm Heel={hw:.1f}mm')
