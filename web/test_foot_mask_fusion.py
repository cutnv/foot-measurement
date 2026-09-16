import cv2
import numpy as np

import app


def main():
    first = np.zeros((80, 80), np.uint8)
    second = np.zeros_like(first)
    third = np.zeros_like(first)
    cv2.rectangle(first, (20, 10), (60, 70), 255, -1)
    cv2.rectangle(second, (20, 10), (58, 70), 255, -1)
    cv2.rectangle(third, (22, 10), (60, 70), 255, -1)
    cv2.circle(first, (40, 40), 8, 0, -1)
    fused = app._fuse_foot_mask_votes([first, second, third])
    assert fused is not None
    assert fused[40, 40] == 255       # 内部漏选被填实
    assert fused[40, 60] == 255
    assert fused[40, 70] == 0
    assert app._fuse_foot_mask_votes([first]) is None
    print('foot mask fusion: ok')


if __name__ == '__main__':
    main()
