import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wafer_display as WD


def synthetic():
    """회색 웨이퍼(반지름 150) 중 오른쪽 띠는 미촬영 검정, 촬영 영역 안에 작은 검정 파티클 하나."""
    img = np.full((400, 400, 3), 170, np.uint8)
    img[:, 290:] = 0                      # 미촬영 (바깥과 이어진 검정)
    img[195:205, 145:155] = 10            # 파티클 (작은 검정)
    return img


class Render(unittest.TestCase):
    def setUp(self):
        self.out = WD.render(synthetic(), (200, 200), 150, 60, 0.0)

    def test_same_frame(self):
        self.assertEqual(self.out.shape, (400, 400, 3))    # 원본과 같은 좌표라야 검출 표시가 맞는다

    def test_unseen_filled_with_surface(self):
        self.assertGreater(self.out[200, 330].mean(), 100)  # 원 안의 검정 미촬영 → 표면색

    def test_defect_kept(self):
        self.assertLess(self.out[200, 150].mean(), 60)      # 작은 검정 결함은 지우지 않는다

    def test_seen_unchanged(self):
        self.assertLess(abs(int(self.out[200, 100].mean()) - 170), 8)

    def test_outside_white(self):
        self.assertGreater(self.out[5, 5].mean(), 250)

    def test_scale(self):
        out = WD.render(synthetic(), (200, 200), 150, 60, 0.0, scale=0.5)
        self.assertEqual(out.shape, (200, 200, 3))


if __name__ == "__main__":
    unittest.main()
