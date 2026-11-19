"""웨이퍼 맵 합성 회귀 테스트. 실행: ./venv/bin/python -m unittest tests.test_stitch -v"""
import math
import unittest

import cv2
import numpy as np

import wafer_stitch as W


def synthetic_scan(step=30.0, n=3, s=4.0, rot_deg=88.0, size=(540, 960), q=0.3):
    """알려진 카메라 모델로 평면 장면을 촬영한 사진을 만든다."""
    h, w = size
    rng = np.random.default_rng(1)
    scene = cv2.GaussianBlur(rng.random((1400, 1400)).astype(np.float32), (0, 0), 3)
    scene = (scene - scene.min()) / np.ptp(scene)
    c, sn = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    A = s * np.array([[c, -sn], [sn, c]])
    Ainv = np.linalg.inv(-A)
    qn = q / (h * h)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - w / 2, yy - h / 2
    ux, uy = W._undistort(dx, dy, qn)
    images, pos = [], []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            C = np.array([i * step, j * step])
            px = Ainv[0, 0] * ux + Ainv[0, 1] * uy + C[0]
            py = Ainv[1, 0] * ux + Ainv[1, 1] * uy + C[1]
            mx, my = (700 + px * 2).astype(np.float32), (700 - py * 2).astype(np.float32)
            images.append(cv2.remap(scene, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
            pos.append(C)
    return images, np.array(pos), A, qn


class LensModel(unittest.TestCase):
    def test_distort_inverts_undistort(self):
        x, y = np.array([120.0, -260.0, 40.0]), np.array([80.0, 170.0, -300.0])
        q = 0.4 / 540 ** 2
        ux, uy = W._undistort(x, y, q)
        rx, ry = W._distort(ux, uy, q)
        np.testing.assert_allclose(rx, x, atol=0.5)
        np.testing.assert_allclose(ry, y, atol=0.5)


class CameraEstimate(unittest.TestCase):
    def test_recovers_rotation_scale_and_distortion(self):
        images, pos, A, q = synthetic_scan()
        light = np.full(images[0].shape, 200, np.float32)
        far = [(i, j) for i in range(0, len(pos), 3) for j in range(0, len(pos), 3)
               if i < j and 20 < np.hypot(*(pos[i] - pos[j])) < 95][:50]
        start = 1.04 * A
        out = W.refine_camera(images, light, pos, start, far)
        scale = np.linalg.norm(out["A"][:, 0])
        self.assertAlmostEqual(scale / np.linalg.norm(A[:, 0]), 1.0, delta=0.01)
        self.assertGreater(out["ncc_after"], 0.8)
        self.assertAlmostEqual(out["q"] * 540 ** 2, 0.3, delta=0.08)


class WaferMeasure(unittest.TestCase):
    def test_circle_fit(self):
        t = np.deg2rad(np.arange(10, 90, 3))
        p = np.c_[300 + 500 * np.cos(t), 200 + 500 * np.sin(t)]
        cx, cy, r = W.fit_circle(p)
        self.assertAlmostEqual(cx, 300, places=3)
        self.assertAlmostEqual(cy, 200, places=3)
        self.assertAlmostEqual(r, 500, places=3)



class GantryGridTest(unittest.TestCase):
    """다이 칸 간격은 갠트리 이동량(웨이퍼 원)에서 정한다 (11-18 격자 측정 실패 4 장·피치 3 % 어긋남)."""

    def test_pitch_from_gantry_wafer_keeps_measured_angle(self):
        wafer = {"r": 1350.0, "from_gantry": True}
        g = W.gantry_grid(wafer, {"angle_deg": 0.5, "pitch_px": 310.0, "phase": {"x": 0.1}, "lines": "bright"})
        self.assertAlmostEqual(g["pitch_px"], 1350.0 * 32 / 135)
        self.assertEqual(g["angle_deg"], 0.5)
        self.assertTrue(g["from_gantry"])

    def test_measure_failed_still_gives_grid(self):
        g = W.gantry_grid({"r": 1350.0, "from_gantry": True}, None)
        self.assertAlmostEqual(g["pitch_px"], 320.0)
        self.assertEqual(g["angle_deg"], 0.0)

    def test_photo_wafer_keeps_measurement(self):
        m = {"angle_deg": 0.25, "pitch_px": 300.0, "phase": {}, "lines": "bright"}
        self.assertIs(W.gantry_grid({"r": 1300.0}, m), m)
        self.assertIsNone(W.gantry_grid(None, None))

    def test_wafer_radius_far_from_spec_is_not_trusted(self):
        s = 10.0                                          # 모자이크 px/mm → 반지름 1350 px 가 정상
        self.assertTrue(W.plausible_wafer({"r": 1340.0, "arc_deg": 120}, s))
        self.assertFalse(W.plausible_wafer({"r": 1134.0, "arc_deg": 165}, s))   # 11-18 14:29: 스테인 고리를 잡음
        self.assertFalse(W.plausible_wafer({"r": 1350.0, "arc_deg": 30}, s))
        self.assertFalse(W.plausible_wafer(None, s))


class CameraPriorOrderTest(unittest.TestCase):
    """기준 카메라 값은 그 스캔보다 먼저 찍은 스캔에서만 (나중 스캔이 끼면 다시 합성할 때 결과가 바뀐다)."""

    def test_only_earlier_scans(self):
        import json, os, tempfile
        root = tempfile.mkdtemp()
        for name, sc in (("20260101_000000_scan", 20.0), ("20260102_000000_scan", 21.0), ("20260103_000000_scan", 99.0)):
            os.makedirs(os.path.join(root, name))
            with open(os.path.join(root, name, "wafer_map.json"), "w") as f:
                json.dump({"camera": {"px_per_mm_full": sc, "rotation_deg": -90.0, "mirrored": False}}, f)
        A = W.camera_prior(root, os.path.join(root, "20260102_120000_scan"))
        self.assertAlmostEqual(float(np.linalg.norm(A[:, 0])), 20.5 * W.WORK_SCALE, places=4)


if __name__ == "__main__":
    unittest.main()
