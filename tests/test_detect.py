"""기준 다이 결함 추출 시험. 실행: ./venv/bin/python -m unittest tests.test_detect -v"""
import unittest

import cv2
import numpy as np

import wafer_detect as W
import wafer_raster as R
from wafer_model import die_layout

P = 170
SIDE = R.GRID_DIES * P


def clean_map(seed=0, line=-25):
    """스크라이브 라인과 조명 기울기, 잡음이 있는 곧은 웨이퍼 맵. line>0 이면 투명 앞면처럼 격자선이 밝다"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIDE, 0:SIDE].astype(np.float32)
    g = 150 + 0.01 * xx + 0.008 * yy
    for k in range(R.GRID_DIES + 1):
        g[:, max(k * P - 2, 0):k * P + 2] += line
        g[max(k * P - 2, 0):k * P + 2, :] += line
    g = g + rng.normal(0, 0.4, g.shape)
    meta = {"wafer": {"center_px": [SIDE / 2 - 0.5, SIDE / 2 - 0.5], "radius_px": 135 / 32 * P},
            "grid": {"pitch_px": float(P), "angle_deg": 0.0}}
    return np.clip(g, 0, 255).astype(np.uint8), meta


def die_at(i):
    return next(d for d in die_layout() if d["id"] == i)


class Detect(unittest.TestCase):
    def test_clean_map_has_no_defects(self):
        g, meta = clean_map()
        r = W.detect_array(g, meta)
        self.assertEqual(len(r["inspected"]), 44)
        self.assertEqual(r["defects"], [])

    def test_particle_is_found_in_the_right_die(self):
        g, meta = clean_map()
        a, b = R.die_slot(die_at(20))
        cv2.circle(g, (b * P + 80, a * P + 70), 4, 60, -1)
        r = W.detect_array(g, meta)
        self.assertEqual({d["die"] for d in r["defects"]}, {20})
        self.assertEqual(r["defects"][0]["type"], "particle")
        self.assertTrue(r["quadrats"])

    def test_scratch_is_long(self):
        g, meta = clean_map()
        a, b = R.die_slot(die_at(25))
        cv2.line(g, (b * P + 30, a * P + 40), (b * P + 140, a * P + 120), 90, 2)
        r = W.detect_array(g, meta)
        self.assertIn("scratch", {d["type"] for d in r["defects"] if d["die"] == 25})

    def test_large_dark_blob_is_stain(self):
        # 크기 기준 (11-17): 10 mm 를 넘는 덩어리는 스테인. 실물처럼 투명 앞면·밝은 격자선·중간 대비로 만든다
        front = dict(dark_only=True, sharp_z=W.DARK_ONLY_Z, min_area_mm2=W.MIN_DARK_AREA_MM2)
        g, meta = clean_map(line=25)
        a, b = R.die_slot(die_at(20))
        cv2.ellipse(g, (b * P + 85, a * P + 80), (30, 22), 20, 0, 360, 110, -1)  # 길이 약 11 mm
        r = W.detect_array(g, meta, **front)
        self.assertEqual([d["type"] for d in r["defects"] if d["die"] == 20], ["stain"])

    def test_die_verdict_by_position(self):
        # docs/specs/die-verdict.md: 중심부 침범 → 불량, 가장자리 띠의 작은 결함 → 재검사, 없음 → 정상
        front = dict(dark_only=True, sharp_z=W.DARK_ONLY_Z, min_area_mm2=W.MIN_DARK_AREA_MM2)
        mm = P / 32
        a, b = R.die_slot(die_at(20))
        g, meta = clean_map(line=25)
        cv2.circle(g, (b * P + 85, a * P + 85), 5, 120, -1)                       # 20: 가운데 (실물만큼 옅게)
        a2, b2 = R.die_slot(die_at(27))
        cv2.circle(g, (int(b2 * P + 2.5 * mm), a2 * P + 85), 5, 120, -1)          # 27: 왼쪽 가장자리 2.5 mm
        v = W.detect_array(g, meta, **front)["verdicts"]
        self.assertEqual((v[20]["verdict"], v[27]["verdict"], v[5]["verdict"]), ("defect", "recheck", "normal"))

    def test_edge_rules_by_kind(self):
        # docs/specs/inspection-criteria.md 3절: 가장자리 띠에만 있을 때 종류별 변수
        self.assertEqual(W.die_verdict(False, 0.01, scratch_edge_mm=17.0), "defect")   # 띠 안 스크래치 16 mm 이상
        self.assertEqual(W.die_verdict(False, 0.01, scratch_edge_mm=10.0), "recheck")
        self.assertEqual(W.die_verdict(False, 0.03, stain_ratio=0.03), "defect")       # 띠 안 스테인 2.5 % 이상
        self.assertEqual(W.die_verdict(False, 0.02, stain_ratio=0.02), "recheck")

    def test_gdbn(self):
        # 정상 다이라도 8 방향 이웃 중 불량이 5 개 이상이면 재검사 (검사 못 한 이웃은 세지 않는다)
        by = {(d["row"], d["col"]): d["id"] for d in die_layout()}
        center = by[(3, 3)]
        ring = [by[(3 + dr, 3 + dc)] for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
        v = {i: {"verdict": "normal"} for i in by.values()}
        for i in ring[:5]:
            v[i]["verdict"] = "defect"
        W.apply_gdbn(v)
        self.assertEqual((v[center]["verdict"], v[center]["gdbn"]), ("recheck", 5))
        v = {i: {"verdict": "normal"} for i in by.values()}
        for i in ring[:4]:
            v[i]["verdict"] = "defect"
        W.apply_gdbn(v)
        self.assertEqual(v[center]["verdict"], "normal")

    def test_die_verdict_rule(self):
        self.assertEqual(W.die_verdict(False, 0.0), "normal")
        self.assertEqual(W.die_verdict(True, 0.002), "defect")       # 중심부에 작은 점
        self.assertEqual(W.die_verdict(False, 0.01), "recheck")      # 가장자리에만 작게
        self.assertEqual(W.die_verdict(False, 0.06), "defect")       # 가장자리라도 너무 많음

    def test_scratch_across_two_dies_is_one_defect(self):
        # 2026-11-16 18:20 실물: 스크래치 하나가 다이 경계를 넘어 조각마다 파티클·스크래치로 따로 잡혔다
        front = dict(dark_only=True, sharp_z=W.DARK_ONLY_Z, min_area_mm2=W.MIN_DARK_AREA_MM2)
        g, meta = clean_map(line=25)
        a, b = R.die_slot(die_at(20))
        cv2.line(g, (b * P + 60, a * P + 70), (b * P + P + 110, a * P + 100), 110, 3)  # 20 → 21 (오른쪽 이웃)
        r = W.detect_array(g, meta, **front)
        parts = [d for d in r["defects"] if d["die"] in (20, 21)]
        self.assertEqual({d["die"] for d in parts}, {20, 21})
        self.assertEqual({d["type"] for d in parts}, {"scratch"})
        self.assertEqual(len({d["obj"] for d in parts}), 1)

    def test_scratch_cut_by_wide_unseen_band_is_one_scratch(self):
        # 2026-11-16 18:20 실물: 반사 띠(최대 5 mm)가 스크래치를 끊어 한쪽은 스크래치, 다른 쪽은 스테인으로 나왔다.
        # 결함 모양은 합성 전체 이미지에서 따라가므로 띠 너머까지 하나로 잡혀야 한다
        front = dict(dark_only=True, sharp_z=W.DARK_ONLY_Z, min_area_mm2=W.MIN_DARK_AREA_MM2)
        g, meta = clean_map(line=25)
        a, b = R.die_slot(die_at(20))
        cv2.line(g, (b * P + 20, a * P + 85), (b * P + P + 110, a * P + 85), 110, 3)
        seen = np.full(g.shape, 255, np.uint8)
        seen[a * P + 60:a * P + 110, b * P + 70:b * P + 100] = 0           # 다이 20 안의 약 5.6 mm 미검사 띠
        r = W.detect_array(g, meta, seen=seen, **front)
        parts = [d for d in r["defects"] if d["die"] in (20, 21)]
        self.assertEqual(len({d["obj"] for d in parts}), 1)
        self.assertEqual({d["type"] for d in parts}, {"scratch"})

    def test_scratch_tail_is_absorbed(self):
        # 2026-11-17 사용자 지적: 스크래치 꼬리가 따로 잡혀 파티클·스테인으로 겹쳐 나왔다. 스크래치 연장선 위의
        # 작은 조각은 스크래치에 흡수한다 (웨이퍼 단위 판단)
        mm = 0.1
        line = {"kind": "scratch", "fx": np.arange(0, 300), "fy": np.full(300, 50), "xs": np.arange(0, 300), "ys": np.full(300, 50)}
        tail = {"kind": "particle", "fx": np.arange(330, 360), "fy": np.full(30, 52), "xs": np.arange(330, 360), "ys": np.full(30, 52)}
        far = {"kind": "particle", "fx": np.arange(100, 130), "fy": np.full(30, 120), "xs": np.arange(100, 130), "ys": np.full(30, 120)}
        out = W._absorb_tails([line, tail, far], mm)
        self.assertEqual(len(out), 2)
        self.assertEqual(sorted(o["kind"] for o in out), ["particle", "scratch"])
        self.assertEqual(len(out[0]["xs"]) + len(out[1]["xs"]), 360)

    def test_fragments_near_stain_are_absorbed(self):
        # 11-17 도넛 스테인: 붓질로 쪼개진 고리 조각이 파티클·스크래치로 나왔다. 스테인 3 mm 안에 절반 이상 있는
        # 조각은 같은 오염으로 흡수한다. 스테인 옆을 지나가기만 하는 긴 스크래치는 그대로 둔다
        mm = 0.1
        stain = {"kind": "stain", "fx": np.repeat(np.arange(100, 200), 50), "fy": np.tile(np.arange(100, 150), 100)}
        frag = {"kind": "particle", "fx": np.arange(205, 215), "fy": np.full(10, 120)}
        line = {"kind": "scratch", "fx": np.arange(0, 600), "fy": np.full(600, 160)}
        far = {"kind": "particle", "fx": np.arange(400, 410), "fy": np.full(10, 400)}
        for o in (stain, frag, line, far):
            o["xs"], o["ys"], o["long_mm"] = o["fx"], o["fy"], 1.0
        out = W._absorb_into_stains([stain, frag, line, far], mm, (500, 700))
        self.assertEqual(sorted(o["kind"] for o in out), ["particle", "scratch", "stain"])
        self.assertEqual(len(next(o for o in out if o["kind"] == "stain")["xs"]), 5000 + 10)

    def test_kind_rules(self):
        self.assertEqual(W._kind(False, 4.0, 2.5, 1.2), "particle")
        self.assertEqual(W._kind(False, 55.0, 11.0, 1.4), "stain")      # 길이로
        self.assertEqual(W._kind(False, 45.9, 7.9, 1.3), "particle")    # 큰 마카 점 (21:15 권역 스캔)
        self.assertEqual(W._kind(False, 70.0, 8.5, 1.1), "particle")    # 크기 기준: 10 mm 미만은 넓어도 파티클 (11-17 정의)
        self.assertEqual(W._kind(False, 24.9, 9.8, 1.6), "particle")    # 붙은 파티클 두 개 (14:46 다이 29)
        self.assertEqual(W._kind(False, 40.0, 10.0, 1.5), "particle")   # 10 mm 까지 파티클 (11-17 정의)
        self.assertEqual(W._kind(False, 45.0, 10.5, 1.5), "stain")      # 10 mm 를 넘으면 스테인
        self.assertEqual(W._kind(False, 12.0, 20.0, 6.0), "scratch")    # 길쭉하면 길어도 스크래치
        self.assertEqual(W._kind(True, 5.0, 3.0, 1.0), "stain")         # 번짐 채널
        self.assertEqual(W._kind(True, 60.0, 40.0, 8.0), "scratch")     # 번짐 조각이 붙은 긴 스크래치
        self.assertEqual(W._kind(False, 870.0, 132.6, 4.4), "stain")    # 도넛 스테인의 넓은 호 (평균 폭 6.6 mm, 11-17)
        self.assertEqual(W._kind(False, 320.0, 101.4, 3.2), "scratch")  # 교차한 두 스크래치 (평균 폭 3.17 mm)
        self.assertEqual(W._kind(False, 56.3, 21.7, 2.6), "scratch")    # 굵은 스크래치: 가늘고 길다 (21.7²/56.3 = 8.4)
        self.assertEqual(W._kind(False, 41.0, 14.3, 2.9), "stain")      # 흐린 스테인: 10 mm 넘고 덜 가늘다 (5.0)

    def test_small_dot_at_die_corner_is_grid_residual(self):
        # 2026-11-16 17:41 실물: 사진 이음매에서 격자선 교차점이 어긋나 다이 모서리마다 작은 점이 오검출됐다
        front = dict(dark_only=True, sharp_z=W.DARK_ONLY_Z, min_area_mm2=W.MIN_DARK_AREA_MM2)   # 실물 투명 앞면 설정
        mm = P / 32
        a, b = R.die_slot(die_at(20))
        g, meta = clean_map(line=25)
        cv2.circle(g, (int(b * P + 2.5 * mm), int(a * P + P - 2.5 * mm)), 5, 60, -1)    # 왼쪽 아래 모서리에서 2.5 mm
        self.assertEqual(W.detect_array(g, meta, **front)["defects"], [])
        g, meta = clean_map(line=25)
        cv2.circle(g, (b * P + 85, a * P + 85), 5, 60, -1)                              # 같은 점이 가운데면 센다
        self.assertEqual([d["type"] for d in W.detect_array(g, meta, **front)["defects"]], ["particle"])

    def test_pieces_split_only_by_unseen_band_are_one(self):
        # 2026-11-16 실물: 카메라 반사 자리(검사 제외)가 스테인을 갈라 안쪽 조각이 따로 세어졌다
        mask = np.zeros((100, 100), bool)
        mask[40:60, 20:40] = mask[40:60, 50:70] = True          # 10 px 떨어진 두 조각
        unseen = np.ones((100, 100), bool)
        unseen[:, 40:50] = False                                  # 사이가 검사 제외 띠
        self.assertEqual(len(W._components(mask, 1, unseen, gap=6)), 1)
        self.assertEqual(len(W._components(mask, 1, np.ones((100, 100), bool), gap=6)), 2)   # 사이가 보이면 둘

    def test_coordinates_are_in_mm(self):
        g, meta = clean_map()
        d = die_at(20)
        a, b = R.die_slot(d)
        cv2.circle(g, (b * P + 85, a * P + 85), 4, 60, -1)
        f = W.detect_array(g, meta)["defects"][0]
        self.assertAlmostEqual(f["x_mm"], d["x_mm"], delta=1.5)
        self.assertAlmostEqual(f["y_mm"], d["y_mm"], delta=1.5)

    def test_too_few_dies_is_an_error(self):
        g, meta = clean_map()
        with self.assertRaises(ValueError):
            W.detect_array(np.zeros_like(g), meta)


class Quadrats(unittest.TestCase):
    def test_matches_die_value_layout(self):
        dies = die_layout()
        x = R.from_quadrats(dies, {(1, 0, 1): 1.0}, 2)
        r, c = R.die_slot(dies[0])
        self.assertEqual(x[0, r * 6 + 0, c * 6 + 3], 1.0)
        self.assertEqual(x[0, r * 6 + 3, c * 6 + 3], 0.0)
        self.assertEqual(x.shape, (3, R.SIZE, R.SIZE))


if __name__ == "__main__":
    unittest.main()
