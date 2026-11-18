"""패턴 근거 검토(adc/geometry.py) 테스트. 실행: ./venv/bin/python -m unittest tests.test_geometry -v"""
import math
import unittest

import wafer_model
from adc import geometry as G
from adc import scoring

DIES = scoring.annotate(wafer_model.die_layout())
BY_ID = {d["id"]: d for d in DIES}
ALL = {d["id"] for d in DIES}
EDGE = {d["id"] for d in DIES if d["zone"] == "edge"}


def die_at(x, y):
    """mm 좌표가 들어가는 다이 번호 (다이 32 mm, 중심 기준)"""
    for d in DIES:
        if abs(x - d["x_mm"]) <= 16 and abs(y - d["y_mm"]) <= 16:
            return d["id"]
    return None


def ring(cx, cy, r, n, kind="particle", start=0.0, span=360.0):
    out = []
    for k in range(n):
        a = math.radians(start + span * k / n)
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        die = die_at(x, y)
        if die:
            out.append({"die": die, "type": kind, "obj": k + 1, "x_mm": x, "y_mm": y, "length_mm": 4.0})
    return out


def blobs(ids, kind="particle"):
    return [{"die": i, "type": kind, "obj": 100 + i, "x_mm": BY_ID[i]["x_mm"], "y_mm": BY_ID[i]["y_mm"], "length_mm": 5.0}
            for i in ids]


def run(defects, labels=(), inspected=ALL):
    dd = {f["die"] for f in defects}
    ev = G.evidence(defects, dd, set(inspected), DIES, list(labels))
    return ev, G.reconcile(list(labels), ev)


class RingEvidence(unittest.TestCase):
    def test_edge_ring_of_particles_overrules_scratch_and_edge_loc(self):
        ev, labels = run(ring(0, 0, 112, 40), labels=["Scratch", "Edge-Loc"])
        self.assertIn("Edge-Ring", ev["add"])
        self.assertIn("Scratch", ev["drop"])           # 검출된 스크래치가 없다
        self.assertIn("Edge-Loc", ev["drop"])          # 고리가 둘레를 거의 다 덮는다
        self.assertEqual(labels, ["Edge-Ring"])

    def test_ring_with_gaps_top_and_bottom_is_still_edge_ring(self):
        d = ring(0, 0, 110, 14, "stain", start=30, span=120) + ring(0, 0, 110, 14, "stain", start=210, span=120)
        ev, _ = run(d)
        self.assertIn("Edge-Ring", ev["add"])

    def test_small_ring_off_center_is_donut(self):
        ev, labels = run(ring(16, -16, 45, 24), labels=["Local"])
        self.assertIn("Donut", ev["add"])
        self.assertIn("Local", ev["drop"])             # 고리가 모든 결함 다이를 설명한다
        self.assertEqual(labels, ["Donut"])

    def test_filled_disc_is_not_a_ring(self):
        d = ring(0, 0, 45, 20) + blobs([18, 19, 26, 27])
        ev, _ = run(d)
        self.assertNotIn("Donut", ev["add"])

    def test_two_donuts_and_scratch(self):
        d = ring(-60, 0, 38, 20, "stain") + ring(60, -10, 40, 22) + [
            {"die": i, "type": "scratch", "obj": 900, "x_mm": BY_ID[i]["x_mm"], "y_mm": BY_ID[i]["y_mm"], "length_mm": 90.0}
            for i in (38, 39, 43, 44)]
        ev, labels = run(d, labels=["Scratch", "Local", "Random"])
        self.assertIn("Donut", ev["add"])
        self.assertEqual(set(labels), {"Donut", "Scratch"})


    def test_three_edge_clusters_are_not_a_ring(self):
        # 11-17 16:51: 위·왼쪽·오른쪽 아래 세 덩어리(Edge-Loc + Local)가 248° 고리로 맞았다
        d = ring(0, 0, 105, 8, start=60, span=60) + ring(0, 0, 105, 8, start=170, span=40) + ring(0, 0, 105, 8, start=290, span=30)
        ev, _ = run(d)
        self.assertNotIn("Edge-Ring", ev["add"])

    def test_thick_particle_donut(self):
        # 11-18 16:23: 반지름 25~42 mm 띠의 입자 도넛
        d = ring(50, 0, 26, 12, start=10) + ring(50, 0, 40, 16)
        ev, _ = run(d)
        self.assertIn("Donut", ev["add"])

    def test_no_inner_donut_inside_edge_ring(self):
        # 11-18 14:16: 엣지 다이 안쪽 가장자리 점들이 두 번째 고리(가짜 도넛)로 맞았다
        d = ring(0, 0, 110, 40) + ring(0, 0, 86, 24)
        ev, _ = run(d)
        self.assertIn("Edge-Ring", ev["add"])
        self.assertNotIn("Donut", ev["add"])


    def test_dense_random_scatter_is_not_edge_ring(self):
        # 11-16 1단계 파티클 시료: 엣지 12 / 안쪽 13 다이에 흩어진 결함이 둘레 고리로 맞았다
        ids = [6, 8, 10, 13, 14, 15, 16, 18, 19, 21, 22, 23, 29, 30, 31, 32, 34, 35, 36, 38, 39, 40, 41, 42, 43]
        ev, _ = run(blobs(ids))
        self.assertNotIn("Edge-Ring", ev["add"])


class FillAndLeftovers(unittest.TestCase):
    def test_near_full(self):
        ev, labels = run(blobs(ALL - {1, 2, 43}), labels=["Random"])
        self.assertIn("Near-full", ev["add"])
        self.assertIn("Near-full", labels)

    def test_scratch_plus_local_cluster(self):
        sc = [{"die": i, "type": "scratch", "obj": 900, "x_mm": BY_ID[i]["x_mm"], "y_mm": BY_ID[i]["y_mm"], "length_mm": 180.0}
              for i in (3, 10, 18, 27, 35, 42)]
        ev, labels = run(sc + blobs([24, 25, 32]), labels=["Scratch", "Random"])
        self.assertIn("Local", ev["add"])
        self.assertIn("Random", ev["drop"])
        self.assertEqual(set(labels), {"Scratch", "Local"})

    def test_edge_ring_plus_scattered_random(self):
        d = blobs(EDGE) + blobs([11, 26, 33], "stain")
        ev, labels = run(d, labels=["Random", "Scratch"])
        self.assertIn("Edge-Ring", ev["add"])
        self.assertIn("Random", labels)
        self.assertNotIn("Scratch", labels)

    def test_center_cluster_keeps_model_center(self):
        ev, labels = run(blobs([18, 19, 26, 27]), labels=["Center"])
        self.assertEqual(labels, ["Center"])
        self.assertEqual(ev["drop"], {})

    def test_clean_wafer(self):
        ev, labels = run([])
        self.assertEqual((ev["add"], ev["drop"], labels), ({}, {}, []))


if __name__ == "__main__":
    unittest.main()
