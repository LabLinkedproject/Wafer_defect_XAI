"""
adc/geometry.py
패턴 근거 검토: 학습 모델(WaferNet) 판정을 검출된 결함의 실제 배치로 다시 따진다.
모델은 다이 격자(8×8) 지도만 보므로 웨이퍼 둘레의 입자 고리를 Scratch 로 읽거나(11-18 17:13·14:16),
다이 격자에 맞지 않는 도넛을 Local 로 읽었다(16:23). 그래서
  1) 결함 위치에 원을 맞춰 속이 빈 고리(둘레 240° 이상)를 찾는다 → 웨이퍼 둘레면 Edge-Ring, 아니면 Donut
  2) 검사 다이의 80 % 이상이 불량이면 Near-full, 3 다이 넘게 잇는 스크래치는 Scratch
  3) 위 근거로 설명되지 않은 불량 다이를 덩어리(Center·Edge-Loc·Local)와 흩어짐(Random)으로 가른다
  4) 근거와 맞지 않는 모델 라벨은 이유와 함께 뺀다 (스크래치가 없는데 Scratch 등)
모든 판단은 근거 문장으로 남겨 화면(XAI)에 보인다. 기준값은 WM-811K 패턴 정의와 11-17 영역 구분을 따른 어림값이다.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

WAFER_R_MM = 135.0
RING_MIN_POINTS = 6          # 고리로 볼 최소 결함 위치 수
RING_MIN_COVER_DEG = 270     # 고리가 둘러싼 각도 (가장 큰 빈틈을 뺀 나머지). 실물 고리 293~344°, 위아래 끊긴 고리 ≈300°,
                             # 세 군데 몰린 Edge-Loc+Local 이 248° 로 고리처럼 맞았다(11-17 16:51)
RING_TOL = 0.25              # 원에서 반지름의 이 비율 안이면 고리 위 (최소 5 mm). 입자 도넛은 띠 폭이 반지름의 40 %(11-18 16:23)
RING_SHARE = 0.5             # 남은 결함 위치 중 이 비율 이상이 고리 위여야 한다
RING_HOLLOW = 0.55           # Donut: 반지름의 이 비율 안쪽에는 결함이 없어야 한다
EDGE_RING_R = 0.70           # 웨이퍼 반지름의 이 비율 이상이고
EDGE_RING_CENTER_MM = 25.0   # 중심이 웨이퍼 중심에서 이 안이면 Edge-Ring
EDGE_RING_FILL = 0.6         # Edge-Ring: 엣지 다이의 이 비율 이상이 불량이고
EDGE_RING_INNER = 0.5        #   안쪽 다이 불량 비율이 엣지 비율의 이 배 이하 (흩어진 1단계 시료 11-16 이 고리로 맞았다)
RING_EXPLAIN = 0.4           # 찾은 고리에서 반지름의 이 비율 안의 결함은 그 고리의 일부로 본다 (띠 가장자리 점)
NEAR_FULL = 0.80             # 검사 다이 중 불량 비율
LONG_SCRATCH_DIES = 3        # 이 수 이상 다이를 잇는 스크래치는 Scratch 패턴
SCRATCH_MIN_MM = 10.0        # 모델의 Scratch 를 받아들일 최소 스크래치 길이
CLUSTER_SHARE = 0.7          # 남은 불량 다이의 이 비율 이상이 한 덩어리면 국소 패턴
CLUSTER_MAX_DIES = 8
CLUSTER_SPAN = 3.0           # 덩어리 안 다이 사이 최대 거리 (다이 칸)
RANDOM_SPAN = 3.0            # 흩어진 다이 사이 최대 거리가 이 이상이면 Random
PRIORITY = ["Near-full", "Edge-Ring", "Donut", "Scratch", "Center", "Edge-Loc", "Local", "Random"]
SPREAD = {"Center", "Edge-Loc", "Local", "Random"}          # 남은 다이 분석이 정하는 라벨


def _circle(p: np.ndarray):
    A = np.c_[2 * p, np.ones(len(p))]
    sol = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
    return float(sol[0]), float(sol[1]), float(math.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 0.0)))


def _cover(p: np.ndarray, cx: float, cy: float) -> float:
    a = np.sort(np.degrees(np.arctan2(p[:, 1] - cy, p[:, 0] - cx)) % 360)
    gaps = np.diff(np.r_[a, a[0] + 360])
    return float(360 - gaps.max())


def _is_edge(cx: float, cy: float, r: float) -> bool:
    return r >= EDGE_RING_R * WAFER_R_MM and math.hypot(cx, cy) <= EDGE_RING_CENTER_MM


def find_ring(p: np.ndarray, others: np.ndarray | None = None):
    """결함 위치 p(mm)에서 가장 많은 점을 지나는 고리. 반환 (cx, cy, r, 고리 위 점 표시) 또는 None.
    후보마다 고리 조건(둘레 270° 이상, 웨이퍼 둘레가 아니면 속이 빔)을 먼저 따진다. 반지름에 비례한 허용 폭 때문에
    큰 원이 떨어진 두 고리를 함께 지나며 점을 더 모았다. others: 속이 비었는지 볼 때 함께 볼 다른 결함 위치"""
    n = len(p)
    if n < RING_MIN_POINTS:
        return None
    allp = p if others is None or not len(others) else np.vstack([p, others])
    triples = list(itertools.combinations(range(n), 3))
    if len(triples) > 2000:
        rng = np.random.default_rng(0)
        triples = [tuple(rng.choice(n, 3, replace=False)) for _ in range(2000)]

    def check(cx, cy, r):
        if not 12 <= r <= WAFER_R_MM:
            return None
        on = np.abs(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r) <= max(5.0, RING_TOL * r)
        if on.sum() < RING_MIN_POINTS or _cover(p[on], cx, cy) < RING_MIN_COVER_DEG:
            return None
        if not _is_edge(cx, cy, r) and (np.hypot(allp[:, 0] - cx, allp[:, 1] - cy) < RING_HOLLOW * r).any():
            return None
        return on

    best, best_n = None, 0
    for t in triples:
        q = p[list(t)]
        if min(np.hypot(*(q[0] - q[1])), np.hypot(*(q[1] - q[2])), np.hypot(*(q[0] - q[2]))) < 8:
            continue
        try:
            cx, cy, r = _circle(q)
        except np.linalg.LinAlgError:
            continue
        on = check(cx, cy, r)
        if on is not None and on.sum() > best_n:
            best, best_n = (cx, cy, r, on), int(on.sum())
    if best is None:
        return None
    cx, cy, r, on = best
    for _ in range(3):                                    # 고리 위 점으로 다시 맞춘다 (조건을 지킬 때만)
        c2 = _circle(p[on])
        on2 = check(*c2)
        if on2 is None or on2.sum() < on.sum():
            break
        (cx, cy, r), on = c2, on2
    return cx, cy, r, on


def _dist(a: dict, b: dict) -> float:
    return math.hypot(a["row"] - b["row"], a["col"] - b["col"])


def _components(ids: set[int], by_id: dict) -> list[set[int]]:
    left, out = set(ids), []
    while left:
        comp, stack = set(), [left.pop()]
        while stack:
            i = stack.pop()
            comp.add(i)
            for j in list(left):
                if max(abs(by_id[i]["row"] - by_id[j]["row"]), abs(by_id[i]["col"] - by_id[j]["col"])) <= 1:
                    left.discard(j)
                    stack.append(j)
        out.append(comp)
    return sorted(out, key=len, reverse=True)


def evidence(defects: list[dict], defect_dies: set[int], inspected: set[int], dies: list[dict],
             labels: list[str] | None = None) -> dict:
    """defects: 검출 목록(다이·종류·obj·x_mm·y_mm·length_mm), defect_dies: 불량 판정 다이, dies: scoring.annotate 결과.
    반환 {"add": {라벨: 근거}, "drop": {모델 라벨: 기각 근거}, "explained": {다이: 설명한 라벨}}"""
    labels = labels or []
    by_id = {d["id"]: d for d in dies}
    add, drop, explained = {}, {}, {}
    bad = set(defect_dies) & set(by_id)
    if not bad:
        return {"add": add, "drop": drop, "explained": explained}

    near_full = len(bad) >= NEAR_FULL * max(len(inspected), 1)
    if near_full:
        add["Near-full"] = f"검사 다이 {len(inspected)}개 중 {len(bad)}개가 불량"
        for i in bad:
            explained.setdefault(i, "Near-full")

    # 1) 고리: 스크래치가 아닌 결함 위치에 원을 맞춘다. 두 개까지 (16:23 스테인 도넛 + 파티클 도넛). 전면이면 찾지 않는다
    pts = [f for f in defects if f["type"] != "scratch" and f["die"] in bad and not near_full]
    p = np.array([[f["x_mm"], f["y_mm"]] for f in pts], float).reshape(-1, 2)
    alive = np.ones(len(p), bool)
    for _ in range(2):
        idx = np.nonzero(alive)[0]
        ring = find_ring(p[idx], p[~alive]) if len(idx) else None
        if ring is None:
            break
        cx, cy, r, on = ring
        if on.sum() < RING_SHARE * len(idx):
            break
        members = idx[on]
        cover = _cover(p[members], cx, cy)
        edge = _is_edge(cx, cy, r)
        if edge:
            zone = {z: [i for i in inspected if i in by_id and by_id[i]["zone"] == z] for z in ("edge", "middle", "center")}
            inner = zone["middle"] + zone["center"]
            f_edge = sum(1 for i in zone["edge"] if i in bad) / max(len(zone["edge"]), 1)
            f_inner = sum(1 for i in inner if i in bad) / max(len(inner), 1)
            if f_edge < EDGE_RING_FILL or f_inner > EDGE_RING_INNER * f_edge:
                alive[members] = False
                continue
        if edge:
            name, why = "Edge-Ring", f"결함 {len(members)}곳이 웨이퍼 둘레 반지름 {r:.0f} mm 고리를 {cover:.0f}° 둘러쌈"
        else:
            name, why = "Donut", f"결함 {len(members)}곳이 반지름 {r:.0f} mm 고리를 {cover:.0f}° 둘러싸고 가운데가 비어 있음"
        add.setdefault(name, why)
        near = np.abs(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r) <= RING_EXPLAIN * r
        objs = {pts[k]["obj"] for k in members} | {pts[k]["obj"] for k in np.nonzero(near & alive)[0]}
        for f in pts:                                    # 고리에 걸친 결함(한 덩어리 스테인 고리)은 그 다이 전부를 설명한다
            if f["obj"] in objs:
                explained[f["die"]] = name
        if edge:                                 # 엣지 다이 칸은 원에서 조금 벗어나도 고리의 일부다 (다이 31·38 은 반지름 93 mm)
            for i in bad:
                if by_id[i]["zone"] == "edge":
                    explained.setdefault(i, name)
        alive[members] = False
        alive &= np.array([f["die"] not in explained for f in pts], bool)   # 고리로 설명된 다이의 나머지 점도 뺀다 (14:16 안쪽 가짜 도넛)

    # 2) 긴 스크래치
    spans: dict[int, set] = {}
    longest = 0.0
    for f in defects:
        if f["type"] == "scratch":
            spans.setdefault(f["obj"], set()).add(f["die"])
            longest = max(longest, float(f.get("length_mm", 0.0)))
    for dd in spans.values():
        for i in dd & bad:
            explained.setdefault(i, "Scratch")
    n_long = max((len(v) for v in spans.values()), default=0)
    if n_long >= LONG_SCRATCH_DIES:
        add["Scratch"] = f"전체 이미지 분석: 다이 {n_long}개를 잇는 스크래치"

    # 3) 구조 근거(고리·전면·스크래치)가 있을 때만, 그것으로 설명되지 않은 불량 다이를 덩어리(Center·Edge-Loc·Local)와
    #    흩어짐(Random)으로 가른다. 구조 근거가 없으면 퍼짐 판정은 모델에 맡긴다 (11-17 16:51 Edge-Loc+Local 을 Random 으로 덮었다)
    structural = bool(add)
    rest = bad - set(explained)
    spread_label = None
    if rest and structural:
        comps = _components(rest, by_id)
        big = comps[0]
        span = max((_dist(by_id[a], by_id[b]) for a in rest for b in rest), default=0.0)
        big_span = max((_dist(by_id[a], by_id[b]) for a in big for b in big), default=0.0)
        if len(big) >= 2 and len(big) >= CLUSTER_SHARE * len(rest) and len(big) <= CLUSTER_MAX_DIES and big_span <= CLUSTER_SPAN:
            zones = [by_id[i]["zone"] for i in big]
            if zones.count("center") >= 2:
                spread_label = "Center"
            elif zones.count("edge") * 2 > len(zones):
                spread_label = "Edge-Loc"
            else:
                spread_label = "Local"
            add.setdefault(spread_label, f"다이 {', '.join(map(str, sorted(big)))} 이 한 덩어리 ({spread_label})")
        elif len(rest) >= 3 and span >= RANDOM_SPAN:
            spread_label = "Random"
            add.setdefault("Random", f"다른 패턴으로 설명되지 않는 불량 다이 {len(rest)}개가 흩어져 있음")

    # 4) 근거와 맞지 않는 모델 라벨
    if "Scratch" in labels and longest < SCRATCH_MIN_MM:
        drop["Scratch"] = "검출된 스크래치가 없다 (결함 배치만 보고 모델이 선으로 읽음)"
    if "Edge-Loc" in labels and "Edge-Ring" in add and spread_label != "Edge-Loc":
        drop["Edge-Loc"] = "엣지 결함이 둘레를 거의 다 덮어 국소가 아니라 고리다"
    if structural:
        for name in SPREAD & set(labels):
            if name == spread_label:
                continue
            if not rest:
                drop[name] = "모든 불량 다이가 " + ", ".join(sorted(set(explained.values()))) + " 로 설명된다"
            elif spread_label:
                drop[name] = f"남은 불량 다이는 {spread_label} 로 보인다"
    return {"add": add, "drop": drop, "explained": explained}


def reconcile(model_labels: list[str], ev: dict) -> list[str]:
    """모델 라벨에서 기각된 것을 빼고 근거 라벨을 더해, 구조가 뚜렷한 순서로 정렬한다."""
    keep = [l for l in model_labels if l not in ev["drop"]]
    out = set(keep) | set(ev["add"])
    return sorted(out, key=lambda l: PRIORITY.index(l) if l in PRIORITY else len(PRIORITY))
