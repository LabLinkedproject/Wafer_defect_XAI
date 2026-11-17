"""
wafer_detect.py
합성 웨이퍼 맵에서 결함을 찾는다. 다이 44개는 모양이 같으므로 라벨 없이 다이끼리 비교할 수 있다.

표준 기준 다이 방식을 따른다.
1. 측정한 격자로 맵을 곧게 펴고 다이마다 잘라낸다.
2. 다이를 서로 미세 정렬한 뒤 화소별 중앙값으로 기준 다이를 만든다. 결함은 다이마다 위치가 달라 중앙값에서 사라진다.
3. 다이와 기준의 차이를 강건한 분산(MAD)으로 나누어, 기준값을 스캔마다 자동으로 정한다.

한계: 여러 다이의 같은 위치에 반복되는 결함은 중앙값에 흡수되어 검출되지 않는다.
웨이퍼 전체를 덮는 Near-full 도 기준이 오염되어 약하다.
"""
from __future__ import annotations
import json
import math
import os

import cv2
import numpy as np

import wafer_raster as R
from wafer_model import WAFER_MODEL, die_layout

SCRIBE_MARGIN = 12            # 스크라이브 라인은 정렬 오차에 민감해 다이 가장자리는 비교에서 뺀다
MIN_DIE_COVERAGE = 0.85       # 다이들의 보통 촬영 비율 대비 이보다 적게 찍힌 다이는 검사하지 않는다 (가장자리 다이는 이웃 사진이 덮는 몫이 없어 0.87 배 안팎)
MIN_DIE_COVERAGE_ABS = 0.5    # 그래도 다이 면적의 절반도 안 찍힌 다이는 뺀다
MAX_SHIFT = 8                 # 미세 정렬 허용 폭 (px)
GOLDEN_MAX_SHIFT_MM = 4.0     # 기준 웨이퍼 같은 다이에 맞출 때 허용 폭 (웨이퍼를 다시 놓은 어긋남)
SHARP_Z = 7.0                 # 선명한 결함 판정선
STAIN_Z = 7.0                 # 번짐 결함 판정선
MIN_CONTRAST = 0.05           # 맵 합성 잔차보다 작은 밝기 차이는 결함으로 보지 않는다
MIN_STAIN_CONTRAST = 0.02
MIN_SHARP_AREA = 5            # px
MIN_STAIN_RATIO = 0.012       # 다이 면적 대비
STAIN_MIN_LENGTH_MM = 10.0    # 크기 기준 (2026-11-17 정의): 10 mm 까지 파티클, 넘으면 스테인(흐림·다이 격자 침범)
UNSEEN_GAP_MM = 2.0           # 검사 제외 띠 양쪽에서 이만큼 뻗어 조각을 잇는다. 2026-11-16 실물 두 스캔에서 0·1.5·2·2.5·3 mm 중 개수 정확도 최고
SCRATCH_ELONGATION = 3.5
SCRATCH_MIN_MM = 4.0
SCRATCH_MAX_WIDTH_MM = 3.3    # 스크래치 평균 폭(면적/길이) 상한. 실물 스크래치 ≤ 3.17(교차), 도넛 스테인의 호 ≥ 3.7 mm (11-17)
SCRATCH_SLENDER = 7.0         # 10 mm 넘는 결함은 길이²/면적(가는 정도)이 이 이상이면 스크래치. 실물 스테인 ≤ 5.3, 스크래치 ≥ 8.3
SCRATCH_CORRIDOR_MM = 2.5     # 스크래치 연장선에서 이 거리 안의 작은 조각은 그 스크래치의 꼬리다
STAIN_HALO_MM = 5.0           # 스테인에서 이 거리 안에 픽셀 절반 이상이 있는 조각은 같은 오염 (흐린 경계)
SCRATCH_TAIL_MM = 8.0         # 스크래치 끝에서 이만큼 더 나간 곳까지 꼬리로 본다
HEAVY_QUADRAT_AREA = 0.05     # 구역 면적 중 이 비율 이상이면 heavy
HEAVY_QUADRAT_COUNT = 3
LINE_GUARD_MM = 5.0           # 다이 경계선에서 이 거리 안의 가는 성분은 격자선 잔차
LINE_RESIDUAL_ELONGATION = 3.0
LINE_SHADOW_MM = 1.5          # 밝은 격자선에서 이 거리 안에
LINE_SHADOW_FRAC = 0.75       # 픽셀 이 비율 이상이 있는
LINE_SHADOW_WIDTH_MM = 1.2    # 이 폭 이하의 가는 성분은 격자선 그림자 잔차
CORNER_GUARD_MM = 4.0         # 다이 모서리(두 경계선 모두)에서 이 거리 안의 작은 점은 격자선 교차점 잔차
CORNER_MAX_AREA_MM2 = 5.0     # 2026-11-16 17:41: 사진 이음매의 교차점 어긋남이 1~4.5 mm² 점 9개로 남았다
EDGE_BAND_MM = 4.0            # 다이 가장자리 띠. 안쪽 24×24 mm 가 중심부 (docs/specs/die-verdict.md)
DEFECT_AREA_RATIO = 0.05      # 다이 안 결함 면적이 이 비율 이상이면 위치와 상관없이 불량
# 가장자리 띠에만 있을 때 종류별 기준 (docs/specs/inspection-criteria.md 3절)
CENTER_MIN_MM2 = 1.0          # 회로 영역(중심부)에 이 면적 이상 들어와야 침범 (최소 검출 크기와 같다)
EDGE_SCRATCH_MM = 16.0        # 띠 안 스크래치가 다이 한 변의 절반 이상이면 불량 (실링을 따라 긁힘)
EDGE_STAIN_RATIO = 0.025      # 띠 안 스테인이 다이 면적의 2.5 % 이상이면 불량 (넓은 오염)
GDBN_MIN = 5                  # 정상 다이라도 8 방향 이웃 중 불량이 이 수 이상이면 재검사 (GDBN 현지화)
FULL_DARK = 0.2               # 합성 전체 이미지에서 주변보다 이 비율 이상 어두우면 결함 몸체 (스크래치·스테인 10 % 지점 0.21~0.24)
GROW_LIMIT_MM = 4.0           # 다이 비교 씨앗에서 이 거리 안에서만 몸체를 따라간다 (검사한 다이 안의 반사 띠·스크라이브)
GROW_FAR_MM = 12.0            # 검사 못 한 다이 자리에서는 더 멀리 따라간다 (11-17: 다이 23 을 지나는 스크래치 꼬리).
                              # 검사한 다이 안까지 12 mm 로 늘리자 스테인이 반사 띠를 따라 길어져 스크래치가 됐다
THIN_WIDTH_MM = 1.5           # 씨앗의 평균 폭(면적/길이)이 이하면 선 모양 (스크래치 조각 1.0~1.1, 스테인 ≥ 2.7 mm)
RIM_MARGIN_MM = 8.0           # 웨이퍼 테두리에서 이 거리 안은 멀리 따라가지 않는다 (테두리 어둠)
GROW_RATIO = 0.6              # 이중 문턱의 낮은 쪽 (높은 문턱 대비)
DARK_BACKGROUND = 120          # 격자선 밝기 정보가 없는 예전 맵: 웨이퍼 안쪽 중앙값이 이보다 어두우면 투명 앞면
CHROMA_GAIN = 3.0             # R−B 차이를 밝기 영상 범위로 늘리는 배율
DARK_ONLY_Z = 6.0             # 어두운 것만 셀 때의 판정선
MIN_DARK_AREA_MM2 = 1.0       # 투명 앞면 고정(surface="front") 때 최소 크기. 2026-11-14 근접 촬영에서 마카 11곳 11/11,
                              # 결함 없는 스캔 오검출 0 (어두운 것만 세면 먼지·흠집은 남지 않았다)
CHROMA_Z = 6.0                # 색 차이 채널의 선명 결함 판정선 (흑백은 SHARP_Z)
MIN_CHROMA_AREA_MM2 = 8.0     # 색 차이 채널은 잡음이 거칠어 이보다 작은 점은 결함으로 세지 않는다 (지름 약 3 mm)
OUTLIER_RATIO = 0.25          # 국소 배경에서 이만큼 벗어난 픽셀은 배경 평균에서 뺀다
MAX_DARK_DEFECT_MM = 12        # 이보다 작은 어두운 구멍은 미촬영이 아니라 검은 결함
BACKGROUND_MAX = 26           # 합성 맵의 미촬영 영역 밝기 (18~20)


def _seen(gray: np.ndarray, pitch_px: float) -> np.ndarray:
    """촬영된 영역. 맵의 미촬영 부분은 어둡게 칠해지지만 검은 결함도 그만큼 어둡다.
    촬영 영역 안에 갇힌 작은 어두운 구멍은 결함으로 보고 촬영된 것으로 친다."""
    seen = gray > BACKGROUND_MAX
    n, lab, stats, _ = cv2.connectedComponentsWithStats((~seen).astype(np.uint8), connectivity=8)
    h, w = gray.shape
    limit = (MAX_DARK_DEFECT_MM * pitch_px / WAFER_MODEL["pitch_mm"]) ** 2
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area <= limit and x > 0 and y > 0 and x + bw < w and y + bh < h:
            seen[lab == i] = True
    return seen


def rectify(gray: np.ndarray, meta: dict, seen: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """맵을 웨이퍼 중심·격자 기준으로 곧게 편다. 반환: 영상, 유효 영역, 다이 한 칸 px
    seen: 맵 합성이 저장한 촬영 범위 마스크. 없으면 밝기로 추정한다(웨이퍼가 밝게 찍힐 때만 맞다)."""
    pitch = meta["grid"]["pitch_px"]
    P = int(round(pitch))
    side = R.GRID_DIES * P
    h = side / 2
    k = pitch / P
    a = math.radians(-meta["grid"]["angle_deg"])
    c, s = math.cos(a) * k, math.sin(a) * k
    cx, cy = meta["wafer"]["center_px"]
    M = np.array([[c, s, cx - h * (c + s)], [-s, c, cy - h * (c - s)]], np.float32)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    img = cv2.warpAffine(gray.astype(np.float32), M, (side, side), flags=flags)
    seen = _seen(gray, pitch) if seen is None else seen > 127
    seen = cv2.warpAffine(seen.astype(np.float32), M, (side, side), flags=flags) > 0.99
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    disc = np.hypot(xx - h + 0.5, yy - h + 0.5) < meta["wafer"]["radius_px"] / k * 0.985
    valid = cv2.erode((seen & disc).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0   # 촬영 경계는 합성 잔차가 크다
    return img, valid, P



def crop_dies(img, valid, P, dies):
    """검사할 수 있는 다이만 잘라낸다.
    근접 촬영(2026-11-14)은 모든 다이의 가운데(카메라 반사)와 네 모서리(조명 반사 고리)가 똑같이 비어
    다이 면적 기준 92 % 를 넘는 다이가 없다. 그래서 다이들의 보통 촬영 비율을 기준으로 덜 찍힌 다이만 뺀다."""
    out = {}
    cover = {}
    for d in dies:
        r, c = R.die_slot(d)
        cover[d["id"]] = float(valid[r * P:(r + 1) * P, c * P:(c + 1) * P].mean())
    seen = [v for v in cover.values() if v >= MIN_DIE_COVERAGE_ABS]
    need = max(MIN_DIE_COVERAGE_ABS, MIN_DIE_COVERAGE * float(np.median(seen))) if seen else 1.0
    for d in dies:
        r, c = R.die_slot(d)
        y, x = r * P, c * P
        v = valid[y:y + P, x:x + P]
        if cover[d["id"]] >= need:
            out[d["id"]] = {"die": d, "y": y, "x": x, "img": img[y:y + P, x:x + P].copy(), "valid": v.copy()}
    return out


def _flatten(a: np.ndarray, sigma: float) -> np.ndarray:
    """국소 배경으로 나눈다. 아주 어두운 결함이 배경 평균을 끌어내려 주변에 가짜 테두리가 생기지 않도록
    배경에서 크게 벗어난 픽셀을 빼고 다시 평균한다."""
    base = cv2.GaussianBlur(a, (0, 0), sigma)
    keep = (np.abs(a - base) < OUTLIER_RATIO * np.maximum(base, 1.0)).astype(np.float32)
    w = cv2.GaussianBlur(keep, (0, 0), sigma)
    base = np.where(w > 0.05, cv2.GaussianBlur(a * keep, (0, 0), sigma) / np.maximum(w, 1e-6), base)
    return a / np.maximum(base, 1.0)


def golden_crops(gray: np.ndarray, meta: dict, seen: np.ndarray | None = None) -> dict:
    """기준 웨이퍼(결함 없는 순정 상태) 맵을 다이별로 잘라 둔다. 같은 위치의 다이끼리 비교하는 데 쓴다."""
    img, valid, P = rectify(gray, meta, seen)
    out = {}
    for d in die_layout():
        r, c = R.die_slot(d)
        v = valid[r * P:(r + 1) * P, c * P:(c + 1) * P]
        if v.mean() >= MIN_DIE_COVERAGE_ABS:
            out[d["id"]] = {"img": img[r * P:(r + 1) * P, c * P:(c + 1) * P].copy(), "valid": v.copy(), "P": P}
    return out


def _golden_for(golden: dict | None, die_id: int, P: int):
    g = (golden or {}).get(die_id)
    if g is None:
        return None
    if g["P"] != P:                                   # 두 맵의 격자 간격이 1 px 다를 수 있다
        g = {"img": cv2.resize(g["img"], (P, P)), "valid": cv2.resize(g["valid"].astype(np.uint8), (P, P)) > 0, "P": P}
    return g


def align(crops: dict, P: int, golden: dict | None = None):
    """다이를 기준에 맞춘다. 합성 잔차가 1~2 px 남기 때문에 필요하다.
    기준 웨이퍼가 있으면 같은 다이에 맞춘다. 웨이퍼를 다시 놓으면 몇 mm 어긋날 수 있어 더 넓게 찾는다."""
    win = cv2.createHanningWindow((P, P), cv2.CV_32F)
    stack = np.stack([c["img"] for c in crops.values()])
    med = _flatten(np.median(stack, 0), P / 6)
    for c in crops.values():
        g = _golden_for(golden, c["die"]["id"], P)
        ref = _flatten(g["img"], P / 6) if g is not None else med
        lim = GOLDEN_MAX_SHIFT_MM * P / 32 if g is not None else MAX_SHIFT
        (dx, dy), _ = cv2.phaseCorrelate(ref.astype(np.float32), _flatten(c["img"], P / 6).astype(np.float32), win)
        dx, dy = float(np.clip(dx, -lim, lim)), float(np.clip(dy, -lim, lim))
        M = np.array([[1, 0, -dx], [0, 1, -dy]], np.float32)
        c["img"] = cv2.warpAffine(c["img"], M, (P, P), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        c["valid"] = cv2.warpAffine(c["valid"].astype(np.uint8), M, (P, P), flags=cv2.INTER_NEAREST) > 0
        c["shift"] = (round(dx, 2), round(dy, 2))


def robust_sigma(a: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def analyse(crops: dict, P: int, golden: dict | None = None) -> dict:
    """다이별 z 지도를 만든다. 기준은 같은 웨이퍼 다이들의 중앙값이고, 기준 웨이퍼가 있으면 그 웨이퍼의 같은 다이다.
    같은 다이끼리 비교하면 그 자리에 늘 있는 아크릴 흠집·카메라 반사·조명 번짐이 상쇄된다(2026-11-16)."""
    stack = np.stack([c["img"] for c in crops.values()])
    ref = np.median(stack, 0)
    inner = np.zeros((P, P), bool)
    inner[SCRIBE_MARGIN:-SCRIBE_MARGIN, SCRIBE_MARGIN:-SCRIBE_MARGIN] = True
    ref_sharp_med = _flatten(ref, P / 6)
    ref_soft_med = cv2.GaussianBlur(ref, (0, 0), P / 14) / np.median(ref[inner])
    sharp_all, soft_all = [], []
    for c in crops.values():
        g = _golden_for(golden, c["die"]["id"], P)
        m = c["valid"] & inner
        if g is not None:
            m = m & g["valid"]
            ref_sharp = _flatten(g["img"], P / 6)
            ref_soft = cv2.GaussianBlur(g["img"], (0, 0), P / 14) / max(float(np.median(g["img"][m])) if m.any() else 1.0, 1.0)
        else:
            ref_sharp, ref_soft = ref_sharp_med, ref_soft_med
        c["mask"] = m
        c["golden"] = g is not None
        c["sharp"] = _flatten(c["img"], P / 6) - ref_sharp
        c["soft"] = cv2.GaussianBlur(c["img"], (0, 0), P / 14) / max(float(np.median(c["img"][m])) if m.any() else 1.0, 1.0) - ref_soft
        sharp_all.append(c["sharp"][m])
        soft_all.append(c["soft"][m])
    s1, s2 = robust_sigma(np.concatenate(sharp_all)), robust_sigma(np.concatenate(soft_all))
    return {"sharp_sigma": s1, "soft_sigma": s2, "reference": ref}


def _components(mask: np.ndarray, min_area: int, valid: np.ndarray | None = None, gap: int = 0):
    """연결 성분. valid 와 gap 을 주면 검사 제외 영역(카메라 반사 자리)만 사이에 둔 조각은 하나로 합친다.
    2026-11-16 실물: U 자 반사 마스크가 스테인을 갈라 안쪽 조각이 다이마다 같은 자리에 따로 세어졌다.
    shortcut: 보이지 않는 영역 안에서 gap*2 이내로 떨어진 두 결함도 하나로 센다. 사이를 볼 수 없어서다."""
    grown = mask
    if valid is not None and gap > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
        grown = mask | (~valid & (cv2.dilate(mask.astype(np.uint8), k) > 0))
    n, lab = cv2.connectedComponents(grown.astype(np.uint8), connectivity=8)
    lab[~mask] = 0
    areas = np.bincount(lab.ravel(), minlength=n)
    return [(lab == i, areas[i]) for i in range(1, n) if areas[i] >= min_area]


def _grid_residual(xs, ys, w, h, area_mm2, P, mm, line_dist=None) -> bool:
    """다이 경계선 옆 띠 안에 갇힌 길쭉한 성분과 모서리의 작은 점은 격자선 위치 차이가 남은 것으로 본다.
    격자선이 다이마다 조금씩 어긋나 있으면 기준 다이와 비교할 때 선이 결함처럼 남는다."""
    guard = LINE_GUARD_MM / mm
    near_x = min(xs.min(), P - 1 - xs.max()) < guard and np.ptp(xs) < guard     # 세로 경계선 옆
    near_y = min(ys.min(), P - 1 - ys.max()) < guard and np.ptp(ys) < guard     # 가로 경계선 옆
    along = max(w, h) >= LINE_RESIDUAL_ELONGATION * max(min(w, h), 1.0)     # 경계선 방향으로 길쭉함
    # 사진 이음매에서 교차점이 어긋나면 모서리에 작은 점이 남는다.
    # shortcut: 모서리 4 mm 안의 5 mm² 미만 실제 파티클도 빠진다. 맵 합성 정렬이 나아지면 없앤다
    cg = CORNER_GUARD_MM / mm
    cx, cy = xs.mean(), ys.mean()
    corner = min(cx, P - 1 - cx) < cg and min(cy, P - 1 - cy) < cg and area_mm2 < CORNER_MAX_AREA_MM2
    # 원점이 어긋나 격자선이 다이 안쪽에 들어오면 밝은 선 옆 그림자가 L 자 '스크래치'로 남는다 (11-17 16:29).
    # 가는 성분이 거의 전부 밝은 격자선에 붙어 있으면 잔차. 실제 스크래치는 선과 건너는 곳에서만 닿는다
    # (실물: 잔차 0.78~1.00, 경계 근처 실제 스크래치 조각 ≤ 0.54)
    on_line = (line_dist is not None and area_mm2 / max(max(w, h) * mm, 1e-6) <= LINE_SHADOW_WIDTH_MM
               and float(np.mean(line_dist[ys, xs] * mm <= LINE_SHADOW_MM)) >= LINE_SHADOW_FRAC)
    return corner or on_line or (near_x or near_y) and (along or area_mm2 < 1.0)


def _kind(soft: bool, area_mm2: float, long_mm: float, elong: float) -> str:
    """결함 종류 (docs/specs/die-verdict.md). 주 기준은 크기다: 길쭉하면 스크래치, 길이 10 mm 를 넘으면 스테인,
    10 mm 까지는 파티클. 흐린 번짐(번짐 채널이 과반)은 크기와 상관없이 스테인.
    면적은 쓰지 않는다: 반사 마스크에 반쯤 가린 스테인은 면적만 줄고(26.8 mm²) 길이는 남았다(14.8 mm)."""
    slender = long_mm * long_mm / max(area_mm2, 1e-6)
    thin = area_mm2 / max(long_mm, 1e-6) <= SCRATCH_MAX_WIDTH_MM       # 스크래치는 가는 선이다
    if long_mm >= SCRATCH_MIN_MM and thin and (elong >= SCRATCH_ELONGATION
                                               or (long_mm > STAIN_MIN_LENGTH_MM and slender >= SCRATCH_SLENDER)):
        return "scratch"          # 다이를 넘어 이은 스크래치에 번짐 조각이 붙어도 스크래치 (2026-11-16 18:20)
    if soft:
        return "stain"
    return "stain" if long_mm > STAIN_MIN_LENGTH_MM else "particle"


def extract(crops: dict, P: int, noise: dict, min_area_mm2: float = 0.0, sharp_z: float = SHARP_Z,
            dark_only: bool = False):
    """z 지도에서 결함 조각을 뽑아 다이마다 c["pixels"](전부)와 c["soft_px"](번짐 채널)에 칠한다.
    종류와 판정은 link_defects 가 웨이퍼 단위로 이은 뒤에 정한다."""
    mm = WAFER_MODEL["pitch_mm"] / P
    for c in crops.values():
        thr = max(sharp_z * noise["sharp_sigma"], MIN_CONTRAST)
        # dark_only: 주변보다 어두워진 것만 결함으로 센다. 투명면에서 표면 흠집·얼룩은 희게, 붙은 결함은 검게 보인다
        z = np.clip(-c["sharp"], 0, None) if dark_only else np.abs(c["sharp"])
        # 이중 문턱: 확실한 픽셀에서 출발해 절반 문턱까지 이어 붙인다. 대비가 약한 면에서 결함 하나가
        # 조각나는 것을 막고, 떨어져 있는 두 결함은 합치지 않는다
        weak = cv2.morphologyEx(((z > thr * GROW_RATIO) & c["mask"]).astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8))
        n, lab = cv2.connectedComponents(weak, connectivity=8)
        seeds = np.unique(lab[(z > thr) & c["mask"] & (weak > 0)])
        sharp = np.isin(lab, seeds[seeds > 0])
        # 번짐도 같은 규칙: 투명면의 하얀 번짐(표면 얼룩·조명 고리 번짐)은 결함이 아니다 (2026-11-14 근접 촬영 오검출 4건)
        soft_z = np.clip(-c["soft"], 0, None) if dark_only else np.abs(c["soft"])
        soft = (soft_z > max(STAIN_Z * noise["soft_sigma"], MIN_STAIN_CONTRAST)) & c["mask"]
        stain_min = int(MIN_STAIN_RATIO * P * P)
        c["pixels"] = np.zeros((P, P), bool)
        c["soft_px"] = np.zeros((P, P), bool)
        # 번짐 지도는 흐림을 거치므로 선명한 결함 둘레에도 반응한다. 그 둘레는 번짐 결함에서 뺀다
        halo = cv2.dilate(sharp.astype(np.uint8), np.ones((2 * (P // 14) + 1,) * 2, np.uint8)) > 0
        for region, kind_min, is_stain in ((sharp, MIN_SHARP_AREA, False), (soft & ~halo, stain_min, True)):
            for comp, area in _components(region, kind_min, c["mask"], int(round(UNSEEN_GAP_MM / mm))):
                if area * mm * mm < min_area_mm2:
                    continue
                ys, xs = np.where(comp)
                pts = np.column_stack([xs, ys]).astype(np.float32)
                (_, _), (w, h), _ = cv2.minAreaRect(pts)
                long_mm, short = max(w, h) * mm, max(min(w, h), 1.0)
                elong = max(w, h) / short
                if "line_dist" not in c:                    # 밝은 격자선(다이 안 상위 2 %)까지 거리, 필요할 때 한 번
                    bright = (c["img"] > np.percentile(c["img"], 98)).astype(np.uint8)
                    c["line_dist"] = cv2.distanceTransform(1 - bright, cv2.DIST_L2, 5)
                if not is_stain and _grid_residual(xs, ys, w, h, area * mm * mm, P, mm, c["line_dist"]):
                    continue
                c["pixels"] |= comp
                if is_stain:
                    c["soft_px"] |= comp


def _within(mask: np.ndarray, r: float) -> np.ndarray:
    """mask 에서 거리 r px 안. 큰 원형 팽창(지름 325 px)은 전체 이미지에서 너무 무거워 거리 변환으로 잰다."""
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) <= r


def _absorb_tails(objs: list[dict], mm: float) -> list[dict]:
    """스크래치 연장선 위의 작은 조각(파티클·스테인으로 분류된 꼬리)을 그 스크래치에 흡수한다.
    스크래치 패턴에 파티클·스테인이 겹쳐 나오면 안 된다 (2026-11-17 사용자 지시). objs: fx·fy(모양), xs·ys(씨앗), kind."""
    cor, tail = SCRATCH_CORRIDOR_MM / mm, SCRATCH_TAIL_MM / mm
    keep = [o for o in objs if o["kind"] == "scratch"]
    for s in keep:
        p = np.column_stack([s["fx"], s["fy"]]).astype(np.float64)
        s["_c"] = p.mean(0)
        s["_u"] = np.linalg.svd(p - s["_c"], full_matrices=False)[2][0]
        t = (p - s["_c"]) @ s["_u"]
        s["_t"] = (t.min(), t.max())
    out = []
    for o in objs:
        if o["kind"] == "scratch":
            continue
        q = np.column_stack([o["fx"], o["fy"]]).astype(np.float64)
        host = None
        for s in keep:
            d = q - s["_c"]
            t = d @ s["_u"]
            off = np.abs(d @ np.array([-s["_u"][1], s["_u"][0]]))
            if off.max() <= cor and t.min() >= s["_t"][0] - tail and t.max() <= s["_t"][1] + tail:
                host = s
                break
        if host is None:
            out.append(o)
            continue
        for k in ("fx", "fy", "xs", "ys"):
            host[k] = np.concatenate([host[k], o[k]])
    for s in keep:
        t = (np.column_stack([s["fx"], s["fy"]]) - s["_c"]) @ s["_u"]
        s["long_mm"] = float(np.ptp(t)) * mm
        for k in ("_c", "_u", "_t"):
            s.pop(k)
    return keep + out


def _absorb_into_stains(objs: list[dict], mm: float, shape: tuple) -> list[dict]:
    """스테인 가까이(STAIN_HALO_MM)에 픽셀 절반 이상이 있는 조각은 그 스테인의 일부로 흡수한다.
    스테인은 경계가 흐리고 모양이 변칙적인 넓은 오염이라 붓질 사이가 끊겨 조각으로 잡힌다
    (11-17 도넛 스테인: 고리 조각이 파티클 17·스크래치 7 로 나왔다)."""
    stains = [o for o in objs if o["kind"] == "stain"]
    if not stains:
        return objs
    lab = np.zeros(shape, np.int32)
    for k, o in enumerate(stains, 1):
        lab[o["fy"], o["fx"]] = k
    dist, near = cv2.distanceTransformWithLabels((lab == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    # 거리 변환의 라벨은 가장 가까운 0 픽셀(=스테인 픽셀) 번호 → 그 픽셀의 스테인 번호로 바꾼다
    ys, xs = np.nonzero(lab)
    owner = np.zeros(near.max() + 1, np.int32)
    owner[near[ys, xs]] = lab[ys, xs]
    out = list(stains)
    for o in objs:
        if o["kind"] == "stain":
            continue
        close = dist[o["fy"], o["fx"]] * mm <= STAIN_HALO_MM
        if close.mean() < 0.5:
            out.append(o)
            continue
        host = stains[np.bincount(owner[near[o["fy"], o["fx"]][close]]).argmax() - 1]
        for k in ("fx", "fy", "xs", "ys"):
            host[k] = np.concatenate([host[k], o[k]])
    return out


def die_verdict(center_hit: bool, area_ratio: float, scratch_edge_mm: float = 0.0, stain_ratio: float = 0.0) -> str:
    """다이 판정 (docs/specs/inspection-criteria.md). 회로 영역(중심부)에 닿거나 결함이 많으면 불량,
    가장자리 띠에만 있으면 종류별 기준, 그 밖은 재검사(사람이 정한다)."""
    if center_hit or area_ratio >= DEFECT_AREA_RATIO:
        return "defect"
    if scratch_edge_mm >= EDGE_SCRATCH_MM or stain_ratio >= EDGE_STAIN_RATIO:
        return "defect"
    return "recheck" if area_ratio > 0 else "normal"


def apply_gdbn(verdicts: dict) -> dict:
    """주변 불량 속의 정상 다이(GDBN)를 재검사로 올린다. verdicts: {다이 id: {"verdict": ...}} (제자리 수정).
    검사한 다이만 이웃으로 센다."""
    pos = {(d["row"], d["col"]): d["id"] for d in die_layout()}
    rc = {i: rc for rc, i in pos.items()}
    for i, v in verdicts.items():
        if v["verdict"] != "normal" or i not in rc:
            continue
        r, c = rc[i]
        n = sum(1 for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
                and verdicts.get(pos.get((r + dr, c + dc)), {}).get("verdict") == "defect")
        if n >= GDBN_MIN:
            v.update(verdict="recheck", gdbn=n)
    return verdicts


def link_defects(crops: dict, P: int, img: np.ndarray, follow_dark: bool = True,
                 ref: np.ndarray | None = None) -> tuple[list[dict], dict]:
    """결함 분석은 합성 전체 이미지, 양불 판정은 다이 단위로 나눈다 (docs/specs/die-verdict.md).
    1. 다이 비교 조각(오검출이 적다)을 씨앗으로 웨이퍼 좌표에 놓는다.
    2. 합성 전체 이미지에서 씨앗 가까이의 어두운 몸체를 따라가 다이 경계·반사 띠로 끊긴 결함을 하나로 잇는다
       (2026-11-16 18:20: 스크래치 하나가 다이마다 스크래치·스테인·파티클로 따로 나왔다).
    3. 종류는 이은 결함에서 전체 이미지상 실제로 어두운 부분의 모양으로 정한다. 다이 비교의 격자선 잔차는 빠진다.
    4. 판정은 다이 단위 분석으로만 낸다: 다이에 걸친 부분은 그 다이의 비교 검출(씨앗) 픽셀만 센다.
       전체 이미지 몸체는 잇기와 종류에만 쓴다 (몸체가 이웃 다이로 번져 오검출된 다이 2·28, 2026-11-16).
       반환: 다이별 조각 목록(같은 결함은 obj 가 같다), {다이: 판정}.
    ref: 기준 다이(중앙값). 모든 다이에 같은 자리로 있는 어둠(렌즈 점 등)은 몸체로 따라가지 않는다.
    follow_dark: 2 를 할지. 격자선이 어두운 불투명면에서는 선을 따라 번지므로 투명 앞면(dark_only)에서만 한다."""
    shape = img.shape[:2]
    mm = WAFER_MODEL["pitch_mm"] / P
    defect, soft, known = np.zeros(shape, bool), np.zeros(shape, bool), np.zeros(shape, bool)
    for c in crops.values():
        win = np.s_[c["y"]:c["y"] + P, c["x"]:c["x"] + P]
        defect[win] |= c["pixels"]
        soft[win] |= c["soft_px"]
        known[win] |= c["mask"]
    img = img.astype(np.float32)
    # 넓은 배경은 1/4 로 줄여 흐린 뒤 키운다 (전체 이미지 시그마 72 px 흐림이 보드를 오래 붙잡았다, 11-17 재부팅)
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), P / 24), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    dark = ((bg - img) / np.maximum(bg, 1.0) > FULL_DARK) & (img > BACKGROUND_MAX)     # 미촬영(검정) 자리는 뺀다
    if not follow_dark:
        dark[:] = False
    elif ref is not None:
        rb = cv2.GaussianBlur(ref.astype(np.float32), (0, 0), P / 6)
        static = cv2.dilate(((rb - ref) / np.maximum(rb, 1.0) > FULL_DARK).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        for d in die_layout():                         # 검사 못 한 다이 자리에도 렌즈 점은 같은 자리에 있다
            r, c = R.die_slot(d)
            dark[r * P:(r + 1) * P, c * P:(c + 1) * P] &= ~static
    seeds = defect.astype(np.uint8)
    lim = int(round(GROW_LIMIT_MM / mm))
    # 몸체는 다이 비교로 볼 수 없던 곳(스크라이브·반사 띠)에서만 따라간다. 검사한 곳에서 비교 검출이 끊었다면
    # 실제로 떨어진 결함이다 (가까운 파티클 두 개가 하나의 스테인으로 이어졌다, 2026-11-16 14:46 다이 32)
    # 멀리 따라가기는 검사한 다이 밖이면서 웨이퍼 테두리에서 떨어진 곳만. 웨이퍼 밖 테두리의 어둠을 따라가면
    # 가장자리 스테인이 길어져 스크래치가 됐다 (11-17 16:50 스캔 다이 22)
    far_ok = np.ones(shape, bool)
    for c in crops.values():
        far_ok[c["y"]:c["y"] + P, c["x"]:c["x"] + P] = False
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    h = R.GRID_DIES / 2 * P
    far_ok &= np.hypot(xx - h, yy - h) * mm <= WAFER_MODEL["diameter_mm"] / 2 - RIM_MARGIN_MM
    # 멀리 따라가는 것은 가는 선 모양 씨앗(스크래치 조각)뿐이다. 덩어리(스테인)는 4 mm 만
    # (스테인이 웨이퍼 가장자리 빈자리로 길어져 스크래치가 됐다, 11-17 16:50 스캔 다이 23)
    thin = np.zeros(shape, bool)
    ns, slab, sst, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
    for i in range(1, ns):
        x0, y0, bw, bh = sst[i, :4]
        py, px = np.nonzero(slab[y0:y0 + bh, x0:x0 + bw] == i)
        if len(px) < MIN_SHARP_AREA:
            continue
        (_, _), (w, h), _ = cv2.minAreaRect(np.column_stack([px, py]).astype(np.float32))
        length = max(w, h, 1.0)
        line_like = length >= SCRATCH_ELONGATION * max(min(w, h), 1.0) or len(px) / length * mm <= THIN_WIDTH_MM
        if length * mm >= SCRATCH_MIN_MM and line_like:          # 길쭉하거나 평균 폭이 가늘면 선 (파티클은 둥글다)
            thin[y0:y0 + bh, x0:x0 + bw] |= slab[y0:y0 + bh, x0:x0 + bw] == i
    near = _within(seeds > 0, lim) | (_within(thin, int(round(GROW_FAR_MM / mm))) & far_ok)
    body = defect | (dark & near & ~known)
    # 검사 제외 띠(스크라이브·반사 띠)를 건너 잇는 것은 가는 선 조각(스크래치)만. 덩어리까지 잇자 격자선 양쪽의
    # 파티클 두 개가 스테인 하나가 됐다 (11-17 격자 맞춤 뒤 14:46 다이 32)
    gap = max(int(round(UNSEEN_GAP_MM / mm)), SCRIBE_MARGIN + 2)    # 스크라이브 여백(px)은 최소 넘어 잇는다
    line_body = body & _within(thin, lim)                 # 선 조각과 그 주변에서 따라간 몸체
    grown = body | (~known & _within(line_body, gap))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grown.astype(np.uint8), connectivity=8)
    objs = []
    for i in range(1, n):
        x0, y0, bw, bh = stats[i, :4]
        win = np.s_[y0:y0 + bh, x0:x0 + bw]
        mine = (lab[win] == i) & body[win]
        if not (mine & defect[win]).any():
            continue                                   # 씨앗 없는 어둠은 결함이 아니다
        form = mine & dark[win]                        # 모양은 실제로 어두운 부분으로 (격자선 잔차 제외)
        fy, fx = np.nonzero(form if form.sum() >= MIN_SHARP_AREA else mine)
        (_, _), (w, h), _ = cv2.minAreaRect(np.column_stack([fx, fy]).astype(np.float32))
        long_mm = max(w, h) * mm
        # 번짐 채널 픽셀이 대부분일 때만 번짐 결함이다. 파티클 둘레 번짐이 이어 붙으면 파티클이 스테인이 됐다(21:15)
        n_soft = int((soft[win] & mine).sum())
        kind = _kind(n_soft * 2 > int(mine.sum()), len(fx) * mm * mm, long_mm, max(w, h) / max(min(w, h), 1.0))
        ys, xs = np.nonzero(mine & defect[win])           # 다이 판정은 다이 비교 검출 픽셀로만
        objs.append({"kind": kind, "long_mm": long_mm, "fx": fx + x0, "fy": fy + y0, "xs": xs + x0, "ys": ys + y0})
    objs = _absorb_tails(objs, mm)
    for _ in range(4):                 # 흡수한 조각이 스테인을 넓히므로 더 바뀌지 않을 때까지 되풀이
        n = len(objs)
        objs = _absorb_into_stains(objs, mm, shape)
        if len(objs) == n:
            break

    # 2차: 다이 단위 판정
    band = int(round(EDGE_BAND_MM / mm))
    found, parts = [], {}
    for obj, o in enumerate(objs, 1):
        kind, long_mm, xs, ys = o["kind"], o["long_mm"], o["xs"], o["ys"]
        for c in crops.values():
            inside = (xs >= c["x"]) & (xs < c["x"] + P) & (ys >= c["y"]) & (ys < c["y"] + P)
            if not inside.any():
                continue
            px, py = xs[inside] - c["x"], ys[inside] - c["y"]
            # 회로 영역에 1 mm² 이상 들어와야 침범이다. 선 가장자리가 경계를 몇 px 넘은 것까지 불량으로 보지 않는다
            center = int(((px >= band) & (px < P - band) & (py >= band) & (py < P - band)).sum()) * mm * mm >= CENTER_MIN_MM2
            did = c["die"]["id"]
            p = parts.setdefault(did, [0, False, 0.0, 0])
            p[0] += int(inside.sum())
            p[1] = p[1] or center
            if kind == "scratch" and inside.sum() >= MIN_SHARP_AREA:
                (_, _), (pw, ph), _ = cv2.minAreaRect(np.column_stack([px, py]).astype(np.float32))
                p[2] = max(p[2], max(pw, ph) * mm)
            elif kind == "stain":
                p[3] += int(inside.sum())
            found.append({"die": did, "type": kind, "obj": obj,
                          "x_mm": round(float((xs[inside].mean() - R.GRID_DIES / 2 * P) * mm), 2),
                          "y_mm": round(float((R.GRID_DIES / 2 * P - ys[inside].mean()) * mm), 2),
                          "area_mm2": round(float(inside.sum()) * mm * mm, 2), "length_mm": round(float(long_mm), 2),
                          "center": center})
    verdicts = {}
    for did in crops:
        area_px, center, scratch_mm, stain_px = parts.get(did, (0, False, 0.0, 0))
        ratio, stain_ratio = area_px / (P * P), stain_px / (P * P)
        verdicts[did] = {"verdict": die_verdict(center, ratio, scratch_mm, stain_ratio), "center": center,
                         "area_ratio": round(ratio, 4), "scratch_edge_mm": round(scratch_mm, 1),
                         "stain_ratio": round(stain_ratio, 4)}
    return found, apply_gdbn(verdicts)


def quadrat_values(crops: dict, found: list[dict], P: int, per_die: int = 2) -> dict:
    """다이를 구역으로 나눈 판정. {(다이 id, 구역 행, 구역 열): 값}. 학습 입력의 light / heavy 와 같은 값이다."""
    out = {}
    for did, c in crops.items():
        q = P // per_die
        pts = [(d["x_mm"], d["y_mm"]) for d in found if d["die"] == did]
        for r in range(per_die):
            for k in range(per_die):
                part = c["pixels"][r * q:(r + 1) * q, k * q:(k + 1) * q]
                if not part.any():
                    continue
                n = _components(part, 1)
                heavy = part.mean() >= HEAVY_QUADRAT_AREA or len(n) >= HEAVY_QUADRAT_COUNT
                out[(did, r, k)] = R.HEAVY_VALUE if heavy else R.LIGHT_VALUE
    return out


def detection_channel(bgr: np.ndarray, meta: dict, use_chroma: bool = False,
                      surface: str | None = None) -> tuple[np.ndarray, dict]:
    """검출에 쓸 한 장의 영상과 검출 설정 (channel, sharp_z, min_area_mm2, dark_only).
    - 밝은 바탕(불투명면): 흑백 그대로.
    - 어두운 바탕(투명 앞면): 흑백에서 주변보다 어두운 것만 센다. 표면 흠집·얼룩·반사는 희게, 붙은 결함은
      검게 보인다. 2026-11-03 앞면 스캔 다섯 번 중 네 번에서 가장 정확했다 (8/8·오검출 0 등).
    - use_chroma: 색 차이(R−B). 카메라가 가깝고 조명 반사가 커서 흠집이 하얗게 빛나던 한 번(9/9)에서만
      나았고, 거리를 띄운 지금 환경에서는 0/8 이었다. 설정(detect_chroma)으로만 켠다."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if surface == "front" and not use_chroma:
        # 투명 앞면으로 고정: 격자선 밝기로 가르지 않는다. 웨이퍼 아래를 흰 판으로 바꾸자 격자선이 어둡게 보여
        # 불투명면으로 오판했고, 흰 흠집·먼지까지 세어 오검출 17 다이가 나왔다(2026-11-14 17:35 스캔)
        return gray, {"channel": "gray-dark", "sharp_z": DARK_ONLY_Z, "min_area_mm2": MIN_DARK_AREA_MM2, "dark_only": True}
    h, w = gray.shape
    (cx, cy), r = meta["wafer"]["center_px"], meta["wafer"]["radius_px"]
    yy, xx = np.mgrid[0:h, 0:w]
    disc = np.hypot(xx - cx, yy - cy) < r * 0.9
    # 격자선이 밝게 보이면 투명 앞면(바닥이 비침), 어둡게 보이면 불투명면이다. 조명을 바꿔도 뒤집히지 않는다.
    # 예전 맵에는 이 값이 없어 밝기로 가른다 (조명을 세게 한 앞면은 중앙값이 91 까지 올라갔다)
    lines = meta.get("grid", {}).get("lines")
    opaque = lines == "dark" if lines else float(np.median(gray[disc])) >= DARK_BACKGROUND
    if opaque:
        return gray, {"channel": "gray", "sharp_z": SHARP_Z, "min_area_mm2": 0.0, "dark_only": False}
    if not use_chroma:
        return gray, {"channel": "gray-dark", "sharp_z": DARK_ONLY_Z, "min_area_mm2": MIN_CHROMA_AREA_MM2, "dark_only": True}
    b, _, rr = cv2.split(bgr.astype(np.float32))
    # 색 차이에서 결함은 가장자리(종이)만 따뜻한 색인 고리로 보여 흑백보다 신호가 약하다. 문턱을 한 단계 낮추고
    # 대신 작은 점은 세지 않는다
    return (np.clip(128 + CHROMA_GAIN * (rr - b), 0, 255).astype(np.uint8),
            {"channel": "chroma", "sharp_z": CHROMA_Z, "min_area_mm2": MIN_CHROMA_AREA_MM2, "dark_only": False})


def detect(map_path: str, meta: dict, seen: np.ndarray | None = None) -> dict:
    img, prof = detection_channel(cv2.imread(map_path), meta)
    return detect_array(img, meta, seen=seen, min_area_mm2=prof["min_area_mm2"], sharp_z=prof["sharp_z"],
                        dark_only=prof["dark_only"])


def detect_array(gray: np.ndarray, meta: dict, per_die: int = 2, seen: np.ndarray | None = None,
                 min_area_mm2: float = 0.0, sharp_z: float = SHARP_Z, dark_only: bool = False,
                 golden: dict | None = None) -> dict:
    img, valid, P = rectify(gray, meta, seen)
    return detect_rectified(img, valid, P, per_die, min_area_mm2, sharp_z, dark_only, golden)


def detect_rectified(img: np.ndarray, valid: np.ndarray, P: int, per_die: int = 2, min_area_mm2: float = 0.0,
                     sharp_z: float = SHARP_Z, dark_only: bool = False, golden: dict | None = None) -> dict:
    dies = die_layout()
    crops = crop_dies(img, valid, P, dies)
    if len(crops) < 8:
        raise ValueError(f"검사 가능한 다이가 {len(crops)}개뿐입니다. 기준 다이를 만들려면 8개 이상 필요합니다.")
    align(crops, P, golden)
    noise = analyse(crops, P, golden)
    extract(crops, P, noise, min_area_mm2, sharp_z, dark_only)
    found, verdicts = link_defects(crops, P, img, follow_dark=dark_only, ref=noise["reference"])
    return {"inspected": sorted(crops), "defects": found, "verdicts": verdicts, "pitch_px": P,
            "noise": {k: round(v, 5) for k, v in noise.items() if k != "reference"},
            "shift_px": round(float(np.max([np.hypot(*c["shift"]) for c in crops.values()])), 2),
            "quadrats": quadrat_values(crops, found, P, per_die)}


def load_meta(scan_dir: str) -> dict:
    with open(os.path.join(scan_dir, "wafer_map.json")) as f:
        return json.load(f)
