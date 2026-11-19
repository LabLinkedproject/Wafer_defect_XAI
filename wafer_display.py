"""
wafer_display.py
화면용 웨이퍼 맵: 합성 맵의 미촬영(검정) 자리를 주변 표면색과 다이 격자선으로 채우고, 원판 바깥은 흰색, 6시 노치를 그린다.
검출용 맵은 건드리지 않는다. 출력은 원본과 같은 좌표(크기만 scale 배)라 검출 표시·격자를 그대로 겹칠 수 있다.
"""
from __future__ import annotations
import math
import os

import cv2
import numpy as np

NAME = "wafer_map_display.jpg"
SCALE = 0.5                     # 화면용이라 절반 크기 (4000 px 맵을 그대로 흐리면 메모리 1 GB 가까이 든다)
DARK = 40                       # 이 밝기 이하 = 검정


def render(img: np.ndarray, center, radius: float, pitch: float, angle_deg: float, scale: float = 1.0) -> np.ndarray:
    """img: 합성 맵(BGR), center·radius·pitch: 원본 픽셀, angle_deg: 격자 각도(cv2 기준, wafer_view.js 와 같은 뜻)."""
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cx, cy, r, pitch = center[0] * scale, center[1] * scale, radius * scale, pitch * scale
    h, w = img.shape[:2]
    canvas = img.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    disc = np.hypot(xx - cx, yy - cy) <= r
    # 미촬영 = 바깥과 이어지거나 다이 반 칸보다 큰 검정 덩어리. 작은 검정(파티클 등 결함)은 그대로 둔다
    dark = (canvas.mean(2) <= DARK).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    big = np.zeros(n, bool)
    edge = (st[1:, 0] == 0) | (st[1:, 1] == 0) | (st[1:, 0] + st[1:, 2] >= w) | (st[1:, 1] + st[1:, 3] >= h)
    big[1:] = (st[1:, cv2.CC_STAT_AREA] > (pitch * 0.5) ** 2) | edge
    k = max(3, int(pitch * 0.07) | 1)                                     # 경계 잔여 그늘까지
    unseen = cv2.dilate(big[lab].astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    seen = ~unseen & disc
    if not seen.any():
        return np.full_like(img, 255)
    # 빈 자리 = 주변 표면색의 부드러운 연장 (정규화 흐림)
    # 큰 반경 흐림은 1/8 크기에서 계산한다 (결과가 부드러운 색이라 차이가 없고, 보드에서 수십 배 빠르다)
    f = 8
    small = (max(w // f, 1), max(h // f, 1))
    s_seen = cv2.resize(seen.astype(np.float32), small, interpolation=cv2.INTER_AREA)
    s_img = cv2.resize(canvas * seen[..., None], small, interpolation=cv2.INTER_AREA)
    sig = max(r * 0.05 / f, 1.0)
    wgt = cv2.GaussianBlur(s_seen, (0, 0), sig)
    base = cv2.GaussianBlur(s_img, (0, 0), sig) / np.maximum(wgt[..., None], 1e-3)
    sample = canvas[seen][::16]
    base = np.where(wgt[..., None] > 0.02, base, np.median(sample, 0))
    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
    # 격자선 연장: 웨이퍼 중심이 교차점, 격자 각도만큼 돌린다 (wafer_view.js toPx 와 같은 변환)
    a = math.radians(-angle_deg)
    u = ((xx - cx) * math.cos(a) - (yy - cy) * math.sin(a)) / pitch      # 다이 칸 단위 좌표
    v = ((xx - cx) * math.sin(a) + (yy - cy) * math.cos(a)) / pitch
    half = max(pitch * 0.01, 1.0) / pitch
    lines = ((np.abs(u - np.round(u)) < half) | (np.abs(v - np.round(v)) < half)).astype(np.float32)[..., None]
    line_col = np.percentile(sample, 97, axis=0)
    fill = base * (1 - 0.55 * lines) + line_col * 0.55 * lines
    feather = cv2.GaussianBlur(seen.astype(np.float32), (0, 0), max(r * 0.008, 1.0))[..., None]
    out = canvas * feather + fill * (1 - feather)
    out[~disc] = 255
    out = np.clip(out, 0, 255).astype(np.uint8)
    lw = max(int(r / 450), 1)
    cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r)), (150, 150, 150), lw, cv2.LINE_AA)
    notch = np.array([[cx - r * 0.035, cy + r], [cx + r * 0.035, cy + r], [cx, cy + r * 0.96]], np.int32)   # 6시 노치
    cv2.fillPoly(out, [notch], (255, 255, 255))
    cv2.polylines(out, [notch], True, (150, 150, 150), lw, cv2.LINE_AA)
    return out


def ensure(scan_dir: str, meta: dict) -> str | None:
    """스캔 폴더에 화면용 맵을 만들어 두고 파일 이름을 돌려준다. 맵이 새로 합성되면 다시 만든다."""
    src = os.path.join(scan_dir, meta.get("view") or meta["map"])
    dst = os.path.join(scan_dir, NAME)
    grid = meta.get("grid") or {}
    if not grid.get("pitch_px") or not os.path.isfile(src):
        return None
    if os.path.isfile(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return NAME
    img = cv2.imread(src)
    if img is None:
        return None
    out = render(img, meta["wafer"]["center_px"], meta["wafer"]["radius_px"], grid["pitch_px"], grid.get("angle_deg", 0.0), SCALE)
    cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return NAME
