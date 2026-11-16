"""
adc/overlay.py
결과 이미지: 순정 웨이퍼(템플릿)의 깨끗한 맵 위에 검출된 결함의 실제 사진 조각을 붙인다.

사용자 요청(2026-11-16): 결함 웨이퍼 결과를 사람이 눈으로 확인하기 쉽게, 순정 상태 템플릿 위에 인식한 결함을 덧씌운다.
두 스캔 사이에 웨이퍼가 밀리거나 조금 돌아가도 맞도록, 사진 좌표가 아니라 웨이퍼 좌표(다이 기준 위치)로 옮긴다.
각 맵이 스스로 잰 웨이퍼 중심·격자 기울기·축척으로 웨이퍼 mm ↔ 맵 픽셀을 바꾼다 (wafer_detect.rectify 와 같은 식).
놓친 결함은 여기 나타나지 않으므로, 사람이 확인할 때는 그 스캔의 실제 사진(화면용 맵)과 나란히 본다.
"""
from __future__ import annotations
import json
import math
import os

import cv2
import numpy as np

import scan_config
import wafer_raster as R
from wafer_model import WAFER_MODEL
from . import scan_input

NAME = "result_view.jpg"
MARGIN_MM = 1.5          # 결함 크기 바깥으로 더 가져오는 폭
DARK_RATIO = 0.18        # 주변 밝기보다 이 비율 이상 어두운 픽셀을 결함으로 옮긴다 (투명면 결함은 검게 보인다)


def _affine(meta: dict) -> np.ndarray:
    """웨이퍼 mm (x 오른쪽, y 위) → 맵 픽셀 (화면용 맵과 같은 좌표)."""
    pitch = meta["grid"]["pitch_px"]
    P = int(round(pitch))
    k = pitch / P
    a = math.radians(-meta["grid"]["angle_deg"])
    c, s = math.cos(a) * k, math.sin(a) * k
    cx, cy = meta["wafer"]["center_px"]
    q = P / WAFER_MODEL["pitch_mm"]                 # 정렬 영상 px/mm
    # u-h = x*q, v-h = -y*q  →  X = cx + c(u-h) + s(v-h),  Y = cy - s(u-h) + c(v-h)
    return np.array([[c * q, -s * q, cx], [-s * q, -c * q, cy]], np.float64)


def _inv(A: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(A)


def template_id(scan_id: str) -> str | None:
    t = scan_config.load().get("template_scan")
    if not t or os.path.basename(str(t)) == os.path.basename(scan_id):
        return None
    try:
        scan_input.scan_dir(t)
    except KeyError:
        return None
    return os.path.basename(str(t))


def build(scan_id: str, tmpl: str | None = None) -> str | None:
    """결과 이미지를 스캔 폴더에 저장하고 파일 이름을 돌려준다. 템플릿이 없으면 None."""
    tmpl = tmpl or template_id(scan_id)
    if not tmpl:
        return None
    d_t, d_g = scan_input.scan_dir(scan_id), scan_input.scan_dir(tmpl)
    m_t, m_g = scan_input.map_meta(scan_id), scan_input.map_meta(tmpl)
    view_t = cv2.imread(os.path.join(d_t, m_t.get("view") or m_t["map"]))
    view_g = cv2.imread(os.path.join(d_g, m_g.get("view") or m_g["map"]))
    if view_t is None or view_g is None:
        return None
    # 결함 웨이퍼 맵을 템플릿 좌표로: 템플릿 픽셀 → 웨이퍼 mm → 결함 맵 픽셀
    A_g, A_t = _affine(m_g), _affine(m_t)
    inv_g = _inv(A_g)
    M = A_t[:, :2] @ inv_g[:, :2]
    M = np.hstack([M, (A_t[:, :2] @ inv_g[:, 2:] + A_t[:, 2:])])
    h, w = view_g.shape[:2]
    warped = cv2.warpAffine(view_t, M.astype(np.float32), (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderValue=(0, 0, 0))
    out = view_g.astype(np.float32)
    wt = warped.astype(np.float32)
    gw = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    px_mm = m_g["grid"]["pitch_px"] / WAFER_MODEL["pitch_mm"]
    mult = np.ones((h, w, 3), np.float32)          # 템플릿에 곱할 어두워짐 (1 = 그대로)
    for d in scan_input.detect(scan_id)["defects"]:
        X, Y = A_g @ np.array([d["x_mm"], d["y_mm"], 1.0])
        r_mm = max(math.sqrt(max(d["area_mm2"], 0.1) / math.pi), d.get("length_mm", 0) / 2) + MARGIN_MM
        r = int(math.ceil(r_mm * px_mm))
        x0, x1 = max(0, int(X) - 2 * r), min(w, int(X) + 2 * r + 1)
        y0, y1 = max(0, int(Y) - 2 * r), min(h, int(Y) + 2 * r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        patch, col = gw[y0:y1, x0:x1], wt[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.hypot(xx - X, yy - Y)
        ring = (dist > r) & (dist < 2 * r) & (patch > 0)
        if not ring.any():
            continue
        bg = float(np.median(patch[ring]))
        bg_c = np.median(col[ring], 0)
        if bg <= 0:
            continue
        # 픽셀을 그대로 옮기면 결함 쪽 배경 밝기까지 따라와 둘레에 밝은 테두리가 생겼다.
        # 배경 대비 어두워진 비율만 템플릿에 입힌다 (결함 색은 채널별 비율로 남는다)
        ratio = np.clip(col / np.maximum(bg_c, 1.0), 0, 1)
        dark = np.clip((bg - patch) / (DARK_RATIO * bg), 0, 1)
        inside = np.clip((r - dist) / max(2.0, 0.3 * r) + 0.5, 0, 1)
        wgt = (dark * inside)[..., None]
        mult[y0:y1, x0:x1] = np.minimum(mult[y0:y1, x0:x1], 1 - wgt * (1 - ratio))
    mult = cv2.GaussianBlur(mult, (0, 0), 0.8)
    out = out * mult
    name = NAME
    path = os.path.join(d_t, name)
    tmp = path + ".tmp.jpg"
    cv2.imwrite(tmp, np.clip(out, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    with open(os.path.join(d_t, "result_view.json"), "w") as f:
        json.dump({"template": tmpl, "image": name}, f)
    return name
