"""
adc/render.py
결함을 유형별 표현으로 변환한다.
    stain            : 연속 면적 결함 → 다이 합집합 폴리곤 (도넛의 구멍 포함)
    particle/scratch : 다이별 박스
"""
from __future__ import annotations
import math

import cv2
import numpy as np

IMAGE_SIZE = 620
PADDING = 22


def px_per_mm(diameter_mm: float) -> float:
    return (IMAGE_SIZE - 2 * PADDING) / diameter_mm


def to_px(x_mm: float, y_mm: float, s: float) -> tuple[int, int]:
    return int(IMAGE_SIZE / 2 + x_mm * s), int(IMAGE_SIZE / 2 - y_mm * s)


def _die_rect(d, pitch, s):
    x, y = to_px(d["x_mm"] - pitch / 2, d["y_mm"] + pitch / 2, s)
    side = int(pitch * s)
    return x, y, side


def _stain_mask(stain_dies, pitch, s):
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    for d in stain_dies:
        x, y, side = _die_rect(d, pitch, s)
        cv2.rectangle(mask, (x, y), (x + side, y + side), 255, -1)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    return cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)[1]


def _distribution(regions: int, has_hole: bool, location: str) -> str:
    if regions == 1:
        if has_hole:
            return "Ring (Donut)"
        return {"center": "Center Cluster", "edge": "Edge Cluster"}.get(location, "Local Cluster")
    if regions == 2:
        return "Dual Cluster"
    return "Multi-cluster" if regions <= 4 else "Scattered"


def detect_regions(dies, defects, pitch, diameter):
    """결함 다이 → 화면 오버레이용 검출 표현 + Stain 영역 지표."""
    s = px_per_mm(diameter)
    by_id = {d["id"]: d for d in dies}
    stain_dies = [by_id[i] for i, c in defects.items() if c.get("stain", 0) > 0]
    out, metrics = [], None

    if stain_dies:
        mask = _stain_mask(stain_dies, pitch, s)
        contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 80:
                continue
            approx = cv2.approxPolyDP(cnt, 0.003 * cv2.arcLength(cnt, True), True)
            polygons.append({"is_hole": bool(hier[0][i][3] != -1),
                             "points": [[int(p[0][0]), int(p[0][1])] for p in approx]})
        regions = sum(1 for p in polygons if not p["is_hole"])
        has_hole = any(p["is_hole"] for p in polygons)
        zones = {}
        for d in stain_dies:
            zones[d["zone"]] = zones.get(d["zone"], 0) + 1
        location = max(zones, key=zones.get)
        m = cv2.moments(mask)
        cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (IMAGE_SIZE / 2, IMAGE_SIZE / 2)
        cx_mm, cy_mm = (cx - IMAGE_SIZE / 2) / s, (IMAGE_SIZE / 2 - cy) / s
        metrics = {
            "region_count": regions, "has_hole": has_hole,
            "area_mm2": round(float((mask > 0).sum()) / s ** 2, 1),
            "location": location,
            "centroid_mm": {"x": round(cx_mm, 1), "y": round(cy_mm, 1)},
            "centroid_distance_mm": round(math.hypot(cx_mm, cy_mm), 1),
            "distribution": _distribution(regions, has_hole, location),
            "die_count": len(stain_dies),
            "coverage_pct": round(len(stain_dies) / len(dies) * 100, 1),
        }
        out.append({"shape": "polygon", "type": "stain", "label": "stain", "polygons": polygons,
                    "die_count": len(stain_dies)})

    for t in ("particle", "scratch"):
        for i, c in defects.items():
            if c.get(t, 0) > 0:
                x, y, side = _die_rect(by_id[i], pitch, s)
                pad = int(side * 0.16)
                out.append({"shape": "bbox", "type": t, "label": t, "die": i, "count": c[t],
                            "bbox": [x + pad, y + pad, side - 2 * pad, side - 2 * pad]})
    return out, metrics

