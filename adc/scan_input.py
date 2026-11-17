"""
adc/scan_input.py
실측 스캔 → 분석 입력. 합성 맵에서 결함을 검출하고 다이별 개수와 구역 판정을 만든다.
검출은 몇 초 걸리므로 결과를 스캔 폴더의 detection.json 에 남겨 두고, 맵이 바뀌지 않았으면 다시 쓴다.
"""
from __future__ import annotations
import glob
import json
import os
from datetime import datetime

import cv2
import numpy as np

import scan_config
import wafer_detect
from .knowledge import DEFECT_TYPES

SCAN_ROOT = os.path.join("captures", "scans")
CACHE = "detection.json"
VERSION = 38                    # 검출 방식이 바뀌면 올린다. 이전 결과는 다시 계산한다


def scan_dir(scan_id: str) -> str:
    d = os.path.join(SCAN_ROOT, os.path.basename(scan_id))
    if not os.path.isfile(os.path.join(d, "wafer_map.json")):
        raise KeyError(scan_id)
    return d


def map_meta(scan_id: str) -> dict:
    d = scan_dir(scan_id)
    with open(os.path.join(d, "wafer_map.json")) as f:
        m = json.load(f)
    m["dir"] = os.path.basename(d)
    m["url"] = f"/scans/{m['dir']}/{m.get('view') or m['map']}"      # 화면용 맵이 있으면 그것을 보여 준다
    res = os.path.join(d, "result_view.json")
    if os.path.isfile(res):
        try:
            with open(res) as f:
                info = json.load(f)
            t = map_meta(info["template"])                     # 결과 이미지는 템플릿 맵 좌표다
            img = os.path.join(d, info["image"])
            t["url"] = f"/scans/{m['dir']}/{info['image']}?v={int(os.path.getmtime(img))}"
            t.pop("result", None)
            m["result"] = t
        except (KeyError, OSError, ValueError):
            pass
    return m


def label(scan_id: str) -> str:
    try:
        t = datetime.strptime(scan_id[:15], "%Y%m%d_%H%M%S").strftime("%m-%d %H:%M")
    except ValueError:
        t = scan_id
    return f"{t} 스캔"


def scans() -> list[dict]:
    """맵이 합성된 스캔, 최신순"""
    out = []
    for meta in sorted(glob.glob(os.path.join(SCAN_ROOT, "*_scan", "wafer_map.json")), reverse=True):
        sid = os.path.basename(os.path.dirname(meta))
        out.append({"id": sid, "name": label(sid)})
    return out


def _reference_for(scan_id: str) -> str | None:
    """설정된 기준 웨이퍼 스캔 (자기 자신이거나 맵이 없으면 쓰지 않는다)."""
    ref = scan_config.load().get("reference_scan")
    if not ref or os.path.basename(str(ref)) == os.path.basename(scan_id):
        return None
    try:
        scan_dir(ref)
    except KeyError:
        return None
    return os.path.basename(str(ref))


def _golden(ref_id: str, cfg: dict) -> dict:
    """기준 웨이퍼 맵을 다이별로 자른다. 같은 검사면 설정으로 같은 채널을 쓴다."""
    d = scan_dir(ref_id)
    meta = wafer_detect.load_meta(d)
    gray, _ = wafer_detect.detection_channel(cv2.imread(os.path.join(d, meta["map"])), meta,
                                             use_chroma=bool(cfg.get("detect_chroma")), surface=cfg.get("surface"))
    mask_path = os.path.join(d, meta.get("mask", ""))
    seen = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if meta.get("mask") and os.path.isfile(mask_path) else None
    return wafer_detect.golden_crops(gray, meta, seen)


def detect(scan_id: str) -> dict:
    """스캔 맵의 결함 검출 결과. 맵이 새로 합성되면 다시 계산한다."""
    d = scan_dir(scan_id)
    path = os.path.join(d, CACHE)
    map_path = os.path.join(d, map_meta(scan_id)["map"])
    if os.path.isfile(path) and os.path.getmtime(path) >= os.path.getmtime(map_path):
        with open(path) as f:
            cached = json.load(f)
        if cached.get("version") == VERSION and cached.get("reference") == _reference_for(scan_id):
            return cached
    meta = wafer_detect.load_meta(d)
    cfg = scan_config.load()
    gray, prof = wafer_detect.detection_channel(cv2.imread(map_path), meta, use_chroma=bool(cfg.get("detect_chroma")),
                                                surface=cfg.get("surface"))
    mask_path = os.path.join(d, meta.get("mask", ""))
    seen = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if meta.get("mask") and os.path.isfile(mask_path) else None
    ref_id = _reference_for(scan_id)
    golden = _golden(ref_id, cfg) if ref_id else None
    weak_path = os.path.join(d, meta.get("weak_mask", ""))
    if golden is None and seen is not None and meta.get("weak_mask") and os.path.isfile(weak_path):
        # 카메라 반사·조명 번짐 자리로만 찍힌 곳은 같은 웨이퍼 다이끼리 비교하면 오검출이 난다(2026-11-14) → 뺀다
        seen = np.where(cv2.imread(weak_path, cv2.IMREAD_GRAYSCALE) > 127, 0, seen).astype(np.uint8)
    r = wafer_detect.detect_array(gray, meta, seen=seen, min_area_mm2=prof["min_area_mm2"], sharp_z=prof["sharp_z"],
                                  dark_only=prof["dark_only"], golden=golden)
    out = {"version": VERSION, "scan": os.path.basename(d), "defects": r["defects"], "inspected": r["inspected"],
           "verdicts": r["verdicts"],
           "per_die": 2, "quadrats": [[k[0], k[1], k[2], v] for k, v in sorted(r["quadrats"].items())],
           "noise": r["noise"], "shift_px": r["shift_px"], "channel": prof["channel"], "reference": ref_id}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, ensure_ascii=False, default=float)
    os.replace(tmp, path)
    return out


def die_defects(det: dict) -> dict[int, dict]:
    """검출 목록 → {다이 id: {유형: 1 있음 / 0 없음}} (분석 엔진 입력).
    개수는 세지 않는다: 개수 오차가 판정 오류로 이어졌다 (docs/specs/die-verdict.md)."""
    out: dict[int, dict] = {}
    for x in det["defects"]:
        out.setdefault(int(x["die"]), {t: 0 for t in DEFECT_TYPES})[x["type"]] = 1
    return out


HISTORY_MIN = 6                  # AEC-Q002: 통계 수율 한계는 최소 6 로트 이력. 그 전에는 고정 기준


def yield_limit(history: list[float], fixed: float = 0.10) -> float:
    """웨이퍼 불량 다이 비율의 한계. 이력이 6 장 미만이면 고정값, 이상이면 평균 + 3σ (SYL1 현지화)."""
    if len(history) < HISTORY_MIN:
        return fixed
    return min(1.0, float(np.mean(history) + 3 * np.std(history)))


def defect_ratio_history(exclude: str | None = None) -> list[float]:
    """검출 결과가 저장된 다른 스캔들의 불량 다이 비율."""
    out = []
    for s in scans():
        if s["id"] == exclude:
            continue
        try:
            with open(os.path.join(scan_dir(s["id"]), CACHE)) as f:
                det = json.load(f)
        except (OSError, ValueError):
            continue
        v = [x["verdict"] for x in det.get("verdicts", {}).values()]
        if v:
            out.append(sum(1 for x in v if x == "defect") / len(v))
    return out


def die_verdicts(det: dict) -> dict[int, str]:
    """{다이 id: normal|defect|recheck}"""
    return {int(k): v["verdict"] for k, v in det.get("verdicts", {}).items()}


def quadrats(det: dict) -> dict[tuple, float]:
    return {(int(a), int(b), int(c)): float(v) for a, b, c, v in det["quadrats"]}
