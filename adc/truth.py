"""
adc/truth.py
실물 검증용 정답 기록. 스캔 폴더의 truth.json 에 "어느 다이에 어떤 결함을 넣었는지, 의도한 패턴"을 남기고
검출 결과와 다이 단위로 비교한다 (TP·FP·FN·TN).

2026-11-16 실물 데이터 수집 계획: 1단계 단일 결함(particle·scratch·stain) → 2단계 대표 패턴 → 3단계 복합 패턴.
결함은 팀이 직접 넣는다. 목표는 최소 검증 수준이며, 모은 정답은 나중에 실물 학습·검증 데이터로 쓸 수 있다.
"""
from __future__ import annotations
import json
import os
from datetime import datetime

from . import scan_input
from .knowledge import DEFECT_TYPES, PATTERNS

NAME = "truth.json"
STAGES = {1: "단일 결함", 2: "대표 패턴", 3: "복합 패턴"}


def _path(scan_id: str) -> str:
    return os.path.join(scan_input.scan_dir(scan_id), NAME)


def load(scan_id: str) -> dict | None:
    try:
        with open(_path(scan_id)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def clean(data: dict) -> dict:
    """입력 검증. dies 는 {다이 번호: [결함 종류]}, patterns 는 패턴 이름 목록."""
    stage = int(data.get("stage") or 1)
    if stage not in STAGES:
        raise ValueError("단계는 1~3 이어야 합니다")
    dies = {}
    for k, kinds in (data.get("dies") or {}).items():
        die = int(k)
        if not 1 <= die <= 44:
            raise ValueError(f"다이 번호 {die} 는 1~44 밖입니다")
        kinds = sorted({str(t) for t in kinds})
        bad = [t for t in kinds if t not in DEFECT_TYPES]
        if bad:
            raise ValueError(f"알 수 없는 결함 종류: {', '.join(bad)}")
        if kinds:
            dies[str(die)] = kinds
    patterns = sorted({str(p) for p in data.get("patterns") or []})
    bad = [p for p in patterns if p not in PATTERNS]
    if bad:
        raise ValueError(f"알 수 없는 패턴: {', '.join(bad)}")
    return {"stage": stage, "dies": dies, "patterns": patterns, "note": str(data.get("note") or "")[:500]}


def save(scan_id: str, data: dict) -> dict:
    """디스크에 확정해 저장한다 (보드가 갑자기 꺼지면 최근 파일이 깨졌다, 2026-11-14)."""
    out = clean(data)
    out["updated"] = datetime.now().isoformat(timespec="seconds")
    path = _path(scan_id)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return out


def compare(scan_id: str, truth: dict | None = None) -> dict | None:
    """정답과 검출 결과의 다이 단위 비교. 검사하지 못한 다이는 세지 않는다."""
    truth = truth or load(scan_id)
    if truth is None:
        return None
    det = scan_input.detect(scan_id)
    inspected = set(det["inspected"])
    found: dict[int, set] = {}
    for d in det["defects"]:
        found.setdefault(int(d["die"]), set()).add(d["type"])
    want = {int(k): set(v) for k, v in truth["dies"].items()}
    tp = sorted(d for d in inspected if d in want and d in found)
    fn = sorted(d for d in inspected if d in want and d not in found)
    fp = sorted(d for d in inspected if d not in want and d in found)
    tn = sum(1 for d in inspected if d not in want and d not in found)
    type_ok = sorted(d for d in tp if want[d] & found[d])
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "type_match": type_ok,
            "not_inspected": sorted(d for d in want if d not in inspected)}
