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
    # 저장 시점의 AI 패턴 판정 (집계 때 모델을 다시 돌리지 않도록)
    judged = sorted({str(p) for p in data.get("judged") or [] if str(p) in PATTERNS})
    # 선택: 다이별 결함 개수 {다이: {종류: 개수}} (1단계 단일 결함 수준 확인용)
    counts = {}
    for k, per in (data.get("counts") or {}).items():
        if str(int(k)) not in dies:
            raise ValueError(f"개수를 적은 다이 {k} 가 결함 다이 목록에 없습니다")
        counts[str(int(k))] = {t: int(n) for t, n in per.items() if t in dies[str(int(k))] and int(n) > 0}
    out = {"stage": stage, "dies": dies, "patterns": patterns, "judged": judged,
           "note": str(data.get("note") or "")[:500]}
    if counts:
        out["counts"] = counts
    return out


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
    out = {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "type_match": type_ok,
           "not_inspected": sorted(d for d in want if d not in inspected)}
    return out


# ── 평가 기록 (화면에는 내지 않고 데이터로만 남긴다) ────────────────
EVAL_LOG = "evaluation.jsonl"          # 스캔 폴더들의 상위(captures/scans)에 누적


def _write_json(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def record(scan_id: str) -> dict | None:
    """정답이 있는 스캔의 혼동행렬(다이 단위 TP·FN·FP·TN)을 truth.json 과 누적 파일에 남긴다.
    사용자 요청(2026-11-16): 결과 화면에는 표기하지 않고, 나중에 꺼내 그림으로 만들 수 있게 데이터로만 저장."""
    t = load(scan_id)
    if t is None:
        return None
    c = compare(scan_id, t)
    det = scan_input.detect(scan_id)
    ev = {"time": datetime.now().isoformat(timespec="seconds"), "scan": os.path.basename(scan_id), "stage": t["stage"],
          "tp": len(c["tp"]), "fn": len(c["fn"]), "fp": len(c["fp"]), "tn": c["tn"],
          "dies": {"tp": c["tp"], "fn": c["fn"], "fp": c["fp"], "not_inspected": c["not_inspected"]},
          "type_match": len(c["type_match"]), "patterns": t["patterns"], "judged": t.get("judged", []),
          "channel": det.get("channel"), "reference": det.get("reference"), "detect_version": det.get("version")}
    t["evaluation"] = ev
    _write_json(_path(scan_id), t)
    log = os.path.join(os.path.dirname(scan_input.scan_dir(scan_id)), EVAL_LOG)
    with open(log, "a") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return ev


# ── 단계별 집계 ─────────────────────────────────────────────────
TARGET = {1: 20, 2: 3, 3: 2}     # 1단계: 종류당 결함 다이 수, 2단계: 패턴당 스캔 수, 3단계: 조합당 스캔 수


def _bound(errors: int, n: int) -> str:
    """오류가 0 이면 3/n 규칙(95 %)으로 상한을 말한다. 그 밖에는 비율만."""
    if n == 0:
        return "표본 없음"
    if errors == 0:
        return f"0/{n} → 95 % 신뢰로 {3 / n * 100:.0f} % 이하"
    return f"{errors}/{n} = {errors / n * 100:.0f} %"


def summary() -> dict:
    rows = []
    for item in scan_input.scans():
        t = load(item["id"])
        if t is None:
            continue
        try:
            c = compare(item["id"], t)
        except Exception as e:                       # 검출 실패한 스캔은 집계에서 빼고 알린다
            rows.append({"scan": item["id"], "stage": t["stage"], "error": str(e)})
            continue
        rows.append({"scan": item["id"], "truth": t, "cmp": c, "stage": t["stage"]})
    out = {"scans": len(rows), "stages": {}}
    for stage in STAGES:
        rs = [r for r in rows if r["stage"] == stage and "cmp" in r]
        tp = sum(len(r["cmp"]["tp"]) for r in rs); fn = sum(len(r["cmp"]["fn"]) for r in rs)
        fp = sum(len(r["cmp"]["fp"]) for r in rs); tn = sum(r["cmp"]["tn"] for r in rs)
        per_type = {k: 0 for k in DEFECT_TYPES}
        for r in rs:
            for kinds in r["truth"]["dies"].values():
                for k in kinds:
                    per_type[k] += 1
        per_pattern: dict[str, int] = {}
        pat_ok = 0
        pat_n = 0
        for r in rs:
            want = r["truth"]["patterns"]
            if want:
                pat_n += 1
                key = " + ".join(want)
                per_pattern[key] = per_pattern.get(key, 0) + 1
                pat_ok += set(want) <= set(r["truth"].get("judged", []))
        out["stages"][stage] = {"scans": len(rs), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                                "type_match": sum(len(r["cmp"]["type_match"]) for r in rs),
                                "miss": _bound(fn, tp + fn), "false_alarm": _bound(fp, fp + tn),
                                "per_type": per_type, "per_pattern": per_pattern,
                                "pattern_ok": pat_ok, "pattern_n": pat_n}
    out["failed"] = [r for r in rows if "error" in r]
    return out


def main():
    s = summary()
    print(f"정답이 기록된 스캔 {s['scans']}개")
    for stage, v in s["stages"].items():
        print(f"\n[{stage}단계 {STAGES[stage]}] 스캔 {v['scans']}개")
        print(f"  검출 TP {v['tp']} · FN {v['fn']} · FP {v['fp']} · TN {v['tn']}, 종류 일치 {v['type_match']}/{v['tp']}")
        print(f"  놓침: {v['miss']} / 오검출: {v['false_alarm']}")
        if stage == 1:
            print("  종류별 결함 다이 (목표 {}): ".format(TARGET[1]) +
                  ", ".join(f"{DEFECT_TYPES[k]['ko']} {n}" for k, n in v["per_type"].items()))
        else:
            print(f"  패턴 판정 맞음 {v['pattern_ok']}/{v['pattern_n']} · 배치별 스캔 수 (목표 {TARGET[stage]}): "
                  + (", ".join(f"{k} {n}" for k, n in v["per_pattern"].items()) or "없음"))
    for r in s["failed"]:
        print(f"\n집계에서 뺀 스캔 {r['scan']}: {r['error']}")


if __name__ == "__main__":
    main()
