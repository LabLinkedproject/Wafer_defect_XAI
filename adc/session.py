"""
adc/session.py
현재 웨이퍼 상태를 들고 있는 서버 세션. 입력(실측 스캔 또는 시나리오)·다이 수정·현장 확인 답변이 여기 쌓인다.
실측 스캔이면 검출 결과를 학습 모델로 판정하고, 시나리오면 규칙 기반으로 판정한다.
다이를 고치면 영상 근거는 다시 계산하되 현장 확인 답변은 유지한다. 새 사실이 들어오면 판단이 그대로 이어지게 하기 위해서다.
"""
from __future__ import annotations
import json
import os
import threading
from datetime import datetime

import scan_config
import wafer_model
import wafer_stitch
from . import cnn, engine, scan_input, scenarios, scoring
from .knowledge import CHECKS, PATTERNS

FEEDBACK_FILE = "adc_feedback.jsonl"
SCAN_ROOT = os.path.join("captures", "scans")
_CHECK_IDS = {c["id"] for c in CHECKS}


class Session:
    def __init__(self):
        self._lock = threading.RLock()
        self.source = "scenario"
        self.scenario = "donut"
        self.scan_id = None
        self.detection = None
        self.quadrats: dict[tuple, float] = {}
        self.defects: dict[int, dict] = {}
        self.verdicts: dict[int, str] = {}        # 검출 판정
        self.overrides: dict[int, str] = {}       # 사람이 확정한 판정 (재검사 → 정상/불량, Z플랜)
        self.answers: list[dict] = []
        self.edited = False
        self.target = None
        self._cache = None
        latest = scan_input.scans()
        try:
            self.load_scan(latest[0]["id"]) if latest else self.load_scenario("donut")
        except Exception as e:                           # 맵이 깨졌어도 대시보드는 열린다
            print("[ADC] 최신 스캔을 불러오지 못했습니다:", e)
            self.load_scenario("donut")

    def _model(self):
        spec = dict(wafer_model.WAFER_MODEL)
        pitch = scan_config.load().get("pitch_mm")
        if pitch:
            spec["pitch_mm"] = float(pitch)
        return spec, scoring.annotate(wafer_model.die_layout(spec))

    def _invalidate(self):
        self._cache = None

    def load_scenario(self, name: str):
        with self._lock:
            if name not in scenarios.SCENARIOS:
                raise KeyError(name)
            _, dies = self._model()
            self.source, self.scenario, self.scan_id, self.detection, self.quadrats = "scenario", name, None, None, {}
            self.defects = scenarios.generate(name, dies)
            self.verdicts, self.overrides = {}, {}
            self.answers, self.edited, self.target = [], False, None
            self._invalidate()

    def load_scan(self, scan_id: str):
        """실측 스캔의 맵에서 결함을 검출해 입력으로 쓴다."""
        with self._lock:
            det = scan_input.detect(scan_id)
            self.source, self.scan_id, self.detection = "scan", det["scan"], det
            self.defects = scan_input.die_defects(det)
            self.quadrats = scan_input.quadrats(det)
            self.verdicts, self.overrides = scan_input.die_verdicts(det), {}
            self.answers, self.edited, self.target = [], False, None
            self._invalidate()

    def set_die(self, die_id: int, verdict: str | None):
        """사람이 다이 판정을 확정한다 (normal|defect). None 이면 검출 판정으로 되돌린다."""
        with self._lock:
            _, dies = self._model()
            if die_id not in {d["id"] for d in dies} or verdict not in (None, "normal", "defect"):
                raise KeyError(die_id)
            if verdict is None:
                self.overrides.pop(die_id, None)
            else:
                self.overrides[die_id] = verdict
            self.edited = bool(self.overrides)
            self._invalidate()

    def die_verdicts(self) -> dict[int, str]:
        return {**self.verdicts, **self.overrides} if self.source == "scan" else {}

    def answer(self, check_id: str, answer: str):
        with self._lock:
            if check_id not in _CHECK_IDS or answer not in ("yes", "no", "unknown"):
                raise KeyError(check_id)
            self.answers = [a for a in self.answers if a["check"] != check_id]
            self.answers.append({"check": check_id, "answer": answer})
            self._invalidate()

    def undo_answer(self):
        with self._lock:
            if self.answers:
                self.answers.pop()
                self._invalidate()

    def set_target(self, target: str | None):
        with self._lock:
            self.target = target if target in PATTERNS else None
            self._invalidate()

    def state(self) -> dict:
        with self._lock:
            if self._cache is None:
                spec, dies = self._model()
                model, verdicts = None, self.die_verdicts()
                if self.source == "scan" and cnn.available():
                    # 패턴 모델 입력은 불량 다이만 (재검사는 정상 취급). 사람이 불량으로 확정한 다이는 다이 전체로 넣는다
                    bad = {i for i, v in verdicts.items() if v == "defect"}
                    q = {k: v for k, v in self.quadrats.items() if k[0] in bad}
                    q.update({(i, a, b): 1.0 for i in bad - {k[0] for k in q} for a in range(2) for b in range(2)})
                    model = cnn.Model(dies, q, scoring.RuleModel(dies, {k: v for k, v in self.defects.items() if k in bad}))
                ctx = None
                if self.source == "scan":
                    # 패턴 근거 검토(adc/geometry.py)에 검출 위치를 넘긴다.
                    # shortcut: 수율 한계는 10 % 고정. 이력이 결함을 일부러 넣은 시험 웨이퍼뿐이라 평균+3σ 가 100 % 가 되어
                    # 불량 판정이 사라졌다(11-17). 생산 웨이퍼 이력이 6 장 이상 쌓이면 AEC-Q002 SYL(평균+3σ)로 바꾼다
                    ctx = {"defects": self.detection["defects"], "inspected": self.detection["inspected"],
                           "uniformity": self.detection.get("uniformity")}
                r = engine.analyze(dies, spec, self.defects, self.answers, self.target, model, verdicts or None, ctx)
                sess = {"source": self.source, "scenario": self.scenario, "edited": self.edited,
                        "scenario_name": scenarios.SCENARIOS[self.scenario]["name"],
                        "scenario_note": scenarios.SCENARIOS[self.scenario]["note"],
                        "feedback_count": self.feedback_count()}
                if self.source == "scan":
                    det = self.detection
                    sess.update(scan_id=self.scan_id, scan_name=scan_input.label(self.scan_id),
                                inspected=len(det["inspected"]), found=len(det["defects"]))
                    r["found"] = det["defects"]
                    # 다이 상세에 판정 근거(중심부 침범·면적)와 사람 확정 여부를 보인다
                    r["die_verdicts"] = {int(k): {**v, "verdict": verdicts.get(int(k), v["verdict"]),
                                                  "by": "operator" if int(k) in self.overrides else "detect"}
                                         for k, v in det.get("verdicts", {}).items()}
                    r["wafer_map"] = scan_input.map_meta(self.scan_id, display=True)
                else:
                    r["found"] = []
                    r["wafer_map"] = wafer_stitch.latest_map(SCAN_ROOT)
                r["session"] = sess
                r["scenarios"] = [{"id": k, **v} for k, v in scenarios.SCENARIOS.items()]
                self._cache = r
            res = dict(self._cache)
            res["scans"] = scan_input.scans()
            return res

    def feedback_count(self) -> int:
        try:
            with open(FEEDBACK_FILE) as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def add_feedback(self, verdict: str, labels: list[str], cause: str | None, note: str) -> int:
        """엔지니어 판정을 재학습 라벨 후보로 기록한다."""
        with self._lock:
            st = self.state()
            rec = {"time": datetime.now().isoformat(timespec="seconds"), "source": self.source,
                   "scenario": self.scenario if self.source == "scenario" else None, "scan": self.scan_id,
                   "model": st["model"]["kind"], "edited": self.edited, "verdict": verdict, "ai_labels": st["labels"],
                   "operator_labels": labels, "operator_cause": cause, "note": note[:500],
                   "die_defects": {str(k): v for k, v in self.defects.items()},
                   "die_verdicts": {str(k): v for k, v in self.die_verdicts().items()},
                   "operator_dies": {str(k): v for k, v in self.overrides.items()},
                   "detections": st.get("found", []),
                   "answers": self.answers}
            with open(FEEDBACK_FILE, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._invalidate()
            return self.feedback_count()
