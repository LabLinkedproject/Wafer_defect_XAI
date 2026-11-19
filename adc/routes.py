"""ADC + XAI 대시보드 API."""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from . import truth
from .knowledge import PATTERNS
from .session import Session


def create_adc_blueprint() -> Blueprint:
    bp = Blueprint("adc", __name__)
    s = Session()

    def body():
        return request.get_json(silent=True) or {}

    def bad(msg):
        return jsonify({"success": False, "error": msg}), 400

    @bp.route("/api/adc/state")
    def state():
        return jsonify(s.state())

    @bp.route("/api/adc/scan", methods=["POST"])
    def scan():
        try:
            s.load_scan(str(body().get("id", "")))
        except KeyError:
            return bad("맵이 합성된 스캔을 찾을 수 없습니다")
        except ValueError as e:
            return bad(f"검출 실패: {e}")
        return jsonify(s.state())

    @bp.route("/api/adc/die", methods=["POST"])
    def die():
        d = body()
        try:
            s.set_die(int(d["id"]), d.get("verdict"))
        except (KeyError, TypeError, ValueError):
            return bad("다이 id 와 판정(normal / defect)을 확인하세요")
        return jsonify(s.state())

    @bp.route("/api/adc/answer", methods=["POST"])
    def answer():
        d = body()
        try:
            s.answer(d.get("check", ""), d.get("answer", ""))
        except KeyError:
            return bad("확인 항목 또는 답변이 올바르지 않습니다")
        return jsonify(s.state())

    @bp.route("/api/adc/answer/undo", methods=["POST"])
    def undo():
        s.undo_answer()
        return jsonify(s.state())

    @bp.route("/api/adc/target", methods=["POST"])
    def target():
        s.set_target(body().get("target"))
        return jsonify(s.state())

    @bp.route("/api/adc/truth")
    def truth_get():
        sid = request.args.get("scan", "")
        try:
            t = truth.load(sid)
            return jsonify({"success": True, "truth": t, "compare": truth.compare(sid, t) if t else None,
                            "stages": truth.STAGES})
        except KeyError:
            return bad("스캔을 찾을 수 없습니다")

    @bp.route("/api/adc/truth", methods=["POST"])
    def truth_save():
        b = body()
        sid = str(b.get("scan", ""))
        try:
            t = truth.save(sid, b)
            truth.record(sid)                      # 혼동행렬은 데이터로만 남긴다
            return jsonify({"success": True, "truth": t, "compare": truth.compare(sid, t)})
        except KeyError:
            return bad("스캔을 찾을 수 없습니다")
        except ValueError as e:
            return bad(str(e))

    @bp.route("/api/adc/feedback", methods=["POST"])
    def feedback():
        d = body()
        verdict = d.get("verdict")
        labels = [x for x in d.get("labels", []) if x in PATTERNS]
        if verdict not in ("confirm", "correct"):
            return bad("verdict 는 confirm 또는 correct")
        n = s.add_feedback(verdict, labels, d.get("cause"), str(d.get("note", "")))
        return jsonify({"success": True, "count": n})

    return bp
