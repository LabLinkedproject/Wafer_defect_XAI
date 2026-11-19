"""
adc/engine.py
다이별 결함 → ADC 판정 → XAI 근거 → 능동 추론 → 공정 피드백까지 한 번에 계산한다.
화면은 이 결과 JSON 만 보고 그린다.
"""
from __future__ import annotations

from wafer_model import active_radius
from . import belief as B
from . import geometry as GEO
from . import render, scoring as S
from .knowledge import (CHECKS, DEFECT_TYPES, HYPOTHESES, LABEL_THRESHOLD, PATTERNS, RECIPE_PARAMS,
                        SEVERITY_ADVICE)

SYSTEMATIC_BONUS = 15          # 구조적 패턴이면 더하는 위험도
ZONE_KO = {"center": "중앙", "middle": "중간", "edge": "엣지"}
ENGINES = {
    "scan": {
        "detector":   {"stage": "검출", "now": "기준 다이 비교 검출 (실측 맵)", "next": "실물 라벨로 검출 기준 보정", "live": True},
        "classifier": {"stage": "분류", "now": "학습 모델 WaferNet (CNN, 8종)", "next": "실물 판정 기록으로 재학습", "live": True},
        "explainer":  {"stage": "설명", "now": "다이 가림 시험 · 반사실 (학습 모델) + 규칙 기반 특징 해석", "next": "Grad-CAM 병행", "live": True},
        "inference":  {"stage": "추론", "now": "베이즈 갱신 + 정보 이득 질의", "next": "로그 연동·우도 학습", "live": True},
    },
    "scenario": {
        "detector":   {"stage": "검출", "now": "시나리오 시뮬레이션", "next": "실측 스캔 입력", "live": False},
        "classifier": {"stage": "분류", "now": "규칙 기반 공간 특징 분류기", "next": "학습 모델 (실측 스캔 입력 시)", "live": False},
        "explainer":  {"stage": "설명", "now": "기여도 분해 · 다이 가림 시험 · 반사실", "next": "Grad-CAM 병행", "live": True},
        "inference":  {"stage": "추론", "now": "베이즈 갱신 + 정보 이득 질의", "next": "로그 연동·우도 학습", "live": True},
    },
}
_HYP = {h["id"]: h for h in HYPOTHESES}
_CHECK = {c["id"]: c for c in CHECKS}


WAFER_DEFECT_RATIO = 0.10     # 불량 다이가 검사 다이의 이 비율 이상이면 웨이퍼 불량 (docs/specs/die-verdict.md)
WAFER_KO = {"pass": "양품", "fail": "불량", "recheck": "재검사"}


def _risk(die_stats: dict, top: str | None) -> dict:
    active = die_stats["active"] or 1
    ratio_term = die_stats["defect"] / active * 60
    recheck_term = min(die_stats["recheck"] * 1.5, 25)
    pattern_term = SYSTEMATIC_BONUS if top not in (None, "Random") else 0
    score = max(0.0, min(100.0, ratio_term + recheck_term + pattern_term))
    level = "Low" if score < 30 else "Medium" if score < 60 else "High" if score < 80 else "Critical"
    return {"score": round(score, 1), "level": level, "advice": SEVERITY_ADVICE[level],
            "terms": [{"name": "불량 다이 비율", "value": round(ratio_term, 1), "max": 60},
                      {"name": "재검사 다이 가산", "value": round(recheck_term, 1), "max": 25},
                      {"name": "구조적 패턴 가산", "value": pattern_term, "max": SYSTEMATIC_BONUS}]}


def _yield(die_stats: dict) -> dict:
    """불량이 아닌 다이 비율. 재검사 다이는 회생 가능으로 보고 넣는다 (사람이 불량으로 바꾸면 빠진다)."""
    active = die_stats["active"] or 1
    pct = round((die_stats["active"] - die_stats["defect"]) / active * 100, 1)
    level = "Excellent" if pct >= 90 else "Good" if pct >= 75 else "Marginal" if pct >= 50 else "Poor"
    return {"yield_pct": pct, "level": level, "good_die": die_stats["normal"], "recheck_die": die_stats["recheck"],
            "defect_die": die_stats["defect"], "active_die": die_stats["active"]}


def _wafer(labels: list, stats: dict, suspect, limit: float = WAFER_DEFECT_RATIO, uniformity: dict | None = None) -> dict:
    """웨이퍼 판정 (docs/specs/inspection-criteria.md 4절).
    Random 외 패턴 → 불량(공정 원인 보류), 불량 다이 비율 ≥ 수율 한계 → 불량,
    재검사 다이·긴 스크래치·의심 → 재검사(사람 판단), 그 밖 양품. Random 은 수율 기준으로만 본다."""
    ratio = stats["defect"] / (stats["active"] or 1)
    systematic = [l for l in labels if l != "Random"]
    if systematic:
        v, why = "fail", f"{' + '.join(systematic)} 패턴 (공정 원인 보류)"
    elif ratio >= limit:
        v, why = "fail", f"불량 다이 {stats['defect']}개 ({ratio * 100:.0f} %, 수율 한계 {limit * 100:.0f} %)"
    elif stats["recheck"]:
        v, why = "recheck", f"재검사 다이 {stats['recheck']}개 — 육안 판정 필요"
    elif suspect:
        v, why = "recheck", f"{suspect['name']} 의심 (점수 {suspect['score']:.2f}) — 육안 판정 필요"
    elif uniformity and uniformity.get("wide_stain"):
        # 자기 점검: 결함을 거의 못 찾았는데 다이 밝기가 크게 흔들리면 넓은 오염이 기준 다이까지 덮었을 수 있다
        v, why = "recheck", (f"다이 밝기가 고르지 않아(변동 {uniformity['cv']:.2f}) 넓은 오염 의심, 어두운 다이 "
                             f"{len(uniformity.get('dark_dies', []))}개 — 다이 비교 검출이 놓칠 수 있어 육안 판정 필요")
    else:
        v, why = "pass", ("Random 분포, 수율 한계 이내" if labels else "패턴·불량 다이 기준 미달")
    return {"verdict": v, "ko": WAFER_KO[v], "reason": why}


SUSPECT_MIN = 0.5   # 판정선 아래라도 이 점수 이상이면 정상과 구분해 '의심'으로 알린다


def _conclusion(labels, top_hyp, zones_top, yield_pct, suspect=None) -> str:
    if not labels and suspect:
        return (f"판정선을 넘는 패턴은 없지만 {suspect['name']} 경향이 있습니다 (점수 {suspect['score']:.2f}, "
                f"판정선 {suspect['threshold']:.2f}). 엔지니어 검토를 권장합니다.")
    if not labels:
        return "뚜렷한 결함 패턴이 없습니다. 공정은 정상 범위로 판단합니다."
    names = " + ".join(labels)
    if not top_hyp:
        return f"{names} 패턴이 검출됐습니다. 예상 수율은 {yield_pct}%입니다."
    return (f"{names} 패턴이 {ZONE_KO.get(zones_top, zones_top)} 존 중심으로 나타났고, 현재 가장 유력한 원인은 "
            f"{top_hyp['process']} 공정의 {top_hyp['name']}입니다.")


def _build_trace(ctx: dict) -> list[dict]:
    """판단이 어떻게 이어졌는지 사람이 읽을 수 있게 기록한다. 값은 모두 실제 계산 결과다."""
    t = [{"kind": "observe", "title": "관찰",
          "text": f"다이 {ctx['active']}개 중 {ctx['defect_dies']}개에서 결함 {ctx['total']}건을 확인했습니다."}]
    if not ctx["labels"]:
        sp, wide = ctx.get("suspect"), ctx.get("wide")
        if wide:                                         # 자기 점검: 검출이 비어도 다이 밝기가 크게 흔들리면 사람에게
            t.append({"kind": "explain", "title": "자기 점검",
                      "text": f"검출 결함은 적지만 다이 밝기 변동이 {wide['cv']:.2f}로 커서, 넓은 오염이 기준 다이까지 덮었을 수 "
                              f"있습니다 (어두운 다이 {len(wide.get('dark_dies', []))}개)."})
        t.append({"kind": "conclude", "title": "결론", "text":
                  f"판정선을 넘는 패턴은 없습니다. {sp['name']} 점수 {sp['score']:.2f}(판정선 {sp['threshold']:.2f})라 검토 대상으로 표시합니다."
                  if sp else "판정선을 넘는 패턴은 없지만 넓은 오염이 의심돼 재검사로 넘깁니다." if wide
                  else "판정선을 넘는 패턴이 없어 정상으로 판단합니다."})
        return t
    for name, why in ctx.get("rejected", {}).items():
        t.append({"kind": "explain", "title": "근거 검토", "text": f"모델의 {name} 판정은 뺐습니다: {why}."})
    for name, why in ctx.get("image_patterns", {}).items():
        t.append({"kind": "explain", "title": "배치 근거", "text": f"{name}: {why}."})
    t.append({"kind": "classify", "title": "분류",
              "text": f"{' + '.join(ctx['labels'])} 판정. 1순위 점수 {ctx['top_score']:.2f}, 2순위와 차이 {ctx['margin']:.2f}."})
    c = ctx["contrib"]
    if c:
        t.append({"kind": "explain", "title": "근거",
                  "text": f"가장 큰 근거는 '{c['label']}'입니다 (값 {c['value']:.2f}, 로그오즈 {c['impact']:+.2f})."})
    if ctx["cf"].get("found"):
        after = ctx["cf"]["after"]
        t.append({"kind": "explain", "title": "반사실",
                  "text": f"결함 다이 {len(ctx['cf']['dies'])}개가 없었다면 판정은 {after}입니다."})
    if ctx["hyp"]:
        h = ctx["hyp"][0]
        t.append({"kind": "hypothesize", "title": "가설",
                  "text": f"영상 근거만으로 본 1순위 원인은 {ctx['image_top_name']}입니다 ({ctx['image_top_p']:.0%})."})
    for e in ctx["history"]:
        ans = {"yes": "예", "no": "아니오"}[e["answer"]]
        change = (f"{e['after_name']} 확률이 {e['before_p']:.0%}에서 {e['after_p']:.0%}가 됐습니다."
                  if e["before_name"] == e["after_name"] else
                  f"1순위가 {e['before_name']}({e['before_p']:.0%})에서 {e['after_name']}({e['after_p']:.0%})가 됐습니다.")
        t.append({"kind": "update", "title": "갱신", "text": f"'{e['ask']}' → {ans}. {change}"})
    if ctx["converged"]:
        h = ctx["hyp"][0]
        t.append({"kind": "conclude", "title": "수렴",
                  "text": f"{h['name']}({h['p']:.0%})로 수렴했습니다. 조치: {h['action']}."})
    elif ctx["checks"]:
        q = ctx["checks"][0]
        t.append({"kind": "ask", "title": "다음 확인",
                  "text": f"아직 원인을 좁히지 못했습니다. '{q['ask']}' 확인이 가장 유용합니다 "
                          f"(기대 정보 이득 {q['gain']:.2f} bit)."})
    return t


def analyze(dies, spec, defects, answers=None, target=None, model=None, verdicts=None, wafer_ctx=None) -> dict:
    """dies: scoring.annotate 결과, defects: {다이 id: {유형: 있음}}, answers: [{check, answer}]
    model: 분류기 (없으면 규칙 기반). 학습 모델이면 판정·다이 가림·반사실을 그 모델로 계산한다.
    verdicts: {다이 id: normal|defect|recheck}. 없으면 결함 다이를 모두 불량으로 본다(시나리오)."""
    answers = answers or []
    defects = {i: c for i, c in defects.items() if sum(c.values()) > 0}
    verdicts = verdicts or {i: "defect" for i in defects}

    # 패턴 모델 입력은 불량 다이만. 재검사 다이는 정상 취급 (회생 가능한 엣지 결함을 패턴으로 읽지 않게)
    model = model or S.RuleModel(dies, {i: c for i, c in defects.items() if verdicts.get(i) == "defect"})
    scores = model.scores()
    labels = S.labels_from(scores, model)
    # 패턴 근거 검토 (adc/geometry.py): 모델 판정을 검출된 결함의 실제 배치로 다시 따진다. 모델 점수는 그대로 두고
    # 근거가 있는 패턴은 더하고, 근거와 맞지 않는 라벨은 이유와 함께 뺀다 (11-18 실측: 둘레 입자 고리를 Scratch 로 읽음)
    ctx = dict(wafer_ctx or {})
    image_patterns, rejected = {}, {}
    if "defects" in ctx:
        bad = {i for i, v in verdicts.items() if v == "defect"}
        geo = GEO.evidence(ctx.pop("defects"), bad, set(ctx.pop("inspected", [])), dies, labels)
        image_patterns, rejected = geo["add"], geo["drop"]
        labels = GEO.reconcile(labels, geo)
    ordered = sorted(PATTERNS, key=lambda c: scores[c]["score"], reverse=True)
    top = labels[0] if labels else None
    second = scores[ordered[1]]["score"]
    top_score = scores[ordered[0]]["score"]
    margin = round(top_score - second, 3)

    # XAI 대상: 지정이 없으면 1순위 라벨
    tgt = target if target in PATTERNS else (top or ordered[0])
    base = scores[tgt]["score"]
    relevance = S.occlusion_map(model, tgt, base) if model.ids else {}
    cf = S.counterfactual(model, tgt, relevance) if (model.ids and base >= model.threshold(tgt)) else {"found": False}

    # 다이·존·유형 집계
    die_rows, stats = [], {"active": len(dies), "normal": 0, "defect": 0, "recheck": 0}
    zones = {z: {"dies": 0, "defect_dies": 0, "count": 0} for z in S.ZONES}
    sectors = [0] * 8
    type_tot = {t: {"count": 0, "dies": 0} for t in DEFECT_TYPES}
    for d in dies:
        c = defects.get(d["id"], {})
        total = sum(c.values())
        st = verdicts.get(d["id"], "normal")
        stats[st] += 1
        zones[d["zone"]]["dies"] += 1
        zones[d["zone"]]["count"] += total
        zones[d["zone"]]["defect_dies"] += 1 if total else 0
        sectors[d["sector"]] += total
        for t in DEFECT_TYPES:
            if c.get(t, 0):
                type_tot[t]["count"] += c[t]
                type_tot[t]["dies"] += 1
        die_rows.append({"id": d["id"], "row": d["row"], "col": d["col"], "x_mm": d["x_mm"], "y_mm": d["y_mm"],
                         "zone": d["zone"], "counts": {t: c.get(t, 0) for t in DEFECT_TYPES}, "total": total,
                         "type": max(c, key=c.get) if total else None, "status": st,
                         "relevance": relevance.get(d["id"], 0.0)})
    for z in zones.values():
        z["fill_pct"] = round(z["defect_dies"] / (z["dies"] or 1) * 100, 1)
    total_defects = sum(v["count"] for v in type_tot.values())
    shares = {t: (v["count"] / total_defects if total_defects else 0.0) for t, v in type_tot.items()}
    dominant_zone = max(zones, key=lambda z: zones[z]["count"]) if total_defects else None
    dominant_type = max(shares, key=shares.get) if total_defects else None

    risk = _risk(stats, top)
    yld = _yield(stats)

    # 능동 추론
    hyp_list, checks, history, converged, h_now = [], [], [], False, None
    image_belief = None
    if labels:
        evidence = {c: (1.0 if c in image_patterns else 0.0 if c in rejected else scores[c]["score"]) for c in PATTERNS}
        image_belief = B.prior_from_evidence(evidence, shares)
        cur = dict(image_belief)
        asked = set()
        for a in answers:
            asked.add(a["check"])
            if a["answer"] not in ("yes", "no"):
                continue
            prev_id = max(cur, key=cur.get)
            prev_p = cur[prev_id]
            cur = B.update(cur, a["check"], a["answer"])
            new_id = max(cur, key=cur.get)
            history.append({"check": a["check"], "ask": _CHECK[a["check"]]["ask"], "answer": a["answer"],
                            "before_name": _HYP[prev_id]["name"], "before_p": round(prev_p, 3),
                            "after_name": _HYP[new_id]["name"], "after_p": round(cur[new_id], 3)})
        ranked = sorted(cur.items(), key=lambda kv: kv[1], reverse=True)
        hyp_list = [{"id": i, "name": _HYP[i]["name"], "process": _HYP[i]["process"], "p": round(p, 4),
                     "image_p": round(image_belief[i], 4), "action": _HYP[i]["action"], "params": _HYP[i]["params"]}
                    for i, p in ranked[:6]]
        h_now = round(B.entropy(cur), 3)
        converged = ranked[0][1] >= B.CONVERGED
        checks = B.rank_checks(cur, asked)
        for c in checks:
            c["target_name"] = _HYP[c["target"]]["name"]

    # 회복 상한: 판정을 지지하는 다이의 결함이 모두 해소된다고 가정한 최댓값 (실제 회복은 이보다 낮을 수 있다)
    support = [d for d in die_rows if d["relevance"] > 0]
    projected = None
    if labels and yld["active_die"]:
        recovered = sum(1 for d in support if d["status"] == "defect")
        projected = {"yield_pct": round(min(100.0, (stats["active"] - stats["defect"] + recovered) / yld["active_die"] * 100), 1),
                     "recovered_die": recovered}

    recipe = []
    related = set(hyp_list[0]["params"]) if hyp_list else set()
    for name, r in RECIPE_PARAMS.items():
        recipe.append({"param": name, **r, "related": name in related})

    image_top = max(image_belief, key=image_belief.get) if image_belief else None
    lead = ordered[0]
    suspect = ({"name": lead, "score": round(scores[lead]["score"], 3), "threshold": model.threshold(lead)}
               if not labels and defects and scores[lead]["score"] >= SUSPECT_MIN else None)
    trace = _build_trace({
        "active": len(dies), "defect_dies": len(defects), "total": total_defects, "labels": labels,
        "top_score": top_score, "margin": margin, "cf": cf,
        "contrib": next((c for c in scores[tgt]["contributions"] if c["impact"] > 0), None),
        "hyp": hyp_list, "history": history, "converged": converged, "checks": checks,
        "image_top_name": _HYP[image_top]["name"] if image_top else None,
        "image_top_p": image_belief[image_top] if image_top else 0, "suspect": suspect,
        "image_patterns": image_patterns, "rejected": rejected,
        "wide": (ctx.get("uniformity") or {}) if (ctx.get("uniformity") or {}).get("wide_stain") else None,
    })

    detections, stain = render.detect_regions(dies, defects, spec["pitch_mm"], spec["diameter_mm"])
    return {
        "engines": ENGINES["scan" if model.kind == "cnn" else "scenario"],
        "model": {"kind": model.kind, "name": "학습 모델 WaferNet" if model.kind == "cnn" else "규칙 기반 분류기"},
        "spec": {"diameter_mm": round(spec["diameter_mm"], 1), "pitch_mm": spec["pitch_mm"],
                 "min_die_area": spec["min_die_area"], "die_count": len(dies), "grid": spec["grid"],
                 "active_radius_mm": round(active_radius(spec), 2)},
        "dies": die_rows,
        "zones": zones, "sectors": sectors,
        "types": {t: {**v, "ko": DEFECT_TYPES[t]["ko"], "share": round(shares[t], 3)} for t, v in type_tot.items()},
        "total_defects": total_defects, "defect_dies": len(defects),
        "dominant_zone": dominant_zone, "dominant_type": dominant_type,
        "threshold": LABEL_THRESHOLD,
        "patterns": [{"name": c, **PATTERNS[c], "score": scores[c]["score"], "channel": scores[c]["channel"],
                      "threshold": model.threshold(c), "is_label": c in labels, "by_image": image_patterns.get(c),
                      "rejected": rejected.get(c)}
                     for c in PATTERNS],
        "image_patterns": image_patterns, "rejected_patterns": rejected,
        "labels": labels, "top": top, "top_score": top_score, "margin": margin, "ambiguous": bool(labels) and margin < 0.2,
        "xai": {"target": tgt, "score": scores[tgt]["score"], "logit": scores[tgt]["logit"], "bias": scores[tgt]["bias"],
                "channel": scores[tgt]["channel"], "contributions": scores[tgt]["contributions"],
                "threshold": model.threshold(tgt), "rule": scores[tgt].get("rule"), "counterfactual": cf},
        "belief": {"hypotheses": hyp_list, "checks": checks, "history": history, "converged": converged,
                   "entropy": h_now, "answered": [a["check"] for a in answers]},
        "risk": risk, "yield": yld, "projected": projected, "recipe": recipe,
        "stain": stain, "detections": detections, "image_size": render.IMAGE_SIZE,
        "suspect": suspect, "wafer": _wafer(labels, stats, suspect, **ctx),
        "conclusion": ("뚜렷한 패턴은 없지만 다이 밝기가 크게 흔들려 넓은 오염이 의심됩니다. 육안 확인이 필요합니다."
                       if not labels and not suspect and (ctx.get("uniformity") or {}).get("wide_stain")
                       else _conclusion(labels, hyp_list[0] if hyp_list else None, dominant_zone, yld["yield_pct"], suspect)),
        "trace": trace,
    }
