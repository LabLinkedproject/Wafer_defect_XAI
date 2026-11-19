/* 실물 검증 기록: 직접 넣은 결함(다이·종류)·의도 패턴을 저장하고 검출 결과와 비교해 보여 준다.
   adc.js 가 화면을 그릴 때마다 보내는 adc:state 이벤트로 지금 보고 있는 스캔을 안다. */
(function () {
  const PATTERNS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Local", "Random", "Scratch", "Near-full"];
  const TYPES = ["particle", "scratch", "stain"];
  const $ = (id) => document.getElementById(id);
  let scan = null;
  let judged = [];

  $("truth-patterns").innerHTML = PATTERNS.map(
    (p) => `<label><input type="checkbox" value="${p}"> ${p}</label>`).join("");

  function dieList(text) {
    return [...new Set(String(text).split(/[^0-9]+/).filter(Boolean).map(Number))].sort((a, b) => a - b);
  }

  function fill(t) {
    $("truth-stage").value = String(t ? t.stage : 1);
    TYPES.forEach((k) => {
      $(`truth-${k}`).value = t ? Object.keys(t.dies).filter((d) => t.dies[d].includes(k)).join(", ") : "";
    });
    $("truth-patterns").querySelectorAll("input").forEach((c) => { c.checked = !!t && t.patterns.includes(c.value); });
    $("truth-note").value = t ? t.note : "";
  }

  function show(t, cmp) {
    const out = $("truth-result");
    if (!t) { out.textContent = "아직 기록이 없습니다."; return; }
    const list = (a) => (a.length ? a.join(", ") : "없음");
    let html = !cmp ? "<b>검출</b> 정답 다이 목록이 없어 다이 단위 비교는 하지 않았습니다 (패턴만 평가)"
      : `<b>검출</b> TP ${cmp.tp.length} · FN ${cmp.fn.length} · FP ${cmp.fp.length} · TN ${cmp.tn}` +
        ` — 놓침: ${list(cmp.fn)} / 오검출: ${list(cmp.fp)} / 종류 일치 ${cmp.type_match.length}/${cmp.tp.length}`;
    if (cmp && cmp.not_inspected.length) html += ` / 검사 못 한 다이: ${list(cmp.not_inspected)}`;
    if (t.patterns.length) {
      const hit = t.patterns.filter((p) => judged.includes(p));
      const extra = judged.filter((p) => !t.patterns.includes(p));
      html += `<br><b>패턴</b> 의도 ${t.patterns.join(" + ")} → 판정 ${judged.length ? judged.join(" + ") : "없음"}` +
        ` (맞음 ${hit.length}/${t.patterns.length}${extra.length ? `, 추가 판정 ${extra.join(", ")}` : ""})`;
    }
    html += ` <span class="muted">· 저장 ${t.updated ? t.updated.replace("T", " ") : ""}</span>`;
    out.innerHTML = html;
  }

  async function load(id) {
    const r = await fetch(`/api/adc/truth?scan=${encodeURIComponent(id)}`).then((x) => x.json()).catch(() => null);
    if (!r || !r.success || id !== scan) return;
    fill(r.truth);
    show(r.truth, r.compare);
  }

  $("truth-save").addEventListener("click", async () => {
    if (!scan) return;
    const dies = {};
    TYPES.forEach((k) => dieList($(`truth-${k}`).value).forEach((d) => { (dies[d] = dies[d] || []).push(k); }));
    const body = {
      scan, dies, stage: Number($("truth-stage").value), note: $("truth-note").value, judged,
      patterns: [...$("truth-patterns").querySelectorAll("input:checked")].map((c) => c.value),
    };
    $("truth-save").disabled = true;
    const r = await fetch("/api/adc/truth", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) }).then((x) => x.json()).catch(() => null);
    $("truth-save").disabled = false;
    if (!r || !r.success) { $("truth-result").textContent = `저장 실패: ${(r && r.error) || "서버 응답 없음"}`; return; }
    fill(r.truth);
    show(r.truth, r.compare);
  });

  document.addEventListener("adc:state", (e) => {
    const S = e.detail;
    const isScan = S.session && S.session.source === "scan";
    $("s-truth").hidden = !isScan;
    judged = S.labels || [];
    if (!isScan) { scan = null; return; }
    if (S.session.scan_id !== scan) {
      scan = S.session.scan_id;
      load(scan);
    }
  });
})();
