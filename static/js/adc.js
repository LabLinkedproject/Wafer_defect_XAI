// ADC + XAI 대시보드. 서버 상태 하나(/api/adc/state)로 모든 패널을 그린다.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const ZONE_KO = { center: "중앙", middle: "중간", edge: "엣지" };
  const TYPE_COLOR = { particle: "--particle", scratch: "--scratch", stain: "--stain" };
  const STATUS_KO = { normal: "정상", defect: "불량", recheck: "재검사" };

  let S = null;
  let mode = "status";
  let selected = null;
  let inputMode = "real";    // real | result

  async function api(url, body) {
    const opt = body === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    };
    const r = await fetch(url, opt);
    const data = await r.json().catch(() => ({}));
    if (!r.ok || data.success === false) throw new Error(data.error || `요청 실패 (${r.status})`);
    return data;
  }

  async function act(url, body) {
    try {
      $("errbar").hidden = true;
      S = await api(url, body);
      render();
    } catch (e) {
      $("errbar").textContent = e.message;
      $("errbar").hidden = false;
    }
  }

  // ── 전체 그리기 ──────────────────────────────────────────────
  function autoSelect() {
    if (S.dies.some((d) => d.id === selected && d.total)) return;
    const best = [...S.dies].filter((d) => d.total).sort((a, b) => b.relevance - a.relevance || b.total - a.total)[0];
    selected = best ? best.id : null;
  }

  function render() {
    autoSelect();
    renderHeader();
    renderVerdict();
    renderInput();
    renderMap();
    renderInspector();
    renderPatterns();
    renderTypes();
    renderZones();
    renderRose();
    renderStain();
    renderXai();
    renderBelief();
    renderTrace();
    renderAction();
    renderEngines();
    document.dispatchEvent(new CustomEvent("adc:state", { detail: S }));   // 실물 검증 기록(truth.js)이 듣는다
  }

  function renderHeader() {
    const sel = $("scenario");
    const opts = S.scans.length
      ? S.scans.map((c) => `<option value="${esc(c.id)}">${esc(c.name)}</option>`).join("")
      : '<option value="">스캔 기록 없음</option>';
    if (sel.dataset.opts !== opts) { sel.innerHTML = opts; sel.dataset.opts = opts; }
    const ss = S.session;
    const scan = ss.source === "scan";
    if (scan) sel.value = ss.scan_id;
    for (const [k, e] of Object.entries(S.engines)) {
      $(`eng-${k}`).textContent = e.now;
    }
    const edited = ss.edited ? " · 엔지니어가 다이 판정을 확정한 상태" : "";
    const y = S.yield;
    $("simbar").textContent = scan
      ? `${ss.scan_name} · 검사 다이 ${ss.inspected}/${S.spec.die_count} · 불량 ${y.defect_die} · 재검사 ${y.recheck_die}${edited} · 패턴 판정 ${S.model.name}`
      : "스캔 기록이 없습니다. 위에서 웨이퍼 스캔을 시작하세요.";
  }

  function renderVerdict() {
    const top = S.top;
    const pat = S.patterns.find((p) => p.name === top);
    const sp = S.suspect;
    const w = S.wafer;
    $("verdict-title").textContent = `웨이퍼 ${w.ko}`;
    $("verdict-title").className = `wv-${w.verdict}`;
    $("verdict-sub").textContent = w.reason;
    const byImage = S.image_patterns?.[top];          // 모델이 아니라 전체 이미지 분석이 낸 패턴
    const tags = [top ? `<span class="tag">패턴 ${esc(top)} · ${esc(pat.ko)}${byImage ? " · 전체 이미지 근거" : ""}</span>` : sp ? `<span class="tag warn">${esc(sp.name)} 의심 ${sp.score.toFixed(2)}</span>` : '<span class="tag">패턴 없음</span>'];
    tags.push(...S.labels.slice(1).map((l) => `<span class="tag">복합 ${esc(l)}</span>`));
    if (S.ambiguous) tags.push('<span class="tag warn">판정 여유 낮음 · 엔지니어 검토 권장</span>');
    $("verdict-labels").innerHTML = tags.join("");

    $("k-score").textContent = byImage ? "이미지" : top ? S.top_score.toFixed(2) : sp ? sp.score.toFixed(2) : "-";
    $("k-margin").textContent = byImage ? byImage : top ? `2순위와 차이 ${S.margin.toFixed(2)}` : "";
    const y = S.yield;
    $("k-yield").textContent = `${y.yield_pct}%`;
    $("k-yield").className = `lv-${y.level}`;
    $("k-yield-sub").textContent = `불량 아닌 다이 · 재검사 ${y.recheck_die} 포함`;
    const r = S.risk;
    $("k-risk").textContent = r.score;
    $("k-risk").className = `lv-${r.level}`;
    $("k-risk-sub").textContent = r.level;
    const h = S.belief.hypotheses[0];
    $("k-cause").textContent = h ? h.name : "해당 없음";
    $("k-cause-sub").textContent = h ? `${pct(h.p)} · ${h.process}` : "";
    $("verdict-line").textContent = S.conclusion;
  }

  // ── 웨이퍼 맵 사진 (실측 / 결과) ─────────────────────────────
  function renderInput() {
    const scan = S.session.source === "scan";
    const usable = !!(scan && S.wafer_map && S.wafer_map.wafer);
    const hasResult = !!(usable && S.wafer_map.result && S.wafer_map.result.wafer);
    if (inputMode !== "result" || !hasResult) inputMode = "real";
    // 결과 화면은 순정 웨이퍼 템플릿 좌표의 이미지라 템플릿 맵 정보로 그린다 (검출 표시·격자도 그 좌표로)
    const map = inputMode === "result" ? S.wafer_map.result : S.wafer_map;
    document.querySelectorAll("#input-modes button").forEach((b) => {
      b.classList.toggle("on", b.dataset.v === inputMode);
      b.setAttribute("aria-pressed", b.dataset.v === inputMode);
      b.disabled = !usable || (b.dataset.v === "result" && !hasResult);
      if (b.dataset.v === "result") b.hidden = !hasResult;
    });
    $("input-title").textContent = inputMode === "result" ? "결과 (순정 템플릿 + 검출 결함)" : "웨이퍼 맵";
    $("map-empty").hidden = usable;
    $("real-view").hidden = !usable;
    if (!usable) { $("real-view").innerHTML = ""; $("input-note").textContent = ""; return; }
    const marks = $("show-overlay").checked ? S.found.map((f) => ({
      x_mm: f.x_mm, y_mm: f.y_mm, r_mm: Math.max(2.5, Math.sqrt(f.area_mm2 / Math.PI) + 1.5),
      color: css(TYPE_COLOR[f.type]), title: `다이 ${f.die} · ${S.types[f.type].ko} · ${f.area_mm2} mm²`,
    })) : null;
    WaferView.draw($("real-view"), map, { grid: $("show-grid").checked, selected: selected, marks, pitchMm: S.spec.pitch_mm });
    // 다이를 넘는 결함은 하나로 센다 (같은 obj)
    const objs = new Map(S.found.map((f) => [f.obj, f.type]));
    const kinds = Object.keys(S.types).map((k) => [S.types[k].ko, [...objs.values()].filter((t) => t === k).length]).filter(([, n]) => n);
    const by = kinds.map(([ko, n]) => `${ko} ${n}`).join(" · ") || "결함 없음";
    $("input-note").textContent = inputMode === "result"
      ? `순정 웨이퍼(${map.dir}) 위에 검출 결함 ${objs.size}개의 실제 모습을 붙인 결과 · 놓친 결함은 '실측 맵'에서 확인하세요`
      : `촬영 ${S.wafer_map.images}장 합성 맵 · 커버리지 ${S.wafer_map.coverage_pct}% · 기준 다이 비교로 결함 ${objs.size}개 (${by}, 다이를 넘는 것은 하나로)`;
  }

  // ── 웨이퍼 맵 ────────────────────────────────────────────────
  function mix(a, b, t) {
    const pa = a.match(/\w\w/g).map((h) => parseInt(h, 16));
    const pb = b.match(/\w\w/g).map((h) => parseInt(h, 16));
    return `rgb(${pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(",")})`;
  }

  function dieFill(d, maxRel) {
    if (mode === "type") {
      if (!d.type) return { fill: css("--normal"), op: 1 };
      return { fill: css(TYPE_COLOR[d.type]), op: d.status === "defect" ? 1 : 0.55 };
    }
    if (mode === "xai") {
      if (!d.total) return { fill: "#eef1f5", op: 1 };
      const v = maxRel ? d.relevance / maxRel : 0;
      return { fill: v >= 0 ? mix("e6eef8", "004898", Math.min(1, Math.abs(v))) : mix("fbe9e4", "c4553f", Math.min(1, Math.abs(v))), op: 1 };
    }
    return { fill: css(`--${d.status}`), op: 1 };
  }

  function drawMap(svg, drawMode, interactive) {
    const prevMode = mode;
    mode = drawMode;                    // dieFill 이 쓰는 표시 방식
    const R = 250;
    const c = 260;
    const s = (R * 2) / S.spec.diameter_mm;
    const p = S.spec.pitch_mm * s;
    const maxRel = Math.max(1e-6, ...S.dies.map((d) => Math.abs(d.relevance)));
    let html = `<circle cx="${c}" cy="${c}" r="${R}" fill="#fbfcfd" stroke="#44505e" stroke-width="2"/>`;
    if (interactive && $("show-zones").checked) {
      const rmax = Math.max(...S.dies.map((d) => Math.hypot(d.x_mm, d.y_mm)));
      for (const k of [0.33, 0.75]) {
        html += `<circle cx="${c}" cy="${c}" r="${k * rmax * s}" fill="none" stroke="#7a8796" stroke-dasharray="5 5" stroke-width="1.2" pointer-events="none"/>`;
      }
    }
    for (const d of S.dies) {
      const x = c + (d.x_mm - S.spec.pitch_mm / 2) * s;
      const y = c - (d.y_mm + S.spec.pitch_mm / 2) * s;
      const { fill, op } = dieFill(d, maxRel);
      // 판정 맵은 색으로만 구분한다 (사용자 지시 2026-11-16). 판정은 aria-label 과 다이 상세에 글로 남는다
      const label = `다이 ${d.id}, ${ZONE_KO[d.zone]} 존, ${STATUS_KO[d.status]}`;
      const attrs = interactive ? `class="die-cell${selected === d.id ? " sel" : ""}" data-id="${d.id}" tabindex="0" role="button" aria-label="${label}"` : "";
      html += `<g ${attrs}>
        <rect x="${x + 1}" y="${y + 1}" width="${p - 2}" height="${p - 2}" rx="2" fill="#fff" stroke="#9aa6b3"/>
        <rect x="${x + 2}" y="${y + 2}" width="${p - 4}" height="${p - 4}" rx="1" fill="${fill}" fill-opacity="${op}"/>
      </g>`;
    }
    html += `<path d="M${c - 8} ${c + R + 2}L${c + 8} ${c + R + 2}L${c} ${c + R - 9}Z" fill="#44505e"/>`;
    svg.innerHTML = html;
    mode = prevMode;
  }

  function renderMap() {
    drawMap($("wafer-map"), mode, true);
    const leg = {
      status: `<span><i style="background:${css("--normal")}"></i>정상</span><span><i style="background:${css("--defect")}"></i>불량 (회로 영역 침범·면적 5 %·가장자리 긴 스크래치·넓은 스테인)</span><span><i style="background:${css("--recheck")}"></i>재검사 (가장자리에 작게·주변 불량, 육안 판정)</span>`,
      type: Object.entries(S.types).map(([k, t]) => `<span><i style="background:${css(TYPE_COLOR[k])}"></i>${esc(t.ko)}</span>`).join("") + "<span>진하면 불량, 옅으면 재검사</span>",
      xai: `<span><i class="grad" style="background:linear-gradient(90deg,#c4553f,#e6eef8,#004898)"></i>방해 ← 없음 → ${esc(S.xai.target)} 판정 지지</span>`,
    };
    $("map-legend").innerHTML = leg[mode];
    document.querySelectorAll("#map-modes button").forEach((b) => {
      const on = b.dataset.mode === mode;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", on);
    });
  }

  function renderInspector() {
    const body = $("inspector-body");
    const d = S.dies.find((x) => x.id === selected);
    if (!d) {
      body.innerHTML = '<p class="empty">웨이퍼 맵에서 다이를 선택하세요. 결함을 바꾸면 판정·근거·원인 확률이 바로 다시 계산됩니다.</p>';
      return;
    }
    const canCrop = S.wafer_map?.wafer && S.wafer_map?.grid;
    const kinds = Object.entries(d.counts).filter(([, v]) => v).map(([k]) => S.types[k].ko).join(", ") || "없음";
    const v = S.die_verdicts?.[d.id];
    // 판정 근거 (docs/specs/inspection-criteria.md)
    const why = !v ? "-" : v.by === "operator" ? "엔지니어 확정"
      : v.center ? "결함이 회로 영역(중심부)에 닿음" : v.area_ratio >= 0.05 ? `결함 면적 ${(v.area_ratio * 100).toFixed(1)} % (5 % 이상)`
      : v.scratch_edge_mm >= 16 ? `가장자리 스크래치 ${v.scratch_edge_mm} mm (16 mm 이상)`
      : v.stain_ratio >= 0.025 ? `가장자리 스테인 ${(v.stain_ratio * 100).toFixed(1)} % (2.5 % 이상)`
      : v.gdbn ? `주변 불량 다이 ${v.gdbn}개 (GDBN)`
      : v.area_ratio > 0 ? `가장자리에만 작게 (${(v.area_ratio * 100).toFixed(1)} %)` : "결함 없음";
    const set = (verdict, label, cls = "") => `<button type="button" class="btn ${cls}" data-verdict="${verdict}">${label}</button>`;
    body.innerHTML = `
      ${canCrop ? `<canvas class="die-photo" id="die-crop" width="320" height="320" role="img" aria-label="합성 맵의 다이 ${d.id} 위치"></canvas><p class="note photo-note">합성 맵의 이 다이 위치</p>` : ""}
      <dl class="facts">
        <dt>다이</dt><dd>${d.id}번 (행 ${d.row + 1}, 열 ${d.col + 1})</dd>
        <dt>위치</dt><dd>${d.x_mm}, ${d.y_mm} mm</dd>
        <dt>존</dt><dd>${ZONE_KO[d.zone]}</dd>
        <dt>판정</dt><dd>${STATUS_KO[d.status]}</dd>
        <dt>근거</dt><dd>${esc(why)}</dd>
        <dt>결함 종류</dt><dd>${esc(kinds)}</dd>
        <dt>패턴 기여</dt><dd>${d.status === "defect" ? `${d.relevance >= 0 ? "+" : ""}${d.relevance.toFixed(3)}` : "-"}</dd>
      </dl>
      <div class="edit-title">${d.status === "recheck" ? "육안으로 보고 판정을 확정하세요" : "엔지니어 판정"}</div>
      <div class="edit-row">
        ${set("normal", "정상으로 확정")}${set("defect", "불량으로 확정")}
        ${v?.by === "operator" ? `<button type="button" class="btn ghost" data-verdict="" style="grid-column:1/-1">검출 판정으로 되돌리기</button>` : ""}
      </div>`;
    if (canCrop) WaferView.cropDie($("die-crop"), S.wafer_map, d.row, d.col, S.spec.grid);
    body.querySelectorAll("[data-verdict]").forEach((b) => {
      b.onclick = () => act("/api/adc/die", { id: d.id, verdict: b.dataset.verdict || null });
    });
  }

  // ── ADC ──────────────────────────────────────────────────────
  function renderPatterns() {
    const box = $("pattern-bars");
    const rows = [...S.patterns].sort((a, b) => b.score - a.score);
    box.innerHTML = rows.map((p) => `
      <button type="button" class="bar-row${S.xai.target === p.name ? " sel" : ""}" data-t="${esc(p.name)}" title="${esc(p.desc)}">
        <span class="name">${esc(p.name)}${p.by_image ? " <small>이미지</small>" : ""}</span>
        <span class="bar-track"><span class="bar-fill${p.is_label ? " on" : ""}" style="width:${p.score * 100}%"></span><span class="bar-mark" style="left:${p.threshold * 100}%"></span></span>
        <span class="bar-val">${p.score.toFixed(2)}</span>
      </button>`).join("");
    box.querySelectorAll("button").forEach((b) => { b.onclick = () => act("/api/adc/target", { target: b.dataset.t }); });
    $("thr-note").textContent = S.model.kind === "cnn"
      ? `${S.model.name} · 눈금 = 패턴별 판정선. 넘은 패턴은 모두 라벨`
      : `눈금 = 판정선 ${S.threshold}. 넘은 패턴은 모두 라벨`;
  }

  function renderTypes() {
    const entries = Object.entries(S.types);
    let start = -Math.PI / 2;
    const r = 46, cx = 60, cy = 60;
    let html = `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#edf0f4" stroke-width="16"/>`;
    for (const [k, t] of entries) {
      if (!t.share) continue;
      if (t.share >= 0.999) {
        html += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${css(TYPE_COLOR[k])}" stroke-width="16"/>`;
        continue;
      }
      const end = start + t.share * 2 * Math.PI;
      const large = t.share > 0.5 ? 1 : 0;
      html += `<path d="M${cx + r * Math.cos(start)} ${cy + r * Math.sin(start)}A${r} ${r} 0 ${large} 1 ${cx + r * Math.cos(end)} ${cy + r * Math.sin(end)}" fill="none" stroke="${css(TYPE_COLOR[k])}" stroke-width="16"/>`;
      start = end;
    }
    html += `<text x="${cx}" y="${cy + 6}" text-anchor="middle" font-size="19" font-weight="700" fill="#17212e">${S.defect_dies}</text>`;
    $("type-ring").innerHTML = html;
    $("type-list").innerHTML = entries.map(([k, t]) => `<li><span><i style="background:${css(TYPE_COLOR[k])}"></i>${esc(t.ko)}</span><span class="num">${t.dies}다이 ${pct(t.share)}</span></li>`).join("");
  }

  function renderZones() {
    $("zone-bars").innerHTML = ["center", "middle", "edge"].map((k) => [k, S.zones[k]]).map(([k, z]) => `
      <div class="bar-row"><span class="name">${ZONE_KO[k]} <small>${z.dies}다이</small></span>
      <span class="bar-track"><span class="bar-fill on" style="width:${z.fill_pct}%"></span></span>
      <span class="bar-val">${z.fill_pct.toFixed(0)}%</span></div>`).join("");
  }

  function renderRose() {
    const c = 100, R = 82;
    const max = Math.max(1, ...S.sectors);
    let html = "";
    for (const f of [0.5, 1]) html += `<circle cx="${c}" cy="${c}" r="${R * f}" fill="none" stroke="#e1e6ec"/>`;
    S.sectors.forEach((v, i) => {
      // 섹터 0 은 각도 -180° 부터 시작 (engine: atan2 기준, 위쪽이 +y)
      const a0 = -Math.PI + (i * Math.PI) / 4, a1 = a0 + Math.PI / 4;
      const r = R * (v / max);
      if (!v) return;
      const pt = (a) => `${c + r * Math.cos(a)} ${c - r * Math.sin(a)}`;
      html += `<path d="M${c} ${c}L${pt(a0)}A${r} ${r} 0 0 0 ${pt(a1)}Z" fill="#004898" fill-opacity=".78" stroke="#fff"><title>${v}건</title></path>`;
    });
    $("rose").innerHTML = html;
  }

  function renderStain() {
    const m = S.stain;
    $("stain-card").style.display = m ? "" : "none";
    if (!m) return;
    $("stain-facts").innerHTML = `
      <dt>분포</dt><dd>${esc(m.distribution)}</dd>
      <dt>영역 수</dt><dd>${m.region_count}${m.has_hole ? " (구멍 있음)" : ""}</dd>
      <dt>면적</dt><dd>${m.area_mm2.toLocaleString()} mm²</dd>
      <dt>위치</dt><dd>${ZONE_KO[m.location]} 존</dd>
      <dt>무게중심</dt><dd>${m.centroid_mm.x}, ${m.centroid_mm.y} mm</dd>
      <dt>점유</dt><dd>${m.die_count}다이 (${m.coverage_pct}%)</dd>`;
  }

  // ── XAI ──────────────────────────────────────────────────────
  function renderXai() {
    const x = S.xai;
    $("xai-title").textContent = `${x.target} 판정 근거`;
    drawMap($("xai-mini"), "xai", false);
    const chan = { all: "전체", stain: "스테인", scratch: "스크래치", particle: "파티클" };
    const ex = x.rule || x;            // 막대 = 공간 특징 기여 (학습 모델일 때는 같은 입력의 규칙 기반 해석)
    $("xai-meta").textContent = x.rule
      ? `학습 모델 점수 ${x.score.toFixed(2)} · 다이 지도와 반사실은 학습 모델 기준 · 막대는 규칙 기반 공간 특징 해석 (규칙 점수 ${x.rule.score.toFixed(2)})`
      : `점수 ${x.score.toFixed(2)} · ${chan[x.channel] || "-"} 채널 기준`;
    const list = (ex.contributions || []).filter((c) => Math.abs(c.impact) >= 0.01);
    const items = [{ label: "기준값", value: null, impact: ex.bias ?? 0 }, ...list];
    const logit = ex.logit ?? 0;
    const max = Math.max(1, ...items.map((c) => Math.abs(c.impact)), Math.abs(logit));
    const bar = (v) => {
      const w = (Math.abs(v) / max) * 50;
      return `<span class="c-bar ${v >= 0 ? "pos" : "neg"}" style="${v >= 0 ? `left:50%` : `left:${50 - w}%`};width:${w}%"></span>`;
    };
    $("contrib").innerHTML = items.map((c) => `
      <div class="c-row"><span class="lbl">${esc(c.label)}${c.value === null ? "" : ` <small>${c.value.toFixed(2)}</small>`}</span>
      <span class="c-axis">${bar(c.impact)}</span><span class="val">${c.impact >= 0 ? "+" : ""}${c.impact.toFixed(2)}</span></div>`).join("") +
      `<div class="c-row total"><span class="lbl">합계 (로그오즈)</span><span class="c-axis">${bar(logit)}</span><span class="val">${logit >= 0 ? "+" : ""}${logit.toFixed(2)}</span></div>`;

    const cf = x.counterfactual;
    let html;
    if (cf.found) {
      html = `<p class="cf">결함 다이 <strong>${cf.dies.length}개</strong>가 없었다면 판정은 <strong>${esc(cf.target)}</strong>에서 <strong>${esc(cf.after)}</strong>로 바뀝니다.</p>
        <div class="ids">${cf.dies.map((i) => `<span>${i}</span>`).join("")}</div>`;
    } else if (x.score >= x.threshold) {
      html = `<p class="cf">결함 다이 ${cf.dies?.length ?? 0}개를 지워도 <strong>${esc(x.target)}</strong> 판정이 유지됩니다. 분포가 넓게 지지하는 판정입니다.</p>`;
    } else {
      html = `<p class="cf">${esc(x.target)}은(는) 판정선 아래라 반사실을 계산하지 않습니다.</p>`;
    }
    $("cf-body").innerHTML = html;

    const r = S.risk;
    $("risk-num").textContent = `${r.score} / 100`;
    $("risk-terms").innerHTML = r.terms.map((t) => `
      <div class="bar-row"><span class="name">${esc(t.name)}</span>
      <span class="bar-track"><span class="bar-fill on" style="width:${(t.value / t.max) * 100}%"></span></span>
      <span class="bar-val">${t.value}/${t.max}</span></div>`).join("");
    $("risk-advice").textContent = r.advice;
  }

  // ── 능동 추론 ────────────────────────────────────────────────
  function renderBelief() {
    const b = S.belief;
    const box = $("hyp-bars");
    box.classList.add("wide-name");
    if (!b.hypotheses.length) {
      box.innerHTML = '<p class="empty">결함 패턴이 없어 원인 가설을 세우지 않습니다.</p>';
    } else {
      box.innerHTML = b.hypotheses.map((h, i) => `
        <div class="bar-row" title="${esc(h.process)}"><span class="name">${esc(h.name)}</span>
        <span class="bar-track"><span class="bar-fill${i === 0 ? " on" : ""}" style="width:${h.p * 100}%"></span>
        <span class="bar-mark" style="left:${h.image_p * 100}%"></span></span>
        <span class="bar-val">${pct(h.p)}</span></div>`).join("");
    }
    $("entropy").textContent = b.entropy === null ? "" : `불확실성 ${b.entropy.toFixed(2)} bit`;

    const c = $("checks");
    let html = "";
    if (b.converged && b.hypotheses[0]) {
      html += `<div class="converged"><b>${esc(b.hypotheses[0].name)}</b>(으)로 수렴했습니다 (${pct(b.hypotheses[0].p)}). 추가 확인으로 더 검증할 수 있습니다.</div>`;
    }
    if (b.checks.length) {
      html += b.checks.map((q) => `
        <div class="check"><p class="check-q">${esc(q.ask)}</p>
        <p class="check-m">${esc(q.where)} · 기대 정보 이득 ${q.gain.toFixed(2)} bit · ${esc(q.target_name)} 구분</p>
        <div class="check-a">
          <button type="button" class="btn small" data-c="${esc(q.id)}" data-a="yes">예</button>
          <button type="button" class="btn small" data-c="${esc(q.id)}" data-a="no">아니오</button>
          <button type="button" class="btn small" data-c="${esc(q.id)}" data-a="unknown">모름</button>
        </div></div>`).join("");
    } else if (!b.converged) {
      html += '<p class="empty">지금 필요한 확인 항목이 없습니다.</p>';
    }
    c.innerHTML = html;
    c.querySelectorAll("[data-c]").forEach((btn) => {
      btn.onclick = () => act("/api/adc/answer", { check: btn.dataset.c, answer: btn.dataset.a });
    });
    $("undo").hidden = !b.answered.length;
  }

  function renderTrace() {
    $("trace").innerHTML = S.trace.map((t) => `<li class="${esc(t.kind)}"><b>${esc(t.title)}</b><span>${esc(t.text)}</span></li>`).join("");
  }

  // ── 조치 ─────────────────────────────────────────────────────
  function renderAction() {
    const h = S.belief.hypotheses[0];
    $("action-body").innerHTML = h
      ? `<p class="action-name">${esc(h.name)}</p><p class="action-proc">${esc(h.process)} · 현재 확률 ${pct(h.p)}</p><p>${esc(h.action)}</p>
         <p class="note">${S.belief.converged ? "원인이 수렴했습니다. 조치를 진행하세요." : "아직 원인이 수렴하지 않았습니다. 4단계 확인 항목을 먼저 진행하세요."}</p>`
      : '<p class="empty">필요한 조치가 없습니다. 정기 모니터링을 유지합니다.</p>';

    const y = S.yield, p = S.projected;
    $("yield-body").innerHTML = `
      <div class="yield-now lv-${y.level}">${y.yield_pct}%</div>
      <div class="yield-track"><span class="yield-up" style="left:${y.yield_pct}%;width:${p ? p.yield_pct - y.yield_pct : 0}%"></span><span class="yield-fill" style="width:${y.yield_pct}%"></span></div>
      ${p ? `<p class="note">빗금: 판정을 지지하는 ${p.recovered_die}다이가 모두 회복될 때의 상한 ${p.yield_pct}%</p>` : ""}
      <div class="dies4">
        <div><b>${y.good_die}</b><span>정상</span></div><div><b>${y.recheck_die}</b><span>재검사</span></div>
        <div><b>${y.defect_die}</b><span>불량</span></div><div><b>${y.active_die}</b><span>전체</span></div>
      </div>`;

    $("fb-count").textContent = `기록 ${S.session.feedback_count}건`;
    const lab = $("fb-labels");
    if (!lab.children.length) {
      lab.innerHTML = S.patterns.map((p) => `<label><input type="checkbox" value="${esc(p.name)}"> ${esc(p.name)}</label>`).join("");
    }
    const cause = $("fb-cause");
    const opts = ['<option value="">모름 / 해당 없음</option>', ...S.belief.hypotheses.map((x) => `<option value="${esc(x.id)}">${esc(x.name)}</option>`)];
    const keep = cause.value;
    cause.innerHTML = opts.join("");
    cause.value = keep;

    $("recipe").innerHTML = S.recipe.map((r) => `
      <div class="r-card${r.related ? " on" : ""}"><b>${esc(r.param)}</b><span>${esc(r.affects)}</span><p>${esc(r.note)}</p></div>`).join("");
  }

  function renderEngines() {
    $("engine-rows").innerHTML = Object.values(S.engines).map((e) => `
      <tr><td>${esc(e.stage)}</td><td>${esc(e.now)}</td><td>${esc(e.next)}</td>
      <td><span class="pill ${e.live ? "live" : "sim"}">${e.live ? "동작 중" : "학습 전 대체"}</span></td></tr>`).join("");
  }

  // ── 이벤트 ───────────────────────────────────────────────────
  $("scenario").onchange = (e) => {
    if (!e.target.value) return;
    selected = null;
    act("/api/adc/scan", { id: e.target.value });
  };
  $("map-modes").onclick = (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    mode = b.dataset.mode;
    renderMap();
  };
  $("show-zones").onchange = renderMap;
  $("show-overlay").onchange = renderInput;
  $("show-grid").onchange = renderInput;
  $("input-modes").onclick = (e) => {
    const b = e.target.closest("button");
    if (!b || b.disabled) return;
    inputMode = b.dataset.v;
    renderInput();
  };
  $("show-xai-map").onclick = () => {
    mode = "xai";
    renderMap();
    $("s-input").scrollIntoView();
  };
  $("undo").onclick = () => act("/api/adc/answer/undo", {});
  const pick = (el) => {
    selected = Number(el.dataset.id);
    renderMap();
    renderInspector();
  };
  $("wafer-map").onclick = (e) => {
    const g = e.target.closest(".die-cell");
    if (g) pick(g);
  };
  $("wafer-map").onkeydown = (e) => {
    const g = e.target.closest?.(".die-cell");
    if (g && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); pick(g); }
  };

  async function feedback(body, msg) {
    try {
      const r = await api("/api/adc/feedback", body);
      $("fb-msg").textContent = msg;
      $("fb-count").textContent = `기록 ${r.count}건`;
      S.session.feedback_count = r.count;
    } catch (e) {
      $("errbar").textContent = e.message;
      $("errbar").hidden = false;
    }
  }
  $("fb-confirm").onclick = () => feedback({ verdict: "confirm", labels: S.labels, cause: S.belief.hypotheses[0]?.id ?? null, note: "" }, "AI 판정을 확정해 재학습 라벨 후보로 저장했습니다.");
  $("fb-toggle").onclick = () => { $("fb-form").hidden = !$("fb-form").hidden; };
  $("fb-form").onsubmit = (e) => {
    e.preventDefault();
    const labels = [...$("fb-labels").querySelectorAll("input:checked")].map((i) => i.value);
    feedback({ verdict: "correct", labels, cause: $("fb-cause").value || null, note: $("fb-note").value }, "수정 판정을 재학습 라벨 후보로 저장했습니다.");
    $("fb-form").hidden = true;
  };

  // ── 새 스캔 감시 ─────────────────────────────────────────────
  // 진행 표시는 위 스캔 구역(scan.js)이 맡고, 여기서는 새 맵이 생기면 결과를 불러오기만 한다
  let newest = null;          // 이 화면이 이미 알고 있는 가장 새 맵
  async function watchScan() {
    if (!S) return;
    let busy = false;
    try {
      const st = await (await fetch("/api/scan/status", { cache: "no-store" })).json();
      const m = st.wafer_map;
      busy = st.job.busy;
      newest ??= S.scans[0]?.id ?? null;
      if (!busy && m?.wafer && m.dir !== newest && m.dir > (newest || "")) {
        newest = m.dir;
        if (!S.session.edited && !S.belief.answered.length) {
          selected = null;
          await act("/api/adc/scan", { id: m.dir });
        }
      }
    } catch (e) { /* 다음 확인에서 다시 시도 */ }
    clearTimeout(watchTimer);
    watchTimer = setTimeout(watchScan, busy ? 1500 : 3000);
  }
  let watchTimer = null;
  watchTimer = setTimeout(watchScan, 1000);

  (async () => {
    try {
      S = await api("/api/adc/state");
      const want = new URLSearchParams(location.search).get("scan");
      const latest = S.scans[0]?.id;
      if (want && want !== S.session.scan_id) S = await api("/api/adc/scan", { id: want });
      else if (!want && latest && (S.session.source !== "scan" || latest !== S.session.scan_id)) S = await api("/api/adc/scan", { id: latest });
      render();
    } catch (e) {
      $("errbar").textContent = `상태를 불러오지 못했습니다: ${e.message}`;
      $("errbar").hidden = false;
    }
  })();
})();
