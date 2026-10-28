import { useState } from "react";
import { api, dt, f, n } from "./api";

type Tab = "explain" | "motifs" | "agent" | "eval" | "diag";
export const STEPS = [
  "Open case",
  "Baseline ranking",
  "Explain history",
  "Replay removal",
  "Motif evidence",
  "Agent + held-out",
  "Reset & export",
];

export const labelClass = (l: string) =>
  l === "Comedy" ? "lab-comedy" : l === "Crime/Thriller" ? "lab-crime" : l === "Mystery" ? "lab-mystery" : "lab-other";

export function Status({ s }: { s?: string }) {
  if (!s) return null;
  const cls = s.startsWith("fresh") ? "st-fresh" : s.startsWith("cached") ? "st-cached" : "st-recorded";
  return <span className={`status ${cls}`}>{s}</span>;
}

/* ---------------------------------------------------------------- left */
export function HistoryPanel({ cd, removedSet, selectedSet }: any) {
  return (
    <section className="panel left">
      <h3><span className="dot blue" />Member history <small>user {cd.user_id} · member {cd.case.member}</small></h3>
      <p className="hint">Model input: rating events strictly before t = {dt(cd.case.t)} ({cd.history.length} in window). Ratings are interactions, not watch logs.</p>
      <ol className="timeline">
        {cd.history.map((h: any) => (
          <li key={h.event_id} className={`ev ${labelClass(h.label)} ${selectedSet.has(h.event_id) ? "sel" : ""} ${removedSet.has(h.event_id) ? "removed" : ""}`}>
            <div className="ev-top"><code>e{h.event_id}</code><span className="tag">{h.label}</span>{removedSet.has(h.event_id) && <span className="rm">removed in replay</span>}</div>
            <div className="ev-title">{h.title}</div>
            <div className="ev-meta">{h.genres} · rated {h.rating} · {dt(h.ts)}</div>
          </li>
        ))}
      </ol>
      <div className="query">
        <h3><span className="dot gold" />Current query <small>{cd.case.qid} · {cd.case.kind}</small></h3>
        <div className="qtext">“{cd.case.query}”</div>
        <p className="hint">Prototype query written for this demo (MovieLens has no searches). Runtime constraint “under 100 minutes”: <b>unavailable</b> — runtime unknown for every title; not applied.</p>
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- center */
export function Dataflow({ cd, model, step, ex, cur, meta, setModel }: any) {
  const M = meta.models[model];
  const box = (x: number, y: number, w: number, h: number, cls: string, t1: string, t2: string, active: boolean) => (
    <g className={`df ${cls} ${active ? "active" : ""}`}>
      <rect x={x} y={y} width={w} height={h} rx={8} />
      <text x={x + w / 2} y={y + h / 2 - 4} textAnchor="middle" className="t1">{t1}</text>
      <text x={x + w / 2} y={y + h / 2 + 12} textAnchor="middle" className="t2">{t2}</text>
    </g>
  );
  return (
    <section className="panel center">
      <h3><span className="dot green" />Dataflow & model target</h3>
      <svg viewBox="0 0 420 300" className="dfsvg">
        <defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#66717D" /></marker></defs>
        {box(10, 20, 120, 46, "hist", "Member history H", `${cd.history.length} events`, step <= 3)}
        {box(10, 86, 120, 46, "query", "Query q", cd.case.qid, step <= 1)}
        {box(160, 50, 110, 56, "model", "Ranker", model === "F0" ? "current (frozen)" : "less history", step === 1 || step === 5)}
        {box(300, 50, 110, 56, "out", "Ranked titles", "30 candidates", step === 1)}
        {box(160, 160, 110, 50, "expl", "Explain + replay", ex ? `S = ${ex.selected_event_ids.length} events` : "TemGX-adapted", step === 2 || step === 3)}
        {box(300, 160, 110, 50, "motif", "Motif evidence", "CATGX-adapted", step === 4)}
        {box(160, 235, 250, 50, "eval", "Evaluated model change", "less history vs current, held-out", step >= 5)}
        <path d="M130,43 L160,70" markerEnd="url(#arr)" className="ln" />
        <path d="M130,109 L160,90" markerEnd="url(#arr)" className="ln" />
        <path d="M270,78 L300,78" markerEnd="url(#arr)" className="ln" />
        <path d="M215,106 L215,160" markerEnd="url(#arr)" className="ln" />
        <path d="M270,185 L300,185" markerEnd="url(#arr)" className="ln" />
        <path d="M355,210 L355,235" markerEnd="url(#arr)" className="ln" />
      </svg>
      <div className="target">
        <div className="seg sm">
          <button className={model === "F0" ? "on" : ""} onClick={() => setModel("F0")}>Current ranker</button>
          <button className={model === "F1" ? "on" : ""} onClick={() => setModel("F1")}>Less history, specific searches</button>
        </div>
        <p><b>score(u,q,i,t)</b> = {model === "F1" ? "gate(q,i)·" : ""}{n(M.weights[0], 3)}·z<sub>beh</sub> + {n(M.weights[1], 3)}·z<sub>rel</sub> + {n(M.weights[2], 3)}·z<sub>pop</sub>
          {model === "F1" && <> ; gate = exp({n(M.gate[0], 2)} {f(M.gate[1], 2)}·spec(q) {f(M.gate[2], 2)}·z<sub>rel</sub>)</>}</p>
        <p className="hint">Units: pairwise log-odds (fit on DEV judged pairs). z<sub>beh</sub> = standardised logit of the frozen temporal backbone (TGAT-style, trained on TRAIN); z<sub>rel</sub> = spaCy vectors cosine (title + genres only); z<sub>pop</sub> = log(1+interactions before t). Component values are model weights × features, not causal contributions.</p>
        <p className="hint">Explanation target: <b>behavioural logit margin A−B</b> of the backbone; full-ranker change reported separately. Config <code>{M.config_hash}</code> · backbone <code>{cur.backbone_sha256}</code> · snapshot <code>{cur.snapshot_hash}</code> <Status s={cur.status} /></p>
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- right */
export function Bar({ v, max, cls }: { v: number; max: number; cls: string }) {
  const w = Math.min(Math.abs(v) / max, 1) * 50;
  return <span className="bar"><span className={`fill ${cls}`} style={{ width: `${w}%`, left: v >= 0 ? "50%" : `${50 - w}%` }} /></span>;
}
export function RankingPanel({ cd, cur, base, A, B, removed }: any) {
  const comp = cur.components;
  const top = cur.order.slice(0, 10) as number[];
  const show = [...top];
  [A, B].forEach((x: number) => { if (x !== undefined && !show.includes(x)) show.push(x); });
  const max = Math.max(...show.flatMap((i: number) => [Math.abs(comp.behaviour[i]), Math.abs(comp.relevance[i]), Math.abs(comp.popularity[i])]), 0.5);
  const ab = cur.ab, ab0 = base.ab;
  return (
    <section className="panel right">
      <h3><span className="dot green" />Ranked results <small>{cur.model_id} · {removed.length ? `${removed.length} events removed` : "original history"}</small></h3>
      {ab && (
        <div className="abcard">
          <div><span className="pill rose">A</span> {cd.candidates[A].title} <b>#{ab.rank_A}</b> {ab0 && ab0.rank_A !== ab.rank_A && <small>(was #{ab0.rank_A})</small>}</div>
          <div><span className="pill teal">B</span> {cd.candidates[B].title} <b>#{ab.rank_B}</b> {ab0 && ab0.rank_B !== ab.rank_B && <small>(was #{ab0.rank_B})</small>}</div>
          <table className="mini">
            <thead><tr><th></th><th>score A</th><th>score B</th><th>margin A−B</th><th>beh. logit margin</th></tr></thead>
            <tbody>
              <tr><td>original</td><td>{n(ab0.score_A)}</td><td>{n(ab0.score_B)}</td><td>{f(ab0.margin)}</td><td>{f(ab0.beh_margin, 2)}</td></tr>
              {removed.length > 0 && <tr className="hl"><td>H \ S</td><td>{n(ab.score_A)}</td><td>{n(ab.score_B)}</td><td>{f(ab.margin)}</td><td>{f(ab.beh_margin, 2)}</td></tr>}
              {removed.length > 0 && <tr><td>change</td><td>{f(ab.score_A - ab0.score_A)}</td><td>{f(ab.score_B - ab0.score_B)}</td><td><b>{f(ab.margin - ab0.margin)}</b></td><td>{f(ab.beh_margin - ab0.beh_margin, 2)}</td></tr>}
            </tbody>
          </table>
          <p className="hint">Margin components (original): behaviour {f(ab0.component_margin.behaviour)}, relevance {f(ab0.component_margin.relevance)}, popularity {f(ab0.component_margin.popularity)}. Inference for this call: {n(cur.latency_ms, 2)} ms (first call in a process is cold).</p>
        </div>
      )}
      <div className="legend"><span><i className="c-beh" />behaviour</span><span><i className="c-rel" />relevance</span><span><i className="c-pop" />popularity</span><span className="hint">grade = rubric judgment (0/1/2)</span></div>
      <ol className="rank">
        {show.map((i: number) => {
          const c = cd.candidates[i];
          return (
            <li key={i} className={i === A ? "isA" : i === B ? "isB" : ""}>
              <span className="rk">#{cur.rank[i]}</span>
              <span className="tt">{i === A && <span className="pill rose">A</span>}{i === B && <span className="pill teal">B</span>}{c.title}
                <small> · {c.genres} · grade {c.grade} · {c.train_interactions} train ratings · runtime unknown</small></span>
              <span className="bars"><Bar v={comp.behaviour[i]} max={max} cls="c-beh" /><Bar v={comp.relevance[i]} max={max} cls="c-rel" /><Bar v={comp.popularity[i]} max={max} cls="c-pop" /></span>
              <span className="sc">{n(comp.total[i], 2)}</span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

/* ---------------------------------------------------------------- tabs */
export function ExplainTab({ cd, ex, budget, setBudget, doExplain, doReplay, doReset, resetCheck, removed }: any) {
  const hist = Object.fromEntries(cd.history.map((h: any) => [h.event_id, h]));
  return (
    <div className="tabbody">
      <div className="row">
        <div className="controls">
          <label>Explanation budget k <input type="number" min={1} max={5} value={budget} onChange={(e) => setBudget(+e.target.value)} /></label>
          <button className="primary" onClick={doExplain}>Explain A over B</button>
          <button onClick={() => doReplay()} disabled={!ex}>Replay with S removed</button>
          <button onClick={doReset}>Reset</button>
        </div>
        <a className="cite" href="https://proceedings.iclr.cc/paper_files/paper/2026/hash/36bc2989d4b3371654fce3cb9a1a6889-Abstract-Conference.html" target="_blank">TemGX, ICLR 2026 ↗</a>
      </div>
      {resetCheck && <div className={`note ${resetCheck.maxdiff === 0 && resetCheck.snapshot_equal ? "ok" : "bad"}`}>Reset re-ran inference on the original history: max |Δscore| = {resetCheck.maxdiff} (tolerance 0), snapshot hash equal: {String(resetCheck.snapshot_equal)}. {resetCheck.at}</div>}
      {!ex ? <p className="hint">Explains the decision “A ranked above B”. Target: the backbone's behavioural logit margin. Candidate pool = the member's events in the model window (L = 1). Influence p<sub>e</sub> is measured by single-event replay; Φ = p · g(TRD) · decay; greedy select-and-verify (each addition must lower the margin on replay).</p> : (
        <>
          <div className="kpis">
            <div><span>selected S</span><b>{ex.selected_event_ids.map((e: number) => `e${e}`).join(", ") || "∅"}</b><small>{ex.selected_event_ids.map((e: number) => hist[e]?.title).join(" · ")}</small></div>
            <div><span>behavioural margin</span><b>{n(ex.beh_margin_before, 2)} → {n(ex.beh_margin_after, 2)}</b><small>change {f(ex.beh_margin_change, 2)} ({n(-100 * ex.beh_margin_change / ex.beh_margin_before, 0)}%)</small></div>
            <div><span>full-ranker margin</span><b>{f(ex.margin_before)} → {f(ex.margin_after)}</b><small>change {f(ex.margin_change)} · order flipped: <b>{String(ex.full_ranker_order_flipped)}</b></small></div>
            <div><span>ranks A / B</span><b>#{ex.rank_change.A[0]}→#{ex.rank_change.A[1]} / #{ex.rank_change.B[0]}→#{ex.rank_change.B[1]}</b><small>budget {ex.budget}, removed {ex.n_removed}</small></div>
            <div><span>latency</span><b>{n(ex.latency_ms.total_explanation, 1)} ms</b><small>{ex.n_model_inferences} model inferences · replay {n(ex.latency_ms.replay_after, 2)} ms</small></div>
          </div>
          <div className="two">
            <table className="tbl">
              <thead><tr><th>event</th><th>title</th><th>label</th><th>single-removal effect</th><th>p</th><th>g(TRD)</th><th>decay</th><th>Φ</th><th>attn</th></tr></thead>
              <tbody>{ex.candidates.map((c: any) => (
                <tr key={c.event_id} className={ex.selected_event_ids.includes(c.event_id) ? "selrow" : ""}>
                  <td><code>e{c.event_id}</code></td><td>{hist[c.event_id]?.title}</td><td><span className={`tag ${labelClass(hist[c.event_id]?.label)}`}>{hist[c.event_id]?.label}</span></td>
                  <td>{f(c.single_removal_effect, 2)}</td><td>{n(c.p, 2)}</td><td>{n(c.g_trd, 2)}</td><td>{n(c.decay, 2)}</td><td><b>{n(c.phi, 3)}</b></td><td>{n(c.attention, 2)}</td></tr>))}</tbody>
            </table>
            <div>
              <h4>Equal-budget comparison (k = {ex.comparison.budget})</h4>
              <p className="hint">Reduction of the behavioural margin when each strategy's events are removed (larger = those events support A more).</p>
              {[["TemGX-adapted", ex.comparison.temgx_adapted], ["Recency (last k)", ex.comparison.recency_topk], ["Attention top-k", ex.comparison.attention_topk], [`Random (mean of ${ex.comparison.random_n})`, ex.comparison.random_mean]].map(([l, v]: any) => {
                const mx = Math.max(Math.abs(ex.comparison.temgx_adapted), Math.abs(ex.comparison.recency_topk), Math.abs(ex.comparison.attention_topk ?? 0), 1);
                return <div key={l} className="cmp"><span>{l}</span><span className="bar wide"><span className={`fill ${v >= 0 ? "c-pos" : "c-neg"}`} style={{ width: `${Math.abs(v) / mx * 50}%`, left: v >= 0 ? "50%" : `${50 - Math.abs(v) / mx * 50}%` }} /></span><b>{f(v, 2)}</b></div>;
              })}
              <p className="hint">Interpretation: the model's prediction depends on these inputs. It is not the effect of a real member changing what they watched. No minimality, Shapley or approximation guarantee is claimed for this adaptation.</p>
              {removed.length > 0 && <p className="note ok">Replay applied: removed {removed.map((e: number) => `e${e}`).join(", ")}; query, candidates, eligibility, t, parameters, calibration and metadata held fixed; history rebuilt chronologically from the edited log.</p>}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export function MotifTab({ cd, mo, doMotifs }: any) {
  if (!mo) return <div className="tabbody"><button className="primary" onClick={doMotifs}>Compute motif evidence</button>
    <p className="hint">Is this an isolated history-sensitive prediction, or do similar model-attribution patterns appear in comparable observed contexts?</p></div>;
  const pm = mo.case.per_motif, ag = mo.aggregate.per_motif;
  return (
    <div className="tabbody">
      <div className="row"><p className="formula">S(m) = Σ<sub>b</sub> ω(b) [ μ(m,b) − μ<sub>base</sub>(b) ] — cells b = recency × duration (doubling bins) × title-activity tertile; g(r) = drop in behavioural margin when r's events are removed.</p>
        <a className="cite" href="https://doi.org/10.1145/3770855.3818100" target="_blank">CATGX, KDD 2026 ↗</a></div>
      <div className="two">
        <div>
          <h4>This prediction <Status s="fresh-inference" /></h4>
          <p className="hint">{mo.case.n_occurrences} occurrences in {mo.case.n_cells} cell(s), exact enumeration (π = 1), {n(mo.case.latency_ms, 1)} ms. G* = {mo.case.G_star_event_ids.map((e: number) => `e${e}`).join(", ")} (behavioural margin change {f(mo.case.G_star_beh_margin_change, 2)}).</p>
          <table className="tbl"><thead><tr><th>motif</th><th>occurrences</th><th>S(m)</th><th>mean g</th></tr></thead>
            <tbody>{Object.entries(pm).map(([k, v]: any) => <tr key={k} className={mo.case.selected_motifs.includes(k) ? "selrow" : ""}><td><b>{k}</b> {v.name}</td><td>{v.n_occurrences}</td><td>{v.S === null ? "–" : f(v.S, 2)}</td><td>{v.mean_g === null ? "–" : n(v.mean_g, 2)}</td></tr>)}</tbody></table>
        </div>
        <div>
          <h4>Comparable contexts: {mo.aggregate.comparable_cases} DEV failure cases <Status s="cached-measurement" /></h4>
          <p className="hint">{mo.aggregate.aggregation_note}</p>
          <table className="tbl"><thead><tr><th>motif</th><th>cases</th><th>members</th><th>mean S</th><th>S&gt;0</th><th>95% CI</th></tr></thead>
            <tbody>{Object.entries(ag).filter(([, v]: any) => v.cases_with_motif).map(([k, v]: any) => (
              <tr key={k}><td><b>{k}</b> {v.name}</td><td>{v.cases_with_motif}</td><td>{v.members}</td><td>{f(v.mean_S, 3)}</td><td>{n(100 * v.frac_S_positive, 0)}%</td>
                <td><CI lo={v.ci95_member_bootstrap[0]} hi={v.ci95_member_bootstrap[1]} span={1.2} /></td></tr>))}</tbody></table>
          <p className="hint">Adapted, not reproduced: no ANN sampling; title-activity instead of entity roles. S(m) is an observed-context-adjusted attribution contrast — not a causal effect of watching in this order.</p>
        </div>
      </div>
    </div>
  );
}

export function CI({ lo, hi, span = 0.2 }: { lo: number; hi: number; span?: number }) {
  const x = (v: number) => 50 + (v / span) * 50;
  const cls = lo > 0 ? "c-pos" : hi < 0 ? "c-neg" : "c-na";
  return <span className="ci"><span className="zero" /><span className={`rng ${cls}`} style={{ left: `${Math.max(0, x(lo))}%`, width: `${Math.max(1, Math.min(100, x(hi)) - Math.max(0, x(lo)))}%` }} /><small>[{f(lo, 3)}, {f(hi, 3)}]</small></span>;
}

export function AgentTab({ inv, setInv, openCall, setOpenCall, meta }: any) {
  const [running, setRunning] = useState(false);
  if (!inv) return <div className="tabbody">No investigation trace found.</div>;
  const byId = Object.fromEntries(inv.trace.map((t: any) => [t.artifact_id, t]));
  return (
    <div className="tabbody">
      <div className="row">
        <div><Status s={inv.status} /> <b>{inv.mode}</b><br /><small>investigation <code>{inv.investigation_id}</code> · exported {inv.exported_at} · {inv.trace.length} tool calls · code <code>{inv.provenance.code_version}</code></small></div>
        <div className="controls">
          {!meta.llm_mode_available && <span className="hint">Live LLM mode: no provider configured — not used.</span>}
          <button disabled={running} onClick={async () => { setRunning(true); try { setInv(await api.runInvestigation()); } finally { setRunning(false); } }}>{running ? "Re-running tools… (~40 s)" : "Re-run scripted workflow now"}</button>
          <a className="btn" href={`/api/export/${inv.investigation_id}`}>Export investigation JSON</a>
        </div>
      </div>
      <div className="two">
        <ol className="narr">
          {inv.narrative.map((s: any, i: number) => (
            <li key={i} className={`k-${s.kind}`}><span className="kind">{s.kind}</span> {s.text}
              <div className="cites">{s.cites.map((c: string) => <button key={c} className="cite-btn" onClick={() => setOpenCall(byId[c]?.call_id ?? null)}>{c}</button>)}</div></li>
          ))}
          <li className="k-diagnosis"><span className="kind">diagnosis</span>
            <b>Evidence:</b><ul>{inv.diagnosis.evidence.map((e: string) => <li key={e}>{e}</li>)}</ul>
            <b>Hypotheses:</b><ul>{inv.diagnosis.hypotheses.map((e: string) => <li key={e}>{e}</li>)}</ul>
            <b>Caveats:</b><ul>{inv.diagnosis.caveats.map((e: string) => <li key={e}>{e}</li>)}</ul></li>
        </ol>
        <div>
          <table className="tbl calls"><thead><tr><th>#</th><th>tool</th><th>artifact</th><th>cache</th><th>ms</th></tr></thead>
            <tbody>{inv.trace.map((t: any) => (
              <tr key={t.call_id} className={openCall === t.call_id ? "selrow" : ""} onClick={() => setOpenCall(t.call_id)}>
                <td>{t.call_id}</td><td>{t.tool}</td><td><code>{t.artifact_id}</code></td><td>{t.cache}</td><td>{n(t.latency_ms, 0)}</td></tr>))}</tbody></table>
          {openCall && <pre className="json">{JSON.stringify((({ outputs, ...r }: any) => ({ ...r, outputs: summarize(outputs) }))(inv.trace[openCall - 1]), null, 1)}</pre>}
        </div>
      </div>
    </div>
  );
}
export function summarize(o: any): any {
  if (o === null || typeof o !== "object") return o;
  if (Array.isArray(o)) return o.length > 12 ? [...o.slice(0, 12).map(summarize), `… ${o.length - 12} more`] : o.map(summarize);
  return Object.fromEntries(Object.entries(o).map(([k, v]) => [k, summarize(v)]));
}

export function EvalTab({ ev, meta }: any) {
  if (!ev) return <div className="tabbody">Loading…</div>;
  const fin = ev.comparisons["qi-final-v1"], dev = ev.comparisons["qi-dev-v1"];
  const slices = ["all", "explicit", "broad", "sparse_history", "dense_history", "explicit&sparse", "explicit&dense", "broad&sparse", "broad&dense"];
  const tableFor = (c: any, title: string) => (
    <div>
      <h4>{title} <small>runs {Object.entries(c.runs).map(([k, v]) => `${k}:${v}`).join(" · ")}</small></h4>
      <table className="tbl"><thead><tr><th>slice</th><th>cases</th><th>members</th><th>current</th><th>variant</th><th>Δ NDCG@10 (95% CI, member bootstrap)</th></tr></thead>
        <tbody>{slices.filter((s) => c.slices[s]).map((s) => {
          const e = c.slices[s], m = e.ndcg10;
          return <tr key={s} className={s === "explicit" || s === "broad" ? "keyrow" : ""}><td>{s}</td><td>{e.n_cases}</td><td>{e.n_members}{e.n_members < 30 && <small className="warn"> small</small>}</td><td>{n(m.baseline, 3)}</td><td>{n(m.variant, 3)}</td><td><CI lo={m.ci95[0]} hi={m.ci95[1]} span={0.16} /> <b>{f(m.delta, 4)}</b></td></tr>;
        })}</tbody></table>
      {c.slices.explicit.loweng_rel_recall10 && <p className="hint">Low-engagement judged-relevant recall@10 (explicit): {n(c.slices.explicit.loweng_rel_recall10.baseline)} → {n(c.slices.explicit.loweng_rel_recall10.variant)} ({f(c.slices.explicit.loweng_rel_recall10.delta, 4)}, CI [{f(c.slices.explicit.loweng_rel_recall10.ci95[0], 4)}, {f(c.slices.explicit.loweng_rel_recall10.ci95[1], 4)}])</p>}
    </div>
  );
  const tr = ev.temporal_recommendation;
  return (
    <div className="tabbody">
      <div className="note warnbox"><b>FINAL holdout, scored once</b> (test-period prediction times, query templates never used for fitting, showcase members excluded). Relying less on viewing history for specific searches <b>regresses the targeted explicit-intent slice</b> and <b>improves broad queries</b>; the query-independent control does the opposite. A tradeoff, not a win.</div>
      <div className="two">
        {tableFor(fin.F0_vs_F1, "FINAL · current vs less viewing history for specific searches")}
        {tableFor(fin.F0_vs_Fctrl, "FINAL · current vs less viewing history for every search (control)")}
      </div>
      <details><summary>DEV (fitting/selection data)</summary><div className="two">{tableFor(dev.F0_vs_F1, "DEV · current vs less history for specific searches")}{tableFor(dev.F0_vs_Fctrl, "DEV · current vs less history for every search")}</div></details>
      <div className="two">
        <div><h4>Regression cases (FINAL explicit, largest drops vs current)</h4><ul className="mono">{ev.regression_cases_F1_explicit.map((r: any) => <li key={r.case_id}>{r.case_id} {f(r.delta_ndcg10, 3)}</li>)}</ul></div>
        <div><h4>Temporal recommendation (backbone, shared by all variants)</h4>
          <p className="hint">{tr.protocol.candidates}</p>
          <table className="tbl"><thead><tr><th></th><th>HR@10</th><th>NDCG@10</th><th>n</th></tr></thead><tbody>
            <tr><td>val · backbone</td><td>{n(tr.val["HR@10"])}</td><td>{n(tr.val["NDCG@10"])}</td><td>{tr.val.n}</td></tr>
            <tr><td>test · backbone</td><td>{n(tr.test_backbone["HR@10"])}</td><td>{n(tr.test_backbone["NDCG@10"])}</td><td>{tr.test_backbone.n}</td></tr>
            <tr><td>test · popularity reference</td><td>{n(tr.test_popularity_reference["HR@10"])}</td><td>{n(tr.test_popularity_reference["NDCG@10"])}</td><td>{tr.test_popularity_reference.n}</td></tr></tbody></table>
          <p className="hint">The small backbone does not beat popularity on the test period. Offline public-data proxy; no Netflix production or A/B evidence.</p></div>
      </div>
      <a className="cite" href="https://arxiv.org/abs/2602.04154" target="_blank">Principle: evaluate a change per context (Materials Segmentation, arXiv:2602.04154) ↗</a>
    </div>
  );
}

export function DiagTab({ ev, neg }: any) {
  const lat = ev?.latency;
  return (
    <div className="tabbody two">
      <div>
        <h4>Text-encoder negation check <Status s={neg?.status} /></h4>
        {neg && <><p>cos(“a light cozy mystery <b>not</b> too intense”, same without “not”) = <b>{n(neg["cos(query, query_without_not)"])}</b></p>
          <p>cos(query, “intense”) = <b>{n(neg["cos(query, 'intense')"])}</b></p><p className="hint">{neg.note} This is why relevance alone can push intense titles up for this query (H3).</p></>}
      </div>
      <div>
        <h4>Measured latency <Status s="cached-measurement" /></h4>
        {lat && <><p className="hint">{lat.hardware.platform} · {lat.hardware.cpu_count} CPU · torch {lat.hardware.torch} (CPU) · created {lat.created}</p>
          <table className="tbl"><thead><tr><th>stage (warm)</th><th>n</th><th>p50 ms</th><th>p95 ms</th></tr></thead><tbody>
            {Object.entries(lat.warm).map(([k, v]: any) => <tr key={k}><td>{k}</td><td>{v.n}</td><td>{n(v.p50_ms, 2)}</td><td>{n(v.p95_ms, 2)}</td></tr>)}
            <tr><td>cold first inference (fresh process)</td><td>1</td><td>{n(lat.cold_first_call_ms.inference, 0)}</td><td>–</td></tr></tbody></table>
          <p className="hint">Workload: {lat.workload.candidates} candidates, {lat.workload.history_events_in_window} history events. {lat.cache_status}</p></>}
      </div>
    </div>
  );
}

export function Sources({ meta, close }: any) {
  return (
    <div className="drawer" onClick={close}>
      <div className="drawer-body" onClick={(e) => e.stopPropagation()}>
        <button className="ghost close" onClick={close}>Close</button>
        <h3>Sources</h3>
        <ul className="src">{meta.sources.map((s: any) => <li key={s.key}><b>{s.key}</b> — {s.cite} <a href={s.url} target="_blank">link ↗</a>{s.code && <> · <a href={s.code} target="_blank">code ↗</a></>}<br /><small>{s.used_for}</small></li>)}</ul>
        <h3>Dataset</h3>
        <p>{meta.dataset.source}; {meta.dataset.n_events.toLocaleString()} ratings, {meta.dataset.n_members} members, {meta.dataset.n_items.toLocaleString()} titles. Chronological split: train &lt; {meta.dataset.cutoffs.train_end_utc}, val &lt; {meta.dataset.cutoffs.val_end_utc}, test after. Raw SHA-256: ratings {meta.dataset.raw_sha256["ratings.csv"].slice(0, 16)}…</p>
        <p className="hint">Obtained from a GitHub mirror whose file counts match the official GroupLens README; licence: research use, no endorsement, citation required, non-commercial without permission. Runtime metadata: none (no permitted source reachable) — runtime constraints are not applied.</p>
        <h3>Backbone</h3>
        <p className="hint">Checkpoint <code>{meta.backbone.checkpoint_sha256}</code>, epoch {meta.backbone.selected_epoch} selected on val; {JSON.stringify(meta.backbone.hyperparameters)}</p>
      </div>
    </div>
  );
}

export function Illustrative() {
  return (
    <div className="illus">
      <div className="banner warn"><b>ILLUSTRATIVE SCENARIO — fictional titles, synthetic story, no model outputs.</b> This view only explains the concept from the slides. It provides no evidence of model performance. Switch to “Measured data” for real inference.</div>
      <div className="illus-grid">
        <div className="panel"><h3><span className="dot blue" />Viewing history H (fictional)</h3>
          <ol className="timeline"><li className="ev lab-comedy"><div className="ev-title">Comedy · t0</div></li><li className="ev lab-crime sel"><div className="ev-title">Crime 1 · t1</div></li><li className="ev lab-crime sel"><div className="ev-title">Crime 2 · t2</div></li></ol>
          <div className="query"><div className="qtext">“I want a light, cozy mystery, not too intense, under 100 minutes.”</div></div></div>
        <div className="panel"><h3><span className="dot green" />Question</h3><p>Why does A outrank B? Which history drives it? Does it recur? Does relying less on viewing history for specific searches help without hurting broad queries?</p>
          <p className="hint">In measured mode the same structure appears with real MovieLens data: a member whose window contains two Crime/Thriller ratings, a light-mystery query, an intense crime title ranked first and a children's comedy-mystery lower.</p></div>
        <div className="panel"><h3><span className="dot green" />Candidates (fictional)</h3>
          <div><span className="pill rose">A</span> “Night Chase” — intense crime, 94 min</div>
          <div><span className="pill teal">B</span> “The Missing Teacup” — cozy mystery, 92 min, sparse engagement</div></div>
      </div>
    </div>
  );
}
