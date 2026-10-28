import { useEffect, useState } from "react";
import { MCOL } from "./SubGraph";

// CATGX structures behind one search, made usable: each structure drawn on its own (real people, titles, dates),
// its causal test (delete it, re-run the model, compare with random deletions), and an LLM explanation of why it
// moves the prediction, written only from that evidence and number-checked against it. /api/structure/{case_id}.
const RED = "#c82434", GREEN = "#2f7a43", BLUE = "#7194b9";
const pct = (x: number) => `${Math.round(100 * x)}%`;
const short = (s: string, k = 18) => (s.length > k ? s.slice(0, k - 1) + "…" : s);
export const PLAIN: Record<string, string> = { POP: "Recent fans piled on", CO: "Two people rated the same movie", SEQ: "A run of ratings in one sitting", BRIDGE: "Shared taste leads to the title", BRIDGEB: "Shared taste leads to the better match" };

// Each CATGX structure drawn as its shape: crowd / shared movie / run of ratings as triangles, a shared-taste path as a
// quadrilateral. Solid edges = the real ratings (stars, date); the dashed edge closes the shape = what the graph model
// infers from it (e.g. "these two people look alike"), which is how the structure reaches the prediction.
const SHAPE: Record<string, { pts: [number, number][]; close: [number, number]; note: string }> = {
  POP: { pts: [[120, 182], [400, 182], [260, 52]], close: [0, 1], note: "both rated it recently → it looks in demand" },
  CO: { pts: [[120, 182], [260, 52], [400, 182]], close: [0, 2], note: "same movie → they look alike to the model" },
  SEQ: { pts: [[260, 52], [120, 182], [400, 182]], close: [1, 2], note: "one after the other → current taste" },
  BRIDGE: { pts: [[92, 182], [190, 52], [330, 52], [428, 182]], close: [0, 3], note: "shared taste → the title is pulled toward this person" },
  BRIDGEB: { pts: [[92, 182], [190, 52], [330, 52], [428, 182]], close: [0, 3], note: "shared taste → the better match is pulled toward this person" },
};

export function Diagram({ s, ev }: any) {
  const ids: number[] = (s.nodes ?? []).map((x: any) => x.id);
  const nm: Record<number, string> = Object.fromEntries((s.nodes ?? []).map((x: any) => [x.id, x.name]));
  const sh = SHAPE[s.motif];
  const col = MCOL[s.motif] ?? "#555";
  const P: Record<number, [number, number]> = {};
  ids.forEach((v, j) => { P[v] = sh?.pts[j] ?? [260 + 150 * Math.cos((2 * Math.PI * j) / ids.length), 120 + 70 * Math.sin((2 * Math.PI * j) / ids.length)]; });
  const cx = ids.reduce((t, v) => t + P[v][0], 0) / ids.length, cy = ids.reduce((t, v) => t + P[v][1], 0) / ids.length;
  const isUser = (v: number) => (nm[v] ?? "").startsWith("user ");
  const lab = (a: [number, number], b: [number, number], gap = 30) => { const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, dx = mx - cx, dy = my - cy, dl = Math.hypot(dx, dy) || 1; return [mx + (gap * dx) / dl, my + (gap * dy) / dl]; };
  const cl = sh && ids.length > Math.max(...sh.close) ? [ids[sh.close[0]], ids[sh.close[1]]] : null;
  return <svg viewBox="0 0 520 262" className="sp-dia">
    <polygon points={ids.map((v) => P[v].join(",")).join(" ")} fill={col} fillOpacity={0.07} stroke="none" />
    {cl && (() => { const x = (P[cl[0]][0] + P[cl[1]][0]) / 2, y = Math.max(P[cl[0]][1], P[cl[1]][1]) + 24; return <g><line x1={P[cl[0]][0]} y1={P[cl[0]][1]} x2={P[cl[1]][0]} y2={P[cl[1]][1]} stroke={col} strokeWidth={2} strokeDasharray="6 5" />
      <text x={x} y={y + 4} textAnchor="middle" className="sp-inf" fill={col}>{sh.note}</text></g>; })()}
    {s.links.map((l: any) => { const a = P[l.from_id], b = P[l.to_id]; if (!a || !b) return null; const [x, y] = lab(a, b);
      return <g key={l.eid}><line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={col} strokeWidth={4.5} strokeLinecap="round" />
        <text x={x} y={y - 2} textAnchor="middle" className="sp-el">{l.rating}★</text><text x={x} y={y + 12} textAnchor="middle" className="sp-el2">{l.date}</text></g>; })}
    {ids.map((v) => { const [x, y] = P[v], u = v === ev.member_id, a = v === ev.A_id, b = v === ev.B_id, top = y < cy;
      return <g key={v}>{isUser(v) ? <circle cx={x} cy={y} r={17} fill={u ? "#dce9f7" : "#fff"} stroke={u ? BLUE : col} strokeWidth={3} />
        : <rect x={x - 17} y={y - 17} width={34} height={34} rx={7} fill={a ? RED : b ? GREEN : "#fff"} stroke={a ? RED : b ? GREEN : col} strokeWidth={3} />}
        <text x={x + (top ? 0 : x < cx ? -6 : 6)} y={top ? y - 26 : y + 40} textAnchor={top ? "middle" : x < cx ? "end" : "start"} className={`sp-nl ${a ? "a" : u ? "u" : ""}`}>{u ? "this member" : a ? "Requiem (A)" : b ? "right title (B)" : short(nm[v] ?? `node ${v}`, 22)}</text></g>; })}
  </svg>;
}

export default function StructurePanel({ cid, onDelete, onSelect }: { cid: string; onDelete: (e: number[]) => void; onSelect: (c: any) => void }) {
  const [d, setD] = useState<any>(null);
  const [j, setJ] = useState(0);
  useEffect(() => {
    setD(null); setJ(0);
    fetch(`/api/structure/${cid}`).then((r) => (r.ok ? r.json() : null)).then((x) => {
      setD(x);
      const st = x?.evidence?.structures ?? [];  // open on the structure with the largest verified effect
      let best = 0, bv = -1e9;
      st.forEach((t: any, i: number) => { const v = (t.p_value <= 0.05 ? 1e3 : 0) + t.drop_logit; if (v > bv) { bv = v; best = i; } });
      setJ(best);
    }).catch(() => setD(null));
  }, [cid]);
  const S = d?.evidence?.structures ?? [];
  const s = S[j];
  useEffect(() => { if (s) onSelect({ motif_cards: [{ motif: s.motif, example: { eids: s.links.map((l: any) => l.eid) } }], G_star_occurrences: [] }); }, [d, j]);
  if (!d) return <div className="cs-card"><p className="cap">Measuring the structures (CATGX + deletion tests)…</p></div>;
  if (!s) return <div className="cs-card"><p className="cap">No structure found for this search.</p></div>;
  const ev = d.evidence, x = d.llm?.[j];
  const cause = s.p_value <= 0.05 && s.p_after < s.p_before;
  const sat = s.p_before > 0.97;
  return <div className="cs-card sp">
    <p className="cs-ct">Structures CATGX found behind this ranking</p>
    <div className="sp-tabs">{S.map((t: any, i: number) => { const c = t.p_value <= 0.05 && t.p_after < t.p_before;
      return <button key={i} className={i === j ? "on" : ""} onClick={() => setJ(i)} style={i === j ? { borderColor: MCOL[t.motif] } : undefined}>
        <i style={{ background: MCOL[t.motif] }} />{PLAIN[t.motif] ?? t.name}<em className={c ? "ok" : "no"}>{c ? "cause" : "not shown"}</em></button>; })}</div>
    <Diagram s={s} ev={ev} />
    <div className="sp-lift">the model relies on this <b>{s.lift != null ? `${s.lift.toFixed(1)}×` : "–"}</b> as much as on equally recent, equally active structures</div>
    <div className="sp-test">
      {([["as is", sat ? s.gap_before : s.p_before, "#8b97a3"], ["delete this structure", sat ? s.gap_after : s.p_after, RED], [`delete ${s.links.length} random links (same period)`, sat ? s.gap_random : s.p_random, "#b8c2cc"]] as [string, number, string][]).map(([l, v, c]) => {
        const mx = sat ? Math.max(s.gap_before, s.gap_after, s.gap_random, 1e-6) : 1;
        return <div key={l} className="cs-g sm"><span>{l}</span><i><b style={{ width: `${(100 * Math.max(0, v)) / mx}%`, background: c }} /></i><em>{sat ? v.toFixed(2) : pct(v)}</em></div>; })}
      <p className="sp-cap">{sat ? "model's score gap, Requiem over the right title (confidence is near 100%, so the gap shows the effect)" : "model's confidence that Requiem beats the right title"} · <b className={cause ? "ok" : "no"}>{cause ? `cause (p = ${s.p_value.toFixed(2)})` : `not shown to be a cause (p = ${s.p_value.toFixed(2)})`}</b></p>
      <button className="ghost sm" onClick={() => onDelete(s.links.map((l: any) => l.eid))}>Delete this structure in the live ranking →</button>
    </div>
    {x ? <div className="sp-llm">
      <div className="sp-llm-h"><span>LLM explanation</span><small>{d.llm_source}</small></div>
      <p className="sp-head">{x.headline}</p>
      <p className="sp-k">Why it moves the prediction</p><p>{x.mechanism}</p>
      <p className="sp-k">Evidence</p><p>{x.evidence}</p>
      <p className="sp-k">What the team can do</p><p>{x.action}</p>
      <p className={`sp-chk ${d.unsupported_numbers?.length ? "bad" : "ok"}`}>{d.unsupported_numbers?.length ? `⚠ ${d.unsupported_numbers.length} number(s) not found in the evidence: ${d.unsupported_numbers.map((b: any) => b[1]).join(", ")}` : "✓ every number checked against the measured evidence; written only from it"}</p>
    </div> : <p className="cap">No LLM explanation for this search ({d.llm_source ?? "set ANTHROPIC_API_KEY on the server to generate one live"}).</p>}
  </div>;
}
