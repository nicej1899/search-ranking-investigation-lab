import { useMemo, useState } from "react";
import { forceCollide, forceLink, forceManyBody, forceRadial, forceSimulation, forceX, forceY } from "d3-force";
import { dt, f, n } from "./api";
import { LCOL } from "./NetGraph";

// The exact subgraph both explainers work on: targets (person u, movies A and B) in the centre, nodes 1, 2, 3 hops
// away on rings. TemGX overlay: explanatory nodes (red), connection nodes (orange), explanation edges (red dashed).
// CATGX overlay: selected motif occurrences, one colour per motif.
export const MCOL: Record<string, string> = { SEQ: "#7a5bb5", CO: "#2f8f83", BRIDGE: "#c08a1e", BRIDGEB: "#5b8fc7", POP: "#c4577a" };
const R_HOP = [0, 140, 232, 300];
const short = (s: string, k = 20) => (s.length > k ? s.slice(0, k - 1) + "…" : s);

export default function SubGraph({ sg, tx, cx, jt, step, removed, ov, onEdge, compact, alt }: any) {
  const [tip, setTip] = useState<any>(null);
  const W = 1000, H = 660, cx0 = 500, cy0 = 330;
  const lay = useMemo(() => {
    if (!sg) return null;
    const ns = sg.nodes.map((d: any) => ({ ...d }));
    const idx: Record<number, any> = Object.fromEntries(ns.map((d: any) => [d.id, d]));
    const u = idx[sg.u], A = idx[sg.A], B = idx[sg.B];
    if (u) { u.fx = cx0 - 70; u.fy = cy0; }
    if (A) { A.fx = cx0 + 60; A.fy = cy0 - 55; }
    if (B) { B.fx = cx0 + 60; B.fy = cy0 + 55; }
    const ls = sg.edges.map((e: any) => ({ ...e, source: e.s, target: e.t }));
    const sim = forceSimulation(ns as any)
      .force("link", forceLink(ls as any).id((d: any) => d.id).distance(70).strength(0.06))
      .force("charge", forceManyBody().strength(-110))
      .force("radial", forceRadial((d: any) => R_HOP[Math.min(d.hop, 3)], cx0, cy0).strength((d: any) => (d.hop ? 0.9 : 0)))
      .force("collide", forceCollide((d: any) => (d.role ? 22 : d.hop === 1 ? 11 : 7)))
      .force("x", forceX(cx0).strength(0.01)).force("y", forceY(cy0).strength(0.01))
      .stop();
    for (let i = 0; i < 350; i++) sim.tick();
    return { ns, idx: Object.fromEntries(ns.map((d: any) => [d.id, d])), ls };
  }, [sg]);
  if (!sg || !lay) return <div className="kg-wrap"><p className="cap" style={{ padding: 30 }}>Loading subgraph…</p></div>;
  const { ns, idx } = lay;
  const showTx = ov ? ov.tx && tx : step >= 2 && tx && step !== 4;
  const showCx = ov ? ov.cx && cx : step >= 4 && cx;
  const expl = new Set<number>(showTx ? tx.explanatory : []);
  const conn = new Set<number>(showTx ? tx.connection : []);
  const Eeps = new Set<number>(showTx ? tx.E_eps : []);
  const R = new Set<number>(removed ?? []);
  const ALT = new Set<number>(alt ?? []);  // links supporting the better title B (contrastive view)
  const txNode: Record<number, any> = Object.fromEntries((tx?.nodes ?? []).map((d: any) => [d.node, d]));
  const txEdge: Record<number, any> = Object.fromEntries((tx?.edges ?? []).map((e: any) => [e.eid, e]));
  const motifOf: Record<number, string> = {};
  if (!ov && step === 5 && jt) {  // joint: each TemGX link completed into the pattern instance it belongs to
    jt.labels.forEach((x: any) => x.instance && x.instance.eids.forEach((e: number) => { motifOf[e] = motifOf[e] ?? x.motif; }));
  } else if (showCx) {
    (cx.motif_cards ?? []).forEach((c: any) => c.example.eids.forEach((e: number) => { motifOf[e] = motifOf[e] ?? c.motif; }));
    cx.G_star_occurrences.forEach((o: any) => o.eids.forEach((e: number) => { motifOf[e] = motifOf[e] ?? o.motif; }));
  }
  const show = (ev: React.MouseEvent, title: string, rows: [string, string][]) => {
    const box = (ev.currentTarget as SVGElement).ownerSVGElement!.getBoundingClientRect();
    setTip({ x: ev.clientX - box.left + 14, y: ev.clientY - box.top + 10, title, rows });
  };
  const hide = () => setTip(null);
  const maxHop = Math.max(...ns.map((d: any) => d.hop));
  return (
    <div className="kg-wrap">
      <svg viewBox={compact ? "170 0 660 660" : `0 0 ${W} ${H}`} className="kg">
        {[1, 2, 3].filter((h) => h <= maxHop).map((h) => <g key={h}>
          <circle cx={cx0} cy={cy0} r={R_HOP[h]} className="sg-ring" />
          <text x={cx0} y={cy0 - R_HOP[h] - 6} textAnchor="middle" className="sg-ringt">{h} hop{h > 1 ? "s" : ""} away</text></g>)}
        {lay.ls.map((l: any) => {
          const a = idx[l.s], b = idx[l.t];
          const inE = Eeps.has(l.eid), rem = R.has(l.eid), mot = motifOf[l.eid];
          const pe = txEdge[l.eid];
          const cls = `sg-e ${inE ? "x" : ""} ${rem ? "rem" : ""} ${mot ? "m" : ""}`;
          return <g key={l.eid} onMouseMove={(ev) => show(ev, `${a.type === "user" ? `user ${a.id}` : short(a.title, 30)} — ${b.type === "user" ? `user ${b.id}` : short(b.title, 30)}`, [
            ["rated", `${l.rating} on ${dt(l.ts)}`],
            ...(pe ? [["effect on the A-vs-B decision", f(pe.signed, 3)], ["influence p", n(pe.p, 4)]] as [string, string][] : []),
            ...(inE ? [["", "part of the TemGX explanation"]] as [string, string][] : []),
            ...(mot ? [["", `part of a ${mot} pattern (CATGX)`]] as [string, string][] : []),
            ...(rem ? [["", "hidden in the re-run"]] as [string, string][] : []),
            ...(onEdge ? [["", rem ? "click to restore this rating" : "click to delete this rating and re-run"]] as [string, string][] : [])])} onMouseLeave={hide}
            onClick={onEdge ? () => onEdge(l.eid) : undefined} style={onEdge ? { cursor: "pointer" } : undefined}>
            <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} className={cls} style={ALT.has(l.eid) && !inE ? { stroke: "#2a78d6", strokeWidth: 3.5, strokeDasharray: "7 4" } : mot && !inE ? { stroke: MCOL[mot] } : undefined} />
            <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="transparent" strokeWidth={onEdge ? 12 : 8} />
            {rem && <text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 + 4} textAnchor="middle" className="ng-x">✕</text>}
          </g>;
        })}
        {ns.map((d: any) => {
          const isU = d.role === "u", isAB = d.role === "A" || d.role === "B";
          const r = isU ? 20 : isAB ? 17 : d.hop === 1 ? 9 : 6;
          const t = txNode[d.id];
          return <g key={d.id} onMouseMove={(ev) => show(ev, d.type === "user" ? (isU ? `the person we investigate (user ${d.id})` : `another person (user ${d.id})`) : d.title, [
            ["distance", d.hop ? `${d.hop} hop${d.hop > 1 ? "s" : ""} from u/A/B` : "target"],
            ...(d.genres ? [["genres", d.genres]] as [string, string][] : []),
            ...(t && t.phi != null ? [["TemGX score Φ", n(t.phi, 5)], ["influence (ICM)", n(t.inf, 4)], ["connectedness g(TRD)", n(t.g_trd, 3)], ["recency weight", n(t.decay, 3)]] as [string, string][] : []),
            ...(expl.has(d.id) ? [["", "TemGX: explanatory node"]] as [string, string][] : []), ...(conn.has(d.id) ? [["", "TemGX: connection node (carries the influence)"]] as [string, string][] : [])])} onMouseLeave={hide}>
            {expl.has(d.id) && <circle cx={d.x} cy={d.y} r={r + 6} className="sg-expl" />}
            {conn.has(d.id) && <circle cx={d.x} cy={d.y} r={r + 5} className="sg-conn" />}
            {isAB && <circle cx={d.x} cy={d.y} r={r + 5} className={`ng-ring ${d.role}`} />}
            <circle cx={d.x} cy={d.y} r={r} fill={compact && d.role === "A" ? "#c82434" : compact && d.role === "B" ? "#2a78d6" : d.type === "user" ? (isU ? "#dce9f7" : "#eef2f7") : LCOL[d.label] ?? "#ccc"} stroke={d.type === "user" ? "#7194b9" : "#fff"} strokeWidth={isU ? 3 : 1.5} />
            {(isU || isAB) && <text x={d.x} y={d.y + 4} textAnchor="middle" className="ng-role" fill={isU ? "#20262d" : "#fff"}>{isU ? "u" : d.role}</text>}
            {(isAB || (d.hop === 1 && d.type === "movie") || expl.has(d.id)) && <text x={d.x} y={d.y + r + 13} textAnchor="middle" className="ng-lab">{d.type === "user" ? `user ${d.id}` : short(d.title)}</text>}
          </g>;
        })}
      </svg>
      {tip && <div className="kg-tip" style={{ left: tip.x, top: tip.y }}><b>{tip.title}</b>
        <table><tbody>{tip.rows.map(([k, v]: any, i: number) => <tr key={i}><td>{k}</td><td>{v}</td></tr>)}</tbody></table></div>}
      {!compact && <div className="key">
        <div className="key-title">How to read this picture</div>
        <div className="key-grid">
          <div className="key-item"><svg width={20} height={16}><circle cx={9} cy={8} r={7} fill="#dce9f7" stroke="#7194b9" strokeWidth={2.5} /></svg><span>u = the person we investigate; A = top result that doesn't fit; B = better match ranked lower</span></div>
          <div className="key-item"><svg width={20} height={16}><circle cx={9} cy={8} r={5} fill="#eef2f7" stroke="#7194b9" /></svg><span>other people · coloured dots = movies (colour = genre)</span></div>
          <div className="key-item"><svg width={20} height={16}><circle cx={9} cy={8} r={7} fill="none" stroke="#9aa3ad" strokeDasharray="3 3" /></svg><span>rings = how many steps away from u, A, B</span></div>
          <div className="key-item"><svg width={20} height={16}><circle cx={9} cy={8} r={6} fill="none" stroke="#c82434" strokeWidth={3} /></svg><span>TemGX: the people/movies that drive the decision</span></div>
          <div className="key-item"><svg width={20} height={16}><circle cx={9} cy={8} r={6} fill="none" stroke="#e08a2c" strokeWidth={2.5} /></svg><span>TemGX: connectors that carry their influence to u, A, B</span></div>
          <div className="key-item"><svg width={20} height={16}><line x1={0} y1={8} x2={18} y2={8} stroke="#c82434" strokeWidth={3} strokeDasharray="5 3" /></svg><span>TemGX explanation edges (✕ = hidden in the re-run)</span></div>
          {Object.entries(MCOL).map(([m, c]) => <div key={m} className="key-item"><svg width={20} height={16}><line x1={0} y1={8} x2={18} y2={8} stroke={c} strokeWidth={4} /></svg><span>CATGX pattern: {({ SEQ: "a person watched two movies in a row", CO: "two people rated the same movie", BRIDGE: "shared-taste path ending at A", BRIDGEB: "shared-taste path ending at B", POP: "several people rated A" } as any)[m]}</span></div>)}
        </div>
        <div className="key-note">Exactly this subgraph is what both methods analyse ({sg.nodes.length} nodes, {sg.edges.length} edges within the last year, up to 10 most recent links per node). Hover for numbers.</div>
      </div>}
    </div>
  );
}
