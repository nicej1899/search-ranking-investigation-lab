import { useMemo, useState } from "react";
import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY } from "d3-force";
import { dt, f, n } from "./api";

// Node-link view of the real temporal interaction graph (force-directed, computed once, deterministic).
// overview: sampled users x their recent rating edges.  focus: the test user's subgraph at time t,
// with the counterfactual subgraph S (TemGX) and the motif occurrence (CATGX) drawn on the same edges.
export const LCOL: Record<string, string> = {
  "Crime/Thriller": "#d08b6c", Comedy: "#9a86b8", Mystery: "#5f9e90", "Kids/Animation": "#73b3a3", Horror: "#a86464",
  "SciFi/Fantasy": "#6f93bb", "Action/Adventure": "#c5a353", Romance: "#d58aa1", Drama: "#8f99a3", Documentary: "#8c9a6e", Other: "#b9c0c7",
};
const SCOL: Record<string, string> = { train: "#c3ccd5", val: "#7f9fc3", test: "#d65a64" };
const short = (s: string, k = 22) => (s.length > k ? s.slice(0, k - 1) + "…" : s);

type Tip = { x: number; y: number; title: string; rows: [string, string][] } | null;

function layout(nodes: any[], links: any[], W: number, H: number, focus: boolean) {
  const ns = nodes.map((d) => ({ ...d }));
  const idx = Object.fromEntries(ns.map((d) => [d.id, d]));
  const ls = links.filter((l) => idx[l.s] && idx[l.t]).map((l) => ({ ...l, ci: l.index, source: l.s, target: l.t }));
  if (focus) {
    const u = ns.find((d) => d.test);
    if (u) { u.fx = W * 0.42; u.fy = H / 2; }
  }
  const sim = forceSimulation(ns as any)
    .force("link", forceLink(ls as any).id((d: any) => d.id).distance((l: any) => (l.kind === "predict" ? 230 : l.kind === "context" ? 55 : focus ? 120 : 45)).strength((l: any) => (l.kind === "predict" ? 0.15 : 0.7)))
    .force("charge", forceManyBody().strength(focus ? -260 : -55))
    .force("collide", forceCollide((d: any) => (d.type === "user" ? (d.test ? 26 : 10) : focus ? 18 : 7)))
    .force("center", forceCenter(W / 2, H / 2))
    .force("x", forceX((d: any) => (d.role ? W * 0.86 : W / 2)).strength((d: any) => (d.role ? 0.5 : 0.03)))
    .force("y", forceY((d: any) => (d.role === "A" ? H * 0.28 : d.role === "B" ? H * 0.72 : H / 2)).strength((d: any) => (d.role ? 0.5 : 0.04)))
    .stop();
  for (let i = 0; i < 400; i++) sim.tick();
  const pad = 30;
  ns.forEach((d: any) => { d.x = Math.max(pad, Math.min(W - pad, d.x)); d.y = Math.max(pad, Math.min(H - pad, d.y)); });
  return { ns, idx: Object.fromEntries(ns.map((d: any) => [d.id, d])), ls };
}

export default function NetGraph(props: any) {
  const { mode } = props;
  const [tip, setTip] = useState<Tip>(null);
  const [hover, setHover] = useState<string | null>(null);
  const W = 1000, H = 620;
  const show = (e: React.MouseEvent, title: string, rows: [string, string][]) => {
    const box = (e.currentTarget as SVGElement).ownerSVGElement!.getBoundingClientRect();
    setTip({ x: e.clientX - box.left + 14, y: e.clientY - box.top + 10, title, rows });
  };
  const hide = () => { setTip(null); setHover(null); };

  // ------------------------------------------------------------ overview
  const ovLay = useMemo(() => {
    if (mode !== "overview" || !props.ov) return null;
    const ov = props.ov;
    const nodes = [...ov.users.map((u: any) => ({ ...u, type: "user" })), ...ov.movies.map((m: any) => ({ ...m, type: "movie" }))];
    return layout(nodes, ov.edges.map((e: any) => ({ ...e, s: e.u, t: e.m })), W, H, false);
  }, [mode, props.ov]);

  // ------------------------------------------------------------ focus
  const fLay = useMemo(() => {
    if (mode !== "focus" || !props.nb) return null;
    return layout(props.nb.nodes, props.nb.edges, W, H, true);
  }, [mode, props.nb]);

  let body: any = null, legend: any = null;
  if (mode === "overview" && ovLay) {
    const { ns, idx, ls } = ovLay;
    const active = hover ?? (props.selected ? `u${props.selected}` : null);
    body = <>
      {ls.map((l: any) => {
        const a = idx[l.s], b = idx[l.t], on = active === l.s;
        return <line key={l.event_id} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={on ? SCOL[l.split] : "#d8dee4"} strokeWidth={on ? 2.2 : 1} opacity={active && !on ? 0.35 : 1} />;
      })}
      {ns.filter((d: any) => d.type === "movie").map((d: any) => (
        <circle key={d.id} cx={d.x} cy={d.y} r={5} fill={LCOL[d.label] ?? "#ccc"} stroke="#fff" strokeWidth={1}
          onMouseMove={(e) => show(e, d.title, [["genres", d.genres]])} onMouseLeave={hide} />))}
      {ns.filter((d: any) => d.type === "user").map((d: any) => {
        const sel = props.selected === d.member, on = active === d.id;
        return <g key={d.id} className="ov-u" onClick={() => props.onPick(d)} onMouseEnter={() => setHover(d.id)} onMouseLeave={hide}
          onMouseMove={(e) => show(e, `User u${d.member}${d.featured ? ` (${d.featured})` : ""}`, [["searched", d.query ? `“${d.query}”` : "–"], ["past ratings", String(d.n_history)], ["problem", d.A_title ? `“${d.A_title}” ranked above the better match “${d.B_title}”` : "none"],
            ...(d.flipped ? [["result", "hiding 2 past ratings fixes the order"]] as [string, string][] : []), ["", "click to investigate"]])}>
          <circle cx={d.x} cy={d.y} r={sel ? 12 : on ? 10 : d.featured ? 9 : 7} className={`ov-uc ${sel ? "sel" : ""} ${d.featured ? "feat" : ""}`} />
          {(sel || d.featured || on) && <text x={d.x} y={d.y - 14} textAnchor="middle" className="ov-ul">u{d.member}</text>}
        </g>;
      })}
    </>;
    legend = <Key items={[
      ["dot-user", "a person (bigger = hand-picked example)"],
      ["dot-movie", "a movie — colour = genre (key below)"],
      ["line-train", "a rating made in the learning period (before 2013)"],
      ["line-val", "a rating in the tuning period (2013–2016)"],
      ["line-test", "a rating in the test period (after 2016)"],
    ]} note={`${props.ov.note} Hover a person to see their search; click to investigate.`} genres />;
  }

  if (mode === "focus" && fLay) {
    const { ns, idx, ls } = fLay;
    const { cd, cur, base, ex, mo, removed, step } = props;
    const S = new Set<number>(step >= 2 && ex ? ex.selected_event_ids : []);
    const R = new Set<number>(removed ?? []);
    const exBy = Object.fromEntries((ex?.candidates ?? []).map((c: any) => [c.event_id, c]));
    const motif = step >= 4 && mo ? mo.case : null;
    const topM = motif?.selected_motifs?.[0];
    const occ = motif ? motif.occurrences.filter((o: any) => o.motif === topM && o.g > 0).sort((a: any, b: any) => b.g - a.g)[0] : null;
    const occOrder: Record<number, number> = occ ? Object.fromEntries(occ.event_ids.map((e: number, i: number) => [e, i + 1])) : {};
    const comp = cur.components;
    const u = ns.find((d: any) => d.test);
    const histTitle: Record<string, any> = Object.fromEntries(cd.history.map((h: any) => [`m${h.item}`, h]));
    body = <>
      {/* context edges first (background) */}
      {ls.filter((l: any) => l.kind === "context").map((l: any) => {
        const a = idx[l.s], b = idx[l.t];
        return <line key={"c" + l.event_id} x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="ng-ctx" />;
      })}
      {/* member rating edges with overlays */}
      {ls.filter((l: any) => l.kind === "rated").map((l: any) => {
        const a = idx[l.s], b = idx[l.t];
        const inS = S.has(l.event_id), rem = R.has(l.event_id), ord = occOrder[l.event_id];
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        const c = exBy[l.event_id];
        return <g key={"r" + l.event_id} onMouseMove={(e) => show(e, `edge e${l.event_id}: u${cd.case.member} rated ${histTitle[l.t]?.title ?? ""}`, [
          ["time", dt(l.ts)], ["rating", String(l.rating)],
          ...(c ? [["how much it pushes A over B", f(c.single_removal_effect, 2)], ["importance score", n(c.phi, 3)]] as [string, string][] : []),
          ...(inS ? [["", "one of the ratings that caused the problem"]] as [string, string][] : []), ...(rem ? [["", "hidden in the re-run"]] as [string, string][] : []),
          ...(ord ? [["pattern", `step ${ord} of ${occ.event_ids.length} in the repeated pattern`]] as [string, string][] : [])])} onMouseLeave={hide}>
          <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} className={`ng-rated ${inS ? "s" : ""} ${rem ? "rem" : ""} ${ord ? "mot" : ""}`} />
          <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="transparent" strokeWidth={12} />
          {rem && <text x={mx} y={my + 5} textAnchor="middle" className="ng-x">✕</text>}
          {ord && !rem && <g><circle cx={mx} cy={my} r={9} className="ng-ord" /><text x={mx} y={my + 4} textAnchor="middle" className="ng-ordt">{ord}</text></g>}
        </g>;
      })}
      {/* prediction edges to A and B */}
      {ls.filter((l: any) => l.kind === "predict").map((l: any) => {
        const a = idx[l.s], b = idx[l.t], i = l.ci;
        const rk = cur.rank[i], rk0 = base.rank[i];
        return <g key={"p" + l.role} onMouseMove={(e) => show(e, `system ranks ${l.role} at #${cur.rank[i]}`, [["rank", `#${rk}${rk !== rk0 ? ` (was #${rk0})` : ""}`],
          ["score", n(comp.total[i], 3)], ["from viewing history", f(comp.behaviour[i])], ["from text match", f(comp.relevance[i])], ["from popularity", f(comp.popularity[i])]])} onMouseLeave={hide}>
          <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} className={`ng-pred ${l.role}`} strokeWidth={Math.max(1.5, 7 - rk / 2)} />
          <text x={(a.x * 0.35 + b.x * 0.65)} y={(a.y * 0.35 + b.y * 0.65) - 8} textAnchor="middle" className={`ng-predt ${l.role}`}>{l.role} #{rk}{rk !== rk0 ? ` (was #${rk0})` : ""} · {n(comp.total[i], 2)}</text>
        </g>;
      })}
      {/* nodes */}
      {ns.filter((d: any) => d.type === "user" && !d.test).map((d: any) => (
        <circle key={d.id} cx={d.x} cy={d.y} r={5} className="ng-ou" onMouseMove={(e) => show(e, `other user u${d.member}`, [["", "rated one of these movies before the search (shown for context)"]])} onMouseLeave={hide} />))}
      {ns.filter((d: any) => d.type === "movie").map((d: any) => {
        const h = histTitle[d.id];
        const inS = h && S.has(h.event_id), rem = h && R.has(h.event_id), ord = h && occOrder[h.event_id];
        const r = d.role ? 16 : 11;
        return <g key={d.id} className={`ng-m ${rem ? "rem" : ""}`} onMouseMove={(e) => show(e, d.title, [["genres", d.genres], ["other people who rated it before", String(d.degree_before_t)],
          ...(h ? [["member's rating", `${h.rating} at ${dt(h.ts)}`]] as [string, string][] : []), ...(d.role ? [["role", d.role === "A" ? "A: ranked high but does not fit the search" : "B: fits the search but ranked lower"]] as [string, string][] : [])])} onMouseLeave={hide}>
          {d.role && <circle cx={d.x} cy={d.y} r={r + 6} className={`ng-ring ${d.role}`} />}
          {inS && <circle cx={d.x} cy={d.y} r={r + 5} className="ng-sring" />}
          {ord && <circle cx={d.x} cy={d.y} r={r + 9} className="ng-mring" />}
          <circle cx={d.x} cy={d.y} r={r} fill={LCOL[d.label] ?? "#ccc"} stroke="#fff" strokeWidth={2} />
          {d.role && <text x={d.x} y={d.y + 5} textAnchor="middle" className="ng-role">{d.role}</text>}
          {(h || d.role) && <text x={d.x} y={d.y + r + 14} textAnchor="middle" className="ng-lab">{short(d.title)}</text>}
        </g>;
      })}
      {u && <g onMouseMove={(e) => show(e, `test user u${cd.case.member}`, [["searched at", dt(cd.case.t)], ["past ratings", String(cd.case.n_history)], ["search", cd.case.query]])} onMouseLeave={hide}>
        <circle cx={u.x} cy={u.y} r={24} className="ng-u" /><text x={u.x} y={u.y + 5} textAnchor="middle" className="ng-ut">u{cd.case.member}</text>
        <text x={u.x} y={u.y - 32} textAnchor="middle" className="ng-q">“{short(cd.case.query, 40)}”</text></g>}
      {occ && u && <text x={u.x} y={u.y + 46} textAnchor="middle" className="ng-mlab">pattern: {motif.per_motif[topM].name.replace(/ -> /g, " → ")}</text>}
    </>;
    legend = <Key items={[
      ["dot-test", "the person we investigate (their search is shown above them)"],
      ["line-rated", "a movie they rated before searching"],
      ["ring-A", "A = the top result that does not fit the search"],
      ["ring-B", "B = a better match that was ranked lower"],
      ["line-pred", "the system's ranking of A and B (label = rank and score)"],
      ["line-s", "the past ratings that pushed A up (found in step 1)"],
      ["x", "a rating hidden when we re-run the system (step 2)"],
      ["line-mot", "a repeated viewing pattern, in time order ①→② (step 3)"],
      ["dot-ctx", "other people who rated the same movies (more grey dots = more popular)"],
    ]} note="Hover any dot or line for details." genres />;
  }

  return (
    <div className="kg-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} className="kg">{body}</svg>
      {tip && <div className="kg-tip" style={{ left: tip.x, top: tip.y }}><b>{tip.title}</b>
        <table><tbody>{tip.rows.map(([k, v], i) => <tr key={i}><td>{k}</td><td>{v}</td></tr>)}</tbody></table></div>}
      {legend}
    </div>
  );
}


const GENRE_KEY = ["Crime/Thriller", "Comedy", "Mystery", "Kids/Animation", "Horror", "SciFi/Fantasy", "Action/Adventure", "Romance", "Drama", "Documentary"];
function Sym({ k }: { k: string }) {
  const m: Record<string, any> = {
    "dot-user": <circle cx={9} cy={8} r={6} fill="#dce9f7" stroke="#7194b9" strokeWidth={2} />,
    "dot-movie": <circle cx={9} cy={8} r={5} fill="#d08b6c" />,
    "dot-test": <circle cx={9} cy={8} r={7} fill="#dce9f7" stroke="#7194b9" strokeWidth={2.5} />,
    "dot-ctx": <circle cx={9} cy={8} r={4} fill="#c7cdd3" />,
    "line-train": <line x1={0} y1={8} x2={18} y2={8} stroke="#c3ccd5" strokeWidth={2.5} />,
    "line-val": <line x1={0} y1={8} x2={18} y2={8} stroke="#7f9fc3" strokeWidth={2.5} />,
    "line-test": <line x1={0} y1={8} x2={18} y2={8} stroke="#d65a64" strokeWidth={2.5} />,
    "line-rated": <line x1={0} y1={8} x2={18} y2={8} stroke="#9fb6cf" strokeWidth={2.5} />,
    "line-s": <line x1={0} y1={8} x2={18} y2={8} stroke="#c82434" strokeWidth={3} strokeDasharray="5 3" />,
    "line-mot": <line x1={0} y1={8} x2={18} y2={8} stroke="#7a5bb5" strokeWidth={4} />,
    "line-pred": <g><line x1={0} y1={5} x2={18} y2={5} stroke="#c98d97" strokeWidth={2.5} strokeDasharray="5 3" /><line x1={0} y1={12} x2={18} y2={12} stroke="#6fa69a" strokeWidth={2.5} strokeDasharray="5 3" /></g>,
    "ring-A": <g><circle cx={9} cy={8} r={7} fill="#fcebed" stroke="#c98d97" strokeWidth={2} /><text x={9} y={11} textAnchor="middle" fontSize={8} fontWeight={800}>A</text></g>,
    "ring-B": <g><circle cx={9} cy={8} r={7} fill="#e8f3ef" stroke="#6fa69a" strokeWidth={2} /><text x={9} y={11} textAnchor="middle" fontSize={8} fontWeight={800}>B</text></g>,
    "x": <text x={9} y={13} textAnchor="middle" fontSize={14} fontWeight={800} fill="#c82434">✕</text>,
  };
  return <svg width={20} height={16} className="sym">{m[k]}</svg>;
}
function Key({ items, note, genres }: { items: [string, string][]; note?: string; genres?: boolean }) {
  return <div className="key">
    <div className="key-title">How to read this picture</div>
    <div className="key-grid">{items.map(([k, t]) => <div key={k} className="key-item"><Sym k={k} /><span>{t}</span></div>)}</div>
    {genres && <div className="key-genres"><span className="key-sub">Movie colour = genre:</span>{GENRE_KEY.map((g) => <span key={g} className="gk"><i style={{ background: LCOL[g] }} />{g}</span>)}</div>}
    {note && <div className="key-note">{note}</div>}
  </div>;
}
