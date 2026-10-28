import { useMemo, useState } from "react";
import { f, n, dt } from "./api";

// Knowledge graph of one ranking decision.
//   member u --rated@t--> title --has_genre--> genre <--has_genre-- candidate A / B <--query
// Overlays: counterfactual subgraph S (TemGX-adapted) and the top motif occurrence (CATGX-adapted).
// Every number shown comes from backend responses passed in as props.

type Tip = { x: number; y: number; title: string; rows: [string, string][] } | null;

const GCOL: Record<string, string> = {
  Crime: "#ce9e89", Thriller: "#ce9e89", "Film-Noir": "#ce9e89", Comedy: "#a697ba", Mystery: "#6fa69a",
  Children: "#6fa69a", Animation: "#6fa69a", Horror: "#b07a7a",
};
const labelFill = (l: string) =>
  l === "Crime/Thriller" ? "#fbe7d8" : l === "Comedy" ? "#eee8f5" : l === "Mystery" || l === "Kids/Animation" ? "#e8f3ef" : l === "Horror" ? "#f6e1e1" : "#eef0f3";
const short = (s: string, k = 24) => (s.length > k ? s.slice(0, k - 1) + "…" : s);

export default function Graph({ cd, cur, base, ex, mo, removed, step, A, B }: any) {
  const [tip, setTip] = useState<Tip>(null);
  const W = 1000, H = 620;
  const hist = cd.history.slice(-12);
  const hidden = cd.history.length - hist.length;
  const ca = cd.candidates[A], cb = cd.candidates[B];
  const g = (s: string) => s.split("|").filter((x: string) => x !== "IMAX" && x !== "(no genres listed)");
  const gA = new Set(g(ca.genres)), gB = new Set(g(cb.genres));

  const genres = useMemo(() => {
    const all = new Set<string>();
    hist.forEach((h: any) => g(h.genres).forEach((x: string) => all.add(x)));
    gA.forEach((x) => all.add(x)); gB.forEach((x) => all.add(x));
    const rank = (x: string) => (gA.has(x) && !gB.has(x) ? 0 : gA.has(x) && gB.has(x) ? 1 : gB.has(x) ? 2 : 3);
    return [...all].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  }, [cd]);

  const hy = (i: number) => 60 + (i * (H - 120)) / Math.max(hist.length - 1, 1);
  const gy = (i: number) => 40 + (i * (H - 80)) / Math.max(genres.length - 1, 1);
  const HX = 290, GX = 560, CX = 790, UX = 70, UY = H / 2;
  const posA = { x: CX, y: 230 }, posB = { x: CX, y: 470 };
  const genreY = Object.fromEntries(genres.map((x, i) => [x, gy(i)]));

  const S = new Set<number>(step >= 2 && ex ? ex.selected_event_ids : []);
  const R = new Set<number>(removed);
  const exByE = Object.fromEntries((ex?.candidates ?? []).map((c: any) => [c.event_id, c]));
  // top motif occurrence (G*) for the overlay
  const motif = step >= 4 && mo ? mo.case : null;
  const topMotif = motif?.selected_motifs?.[0];
  const occ = motif ? motif.occurrences.filter((o: any) => o.motif === topMotif && o.g > 0).sort((a: any, b: any) => b.g - a.g)[0] : null;
  const occSet = new Set<number>(occ ? occ.event_ids : []);

  const show = (e: React.MouseEvent, title: string, rows: [string, string][]) => {
    const box = (e.currentTarget as SVGElement).ownerSVGElement!.getBoundingClientRect();
    setTip({ x: e.clientX - box.left + 14, y: e.clientY - box.top + 10, title, rows });
  };
  const hide = () => setTip(null);
  const comp = cur.components;
  const ab = cur.ab, ab0 = base.ab;

  const candCard = (idx: number, p: { x: number; y: number }, who: "A" | "B") => {
    const c = cd.candidates[idx];
    const vals = [["history", comp.behaviour[idx], "#7194b9"], ["text match", comp.relevance[idx], "#c5a353"], ["popularity", comp.popularity[idx], "#9aa3ad"]] as const;
    const rank = cur.rank[idx], rank0 = base.rank[idx];
    return (
      <g className={`kg-cand ${who}`} transform={`translate(${p.x - 10},${p.y - 58})`}
        onMouseMove={(e) => show(e, `${who}: ${c.title}`, [["genres", c.genres], ["rank", `#${rank}${rank !== rank0 ? ` (was #${rank0})` : ""}`],
          ["total score", n(comp.total[idx], 3)], ["behaviour", f(comp.behaviour[idx])], ["relevance", f(comp.relevance[idx])], ["popularity", f(comp.popularity[idx])],
          ["judgment", `grade ${c.grade} (rubric)`], ["training ratings", String(c.train_interactions)], ["runtime", "unknown"]])} onMouseLeave={hide}>
        <rect width={180} height={116} rx={10} />
        <text x={12} y={22} className="kg-who">{who} · #{rank}{rank !== rank0 && <tspan className="kg-was"> (was #{rank0})</tspan>}</text>
        <text x={12} y={42} className="kg-ct">{short(c.title, 22)}</text>
        {vals.map(([k, v, col], i) => (
          <g key={k} transform={`translate(12,${56 + i * 16})`}>
            <rect x={60} y={-8} width={80} height={8} fill="#eef0f3" rx={2} />
            <rect x={v >= 0 ? 100 : 100 + Math.max(v * 40, -40)} y={-8} width={Math.min(Math.abs(v) * 40, 40)} height={8} fill={col} rx={2} />
            <text x={0} y={0} className="kg-small">{k}</text>
          </g>
        ))}
        <text x={168} y={22} textAnchor="end" className="kg-score">{n(comp.total[idx], 2)}</text>
      </g>
    );
  };

  return (
    <div className="kg-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} className="kg">
        <defs>
          <marker id="kgarr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="#7a5bb5" /></marker>
        </defs>
        {/* column captions */}
        <text x={UX} y={20} className="kg-cap" textAnchor="middle">member</text>
        <text x={HX} y={20} className="kg-cap" textAnchor="middle">rating history (time ↓)</text>
        <text x={GX} y={20} className="kg-cap" textAnchor="middle">genre</text>
        <text x={CX + 85} y={20} className="kg-cap" textAnchor="middle">query → candidates</text>

        {/* member -> history edges */}
        {hist.map((h: any, i: number) => {
          const sel = S.has(h.event_id), rem = R.has(h.event_id);
          return <path key={"uh" + h.event_id} d={`M${UX + 26},${UY} C${UX + 120},${UY} ${HX - 160},${hy(i)} ${HX - 95},${hy(i)}`}
            className={`kg-e ${sel ? "kg-e-sel" : ""} ${rem ? "kg-e-rem" : ""}`} />;
        })}
        {/* history -> genre edges */}
        {hist.map((h: any, i: number) => g(h.genres).map((x: string) => {
          const toA = gA.has(x), sel = S.has(h.event_id) && toA, rem = R.has(h.event_id);
          return <line key={`hg${h.event_id}${x}`} x1={HX + 95} y1={hy(i)} x2={GX - 42} y2={genreY[x]}
            className={`kg-g ${sel ? "kg-g-sel" : ""} ${rem ? "kg-e-rem" : ""}`} stroke={sel ? undefined : toA || gB.has(x) ? GCOL[x] ?? "#c7ccd2" : "#e3e6ea"} />;
        }))}
        {/* genre -> candidate edges */}
        {genres.map((x) => (
          <g key={"gc" + x}>
            {gA.has(x) && <line x1={GX + 42} y1={genreY[x]} x2={CX - 10} y2={posA.y} className={`kg-g ${S.size && hist.some((h: any) => S.has(h.event_id) && g(h.genres).includes(x)) ? "kg-g-sel" : ""}`} stroke={GCOL[x] ?? "#c7ccd2"} />}
            {gB.has(x) && <line x1={GX + 42} y1={genreY[x]} x2={CX - 10} y2={posB.y} className="kg-g" stroke={GCOL[x] ?? "#c7ccd2"} />}
          </g>
        ))}
        {/* query -> candidates (relevance) */}
        <g onMouseMove={(e) => show(e, "Query q", [["text", cd.case.query], ["kind", cd.case.kind], ["relevance A", f(comp.relevance[A])], ["relevance B", f(comp.relevance[B])],
          ["encoder", "spaCy averaged word vectors — ignores 'not'"]])} onMouseLeave={hide}>
          <path d={`M${CX + 40},${88} L${CX + 40},${posA.y - 58}`} className="kg-q" strokeWidth={1 + 3 * Math.max(0, comp.relevance[A] + 1)} />
          <path d={`M${CX + 175},${70} C${CX + 205},${120} ${CX + 205},${posB.y - 80} ${CX + 140},${posB.y - 58}`} className="kg-q" strokeWidth={1 + 3 * Math.max(0, comp.relevance[B] + 1)} />
          <rect x={CX - 10} y={50} width={190} height={38} rx={8} className="kg-qbox" />
          <text x={CX + 85} y={66} textAnchor="middle" className="kg-small">query q</text>
          <text x={CX + 85} y={80} textAnchor="middle" className="kg-qt">“{short(cd.case.query, 30)}”</text>
        </g>

        {/* member node */}
        <g onMouseMove={(e) => show(e, `Member ${cd.case.member} (MovieLens user ${cd.user_id})`, [["prediction time", dt(cd.case.t)], ["events before t", String(cd.case.n_history)],
          ["model window", `${cd.history.length} most recent`]])} onMouseLeave={hide}>
          <circle cx={UX} cy={UY} r={26} className="kg-u" /><text x={UX} y={UY + 5} textAnchor="middle" className="kg-ut">u{cd.case.member}</text>
        </g>
        {hidden > 0 && <text x={HX} y={44} textAnchor="middle" className="kg-small">+{hidden} older events in window</text>}

        {/* history nodes */}
        {hist.map((h: any, i: number) => {
          const sel = S.has(h.event_id), rem = R.has(h.event_id), m = occSet.has(h.event_id), c = exByE[h.event_id];
          return (
            <g key={h.event_id} className={`kg-h ${sel ? "sel" : ""} ${rem ? "rem" : ""} ${m ? "mot" : ""}`} transform={`translate(${HX - 95},${hy(i) - 17})`}
              onMouseMove={(e) => show(e, h.title, [["event", `e${h.event_id} · rated ${h.rating} · ${dt(h.ts)}`], ["genres", h.genres],
                ...(c ? [["removal effect on A−B", f(c.single_removal_effect, 2)], ["Φ (TemGX score)", n(c.phi, 3)], ["in S", sel ? "yes" : "no"]] as [string, string][] : []),
                ...(rem ? [["status", "removed in replay"]] as [string, string][] : [])])} onMouseLeave={hide}>
              <rect width={190} height={34} rx={7} fill={labelFill(h.label)} />
              <text x={10} y={14} className="kg-ht">{short(h.title, 26)}</text>
              <text x={10} y={27} className="kg-small">t{cd.history.length - hist.length + i + 1} · {h.label}</text>
              {sel && <text x={182} y={14} textAnchor="end" className="kg-badge">S</text>}
            </g>
          );
        })}

        {/* motif occurrence overlay: ordered arc between events */}
        {occ && (() => {
          const idx = occ.event_ids.map((e: number) => hist.findIndex((h: any) => h.event_id === e)).filter((i: number) => i >= 0);
          if (idx.length < 2) return null;
          const pm = motif.per_motif[topMotif];
          return (
            <g onMouseMove={(e) => show(e, `Motif ${topMotif}: ${pm.name}`, [["occurrence", occ.event_ids.map((x: number) => `e${x}`).join(" → ")], ["g(r) margin drop", n(occ.g, 2)],
              ["context cell", `recency ${occ.cell[0]} · duration ${occ.cell[1]} · activity ${occ.cell[2]}`], ["adjusted S(m)", f(pm.S, 2)],
              ["DEV recurrence", mo ? `${mo.aggregate.per_motif[topMotif]?.cases_with_motif} cases, CI [${n(mo.aggregate.per_motif[topMotif]?.ci95_member_bootstrap?.[0], 2)}, ${n(mo.aggregate.per_motif[topMotif]?.ci95_member_bootstrap?.[1], 2)}]` : ""]])}
              onMouseLeave={hide}>
              {idx.slice(1).map((j: number, k: number) => {
                const i0 = idx[k];
                return <path key={k} d={`M${HX + 95},${hy(i0) + 6} C${HX + 160},${hy(i0) + 6} ${HX + 160},${hy(j) - 6} ${HX + 100},${hy(j) - 6}`} className="kg-motif" markerEnd="url(#kgarr)" />;
              })}
              <rect x={HX + 118} y={(hy(idx[0]) + hy(idx[idx.length - 1])) / 2 - 12} width={92} height={22} rx={11} className="kg-mbox" />
              <text x={HX + 164} y={(hy(idx[0]) + hy(idx[idx.length - 1])) / 2 + 3} textAnchor="middle" className="kg-mt">{topMotif} S={f(pm.S, 1)}</text>
            </g>
          );
        })()}

        {/* genre nodes */}
        {genres.map((x) => (
          <g key={x} onMouseMove={(e) => show(e, `Genre: ${x}`, [["linked to A", gA.has(x) ? "yes" : "no"], ["linked to B", gB.has(x) ? "yes" : "no"],
            ["history titles", String(hist.filter((h: any) => g(h.genres).includes(x)).length)]])} onMouseLeave={hide}>
            <rect x={GX - 42} y={genreY[x] - 11} width={84} height={22} rx={11} fill="#fff" stroke={GCOL[x] ?? "#c7ccd2"} strokeWidth={gA.has(x) || gB.has(x) ? 2 : 1} />
            <text x={GX} y={genreY[x] + 4} textAnchor="middle" className="kg-gt">{x}</text>
          </g>
        ))}

        {candCard(A, posA, "A")}
        {candCard(B, posB, "B")}
        {ab && <text x={CX + 85} y={(posA.y + posB.y) / 2 + 6} textAnchor="middle" className="kg-margin">A−B margin {f(ab.margin, 3)}{removed.length > 0 && ab0 ? ` (was ${f(ab0.margin, 3)})` : ""}</text>}
      </svg>
      {tip && (
        <div className="kg-tip" style={{ left: tip.x, top: tip.y }}>
          <b>{tip.title}</b>
          <table><tbody>{tip.rows.map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}</tbody></table>
        </div>
      )}
      <div className="kg-legend">
        <span><i className="sw sel" />counterfactual subgraph S (TemGX)</span>
        <span><i className="sw mot" />motif occurrence (CATGX)</span>
        <span><i className="sw rem" />removed in replay</span>
        <span className="muted">hover any node for numbers</span>
      </div>
    </div>
  );
}
