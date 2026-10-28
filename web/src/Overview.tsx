import { useMemo, useState } from "react";
import { dt } from "./api";

// Step 0: the temporal interaction graph itself. Inner ring = users (members), outer ring = movies grouped by
// genre label, edge = one timestamped rating. Click a user to make it the test node.
const LCOL: Record<string, string> = {
  "Crime/Thriller": "#ce9e89", Comedy: "#a697ba", Mystery: "#6fa69a", "Kids/Animation": "#7fb3a6", Horror: "#b07a7a",
  "SciFi/Fantasy": "#7194b9", "Action/Adventure": "#c5a353", Romance: "#d8a0b0", Drama: "#9aa3ad", Documentary: "#8c9a6e", Other: "#c7ccd2",
};
const SCOL: Record<string, string> = { train: "#b8c2cc", val: "#7194b9", test: "#c82434" };

export default function Overview({ ov, selected, onPick }: any) {
  const [hover, setHover] = useState<string | null>(null);
  const [tip, setTip] = useState<any>(null);
  const W = 1000, H = 620, cx = 500, cy = 310;
  const lay = useMemo(() => {
    if (!ov) return null;
    const us = ov.users, ms = [...ov.movies].sort((a: any, b: any) => a.label.localeCompare(b.label) || a.title.localeCompare(b.title));
    const pu: Record<string, [number, number]> = {}, pm: Record<string, [number, number, number]> = {};
    us.forEach((u: any, i: number) => { const a = (2 * Math.PI * i) / us.length - Math.PI / 2; pu[u.id] = [cx + 150 * Math.cos(a), cy + 150 * Math.sin(a)]; });
    ms.forEach((m: any, i: number) => { const a = (2 * Math.PI * i) / ms.length - Math.PI / 2; pm[m.id] = [cx + 285 * Math.cos(a), cy + 285 * Math.sin(a), a]; });
    // genre arcs
    const groups: any[] = [];
    ms.forEach((m: any, i: number) => { const g = groups[groups.length - 1]; if (g && g.label === m.label) g.end = i; else groups.push({ label: m.label, start: i, end: i }); });
    return { us, ms, pu, pm, groups, n: ms.length };
  }, [ov]);
  if (!ov || !lay) return <div className="kg-wrap"><p className="cap">Loading graph…</p></div>;
  const active = hover ?? (selected ? `u${selected}` : null);
  const show = (e: React.MouseEvent, title: string, rows: [string, string][]) => {
    const box = (e.currentTarget as SVGElement).ownerSVGElement!.getBoundingClientRect();
    setTip({ x: e.clientX - box.left + 14, y: e.clientY - box.top + 10, title, rows });
  };
  return (
    <div className="kg-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} className="kg">
        {lay.groups.map((g: any) => {
          const a0 = (2 * Math.PI * (g.start - 0.5)) / lay.n - Math.PI / 2, a1 = (2 * Math.PI * (g.end + 0.5)) / lay.n - Math.PI / 2, r = 300;
          const mid = (a0 + a1) / 2, big = a1 - a0 > Math.PI ? 1 : 0;
          return <g key={g.label}>
            <path d={`M${cx + r * Math.cos(a0)},${cy + r * Math.sin(a0)} A${r},${r} 0 ${big} 1 ${cx + r * Math.cos(a1)},${cy + r * Math.sin(a1)}`} stroke={LCOL[g.label] ?? "#ccc"} strokeWidth={5} fill="none" />
            {g.end - g.start >= 3 && <text x={cx + 322 * Math.cos(mid)} y={cy + 322 * Math.sin(mid) + 4} textAnchor={Math.cos(mid) > 0.2 ? "start" : Math.cos(mid) < -0.2 ? "end" : "middle"} className="ov-gl">{g.label}</text>}
          </g>;
        })}
        {ov.edges.map((e: any) => {
          const [x1, y1] = lay.pu[e.u], [x2, y2] = lay.pm[e.m];
          const on = active === e.u;
          return <line key={e.event_id} x1={x1} y1={y1} x2={x2} y2={y2} stroke={on ? SCOL[e.split] : "#d5dbe1"} strokeWidth={on ? 2 : 0.8} opacity={active && !on ? 0.25 : 0.9} />;
        })}
        {lay.ms.map((m: any) => {
          const [x, y] = lay.pm[m.id];
          const on = active && ov.edges.some((e: any) => e.u === active && e.m === m.id);
          return <circle key={m.id} cx={x} cy={y} r={on ? 6 : 4} fill={LCOL[m.label] ?? "#ccc"} stroke="#fff"
            onMouseMove={(e) => show(e, m.title, [["genres", m.genres], ["label", m.label]])} onMouseLeave={() => setTip(null)} />;
        })}
        {lay.us.map((u: any) => {
          const [x, y] = lay.pu[u.id];
          const sel = selected === u.member, on = active === u.id;
          return <g key={u.id} className="ov-u" onClick={() => onPick(u)} onMouseEnter={() => setHover(u.id)}
            onMouseLeave={() => { setHover(null); setTip(null); }}
            onMouseMove={(e) => show(e, `User u${u.member}${u.featured ? ` (${u.featured})` : ""}`, [["ratings before t", String(u.n_history)], ["prediction time", dt(u.t)],
              ["mostly watches", u.top_label ?? "–"], ["ranking failure", u.A_title ? `A "${u.A_title}" above B "${u.B_title}" (#${u.rank_B})` : "none"], ["", "click to use as test node"]])}>
            <circle cx={x} cy={y} r={sel ? 13 : on ? 11 : u.featured ? 9 : 7} className={`ov-uc ${sel ? "sel" : ""} ${u.featured ? "feat" : ""}`} />
            {(sel || u.featured || on) && (() => { const dx = x - cx, dy = y - cy, d = Math.hypot(dx, dy) || 1; return <text x={cx + dx * (d - 30) / d} y={cy + dy * (d - 30) / d + 4} textAnchor="middle" className="ov-ul">u{u.member}</text>; })()}
          </g>;
        })}
        <text x={cx} y={cy - 6} textAnchor="middle" className="ov-c1">{ov.users.length} test users</text>
        <text x={cx} y={cy + 12} textAnchor="middle" className="ov-c2">inner ring = users · outer ring = movies</text>
      </svg>
      {tip && <div className="kg-tip" style={{ left: tip.x, top: tip.y }}><b>{tip.title}</b>
        <table><tbody>{tip.rows.map(([k, v]: any) => <tr key={k + v}><td>{k}</td><td>{v}</td></tr>)}</tbody></table></div>}
      <div className="kg-legend">
        <span><i className="sw" style={{ borderTop: "2.5px solid #b8c2cc" }} />edge rated in train period</span>
        <span><i className="sw" style={{ borderTop: "2.5px solid #7194b9" }} />validation period</span>
        <span><i className="sw" style={{ borderTop: "2.5px solid #c82434" }} />test period</span>
        <span className="muted">{ov.note} Hover a user, click to select.</span>
      </div>
    </div>
  );
}
