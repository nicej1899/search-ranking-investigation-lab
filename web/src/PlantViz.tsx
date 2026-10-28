import { useMemo, useState } from "react";
import { n } from "./api";

// Interactive picture of the planted-cause test: one member's history as a star graph (person in the middle, rated
// movies around), the two planted ratings marked ★, the chosen method's two picks drawn and "deleted", and the model's
// re-run answer as a gauge. Below: every member, every method, as dots (how much of the possible effect it removes).

const M = [
  { k: "TemGX", label: "TemGX", col: "#c82434" },
  { k: "LLM (reads the history as text)", label: "LLM", col: "#7a5bb5" },
  { k: "Attention top-2", label: "Attention", col: "#6fa69a" },
  { k: "Best single removals", label: "One at a time", col: "#c08a1e" },
  { k: "Try every pair", label: "Try every pair", col: "#20262d" },
];
const pct = (x: number | null | undefined) => (x == null ? "–" : `${n(100 * x, 0)}%`);
const short = (s: string, k = 22) => (s.length > k ? s.slice(0, k - 1) + "…" : s);
const title = (l: string) => l.replace(/^\d+\.\s*\S+\s·\s/, "").replace(/\s\([^()]*\)\s·\srated.*$/, "");
const rating = (l: string) => (l.match(/rated ([\d.]+)/) || [])[1];

export default function PlantViz({ pr, simple = false }: any) {
  const [pi, setPi] = useState(0);
  const P = pr.patterns[pi];
  const defIdx = useMemo(() => {
    let best = 0, gap = -1e9;
    P.viz.forEach((v: any, i: number) => {
      const l = v.fidelity["LLM (reads the history as text)"], t = v.fidelity.TemGX;
      const hit = v.picks["LLM (reads the history as text)"].some((x: number) => v.planted_lines.includes(x));
      if (v.plant_is_models_reason && hit && l != null && t != null && t - l > gap) { gap = t - l; best = i; }
    });
    return best;
  }, [P]);
  const [idx, setIdx] = useState<number | null>(null);
  const [mk, setMk] = useState("TemGX");
  const i = idx ?? defIdx;
  const v = P.viz[i];
  const meths = simple ? M.filter((m) => m.k === "TemGX" || m.k.startsWith("LLM")) : M;
  const meth = M.find((m) => m.k === mk)!;
  const picks: number[] = v.picks[mk] ?? [];
  const after = v.p_after[mk];
  const fid = v.fidelity[mk];
  const N = v.lines.length;
  const W = 640, H = 330, cx = 300, cy = 165, R = 125;
  const pos = v.lines.map((_: string, j: number) => {
    const ang = Math.PI * (1.65 - 1.3 * (j / Math.max(N - 1, 1)));  // oldest at top-right, around the left, newest bottom-right
    return { x: cx + R * Math.cos(ang), y: cy + R * Math.sin(ang) };
  });
  const Zx = 575, Zy = cy;
  return (
    <div className="pviz">
      <div className="pviz-top">
        {simple ? <span className="cap">One real member. Two planted ratings (★). Who does each method blame?</span> : <>
        <div className="seg sm">{pr.patterns.map((p: any, j: number) => <button key={p.pattern} className={pi === j ? "on" : ""} onClick={() => { setPi(j); setIdx(null); }}>{p.pattern === "semantic" ? "pattern that makes sense" : "pattern that makes no sense"}</button>)}</div>
        <div className="pviz-nav"><button className="sm" onClick={() => setIdx((i - 1 + P.viz.length) % P.viz.length)}>‹</button><span>member {i + 1} of {P.viz.length}{!v.plant_is_models_reason && " · plant not the model's main reason here"}</span><button className="sm" onClick={() => setIdx((i + 1) % P.viz.length)}>›</button></div></>}
        <div className="seg big2">{meths.map((m) => <button key={m.k} className={mk === m.k ? "on" : ""} onClick={() => setMk(m.k)}>{m.label}</button>)}</div>
      </div>
      <div className="pviz-main">
        <svg viewBox={`0 0 ${W} ${H}`} className="pviz-svg">
          <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#9aa3ad" /></marker></defs>
          {v.lines.map((l: string, j: number) => {
            const p = pos[j], picked = picks.includes(j + 1);
            return <g key={j}>
              <line x1={cx} y1={cy} x2={p.x} y2={p.y} stroke={picked ? meth.col : "#d3dbe3"} strokeWidth={picked ? 4 : 1.5} strokeDasharray={picked ? "6 4" : undefined} opacity={picked ? 0.95 : 1} />
              {picked && <text x={(cx + p.x) / 2} y={(cy + p.y) / 2 + 5} textAnchor="middle" className="pv-x" fill={meth.col}>✕</text>}
            </g>;
          })}
          <line x1={cx + 24} y1={cy} x2={Zx - 30} y2={Zy} stroke="#9aa3ad" strokeWidth={1.5} strokeDasharray="3 4" markerEnd="url(#arr)" />
          <text x={(cx + Zx) / 2 + 20} y={cy - 8} textAnchor="middle" className="pv-small">model's pick</text>
          {v.lines.map((l: string, j: number) => {
            const p = pos[j], planted = v.planted_lines.includes(j + 1);
            const left = p.x < cx - 5;
            return <g key={"n" + j}>
              <rect x={p.x - 9} y={p.y - 9} width={18} height={18} rx={4} fill={planted ? "#c5a353" : "#b8c2cc"} stroke={planted ? "#8a6d17" : "#fff"} strokeWidth={planted ? 2 : 1} />
              {planted && <text x={p.x} y={p.y + 4} textAnchor="middle" className="pv-star">★</text>}
              <text x={p.x + (left ? -14 : 14)} y={p.y + 4} textAnchor={left ? "end" : "start"} className={`pv-lab ${planted ? "pl" : ""}`}>{short(title(l))} · {rating(l)}</text>
            </g>;
          })}
          <circle cx={cx} cy={cy} r={22} fill="#dce9f7" stroke="#7194b9" strokeWidth={3} />
          <text x={cx} y={cy + 5} textAnchor="middle" className="pv-u">person</text>
          <rect x={Zx - 28} y={Zy - 22} width={56} height={44} rx={8} fill="#c82434" />
          <text x={Zx} y={Zy + 4} textAnchor="middle" className="pv-z">title 3</text>
          <text x={Zx} y={Zy + 40} textAnchor="middle" className="pv-small">{short(v.Z, 26)}</text>
        </svg>
        <div className="pviz-side">
          <p className="cap"><b style={{ color: meth.col }}>{meth.label}</b> says the model ranked <b>{short(v.Z, 30)}</b> high because of the two dashed ratings. Delete them and re-run the model:</p>
          <div className="gauge"><span>before</span><i><b style={{ width: `${100 * v.p_before}%` }} /></i><em>{pct(v.p_before)}</em></div>
          <div className="gauge"><span>after</span><i><b style={{ width: `${100 * (after ?? 0)}%`, background: meth.col }} /><u style={{ left: `${100 * v.p_best}%` }} title="best possible" /></i><em>{pct(after)}</em></div>
          <p className="cap">Model's confidence that “{short(v.Z, 24)}” beats “{short(v.B, 24)}”. Black tick = lowest any two deletions can reach.</p>
          <div className={`verdict ${fid != null && fid >= 0.7 ? "ok" : "warn"}`}>{fid == null ? "–" : <>Removes <b>{pct(Math.max(-9.99, fid))}</b> of what's possible{picks.some((x) => v.planted_lines.includes(x)) ? " · named a planted ★ rating" : " · missed the planted ★ ratings"}.</>}</div>
        </div>
      </div>
      {!simple && <Strip pr={pr} />}
    </div>
  );
}

export function Strip({ pr }: any) {
  const W = 1040, H = 46 * M.length + 40, L = 150, Rr = 30;
  const x = (f: number) => L + (Math.max(-0.5, Math.min(1.05, f)) + 0.5) / 1.55 * (W - L - Rr);
  const rows = M.map((m) => {
    const vals: { f: number; pat: string }[] = [];
    pr.patterns.forEach((p: any) => p.viz.forEach((v: any) => { if (v.plant_is_models_reason && v.fidelity[m.k] != null) vals.push({ f: v.fidelity[m.k], pat: p.pattern }); }));
    const s = vals.map((z) => z.f).sort((a, b) => a - b);
    return { ...m, vals, med: s.length ? s[Math.floor(s.length / 2)] : null };
  });
  return (
    <div className="strip">
      <p className="cap"><b>Every member, every method</b> (only members where the planted ratings are the model's main reason; ● pattern that makes sense, ○ makes no sense). Further right = deleting the method's answer removes more of the model's preference. Bar = median.</p>
      <svg viewBox={`0 0 ${W} ${H}`} className="strip-svg">
        {[-0.5, 0, 0.5, 1].map((t) => <g key={t}><line x1={x(t)} x2={x(t)} y1={10} y2={H - 22} className="sc-grid" /><text x={x(t)} y={H - 6} textAnchor="middle" className="sc-ax">{t === -0.5 ? "≤ −50%" : `${t * 100}%`}</text></g>)}
        {rows.map((r, j) => {
          const y = 28 + j * 46;
          return <g key={r.k}>
            <text x={L - 12} y={y + 4} textAnchor="end" className="strip-l" fill={r.col}>{r.label}</text>
            {r.vals.map((z, q) => <circle key={q} cx={x(z.f)} cy={y + ((q * 7) % 15) - 7} r={4.5} fill={z.pat === "semantic" ? r.col : "#fff"} stroke={r.col} strokeWidth={1.5} opacity={0.85} />)}
            {r.med != null && <line x1={x(r.med)} x2={x(r.med)} y1={y - 14} y2={y + 14} stroke={r.col} strokeWidth={4} />}
            {r.med != null && <text x={W - 4} y={y + 4} textAnchor="end" className="strip-m" fill={r.col}>{pct(r.med)}</text>}
          </g>;
        })}
      </svg>
    </div>
  );
}
