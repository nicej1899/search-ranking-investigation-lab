import { n } from "./api";

// Measured scalability (artifacts/reports/scalability.json). Paper-reported numbers are labelled separately.
export default function Scale({ sc, cmp }: any) {
  const cs = cmp?.scale?.rows as any[] | undefined;
  if (!sc) return null;
  const s = sc.A_explainer_vs_exhaustive.summary as any[];
  const b = sc.B_batch_audit;
  const W = 420, H = 210, L = 46, Rr = 12, T = 14, Bt = 34;
  const xs = (w: number) => L + ((w - s[0].w) / (s[s.length - 1].w - s[0].w)) * (W - L - Rr);
  const ys = (v: number) => T + (1 - (Math.log10(v) - 1) / 4) * (H - T - Bt); // 10 .. 100k
  const line = (k: string) => s.map((r, i) => `${i ? "L" : "M"}${xs(r.w)},${ys(r[k])}`).join(" ");
  const last = s[s.length - 1];
  return (
    <section className="scale">
      <div className="cap-card">
        <h3>Finds the cause without trying every combination</h3>
        <p className="does">Checking every possible group of past ratings explodes as history grows (grey). Our method grows slowly (red) and finds the same answer.</p>
        <svg viewBox={`0 0 ${W} ${H}`} className="sc-chart">
          {[10, 100, 1000, 10000, 100000].map((v) => <g key={v}><line x1={L} x2={W - Rr} y1={ys(v)} y2={ys(v)} className="sc-grid" /><text x={L - 6} y={ys(v) + 4} textAnchor="end" className="sc-ax">{v >= 1000 ? `${v / 1000}k` : v}</text></g>)}
          {s.map((r) => <text key={r.w} x={xs(r.w)} y={H - Bt + 16} textAnchor="middle" className="sc-ax">{r.w}</text>)}
          <text x={(W + L) / 2} y={H - 2} textAnchor="middle" className="sc-ax">number of past ratings considered (looking for the best ≤3)</text>
          <text x={12} y={T + 70} className="sc-ax" transform={`rotate(-90 12 ${T + 70})`}>times the model is run</text>
          <path d={line("exhaustive_calls")} className="sc-l ex" />
          <path d={line("temgx_calls")} className="sc-l tg" />
          {s.map((r) => <circle key={"e" + r.w} cx={xs(r.w)} cy={ys(r.exhaustive_calls)} r={3} className="sc-d ex" />)}
          {s.map((r) => <g key={"t" + r.w}><circle cx={xs(r.w)} cy={ys(r.temgx_calls)} r={3} className="sc-d tg" />
            <text x={xs(r.w)} y={ys(r.temgx_calls) - 8} textAnchor="middle" className="sc-q">{n(100 * r.quality_mean, 0)}%</text></g>)}
          <text x={xs(last.w) - 4} y={ys(last.exhaustive_calls) - 8} textAnchor="end" className="sc-lab ex">try everything: {last.exhaustive_calls.toLocaleString()} runs</text>
          <text x={xs(last.w) - 4} y={ys(last.temgx_calls) + 18} textAnchor="end" className="sc-lab tg">our method: {n(last.temgx_calls, 0)} runs</text>
        </svg>
        <p className="cap">Red % = how close our answer is to the best possible (average of {last.n_cases} people with long histories). Honest caveat: with only 10 ratings, one case was a clear miss.</p>
      </div>
      <div className="cap-card tiles">
        <h3>Checks every failure, not just one</h3>
        <p className="does">After an evaluation run, investigate all failed searches automatically instead of one by one.</p>
        <div className="big"><b>{b.n_cases}</b><span>failed searches investigated (steps 1 and 3)</span></div>
        <div className="big accent"><b>{n(b.wall_s, 1)} s</b><span>in total, on an ordinary {sc.hardware.cpu_count}-core CPU, no GPU</span></div>
        <div className="big"><b>{n(b.explain_ms.p50, 0)} / {n(b.motif_ms.p50, 0)} ms</b><span>typical time per search: find cause / check pattern</span></div>
        <div className="big"><b>{(b.occurrences.total as number).toLocaleString()}</b><span>viewing patterns compared across those searches</span></div>
      </div>
      {cs ? <div className="cap-card">
        <h3>Patterns stay cheap as the graph grows (CATGX)</h3>
        <p className="does">Checking every pattern instance with the model gets expensive fast. CATGX retrieves similar instances and samples them, so the model-run count stays small. Measured on the {cmp.scale.backbone.toUpperCase()} model, 3 hops.</p>
        <table className="tbl"><thead><tr><th>links in view</th><th>pattern instances</th><th>check all: runs · time</th><th>CATGX sampled: runs · time</th><th>same top pattern</th></tr></thead><tbody>
          {cs.map((r: any) => <tr key={r.cap}><td>{r.edges.toLocaleString()}</td><td>{r.occurrences.toLocaleString()}</td>
            <td>{r.exact ? `${r.exact.calls.toLocaleString()} · ${n(r.exact.ms / 1000, 1)} s` : "not run (too slow)"}</td>
            <td><b>{r.sampled.calls.toLocaleString()} · {n(r.sampled.ms / 1000, 1)} s</b></td>
            <td>{r.exact ? (r.exact.top[0] === r.sampled.top[0] ? "yes" : "no") : "–"}</td></tr>)}
        </tbody></table>
        <p className="cap">Published results (not measured here): TemGX 8.2 s per prediction vs 35–88 s for prior methods; CATGX 48.2 s on a one-million-edge graph.</p>
      </div> : <div className="cap-card tiles">
        <h3>Works on much bigger graphs <small>(from the published papers, not measured in this demo)</small></h3>
        <div className="big"><b>8.2 s</b><span>to explain one prediction (TemGX), vs 35–88 s for other published methods</span></div>
        <div className="big"><b>48.2 s</b><span>to find patterns in a graph with one million edges (CATGX)</span></div>
      </div>}
    </section>
  );
}
