import { n } from "./api";
import PlantViz from "./PlantViz";

// Planted-cause test: we bury a known cause, retrain, and ask each explainer to find it. Conclusion first.
const pct = (x: number | null | undefined) => (x == null ? "–" : `${n(100 * x, 0)}%`);
const LABEL: Record<string, string> = {
  "TemGX": "TemGX (ours)", "CATGX": "CATGX (ours)", "Best single removals": "Remove one rating at a time", "Attention top-2": "Model's attention weights",
  "Most recent 2": "Two most recent ratings", "Random 2": "Two random ratings", "LLM (reads the history as text)": "LLM reading the history", "Try every pair": "Try every pair (best possible, slow)",
};

export default function Planted({ pr }: any) {
  if (!pr) return null;
  const [S, A] = pr.patterns;
  const sAll = S.all, aEff = A.where_planted_is_models_reason;
  return (
    <>
      <div className="sec-k" style={{ marginBottom: -6 }}>5 · Does the explanation find a cause we planted ourselves?</div>
      <section className="cap-card">
        <h3 className="take">We hid a cause in the data and retrained the model. Delete the ratings TemGX blames and the model changes its mind; delete the ones an LLM blames and it mostly doesn't.</h3>
        <PlantViz pr={pr} simple />
        <div className="psum">{[["TemGX", "TemGX (reads the model)", "#c82434"], ["LLM (reads the history as text)", "LLM (reads the history)", "#7a5bb5"]].map(([k, l, c]) => {
          const vals: number[] = [];
          pr.patterns.forEach((p: any) => p.viz.forEach((v: any) => { if (v.plant_is_models_reason && v.fidelity[k] != null) vals.push(v.fidelity[k]); }));
          vals.sort((x, y) => x - y);
          const med = vals[Math.floor(vals.length / 2)];
          return <div key={k} className="psrow"><span style={{ color: c }}>{l}</span><i><b style={{ width: `${Math.max(0, med) * 100}%`, background: c }} /></i><em>{pct(med)}</em></div>;
        })}<p className="cap">How much of the model's preference disappears when each method's answer is deleted (median over {pr.patterns.reduce((a: number, p: any) => a + p.viz.filter((v: any) => v.plant_is_models_reason).length, 0)} members; 100% = the best any two deletions can do).</p></div>
        <details className="pdet"><summary>Full results: both planted patterns, every method, every member</summary>
          <div className="plant2">
            {[S, A].map((p: any) => <div key={p.pattern} className="pp">
              <b>{p.pattern === "semantic" ? "A pattern that makes sense" : "A pattern that makes no sense"}</b>
              <span className="chain">{p.titles[0]} → {p.titles[1]} → <em>{p.titles[2]}</em></span>
              <small>{p.pattern === "semantic" ? `designed by an LLM: “${p.rationale}”` : "three unrelated titles picked at random"}</small>
              <small>Removing the two planted ratings lowers the model's preference for title 3 by {n(p.sanity.mean_margin - p.sanity.mean_margin_without_planted, 1)} on average{p.pattern === "arbitrary" ? `; the plant was the model's main reason in ${p.n_effective} of ${p.probes} members (only those are scored)` : ` (${p.probes} members)`}.</small>
            </div>)}
          </div>
          <table className="tbl mini plantt"><thead><tr><th>method</th><th>found a planted rating<br /><small>makes sense / no sense</small></th><th>effect removed (mean)<br /><small>makes sense / no sense · below 0 = model liked title 3 even more</small></th></tr></thead><tbody>
            {pr.order.map((k: string) => {
              const s2 = sAll[k], a2 = aEff[k];
              const cls = k === "TemGX" ? "ours" : k.startsWith("LLM") ? "llm" : "";
              return <tr key={k} className={cls}><td>{LABEL[k] ?? k}</td><td>{pct(s2.found_any)} / {pct(a2.found_any)}</td>
                <td><span className="fbar"><i style={{ width: `${Math.max(0, Math.min(1, s2.fidelity)) * 100}%` }} /></span>{pct(s2.fidelity)} / {pct(a2.fidelity)}</td></tr>;
            })}
          </tbody></table>
          <PlantViz pr={pr} />
          <p className="cap">Honest notes: {S.probes} + {A.n_effective} members, so treat these as indicative; the model learned the pattern mostly through title 1; CATGX on this 1-layer model uses attention as its importance signal and misses the pattern that makes no sense, like attention itself. LLM = {S.llm_model}.</p>
        </details>
      </section>
    </>
  );
}
