import { f, n } from "./api";

// "Closer to production": full-catalog retrieval -> multi-feature LambdaMART -> SHAP triage, plus standard baselines.
// Conclusion-first, plain English; numbers from /api/evalreport (ltr.json, baselines_*.json).

const pct = (x: number, d = 0) => `${n(100 * x, d)}%`;
const MODEL: Record<string, string> = { popularity: "Most popular", text: "Text match only", behaviour: "Viewing history only (graph model)", F0: "Current ranker (3 signals)", LambdaMART: "LambdaMART, 45 signals" };
const FAMC: Record<string, string> = { text: "#c5a353", popularity: "#9aa3ad", history: "#7194b9", "title content": "#a697ba", retrieval: "#6fa69a" };
const FAMN: Record<string, string> = { text: "text match", popularity: "popularity", history: "viewing history", "title content": "title content (genre, year)", retrieval: "which retriever found it" };
const CAUSE: Record<string, string> = {
  retrieval_miss: "Retrieval never found a right title", text: "We misread the request", popularity: "A more popular movie won",
  history: "Their past viewing took over", "title content": "Genre/year signals misfired", retrieval: "Retriever flags misled the ranker",
};
const BL: Record<string, string> = { popularity: "Most popular", bpr_mf: "Matrix factorisation (BPR-MF)", lightgcn: "LightGCN", sasrec: "SASRec (sequence transformer)", tgat1: "TGAT · 1 layer (ours)", tgat2: "TGAT · 2 layers (ours)", tgn: "TGN · memory (ours)" };

export default function Production({ ltr, baselines }: any) {
  if (!ltr && !baselines?.final) return null;
  const S = ltr?.final;
  const all = S?.["all searches"];
  const lm = all?.LambdaMART, f0 = all?.F0;
  const R = ltr?.retrieval;
  const tri = ltr?.triage ?? ltr?.triage_dev;
  const B = baselines?.final?.results;
  const best = B ? Object.entries(B).sort((a: any, b: any) => b[1].ndcg10.mean - a[1].ndcg10.mean) : [];
  return (
    <>
      <div className="sec-k" style={{ marginBottom: -6 }}>4 · Closer to production</div>
      {ltr && <section className="cap-card">
        <h3 className="take">On the full catalog, a {ltr.n_features}-signal LambdaMART ranker beats the simple 3-signal ranker in every group of held-out searches ({f(lm.delta_vs_F0, 2)} ranking quality overall), and the same “why did it fail” analysis still works, now with SHAP.</h3>
        <p className="how">Instead of re-ordering 30 hand-picked candidates, every search now runs like a real system: retrieve a few hundred titles from the whole catalog, rank them with a gradient-boosted model (LambdaMART) on 45 signals, score against every relevant title in the catalog. Trained on development searches, scored once on held-out ones.</p>
        <div className="pipe">
          <div><b>{(R.catalog_size || 0).toLocaleString()}</b><span>titles in the catalog</span></div><i>→</i>
          <div><b>~{Math.round(R.mean_candidates)}</b><span>retrieved per search: text match 150 · personal 100 · popular 50</span></div><i>→</i>
          <div><b>{ltr.n_features} signals</b><span>LambdaMART ranks them</span></div><i>→</i>
          <div><b>top 10</b><span>shown</span></div>
        </div>
        <p className="verdict warn">Caution: gains on searches that name a genre partly reflect our genre-based labels. Clean number, vague searches: {S?.["vague searches"] ? `${n(S["vague searches"].F0.ndcg10, 2)} → ${n(S["vague searches"].LambdaMART.ndcg10, 2)} (most-popular ${n(S["vague searches"].popularity.ndcg10, 2)})` : ""}.</p>
        <p className="cap">Retrieval finds at least one right title for {pct(R.searches_with_a_relevant_title_retrieved)} of searches; a perfect ranker on what was retrieved would reach NDCG@10 {n(R.oracle_ndcg10_on_retrieved, 2)} (the ceiling retrieval sets).</p>
        <div className="prod2">
          <div>
            <table className="tbl mini"><thead><tr><th>held-out searches · NDCG@10 on the full catalog</th>{["popularity", "text", "F0", "LambdaMART"].map((m) => <th key={m}>{MODEL[m]}</th>)}</tr></thead><tbody>
              {Object.entries(S).map(([sl, e]: any) => <tr key={sl}><td>{sl} <small>({e.F0.n})</small></td>
                {["popularity", "text", "F0", "LambdaMART"].map((m) => <td key={m} className={m === "LambdaMART" ? (e[m].ci95?.[0] > 0 ? "good" : e[m].ci95?.[1] < 0 ? "bad" : "") : ""}>{n(e[m].ndcg10, 3)}{m === "LambdaMART" && e[m].delta_vs_F0 != null ? <small> ({f(e[m].delta_vs_F0, 3)})</small> : null}</td>)}</tr>)}
            </tbody></table>
            <p className="cap">Green = LambdaMART better than the current ranker with the 95% interval above zero (members resampled). Wrong #1 on searches that name what they want: current ranker {pct(S["searches that name what they want"].F0.top1_wrong)} → LambdaMART {pct(S["searches that name what they want"].LambdaMART.top1_wrong)}.</p>
          </div>
          <div>
            <p className="cap"><b>What the ranker leans on</b> (share of total split gain)</p>
            {Object.entries(ltr.family_importance).map(([k, v]: any) => <div key={k} className="famrow"><span>{FAMN[k] ?? k}</span><i><b style={{ width: pct(v), background: FAMC[k] }} /></i><em>{pct(v)}</em></div>)}
          </div>
        </div>
        {tri && <div className="prodtri">
          <p className="take sm">Why the {ltr.n_features}-signal ranker still fails ({pct(tri.fail_rate)} of held-out searches put a wrong title first): each failure's lead splits exactly into feature-family parts with TreeSHAP, so the triage from section 1 carries over to a non-linear ranker.</p>
          <div className="stack">{Object.entries(tri.causes).sort((a: any, b: any) => b[1].n - a[1].n).map(([k, v]: any) => (
            <div key={k} className="stk" style={{ flex: v.n, background: k === "retrieval_miss" ? "#c82434" : FAMC[k] ?? "#999" }}>{v.share > 0.1 ? `${CAUSE[k] ?? k} · ${pct(v.share)}` : pct(v.share)}</div>))}</div>
          <p className="cap">Top signals behind each cause: {Object.entries(tri.causes).filter(([k]) => k !== "retrieval_miss").map(([k, v]: any) => `${CAUSE[k] ?? k}: ${v.top_features.slice(0, 2).map((z: any) => z[0]).join(", ")}`).join(" · ")}.</p>
        </div>}
      </section>}
      {B && <section className="cap-card">
        <h3 className="take">{best[0][0] === "popularity" ? "On this small dataset, “most popular” is hard to beat;" : `${BL[best[0][0]]} ranks best;`} our TGN {B.tgn ? `reaches ${n(B.tgn.ndcg10.mean, 3)} ranking quality (most popular: ${n(B.popularity.ndcg10.mean, 3)}) and finds the most future favourites in its top 50 (${pct(B.tgn.recall50.mean)})` : ""}. Most differences are within the noise of {B.popularity.n} members.</h3>
        <p className="how">Next-title prediction: for each member, rank every title they haven't rated and check which ones they rated 4+ in the next 90 days. Same members, same time, trained on the same period. Most held-out members are new, so ID-based models (BPR-MF, LightGCN) build a member from their recent titles.</p>
        <table className="tbl mini"><thead><tr><th>model</th><th>NDCG@10</th><th>Recall@10</th><th>Recall@50</th></tr></thead><tbody>
          {best.map(([k, v]: any) => <tr key={k} className={k.startsWith("tg") ? "tgt" : ""}><td>{BL[k] ?? k}</td>
            <td>{n(v.ndcg10.mean, 3)} <small>[{n(v.ndcg10.ci95[0], 3)}, {n(v.ndcg10.ci95[1], 3)}]</small></td><td>{n(v.recall10.mean, 3)}</td><td>{n(v.recall50.mean, 3)}</td></tr>)}
        </tbody></table>
        <p className="cap">{baselines.final.results.popularity.n} held-out members, {baselines.final.catalog_size.toLocaleString()} titles; brackets: 95% interval (members resampled). Small sample — intervals overlap for most pairs.</p>
      </section>}
    </>
  );
}
