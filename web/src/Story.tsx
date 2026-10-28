import { f, n } from "./api";
import { nm } from "./names";
import MotifCard from "./MotifCard";
import { MCOL } from "./SubGraph";

// Right-hand panel. Plain English first ("what this step does"), then 1-3 big numbers from the backend.
const pct = (x: number) => `${n(100 * x, 0)}%`;

export default function Story({ sig, jt, kTx, sweep, rep, parts, ov, step, sg, tx, cx, cxe, cmp, ev, inv, bk, L, showcase, bkInfo }: any) {
  const big = (b: string, s: string, cls = "") => <div className={`big ${cls}`}><b>{b}</b><span>{s}</span></div>;
  let tag = "", head = "", does = "", body: any = null;
  const cell = cmp?.cells?.[`${bk}|L${L}`];
  if (step === 0) {
    const st = ov?.stats;
    tag = "The data"; head = "Who rated what, and when";
    does = "Each line is one person rating one movie at a specific moment. The models learn only from the past, so for any search we only use ratings made before it.";
    body = st && <>
      {big(`${st.users} people · ${st.movies.toLocaleString()} movies`, "the dots")}
      {big(`${st.edges.toLocaleString()} ratings`, "the lines, each with a timestamp", "accent")}
      {showcase && <p className="cap">Shown: {showcase.cases.length} searches where the explanation is clearest, hand-picked from {showcase.n_checked.toLocaleString()} (not a random sample).</p>}
      <p className="cta">Click a person to investigate their search →</p></>;
  } else if (step === 1 && sg) {
    tag = "The problem"; head = "The top result doesn't fit the search";
    does = `This person searched “${sg.query}”. Search put A at #${sg.rank_A} and the better match B at #${sg.rank_B}. Below: how strongly the chosen graph model prefers A, and how far it can “see”.`;
    body = <>
      {big(pct(sg.prob_A_over_B), `the ${bkInfo?.label ?? bk} model's confidence that A beats B`, "accent")}
      {parts && (() => {
        const cause = parts.relevance > 0 ? "The search text was misread" : parts.behaviour >= parts.popularity ? "Viewing history outweighed the search" : "Popularity outweighed the search";
        const mx = Math.max(...Object.values(parts).map((v: any) => Math.abs(v)), 1e-9);
        return <div className="whyA"><p className="mh2">Why the ranker put A above B</p>
          {([["behaviour", "viewing history"], ["popularity", "popularity"], ["relevance", "match to the search text"]] as const).map(([k, l]) => (
            <div key={k} className="wa"><span>{l}</span><i><b className={(parts as any)[k] > 0 ? "pos" : "neg"} style={{ width: `${Math.abs((parts as any)[k]) / mx * 100}%` }} /></i><em>{f((parts as any)[k], 2)}</em></div>))}
          <p className="mh2">→ {cause}</p></div>;
      })()}
      {big(`${sg.nodes.length} nodes · ${sg.edges.length} links`, `within ${L} step${L > 1 ? "s" : ""} of u, A and B (last 12 months)`)}
      <p className="cap"><b>{bkInfo?.label}:</b> {bkInfo?.plain}.</p></>;
  } else if (step === 2 && tx) {
    tag = "Step 1 · Find the cause  (TemGX)"; head = "Which specific links push A up?";
    does = `TemGX looks at everyone and everything up to ${L} hop${L > 1 ? "s" : ""} away, scores how strongly each one's influence travels along time-ordered paths to this decision, picks a few, and checks every pick by re-inference of the model.`;
    const hops = Object.values(tx.explanatory_hops) as number[];
    const far = hops.filter((h) => h >= 2).length;
    body = <>
      {big(`${tx.explanatory.length} drivers + ${tx.connection.length} connectors`, far ? `${far} are 2+ hops away` : (bkInfo?.hops === 1 ? "all 1 hop away (1-layer model)" : "all 1 hop away"))}
      {big(`${pct(tx.prob_full)} → ${pct(tx.prob_without)}`, `A beats B · after deleting them`, "accent")}
      <div className="ksweep"><p className="mh2">Bigger k → bigger drop</p>
        {[1, 2, 3, 5, 8].map((k) => { const r = sweep?.[k]; return <div key={k} className={`kr ${k === kTx ? "on" : ""}`}><span>k = {k}</span>
          <i>{r && <b style={{ width: `${100 * r.prob_without}%` }} className={r.flipped ? "flip" : ""} />}</i><em>{r ? `${pct(r.prob_without)}${r.flipped ? " · flips" : ""}` : "…"}</em></div>; })}</div>
      <p className="mh2">Better than simple guesses (drop in A's lead)</p>
      <div className="vs">{[["TemGX", tx.baselines.temgx], ["most recent", tx.baselines.most_recent], ["random", tx.baselines.random_mean], ...(tx.baselines.attention != null ? [["attention", tx.baselines.attention]] : [])].map(([k, v]: any) => {
        const mx = Math.max(tx.baselines.temgx, tx.baselines.most_recent, tx.baselines.random_mean, tx.baselines.attention ?? 0, 1e-9);
        return <div key={k}><span>{k}</span><i style={{ width: `${Math.max(0, v) / mx * 100}%` }} /><em>{f(v, 2)}</em></div>;
      })}</div>
      <p className="cap">{tx.n_model_inferences} model runs · {n(tx.latency_ms.total / 1000, 1)} s · searched {tx.n_edges} links, kept {n(100 * tx.sparsity, 0)}%</p></>;
  } else if (step === 3 && tx) {
    tag = "Step 2 · Check it  (replay)"; head = tx.flipped ? "Remove them → B now beats A" : "Remove them → A's lead shrinks";
    does = "Same person, same search, same model. We delete only the explanation links (✕) and run the model again — the explanation is tested, not trusted.";
    body = <>
      {big(`${pct(tx.prob_full)} → ${pct(tx.prob_without)}`, "confidence that A beats B", tx.flipped ? "good" : "accent")}
      {big(`${pct(tx.prob_only)}`, "confidence when the model sees ONLY the explanation links (they are enough on their own)")}
      <p className="cap">{tx.flipped ? "The decision flips: these links are the cause." : "A real cause, but not the only one."}</p></>;
  } else if (step === 4 && cx) {
    tag = `Step 3 · Find the pattern  (CATGX${sig === "do" ? " + TemGX replay" : ""})`; head = "What kind of structure is behind it?";
    does = (sig === "do" ? "Importance of each pattern instance = how much the decision changes when it is deleted and the model runs re-inference (TemGX's test), not attention. " : "Importance = the model's attention on the instance (as in the paper). ") + "CATGX looks for recurring shapes in the graph and asks which shapes the model leans on more than other shapes that are equally recent and involve equally active people and movies. Each shape is shown on its own below, with a real example.";
    const cards = cx.motif_cards ?? [];
    body = <>
      {!cx.motif_cards ? <p className="banner warn">This result came from an older backend. Restart the server (Ctrl+C, then <code>make serve</code>) and reload to see the pattern cards.</p> : cards.length ? <MotifCard card={cards[0]} meta={cx.meta} sg={sg} color={MCOL[cards[0].motif]} /> : big("no pattern", "none found in this subgraph")}
      {cards.slice(1).map((cd: any) => <MotifCard key={cd.motif} card={cd} meta={cx.meta} sg={sg} color={MCOL[cd.motif]} compact />)}
      <p className="cap">{cxe ? `Checked ${cx.occurrences_used} of ${cxe.occurrences_total} instances (sampling, ${(cxe.selected_motifs?.[0] === cx.selected_motifs?.[0]) ? "same" : "different"} top shape as checking all) · ` : ""}{cx.n_model_inferences} model runs · {n(cx.latency_ms.total, 0)} ms</p></>;
  } else if (step === 5) {
    tag = "Together · causal check"; head = "Which links, what kind, and is it the cause?";
    does = "Causal means: change the input, run re-inference on the model. We delete each explanation's links and compare with deleting the same number of random links from the same time period (placebo, grey band).";
    const ARM: [string, string, string][] = [["temgx", "TemGX links", "#c82434"], ["catgx_do", "CATGX pattern (replay)", "#c08a1e"], ["catgx", "CATGX pattern (attention)", "#c9b48a"], ["attention", "highest-attention links", "#8b97a3"]];
    const PL: Record<string, string> = { SEQ: "watched in a row", CO: "rated the same movie", BRIDGE: "taste path to A", BRIDGEB: "taste path to B", POP: "crowd rating A" };
    const nm = (v: number) => { const m = jt?.meta?.[String(v)]; const t = m ? m.title : `node ${v}`; return t.length > 18 ? t.slice(0, 17) + "…" : t; };
    const Sm = jt?.summary;
    body = !jt ? <p className="cap">Running the causal check…</p> : <>
      <div className="dol">
        <div className="dorow obs"><span>nothing removed</span><i><b style={{ width: pct(jt.p0) }} /></i><em>{pct(jt.p0)}</em><strong /></div>
        {ARM.filter(([k]) => jt.arms[k]).map(([k, l, c]) => {
          const a = jt.arms[k], ok = a.p_value <= 0.05 && a.drop > 0;
          return <div key={k} className="dorow"><span>remove {l} <small>· {a.n}</small></span>
            <i><u style={{ left: pct(a.placebo_p_lo), width: `${Math.max(0.6, 100 * (a.placebo_p_hi - a.placebo_p_lo))}%` }} /><b style={{ width: pct(a.p), background: c }} /></i>
            <em>{pct(a.p)}</em><strong className={ok ? "ok" : "no"}>{ok ? "cause" : "≈ random"}</strong></div>;
        })}
        <p className="dolab">A beats B · after removal</p>
      </div>
      <p className="mh2">Each TemGX link → the pattern it is part of</p>
      <div className="lnk">{jt.labels.map((x: any) => <div key={x.eid}><span>{nm(x.s)} — {nm(x.t)}</span>{x.motif ? <b style={{ color: MCOL[x.motif] }}>■ {PL[x.motif]}</b> : <b>–</b>}</div>)}</div>
      {Sm && <><p className="mh2">Over {Sm.n_cases} searches ({bkInfo?.label ?? bk}, {L} hop{L > 1 ? "s" : ""})</p>
        <table className="tbl tog"><thead><tr><th></th><th>beats random</th><th>patterns covered</th><th>model runs</th></tr></thead><tbody>
          {ARM.filter(([k]) => Sm.arms[k]).map(([k, l, c]) => <tr key={k}><td style={{ color: c, fontWeight: 600 }}>{l}</td>
            <td><b>{Math.round(Sm.arms[k].share_beats_placebo_p05 * Sm.arms[k].n_cases)}/{Sm.arms[k].n_cases}</b></td>
            <td>{n(100 * (Sm.coverage[k] ?? 0), 1)}%</td>
            <td>{Sm.calls ? (k === "temgx" ? n(Sm.calls.temgx, 0) : k === "catgx_do" ? n(Sm.calls.catgx_do, 0) : k === "catgx" ? n(Sm.calls.catgx, 0) : "1") : "–"}</td></tr>)}
        </tbody></table>
        <p className="cap">Beats random = removal lowers the model's preference for A more than 95% of 40 random, same-period removals of the same size. Patterns covered = share of all pattern instances in the subgraph fully inside the explanation.</p></>}
    </>;
  } else if (step === 6) {
    const fin = ev?.comparisons?.["qi-final-v1"]?.F0_vs_F1?.slices;
    const FX = rep?.fixes;
    const key = parts ? (parts.relevance > 0 ? ["F3", "Read the “not …” part of the search", ["original with 'not'/'nothing'", "conversational"]]
      : parts.behaviour >= parts.popularity ? ["F2", "Trust viewing history less when it is short", ["coldstart (1-4 ratings)"]]
      : ["F4", "Lean less on popularity when the search names what it wants", ["original explicit"]]) : null;
    const LBL: Record<string, string> = { "original explicit": "searches that name what they want", "original broad ('what should I watch')": "vague searches",
      "original with 'not'/'nothing'": "searches with “not / nothing”", "coldstart (1-4 ratings)": "new members (1–4 ratings)", "conversational": "chat-style requests" };
    const cmp = key && FX ? FX[`${key[0]}_vs_F0`] : null;
    const tgt = cmp ? (key![2] as string[]).map((sname) => [sname, cmp[sname]]).filter(([, e]: any) => e) : [];
    const worst = cmp ? Object.entries(cmp).filter(([sname, e]: any) => LBL[sname] && e.ndcg10).sort((a: any, b: any) => a[1].ndcg10.delta - b[1].ndcg10.delta)[0] : null;
    tag = "Step 4 · Try a fix  (agent)"; head = "Fix this cause, test on searches never seen";
    does = key ? `This search failed because of “${parts.relevance > 0 ? "the search text was misread" : parts.behaviour >= parts.popularity ? "viewing history" : "popularity"}”. The agent's targeted fix: ${nm(key[0] as string)}: ${key[1]}. Tuned on development searches, scored once on held-out searches.` : "";
    body = <>
      {tgt.map(([sname, e]: any) => big(`${f(e.ndcg10.delta, 3)}`, `ranking quality on ${LBL[sname]} (95%: ${f(e.ndcg10.ci95[0], 3)} to ${f(e.ndcg10.ci95[1], 3)})`, e.ndcg10.ci95[0] > 0 ? "good" : e.ndcg10.ci95[1] < 0 ? "bad" : ""))}
      {worst && (worst[1] as any).ndcg10.ci95[1] < 0 && !tgt.some(([sname]: any) => sname === worst[0]) && big(`${f((worst[1] as any).ndcg10.delta, 3)}`, `on ${LBL[worst[0] as string]} — the cost`, "bad")}
      <p className="cap"><b>Verdict:</b> {worst && (worst[1] as any).ndcg10.ci95[1] < 0 ? "trade-off — limit the fix to where it helps, then A/B test." : "no group got worse — candidate for an A/B test."}</p>
      {fin && <p className="cap">Earlier attempt (rely less on viewing history for specific searches): {f(fin.broad.ndcg10.delta, 3)} vague, {f(fin.explicit.ndcg10.delta, 3)} specific — rejected.</p>}</>;
  } else if (step >= 7) {
    tag = "Reproduce"; head = "Undo, and save the evidence";
    does = "Every number came from a model run that can be repeated exactly; the investigation is exported with fingerprints.";
    body = <>{big("identical", "result after undo")}{big(`${inv?.trace?.length ?? "–"} steps`, "saved in the exported report")}</>;
  }
  return <aside className="story"><div className="algo">{tag}</div><h2>{head}</h2>{does && <p className="does">{does}</p>}{body}</aside>;
}
