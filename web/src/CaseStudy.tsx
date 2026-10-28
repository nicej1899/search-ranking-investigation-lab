import { useEffect, useMemo, useState } from "react";
import { NAME, sn } from "./names";
import { f, n } from "./api";
import StructurePanel, { Diagram, PLAIN } from "./StructurePanel";
import SubGraph, { MCOL } from "./SubGraph";

// Case study in four moves: Problem → Evidence → Verification → Decision. Left: the real graph around one affected
// member (click a link to delete it). Right: one picture per move, computed live (/api/live re-ranks on every change).
// Evidence = TemGX (which links) + CATGX (what structure, with an LLM reading of it); Verification = delete and re-run
// against random deletions; Decision = the targeted fix, tested on held-out searches.
const RED = "#c82434", GREEN = "#2a78d6", POP = "#8b97a3", TXT = "#c5a353", HIS = "#5b8fc7", GREY = "#b8c2cc", GOLD = "#c08a1e";
const BK = "tgat2", L = 2;
const pct = (x: number) => `${n(100 * x, 0)}%`;
const short = (s: string, k = 30) => (s.length > k ? s.slice(0, k - 1) + "…" : s);
const STEPS: [string, string][] = [["Problem", "what went wrong"], ["Evidence", "why it happened"], ["Verification", "is that really why?"], ["Decision", "what to ship"]];
const MODELS: [string, string][] = [["F0", NAME.F0], ["F3", NAME.F3], ["F4", NAME.F4], ["F5", NAME.F5], ["F4+F5", NAME["F4+F5"]]];
const post = (u: string, b: any) => fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) }).then((r) => r.json());

export default function CaseStudy() {
  const [d, setD] = useState<any>(null);
  const [cid, setCid] = useState("");
  const [step, setStep] = useState(0);
  const [sg, setSg] = useState<any>(null);
  const [tx, setTx] = useState<any>(null);
  const [st, setSt] = useState<any>(null);
  const [base, setBase] = useState<any>(null);
  const [live, setLive] = useState<any>(null);
  const [model, setModel] = useState("F0");
  const [removed, setRemoved] = useState<number[]>([]);
  const [ver, setVer] = useState<any>(null);
  const [more, setMore] = useState(false);
  const [lerr, setLerr] = useState("");
  const [kT, setKT] = useState(3);   // TemGX: how many links it may keep
  const [kC, setKC] = useState(2);   // CATGX: edge budget for the patterns it keeps (knapsack)
  const [cxk, setCxk] = useState<any>(null);
  const [ctr, setCtr] = useState<any>(null);
  const [tg, setTg] = useState<any>(null);  // tests generated from the explanations
  const [sim, setSim] = useState<any>(null);  // similar diagnosed failures from the verified-explanation store  // contrastive diagnosis: what supports A vs what supports B

  const [f5, setF5] = useState<any>(null);
  useEffect(() => { fetch("/api/f5").then((r) => (r.ok ? r.json() : null)).then(setF5).catch(() => {}); }, []);
  useEffect(() => { fetch("/api/testgen").then((r) => (r.ok ? r.json() : null)).then(setTg).catch(() => {}); }, []);
  useEffect(() => { fetch("/api/casestudy").then((r) => (r.ok ? r.json() : null)).then((x) => { setD(x); if (x) setCid(x.demo_member ?? x.featured.case_id); }).catch(() => {}); }, []);
  useEffect(() => {
    if (!cid) return;
    setSg(null); setTx(null); setSt(null); setBase(null); setVer(null); setRemoved([]); setModel("F0");
    fetch(`/api/subgraph/${cid}?backbone=${BK}&L=${L}`).then((r) => r.json()).then(setSg);
    post("/api/live", { case_id: cid, backbone: BK, model_id: "F0", removed: [] }).then((r) => r.rows ? setBase(r) : setLerr(r.detail ?? "no result"));
    fetch(`/api/structure/${cid}`).then((r) => (r.ok ? r.json() : null)).then(setSt);
  }, [cid]);
  useEffect(() => { if (cid) { setTx(null); setVer(null); post("/api/temgx", { case_id: cid, backbone: BK, L, k: kT }).then(setTx); } }, [cid, kT]);
  const [rag, setRag] = useState<any>(null);
  const [rtv, setRtv] = useState<any>(null);
  const [recRun, setRecRun] = useState<any>(null);  // live ranking under the fix retrieval recommends
  useEffect(() => { setRecRun(null); if (cid && sim?.recommend) post("/api/live", { case_id: cid, backbone: BK, model_id: sim.recommend, removed: [] }).then((r) => r.rows && setRecRun(r)); }, [cid, sim]);  // GraphRAG retrievers compared at equal size
  useEffect(() => { if (cid) { setRtv(null); fetch(`/api/retrievers/${cid}`).then((r) => (r.ok ? r.json() : null)).then(setRtv).catch(() => {}); } }, [cid]);
  useEffect(() => { if (cid) { setRag(null); fetch(`/api/rag/${cid}`).then((r) => (r.ok ? r.json() : null)).then(setRag).catch(() => {}); } }, [cid]);
  useEffect(() => { if (cid) { setSim(null); fetch(`/api/similar/${cid}`).then((r) => (r.ok ? r.json() : null)).then(setSim).catch(() => {}); } }, [cid]);
  useEffect(() => { if (cid) { setCtr(null); fetch(`/api/contrast/${cid}?k=${kT}`).then((r) => (r.ok ? r.json() : null)).then(setCtr).catch(() => {}); } }, [cid, kT]);
  useEffect(() => { if (cid) { setCxk(null); setVer(null); post("/api/catgx_do", { case_id: cid, backbone: BK, L, k: 3, kc: kC }).then(setCxk); } }, [cid, kC]);
  useEffect(() => {
    if (!cid) return;
    let alive = true;
    post("/api/live", { case_id: cid, backbone: BK, model_id: model, removed })
      .then((r) => { if (!alive) return; if (r?.rows) { setLive(r); setLerr(""); } else setLerr(r?.detail ?? "no result"); })
      .catch((e) => alive && setLerr(String(e)));
    return () => { alive = false; };
  }, [cid, model, removed]);

  // the structure CATGX found with the largest verified effect
  const S = useMemo(() => {
    const ss = st?.evidence?.structures ?? [];
    let best = -1, bv = -1e9;
    ss.forEach((t: any, i: number) => { const v = (t.p_value <= 0.05 ? 1e3 : 0) + t.drop_logit; if (v > bv) { bv = v; best = i; } });
    return best >= 0 ? { s: ss[best], x: st.llm?.[best] } : null;
  }, [st]);
  const sEids: number[] = cxk?.G_star ?? [];

  // verification: TemGX links, the CATGX structure, and random links of the same size (8 draws), each re-run live
  useEffect(() => {
    if (step !== 2 || !tx || !sg || !cxk || !ctr || !rtv || ver) return;
    (async () => {
      const pool = sg.edges.map((e: any) => e.eid).filter((e: number) => !tx.E_eps.includes(e) && !sEids.includes(e));
      const draw = (k: number) => { const p: number[] = []; while (p.length < Math.min(k, pool.length)) { const e = pool[Math.floor(Math.random() * pool.length)]; if (!p.includes(e)) p.push(e); } return p; };
      const run = (rm: number[]) => post("/api/live", { case_id: cid, backbone: BK, model_id: "F0", removed: rm }).then((r) => r.p_A_over_B as number);
      const t = await run(tx.E_eps), c = await run(sEids), bS = await run(ctr.side_B.links.map((l: any) => l.eid));
      const rs: number[] = [];
      for (let i = 0; i < 8; i++) rs.push(await run(draw(tx.E_eps.length)));
      const arms = rtv?.row?.arms ?? {};
      const g1 = arms["1-hop, newest around the title"] ? await run(arms["1-hop, newest around the title"].eids) : null;
      const at = arms["attention top-n"] ? await run(arms["attention top-n"].eids) : null;
      setVer({ t, c, b: bS, g1, at, n1: arms["1-hop, newest around the title"]?.n, r: rs.reduce((a, b) => a + b, 0) / rs.length, rmin: Math.min(...rs), rmax: Math.max(...rs) });
    })();
  }, [step, tx, sg, cxk, ctr, rtv, ver]);

  if (!d) return <div className="cs"><p className="cs-load">Loading the case study… (if this stays, run <code>python -m srlab.casestudy</code> and restart the server)</p></div>;
  const a = d.alarm, ac = d.across;
  const rows = live?.rows ?? [];
  const A = rows.find((r: any) => sg && r.title === sg.A_title), B = rows.find((r: any) => sg && r.title === sg.B_title);
  const b0 = (base?.rows ?? []);
  const A0 = b0.find((r: any) => sg && r.title === sg.A_title), B0 = b0.find((r: any) => sg && r.title === sg.B_title);
  const lead = A0 && B0 ? { relevance: A0.parts.relevance - B0.parts.relevance, popularity: A0.parts.popularity - B0.parts.popularity, behaviour: A0.parts.behaviour - B0.parts.behaviour } : null;
  const toggle = (e: number) => setRemoved((r) => (r.includes(e) ? r.filter((x) => x !== e) : [...r, e]));
  const ovCx = cxk ? { motif_cards: [], G_star_occurrences: cxk.G_star_occurrences } : null;
  const occ = ([...(cxk?.G_star_occurrences ?? [])] as any[]).sort((x, y) => y.g - x.g);
  const occCount = occ.reduce((m: Record<string, number>, o: any) => ({ ...m, [o.motif]: (m[o.motif] ?? 0) + 1 }), {});
  const kbar = (step === 1 || step === 2) && <div className="cs4-k2">
    <label>How many ratings TemGX shows<span className="seg sm">{[1, 2, 3, 5, 8].map((k) => <button key={k} className={kT === k ? "on" : ""} onClick={() => setKT(k)}>{k}</button>)}</span></label>
    {step === 2 && <label>Pattern size CATGX keeps<span className="seg sm">{[2, 4, 6, 8].map((k) => <button key={k} className={kC === k ? "on" : ""} onClick={() => setKC(k)}>{k}</button>)}</span></label>}</div>;
  const ov = { tx: step === 1 || step === 2, cx: false };
  const altB: number[] = step === 1 && ctr ? ctr.side_B.links.map((l: any) => l.eid) : [];
  const F3 = d.fix.conversational_heldout.F3, F4 = d.fix.conversational_heldout.F4;
  const At = sg ? short(sg.A_title.replace(/ \(\d{4}\)$/, ""), 28) : "the wrong title", Bt = sg ? short(sg.B_title.replace(/ \(\d{4}\)$/, ""), 28) : "the right title";
  const nBad = rows.slice(0, 3).filter((r: any) => r.breaks_exclusion).length;
  const KL = [(n: number) => `${n} recent fan${n > 1 ? "s" : ""} of the film`, (n: number) => `${n} of the member's own ratings`, (n: number) => `${n} other viewer${n > 1 ? "s'" : "'s"} rating${n > 1 ? "s" : ""}`];
  const kindTxt = (k: Record<string, number>) => KINDS.map((x, i) => (k?.[x] ? KL[i](k[x]) : "")).filter(Boolean).join(", ");
  const top = lead ? (Object.entries(lead) as [string, number][]).sort((x, y) => y[1] - x[1])[0] : null;
  const recA = recRun?.rows.find((r: any) => sg && r.title === sg.A_title), recB = recRun?.rows.find((r: any) => sg && r.title === sg.B_title);
  const DRV: Record<string, string> = { relevance: "matching the words", popularity: "being popular", behaviour: "fitting this member's history" };
  const head = [
    <>Asked for “{d.query}”, this member gets <b style={{ color: RED }}>{At}</b> at #1{nBad > 1 ? <>, and {nBad} of the top 3 are what they ruled out</> : null}.</>,
    <>Why? {At} wins mainly by <b>{top ? DRV[top[0]] : "…"}</b>{ctr ? <>, and in the graph it is pushed up by <b style={{ color: RED }}>{kindTxt(ctr.side_A.kinds)}</b></> : null}.</>,
    <>Remove what we found and the model <b>changes its mind</b>. Remove anything else and it <b>doesn't</b>.</>,
    sim && recA && recB ? <>Similar past failures say <b style={{ color: GREEN }}>“{NAME[sim.recommend]}”</b>.{recB.rank === 1 ? <> Applied live: {At} drops to #{recA.rank}, <b style={{ color: GREEN }}>{Bt}</b> is #1.</> : <> For this member it is not enough ({Bt} #{recB.rank}).</>}</>
      : <>Pick the fix from similar past failures, then check it on searches it never saw.</>,
  ][step];

  return (
    <div className="cs live v4">
      <nav className="cs4-steps">{STEPS.map(([s, sub], j) => <button key={s} className={`${j === step ? "on" : ""} ${j < step ? "done" : ""}`} onClick={() => { setStep(j); if (j !== 2) setRemoved([]); if (j !== 3) setModel("F0"); }}>
        <span>{j + 1}</span><div><b>{s}</b><small>{sub}</small></div></button>)}
        <label className="hsel cs4-pick">member<select className="casesel" value={cid} onChange={(e) => setCid(e.target.value)}>
          {ac.rows.map((r: any) => <option key={r.case_id} value={r.case_id}>{r.case_id === (d.demo_member ?? d.featured.case_id) ? "★ " : ""}member {r.member}</option>)}
        </select></label></nav>
      <h2 className="cs4-h">{head}</h2>
      {kbar}
      {lerr && <p className="banner warn">The live ranking did not load ({lerr}). The server is older than this page: restart it (<code>make serve</code>) and reload.</p>}
      <div className="cs-work">
        <div className="cs-graph">
          {sg ? <SubGraph sg={sg} tx={tx} cx={ovCx} step={step} ov={ov} removed={removed} onEdge={toggle} compact alt={altB} /> : <div className="kg-wrap"><p className="cap" style={{ padding: 30 }}>Loading the graph…</p></div>}
          <div className="cs-gkey"><span><i className="u" />the member</span><span><i className="a" />{At} (wrong #1)</span><span><i className="b" />{Bt} (right answer)</span>
            {ov.tx && <span><i className="x" />{step === 1 ? `pushes ${At} up` : "ratings TemGX found"}</span>}{step === 1 && <span><i style={{ background: GREEN, borderRadius: 0, height: 3, width: 18 }} />pushes {Bt} up</span>}{ov.cx && Object.keys(occCount).map((m) => <span key={m}><i style={{ background: MCOL[m], borderRadius: 0, height: 4, width: 18 }} />CATGX: {PLAIN[m] ?? m}</span>)}
            <span className="hint2">click any rating to remove it</span></div>
        </div>
        <aside className="cs-side">
          {step === 0 && <div className="cs-card">
            <p className="cs4-nf">Netflix search now takes requests typed or spoken in a chat. Requests like this one are where rankers fail most often.</p>
            <div className="cs-q sm">“{d.query}”</div>
            <Results rows={rows} A={A} B={B} k={3} />
            <p className="cs4-say">The member ruled out depressing films; the top of the list is exactly that.</p>
            <div className="cs4-big"><b>{ac.n}<small>/{a.template_searches}</small></b><span>members who asked this got {At} first</span></div>
            <div className="cs4-big sm"><b>{pct(a.by_group.conversational.fail_rate)}</b><span>of chat-style requests start with a wrong title, vs {pct(a.by_group.original.fail_rate)} for regular searches</span></div>
            <p className="cs4-take">Not one unlucky member: the ranker has a pattern. Next, why.</p>
          </div>}
          {step === 1 && <div className="cs-card">
            {lead && <><p className="cs4-k">① What the ranker added up</p><Tug lead={lead} /></>}
            <p className="cs4-k">② What in the graph pushes each title up <span className="cs4-tag">TemGX</span><span className="cs4-tag">CATGX</span></p>
            <p className="cs4-say">TemGX picks the few ratings that move the graph model the most; CATGX names the pattern they form.</p>
            {!ctr ? <p className="cap">Explaining both titles…</p> : <>
              <div className="cs4-vs">
                <Side title={`Pushes ${At} up`} col={RED} side={ctr.side_A} ev={{ member_id: ctr.member_id, A_id: ctr.A_id, B_id: ctr.B_id }} />
                <Side title={`Pushes ${Bt} up`} col={GREEN} side={ctr.side_B} ev={{ member_id: ctr.member_id, A_id: ctr.A_id, B_id: ctr.B_id }} />
              </div>
              <p className="cs4-take">{At} gets help from <b style={{ color: RED }}>{kindTxt(ctr.side_A.kinds)}</b>; {Bt} from <b style={{ color: GREEN }}>{kindTxt(ctr.side_B.kinds)}</b>.
                {ctr.summary && <> Across all {ctr.summary.n} such searches, <b style={{ color: RED }}>{pct(share(ctr.summary.kinds.A, KINDS[0]))}</b> of what pushes the wrong title is its recent fans, vs {pct(share(ctr.summary.kinds.B, KINDS[0]))} for the right one.</>}</p>
            </>}
            {S?.x && <p className="cs4-llm"><b>In plain words</b> <small>(an LLM, from this evidence only)</small>: {S.x.mechanism.split(". ")[0]}.</p>}
            <button className="linkb" onClick={() => setMore(!more)}>{more ? "hide" : "see every pattern CATGX found"}</button>
            {more && <StructurePanel cid={cid} onDelete={setRemoved} onSelect={() => {}} />}
          </div>}
          {step === 2 && <div className="cs-card">
            <p className="cs4-k">Remove those ratings and ask the model again</p>
            <p className="cs4-say">Each bar: how sure the model is that <b style={{ color: RED }}>{At}</b> beats <b style={{ color: GREEN }}>{Bt}</b>. If we found the real reason, removing it lowers the bar; removing random ratings doesn't. Click a row to see it on the graph.</p>
            <Bar l="nothing removed" v={base?.p_A_over_B} c={POP} />
            <p className="cs4-sub">Our explanations</p>
            <Bar l={`the ${tx?.E_eps.length ?? kT} ratings TemGX found`} v={ver?.t} c={RED} strong d={base && ver ? ver.t - base.p_A_over_B : null} on={() => tx && setRemoved(tx.E_eps)} />
            <Bar l={`the pattern CATGX found (${sEids.length} ratings)`} v={ver?.c} c={GOLD} strong d={base && ver ? ver.c - base.p_A_over_B : null} on={() => setRemoved(sEids)} />
            <Bar l={`what pushes ${Bt} up (should rise)`} v={ver?.b} c={GREEN} d={base && ver ? ver.b - base.p_A_over_B : null} on={() => ctr && setRemoved(ctr.side_B.links.map((l: any) => l.eid))} />
            <p className="cs4-sub">What others would pick, same number of ratings</p>
            <Bar l={`the ${ver?.n1 ?? 3} newest ratings of ${At} (typical GraphRAG)`} v={ver?.g1} c="#7f8c99" d={base && ver?.g1 != null ? ver.g1 - base.p_A_over_B : null} on={() => rtv?.row && setRemoved(rtv.row.arms["1-hop, newest around the title"].eids)} />
            <Bar l={`the ${ver?.n1 ?? 3} the model pays most attention to`} v={ver?.at} c="#a3adb7" d={base && ver?.at != null ? ver.at - base.p_A_over_B : null} on={() => rtv?.row && setRemoved(rtv.row.arms["attention top-n"].eids)} />
            <Bar l={`${tx?.E_eps.length ?? kT} random ratings from the same weeks`} v={ver?.r} c={GREY} d={base && ver ? ver.r - base.p_A_over_B : null} band={ver ? [ver.rmin, ver.rmax] : undefined} />
            <p className="cs4-take">Only our explanations move the model. Over all {ac.n} searches: TemGX beats random ratings in <b>{ac.temgx_beats_random}/{ac.n}</b>{rtv && <>, and beats the typical GraphRAG pick in <b>{rtv.agg["1-hop, newest around the title"].temgx_moves_more}/{rtv.n_cases}</b></>}. When we planted a known cause, TemGX recovered <b>{pct(d.trust.plant_median.TemGX)}</b> of it; an LLM reading the history, {pct(d.trust.plant_median["LLM (reads the history as text)"])}.</p>
          </div>}
          {step === 3 && <div className="cs-card">
            <p className="cs4-k">Try a fix on this member, live</p>
            <div className="seg sm cs-model">{MODELS.map(([k, l]) => <button key={k} className={model === k ? "on" : ""} onClick={() => setModel(k)}>{l}</button>)}</div>
            <Results rows={rows} A={A} B={B} k={3} base={base} />
            {sim && <Similar sim={sim} rag={rag} gr={rtv?.graphrag} model={model} setModel={setModel} />}
          </div>}
          {step === 3 && <div className="cs-card">
            <p className="cs4-k">Before the A/B test: what the team would know</p>
            <div className="cs4-loop">
              <div><b>{pct(a.by_group.conversational.fail_rate)}</b><span><em>Where it fails</em>chat-style requests that start with a wrong title (regular search {pct(a.by_group.original.fail_rate)})</span></div>
              <div><b>{ac.temgx_beats_random}/{ac.n}</b><span><em>Why, checked</em>failures where the reason we found really moves the model (removing it beats removing random ratings)</span></div>
              {tg && <div><b>{pct(tg.pass.F3.T1.rate)}<small> vs {pct(tg.pass.F0.T1.rate)}</small></b><span><em>Does the fix hold on new requests</em>ruled-out title kept out of the top 3, “understand ‘not’” vs today; {tg.n_tests.toLocaleString()} such checks stay as a regression suite</span></div>}
            </div>
            <p className="cs4-fn">Any model change, by a person or a coding agent, can be scored on these checks before it reaches an A/B test. We have not run an agent on it.</p>
            <details className="cs4-more"><summary>more: every fix on every check, and a fix the graph explanation itself suggested</summary>
            {tg && <>
              <p className="cs4-k">Tests generated from the explanations · {tg.n_tests.toLocaleString()} tests on {tg.n_searches} held-out chat requests</p>
              <table className="cs4-t tg"><thead><tr><th>test (where it came from)</th>{MODELS.map(([k, l]) => <th key={k} className={model === k ? "on" : ""}>{sn(k)}</th>)}</tr></thead><tbody>
                {([["T1", "excluded title not in top 3", "text right but outvoted"], ["T2", "right title above the excluded one", "contrast: crowd vs own taste"], ["T3", "order holds without 3 strangers' ratings", "TemGX / CATGX: crowd links"]] as [string, string, string][]).map(([t, l, src]) =>
                  <tr key={t}><td><b>{t}</b> {l}<small>{src}</small></td>{MODELS.map(([k]) => { const v = tg.pass[k]?.[t]; if (!v) return <td key={k}>–</td>; const byc = (m: string) => t === "T3" && m.includes("F5"); const best = Math.max(...MODELS.filter(([m]) => !byc(m)).map(([m]) => tg.pass[m]?.[t]?.rate ?? 0));
                    return <td key={k} className={`${model === k ? "on" : ""} ${v.rate === best && k !== "F0" && !byc(k) ? "g" : ""}`} title={byc(k) ? "Ignoring strangers' latest ratings already removes these links, so it passes automatically" : undefined}>{pct(v.rate)}{byc(k) ? "*" : ""}</td>; })}</tr>)}
              </tbody></table>
              <p className="cs4-fn">* Ignoring strangers' latest ratings passes T3 automatically (it already removes those ratings), so T3 says nothing about it.</p>
              <p className="cs4-agg">Ranking quality on held-out chat requests: understanding “not” {f(F3.ndcg10.delta, 3)}, relying less on popularity {f(F4.ndcg10.delta, 3)} · plus {tg.n_training_pairs} labelled pairs (right title over ruled-out title, with the reason), not yet used for training.</p>
            </>}
            {f5 && <div className="cs4-f5">
              <p className="cs4-k">A fix the explanation itself suggested: ignore each title's {f5.choice.n} most recent ratings by other people (n chosen on development data) · held-out NDCG@10 gain</p>
              <table className="cs4-t"><thead><tr><th></th>{["chat", "regular", "new members"].map((g) => <th key={g}>{g}</th>)}</tr></thead><tbody>
                {([["F5", "Ignore strangers' latest ratings, vs current", "delta_vs_F0"], ["F4+F5", "Adding it to “less popularity”, vs “less popularity” alone", "delta_vs_F4"]] as [string, string, string][]).map(([m, l, dk]) =>
                  <tr key={m}><td><b>{l}</b></td>{["chat", "regular", "new members"].map((g) => { const d = f5.final.results[m]?.[g]?.[dk]; if (!d) return <td key={g}>–</td>;
                    const sig = d.ci95[0] > 0; return <td key={g} className={sig ? "g" : ""} title={`95% CI ${d.ci95[0].toFixed(3)} to ${d.ci95[1].toFixed(3)}`}>{d.mean >= 0 ? "+" : ""}{d.mean.toFixed(3)}{sig ? "" : " (n.s.)"}</td>; })}</tr>)}
              </tbody></table>
            </div>}
            <p className="cs4-agg">Understanding “not” passes most exclusion tests; relying less on popularity passes most intent tests and improves quality. Ignoring strangers' latest ratings, which the graph explanation itself suggested, adds a small held-out gain on regular searches and new members, not on chat requests.</p>
            </details>
          </div>}
        </aside>
      </div>
    </div>
  );
}

const KINDS = ["the title's recent raters", "the member's own ratings", "other people's ratings"];
const KCOL: Record<string, string> = { [KINDS[0]]: RED, [KINDS[1]]: HIS, [KINDS[2]]: GREY };
const KSHORT: Record<string, string> = { [KINDS[0]]: "recent fans", [KINDS[1]]: "member's own", [KINDS[2]]: "other viewers" };
const share = (k: Record<string, number>, key: string) => (k[key] ?? 0) / Math.max(1, Object.values(k).reduce((a, b) => a + b, 0));

const rateTxt = (l: any) => { const u = /^user \d+$/.test(l.from) ? l.from : l.to, t = u === l.from ? l.to : l.from; return `${l.kind === KINDS[1] ? "the member" : u} rated ${short(t, 22)}`; };

function Side({ title, col, side, ev }: any) {
  const n = side.links.length || 1;
  return <div className="cs4-side" style={{ borderColor: col }}>
    <p className="cs4-st" style={{ color: col }}>{title}</p>
    <div className="cs4-kbar">{KINDS.map((k) => side.kinds[k] ? <b key={k} style={{ flex: side.kinds[k] / n, background: KCOL[k] }}>{KSHORT[k]} {side.kinds[k]}</b> : null)}</div>
    <ul className="cs4-links">{side.links.map((l: any) => <li key={l.eid}><i style={{ background: KCOL[l.kind] }} />{rateTxt(l)} <small>{l.rating}★</small></li>)}</ul>
    {side.structure && <><p className="cs4-sn" style={{ color: MCOL[side.structure.motif] }}>Pattern: {(PLAIN[side.structure.motif] ?? side.structure.motif).toLowerCase()}</p>{side.structure.lift ? <p className="cs4-lift">the model leans on this {side.structure.lift.toFixed(1)}× more than on similar patterns</p> : null}
      <Diagram s={side.structure} ev={ev} /></>}
  </div>;
}

const MN: Record<string, string> = { POP: "recent fans", CO: "shared movie", SEQ: "run of ratings", BRIDGE: "taste path", BRIDGEB: "taste path to B", none: "–" };
const FN = NAME;

function Cite({ text }: { text: string }) {
  return <>{text.split(/(\[\d\])/).map((t, i) => /^\[\d\]$/.test(t) ? <sup key={i} className="cs4-cite">{t.slice(1, -1)}</sup> : t)}</>;
}

function Similar({ sim, rag, gr, model, setModel }: any) {
  const ev = sim.eval;
  const a = rag?.answer, bad = rag?.check?.length ?? 0;
  return <div className="cs4-sim">
    <p className="cs4-k">Have we seen this before? <span className="cs4-tag">RAG over verified explanations</span></p>
    <ol className="cs4-pipe">
      <li><b>Find</b><small>the 3 most similar of {sim.n_store} past failures we already explained and checked</small></li>
      <li><b>Collect</b><small>what fixed each of them, next to this search's evidence</small></li>
      <li><b>Write</b><small>an LLM sums it up from that evidence only{rag?.check ? <> · <span className={bad ? "n" : "y"}>{bad ? `${bad} unsupported` : "every number fact-checked ✓"}</span></> : null}</small></li>
    </ol>
    <p className="cs4-nbh"><span>similar past failure</span><em>what pushed it</em><i>fixed by</i></p>
    <ul className="cs4-nb">{sim.neighbours.slice(0, 3).map((n: any, i: number) => <li key={n.case_id}>
      <span><sup className="cs4-cite">{i + 1}</sup><b>{short(n.A, 20)}</b> over {short(n.B, 18)}<small>“{short(n.query, 34)}” · member {n.member}</small></span>
      <em style={{ color: MCOL[n.motif_A] ?? "#555" }}>{MN[n.motif_A] ?? n.motif_A}</em>
      <i>{(["F2", "F3", "F4"] as const).filter((m) => n.fixed[m]).map(sn).join(", ") || "none"}</i></li>)}</ul>
    {a ? <div className="cs4-gen">
      <p><Cite text={a.diagnosis} /></p>
      <p><Cite text={a.precedent} /></p>
      <p><b><Cite text={a.recommendation} /></b></p>
      <p className="cav"><Cite text={a.caveat} /></p>
      <details><summary>what the LLM was given · {rag.source}</summary><pre>{rag.prompt}</pre></details>
    </div> : rag && <p className="cs4-fn">No generated text for this search (pre-generated for the case-study searches; set ANTHROPIC_API_KEY to generate live). Retrieval and the recommendation below do not need the LLM.</p>}
    <div className="cs4-rec"><span>Recommended: fixed {Math.round(sim.votes[sim.recommend] * sim.rec_k)} of the {sim.rec_k} most similar past failures</span>
      <button className={`rec ${model === sim.recommend ? "on" : ""}`} onClick={() => setModel(sim.recommend)}>Try “{FN[sim.recommend] ?? sim.recommend}”</button></div>
    <p className="cs4-chk">Works for this member? {(["F2", "F3", "F4"] as const).map((m) => <span key={m} className={sim.me.fixed[m] ? "y" : "n"}>{sn(m)} {sim.me.fixed[m] ? "✓" : "✗"}</span>)}</p>
    {ev && <p className="cs4-agg">On {ev.n_new} failures it never saw, this lookup picked a fix that worked <b>{pct(ev.success.retrieval)}</b> of the time, vs {pct(ev.success["always_" + ev.always])} for one fix for everything. The LLM only writes the summary; the choice comes from the lookup.</p>}
    {gr && <p className="cs4-fn">We also tried finding similar failures by walking the graph (shared members, titles, fans): {pct(gr.chosen_graph.final)}, no better for choosing fixes.</p>}
  </div>;
}

function Results({ rows, A, B, k, base }: any) {
  const br: Record<number, number> = Object.fromEntries((base?.rows ?? []).map((r: any) => [r.index, r.rank]));
  return <><ol className="cs-res sm">{rows.slice(0, k).map((x: any) => { const isA = A && x.index === A.index, isB = B && x.index === B.index, mv = br[x.index] ? br[x.index] - x.rank : 0;
    const bad = isA || x.breaks_exclusion;
    return <li key={x.index} className={bad ? "bad" : isB ? "good" : ""}><b>#{x.rank}</b><span>{short(x.title, 30)}<small>{x.genres.replace(/\|/g, " · ")}</small></span>
      <em className={`mv ${mv > 0 ? "up" : mv < 0 ? "down" : ""}`}>{bad ? "ruled out" : isB ? "right answer" : mv > 0 ? `▲${mv}` : mv < 0 ? `▼${-mv}` : ""}</em></li>; })}</ol>
    {A && A.rank > k && <p className="cs-out">{short(A.title.replace(/ \(\d{4}\)$/, ""), 26)} → <b>#{A.rank}</b></p>}
    {B && B.rank > k && <p className="cs-out bad">right title → <b>#{B.rank}</b></p>}</>;
}

function Tug({ lead }: any) {
  const items: [string, number, string][] = [["matches the words", lead.relevance, TXT], ["popular", lead.popularity, POP], ["fits their history", lead.behaviour, HIS]];
  const neg = items.filter(([, v]) => v < 0), pos = items.filter(([, v]) => v >= 0);
  const tot = items.reduce((s, [, v]) => s + Math.abs(v), 0) || 1;
  return <div className="tug">
    <div className="tug-bar">{neg.map(([l, v]) => <b key={l} style={{ flex: Math.abs(v), background: GREEN }} title={`${l} ${f(v, 2)}`}>{l}</b>)}
      <i />{pos.map(([l, v, c]) => <b key={l} style={{ flex: v, background: c }} title={`${l} ${f(v, 2)}`}>{Math.abs(v) / tot > 0.18 ? l : ""}</b>)}</div>
    <div className="tug-lab"><span style={{ color: GREEN }}>← favours the right title</span><span style={{ color: RED }}>favours the wrong #1 →</span></div>
    <div className="tug-num">{items.map(([l, v, c]) => <span key={l}><i style={{ background: v < 0 ? GREEN : c }} />{l} {f(v, 2)}</span>)}</div>
  </div>;
}

function Bar({ l, v, c, strong, band, on, d }: any) {
  return <div className={`cs4-bar ${on ? "click" : ""}`} onClick={on}><span>{l}</span>
    <i>{band && <u style={{ left: pct(band[0]), width: `${100 * (band[1] - band[0])}%` }} />}<b style={{ width: v != null ? pct(v) : 0, background: c }} /></i>
    <em className={strong ? "st" : ""}>{v != null ? pct(v) : "…"}{d != null && <small>{d < 0 ? "−" : "+"}{n(Math.abs(100 * d), 0)} pts</small>}</em></div>;
}
