import { useEffect, useState } from "react";
import { api } from "./api";
import Graph from "./Graph";
import Story from "./Story";
import Scale from "./Scale";
import Report from "./Report";
import CaseStudy from "./CaseStudy";
import Demo from "./Demo";
import NetGraph from "./NetGraph";
import SubGraph from "./SubGraph";
import { AgentTab, DiagTab, EvalTab, ExplainTab, Illustrative, MotifTab, RankingPanel, Sources } from "./Details";

const STEPS = ["The data", "The problem", "Find the cause", "Check it", "Find the pattern", "Together", "Try a fix", "Reproduce"];
const LAST = STEPS.length - 1;
type Tab = "rank" | "explain" | "motifs" | "agent" | "eval" | "diag";

export default function App() {
  const [meta, setMeta] = useState<any>(null);
  const [mode, setMode] = useState<"measured" | "illustrative">("measured");
  const [caseId, setCaseId] = useState("");
  const [cd, setCd] = useState<any>(null);
  const [base, setBase] = useState<any>(null);
  const [cur, setCur] = useState<any>(null);
  const [removed, setRemoved] = useState<number[]>([]);
  const [ex, setEx] = useState<any>(null);
  const [mo, setMo] = useState<any>(null);
  const [inv, setInv] = useState<any>(null);
  const [ev, setEv] = useState<any>(null);
  const [neg, setNeg] = useState<any>(null);
  const [sc, setSc] = useState<any>(null);
  const [step, setStep] = useState(0);
  const [qid, setQid] = useState("showcase");
  const [tpls, setTpls] = useState<any[]>([]);
  const [ulist, setUlist] = useState<any>(null);
  const [ov, setOv] = useState<any>(null);
  const [pending, setPending] = useState(false);
  const [nb, setNb] = useState<any>(null);
  const [view, setView] = useState<"graph" | "schematic">("graph");
  const [sc50, setSc50] = useState<any>(null);
  const [page, setPage] = useState<"case" | "report" | "investigate">("case");
  const [rep, setRep] = useState<any>(null);
  const [extraOpt, setExtraOpt] = useState<{ case_id: string; label: string } | null>(null);
  const [full, setFull] = useState(false);  // case page: simple 4-step demo by default, full investigation on request
  const [notes, setNotes] = useState(false);  // presentation mode by default: only conclusions, numbers and pictures
  const [kTx, setKTx] = useState(3);
  const [sweep, setSweep] = useState<Record<number, any>>({});
  const [bk, setBk] = useState("tgat2");
  const [L, setL] = useState(2);
  const [bks, setBks] = useState<any[]>([]);
  const [sg, setSg] = useState<any>(null);
  const [tx, setTx] = useState<any>(null);
  const [cx, setCx] = useState<any>(null);
  const [cxe, setCxe] = useState<any>(null);
  const [cxd, setCxd] = useState<any>(null);  // CATGX scored by do-removal replay (TemGX-style) instead of attention
  const [jt, setJt] = useState<any>(null);    // TemGX + CATGX joint, causal check
  const [sig, setSig] = useState<"do" | "att">("do");
  const [cmp, setCmp] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [drawer, setDrawer] = useState(false);
  const [tab, setTab] = useState<Tab>("rank");
  const [budget, setBudget] = useState(2);
  const [resetCheck, setResetCheck] = useState<any>(null);
  const [openCall, setOpenCall] = useState<number | null>(null);

  useEffect(() => {
    api.meta().then((m) => { setMeta(m); setCaseId(m.selected_cases.primary.case_id); }).catch((e) => setErr(String(e)));
    api.evalSummary().then(setEv).catch(() => {});
    api.investigation("inv-primary").then(setInv).catch(() => {});
    api.negation().then(setNeg).catch(() => {});
    fetch("/api/scalability").then((r) => r.json()).then(setSc).catch(() => {});
    fetch("/api/templates").then((r) => r.json()).then(setTpls).catch(() => {});
    fetch("/api/showcase").then((r) => r.json()).then(setSc50).catch(() => {});
    fetch("/api/backbones").then((r) => r.json()).then((d) => { setBks(d.backbones); if (!d.backbones.find((b: any) => b.key === "tgat2")?.available) setBk("tgat1"); }).catch(() => {});
    fetch("/api/compare").then((r) => r.ok ? r.json() : null).then(setCmp).catch(() => {});
    fetch("/api/evalreport").then((r) => r.ok ? r.json() : null).then(setRep).catch(() => {});
  }, []);
  useEffect(() => {
    setOv(null);
    fetch(`/api/overview?qid=${qid}`).then((r) => r.json()).then(setOv).catch(() => {});
  }, [qid]);
  function pick(caseIdNew: string) {
    if (caseIdNew === caseId) { go(1); return; }
    setPending(true); setCaseId(caseIdNew);
  }

  const A = cd?.A_index, B = cd?.B_index;
  const post = (url: string, body: any) => fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => r.json());
  useEffect(() => {
    if (!caseId) return;
    setSg(null); setTx(null); setCx(null); setCxe(null); setCxd(null); setJt(null); setRemoved([]);
    fetch(`/api/subgraph/${caseId}?backbone=${bk}&L=${L}`).then((r) => r.json()).then(setSg).catch((e) => setErr(String(e)));
    setStep((st) => (st >= 2 ? 1 : st));
  }, [caseId, bk, L]);
  const score = (rem: number[]) => api.score({ case_id: cd.case.case_id, model_id: "F0", removed: rem, A_index: A, B_index: B });

  useEffect(() => {
    if (!caseId) return;
    (async () => {
      setBusy(true); setEx(null); setMo(null); setRemoved([]); setResetCheck(null);
      const c = await api.case(caseId);
      fetch(`/api/neighborhood/${caseId}`).then((r) => r.json()).then(setNb).catch(() => {});
      const s = await api.score({ case_id: caseId, model_id: "F0", removed: [], A_index: c.A_index, B_index: c.B_index });
      setCd(c); setBase(s); setCur(s); setBusy(false);
      setStep((st) => (pending || st > 0 ? 1 : 0)); setPending(false);
    })().catch((e) => setErr(String(e)));
  }, [caseId]);

  useEffect(() => { setSweep({}); }, [caseId, bk, L]);
  useEffect(() => {
    if (!cd || step < 2 || !tx || tx.k_requested === kTx) return;
    let alive = true;
    setBusy(true);
    post("/api/temgx", { case_id: cd.case.case_id, backbone: bk, L, k: kTx }).then((t) => {
      if (!alive) return;
      t.k_requested = kTx; setTx(t); setJt(null); if (step === 3) setRemoved(t.E_eps);
    }).finally(() => alive && setBusy(false));
    return () => { alive = false; };
  }, [kTx]);
  useEffect(() => {  // background sweep over k so the panel can show how the explanation grows
    if (!cd || step < 2 || step > 3) return;
    let alive = true;
    (async () => {
      for (const k of [1, 2, 3, 5, 8]) {
        if (!alive || sweep[k]) continue;
        const t = await post("/api/temgx", { case_id: cd.case.case_id, backbone: bk, L, k }).catch(() => null);
        if (alive && t) setSweep((sw) => ({ ...sw, [k]: { prob_without: t.prob_without, prob_full: t.prob_full, n: t.E_eps.length, flipped: t.flipped } }));
      }
    })();
    return () => { alive = false; };
  }, [step, cd, bk, L]);

  async function go(target: number) {
    if (!cd || busy) return;
    setBusy(true);
    try {
      let t1 = tx, c1 = cx;
      const body = { case_id: cd.case.case_id, backbone: bk, L, k: 3 };
      if (target >= 2 && !t1) { t1 = await post("/api/temgx", { ...body, k: kTx }); t1.k_requested = kTx; setTx(t1); }
      if (target >= 4 && !c1) {
        c1 = await post("/api/catgx", body); setCx(c1);
        post("/api/catgx", { ...body, exact: true }).then(setCxe).catch(() => {});
      }
      if (target >= 4 && !cxd) setCxd(await post("/api/catgx_do", body));
      if (target >= 5 && (!jt || jt.k !== kTx)) { const j = await post("/api/joint", { ...body, k: kTx }); j.k = kTx; setJt(j); }
      setRemoved(target === 3 && t1 ? t1.E_eps : []);
      if (target === 0) { setCur(base); }
      if (target >= LAST) {
        const s = await score([]);
        const d = Math.max(...s.components.total.map((v: number, i: number) => Math.abs(v - base.components.total[i])));
        setResetCheck({ maxdiff: d, snapshot_equal: s.snapshot_hash === base.snapshot_hash, at: s.computed_at });
        setCur(s);
      }
      setStep(target);
    } catch (e) { setErr(String(e)); }
    setBusy(false);
  }
  async function rerunExplain() {
    const e = await api.explain({ case_id: cd.case.case_id, A_index: A, B_index: B, budget });
    setEx(e); setStep(2); setRemoved([]); setCur(base);
  }

  if (err) return <div className="err">Backend error: {err}. Is the API running on :8000?</div>;
  if (!meta) return <div className="loading">Loading…</div>;
  const cases = [meta.selected_cases.primary, ...meta.selected_cases.backups];

  return (
    <div className={`app ${notes ? "notes-on" : ""}`}>
      <header>
        <div className="brand"><h1>Demo</h1>
          <span className="sub">Why did search rank the wrong movie first — and what should we change? · real MovieLens data, live model</span></div>
        <div className="hdr-right">
          <div className="seg pagesel">
            <button className={page === "case" ? "on" : ""} onClick={() => setPage("case")}>▶ Case study</button>
            <button className={page === "report" ? "on" : ""} onClick={() => setPage("report")}>① Eval report</button>
            <button className={page === "investigate" ? "on" : ""} onClick={() => setPage("investigate")}>② Investigate</button>
          </div>
          {page === "investigate" && <label className="hsel">person &amp; search to investigate
            <select value={caseId} onChange={(e) => pick(e.target.value)} className="casesel wide">
              {extraOpt && !(sc50?.cases ?? cases).some((u: any) => u.case_id === extraOpt.case_id) && <option value={extraOpt.case_id}>from report · {extraOpt.label}</option>}
              {(sc50?.cases ?? cases).map((u: any, i: number) => (
                <option key={u.case_id} value={u.case_id}>{i === 0 ? "★ " : ""}u{u.member} · “{u.query}”</option>))}
            </select></label>}
          <div className="seg">
            <button className={mode === "measured" ? "on" : ""} onClick={() => setMode("measured")}>Measured</button>
            <button className={mode === "illustrative" ? "on" : ""} onClick={() => setMode("illustrative")}>Illustrative</button>
          </div>
          <button className={`ghost ${notes ? "on" : ""}`} onClick={() => setNotes(!notes)} title="show or hide explanatory small text">{notes ? "Hide notes" : "Notes"}</button>
          <button className="ghost" onClick={() => setDrawer(true)}>Sources</button>
        </div>
      </header>

      {page === "case" && mode !== "illustrative" ? (full ? <><button className="ghost dm-back" onClick={() => setFull(false)}>← back to the 4-step demo</button><CaseStudy /></> : <Demo onFull={() => setFull(true)} />) : page === "report" && mode !== "illustrative" ? <Report rep={rep} onInvestigate={(id, label) => { setExtraOpt({ case_id: id, label }); setPage("investigate"); window.scrollTo(0, 0); pick(id); }} />
        : mode === "illustrative" ? <Illustrative /> : cd && cur && (
        <>
          <nav className="stepper">
            {STEPS.map((s, i) => (
              <button key={s} className={`st ${i === step ? "on" : ""} ${i < step ? "done" : ""}`} disabled={busy} onClick={() => go(i)}>
                <span>{i}</span>{s}
              </button>
            ))}
            <div className="spacer" />
            {step > 0 && <>
              <label className="hsel">graph model
                <select className="casesel" value={bk} onChange={(e) => setBk(e.target.value)} disabled={busy}>
                  {bks.map((b) => <option key={b.key} value={b.key} disabled={!b.available}>{b.label}{b.available ? "" : " (not trained)"}</option>)}
                </select></label>
              <label className="hsel">look how far
                <div className="seg sm">{[1, 2, 3].map((h) => <button key={h} className={L === h ? "on" : ""} disabled={busy} onClick={() => setL(h)}>{h} hop{h > 1 ? "s" : ""}</button>)}</div></label>
            </>}
            {step >= 2 && step <= 3 && <label className="hsel">TemGX keeps k links
              <div className="seg sm">{[1, 2, 3, 5, 8].map((k) => <button key={k} className={kTx === k ? "on" : ""} disabled={busy} onClick={() => setKTx(k)}>{k}</button>)}</div></label>}
            {step === 4 && <label className="hsel">CATGX signal
              <div className="seg sm"><button className={sig === "do" ? "on" : ""} onClick={() => setSig("do")}>replay</button><button className={sig === "att" ? "on" : ""} onClick={() => setSig("att")}>attention</button></div></label>}
            {busy && <span className="busy">running…</span>}
            {step < LAST && <button className="primary" disabled={busy} onClick={() => go(step + 1)}>Next →</button>}
          </nav>

          <main className="stage">
            {step === 0 ? <NetGraph mode="overview" ov={ov} selected={cd?.case.member} onPick={(u: any) => pick(u.case_id)} />
              : A === undefined || A === null ? <div className="kg-wrap"><p className="cap" style={{ padding: 30 }}>No ranking problem found for this person and search. Pick another one.</p></div>
              : <SubGraph sg={sg} tx={tx} cx={sig === "do" && cxd ? cxd : cx} jt={jt} step={step} removed={removed} />}
            <Story kTx={kTx} sweep={sweep} rep={rep} parts={base && A != null && B != null ? Object.fromEntries(["behaviour", "relevance", "popularity"].map((k) => [k, base.components[k][A] - base.components[k][B]])) : null} ov={ov} showcase={sc50} step={step} sg={sg} tx={tx} cx={sig === "do" && cxd ? cxd : cx} sig={sig} jt={jt} cxe={sig === "do" ? null : cxe} cmp={cmp} ev={ev} inv={inv} bk={bk} L={L} bkInfo={bks.find((b) => b.key === bk)} />
          </main>

          <Scale sc={sc} cmp={cmp} />

          <details className="more">
            <summary>Details for questions (full tables, agent log, test results, diagnostics)</summary>
            <nav className="tabs">
              {([["rank", "Full ranking"], ["explain", "Explanation"], ["motifs", "Motifs"], ["agent", "Agent trace"], ["eval", "Held-out evaluation"], ["diag", "Diagnostics"]] as [Tab, string][]).map(([k, l]) => (
                <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>))}
            </nav>
            {tab === "rank" && <div className="tabbody"><RankingPanel cd={cd} cur={cur} base={base} A={A} B={B} removed={removed} /></div>}
            {tab === "explain" && <ExplainTab cd={cd} ex={ex} budget={budget} setBudget={setBudget} doExplain={rerunExplain} doReplay={() => go(3)} doReset={() => go(LAST)} resetCheck={resetCheck} removed={removed} cur={cur} base={base} />}
            {tab === "motifs" && <MotifTab cd={cd} mo={mo} doMotifs={() => go(4)} />}
            {tab === "agent" && <AgentTab inv={inv} setInv={setInv} openCall={openCall} setOpenCall={setOpenCall} meta={meta} />}
            {tab === "eval" && <EvalTab ev={ev} meta={meta} />}
            {tab === "diag" && <DiagTab ev={ev} neg={neg} meta={meta} />}
          </details>
        </>
      )}
      {drawer && <Sources meta={meta} close={() => setDrawer(false)} />}
      <footer>Offline prototype on public data · no Netflix data · no production or A/B claims</footer>
    </div>
  );
}
