import { useEffect, useState } from "react";
import { NAME } from "./names";
import { n } from "./api";

// The 4-minute demo: one story, four steps. Each step = the team's pain point → what we do → one picture → one sentence.
// Everything is computed live by the same backend as the full view (kept behind "full investigation" for Q&A).
const RED = "#c82434", BLUE = "#2a78d6", GREY = "#b8c2cc", INK = "#20262d";
const BK = "tgat2", L = 2;
const pct = (x: number) => `${n(100 * x, 0)}%`;
const clean = (s: string) => s.replace(/ \(\d{4}\)$/, "").replace(/ \([^)]*\)$/, "").replace(/^(.*), (The|A|An)$/, "$2 $1");
const post = (u: string, b: any) => fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) }).then((r) => r.json());

// The clearest cases, each the best on one criterion:
const CASES: [string, string, string][] = [
  ["C11-m304-t1460134091", "member 304", "the full loop: 3 ratings flip the model (73% → 46%), two patterns survive matching, and the suggested fix puts the right answer first"],
  ["C07-m1-t1445714882", "member 1", "another request: TemGX's 7 ratings flip the model's decision (70% → 47%)"],
  ["C07-m290-t1453051417", "member 290", "the fix puts the right title first and the ruled-out one out of the top 3"],
];

const STEPS = [
  { t: "Something is wrong", sub: "the failure", pain: "An A/B test says a metric dropped. It does not say why." },
  { t: "Which ratings?", sub: "TemGX · counterfactual, size k", pain: "Hundreds of signals and millions of past interactions. Attention weights do not point to the cause." },
  { t: "Which pattern?", sub: "CATGX · causal adjustment", pain: "A pattern can look important only because it is recent or its people are active." },
  { t: "Put together", sub: "joint diagnosis", pain: "One explanation alone can mislead. The team needs a reason it can act on." },
  { t: "Decide what to test", sub: "the next A/B test", pain: "Every A/B test costs weeks. Which change deserves one?" },
];
const ANSWER = [
  "Find the searches that fail, and how often.",
  "TemGX finds a compact set of k past ratings whose removal changes the history model's preference, and checks it against random removals.",
  "CATGX compares each pattern only with patterns of the same timing and activity, then removes it to test it.",
  "Read the two together with the ranker's score split: which evidence, which mechanism, which part of the ranker.",
  "Reuse what fixed similar past failures, and check the fix on new requests before any A/B test.",
];
const PURPLE = "#7a5bb5";

export default function Demo({ onFull }: { onFull: () => void }) {
  const [d, setD] = useState<any>(null);
  const [cid, setCid] = useState("");
  const [step, setStep] = useState(0);
  const [sg, setSg] = useState<any>(null);
  const [base, setBase] = useState<any>(null);
  const [sim, setSim] = useState<any>(null);
  const [fix, setFix] = useState<any>(null);
  const [tg, setTg] = useState<any>(null);
  const [k, setK] = useState(3);
  const [kd, setKd] = useState<any>(null);
  const [pi, setPi] = useState(0);

  useEffect(() => { fetch("/api/casestudy").then((r) => r.json()).then((x) => { setD(x); setCid(CASES[0][0]); }).catch(() => {}); }, []);
  useEffect(() => { fetch("/api/testgen").then((r) => (r.ok ? r.json() : null)).then(setTg).catch(() => {}); }, []);
  useEffect(() => {
    if (!cid) return;
    setSg(null); setBase(null); setSim(null); setFix(null);
    fetch(`/api/subgraph/${cid}?backbone=${BK}&L=${L}`).then((r) => r.json()).then(setSg);
    post("/api/live", { case_id: cid, backbone: BK, model_id: "F0", removed: [] }).then((r) => r.rows && setBase(r));
    fetch(`/api/similar/${cid}`).then((r) => (r.ok ? r.json() : null)).then(setSim).catch(() => {});
  }, [cid]);
  useEffect(() => { if (!cid) return; setKd(null); setPi(0); fetch(`/api/kdemo/${cid}?k=${k}`).then((r) => (r.ok ? r.json() : null)).then(setKd).catch(() => {}); }, [cid, k]);
  useEffect(() => { if (cid && sim?.recommend) post("/api/live", { case_id: cid, backbone: BK, model_id: sim.recommend, removed: [] }).then((r) => r.rows && setFix(r)); }, [cid, sim]);

  if (!d || !sg || !base || !cid) return <div className="dm"><p className="cs-load">Loading the demo…</p></div>;
  const a = d.alarm;
  const At = clean(sg.A_title), Bt = clean(sg.B_title);
    const top = kd?.patterns?.[pi] ?? kd?.patterns?.[0];
  const maxDrop = Math.max(0.001, ...(kd?.evidence ?? []).map((l: any) => l.drop));
  const pmax = Math.max(1, ...(kd?.patterns ?? []).flatMap((x: any) => [x.raw ?? 0, x.adjusted ?? 0]));
  const recName = sim ? NAME[sim.recommend] : "";
  const go = (j: number) => setStep(j);
  const fixA = fix?.rows.find((r: any) => r.title === sg.A_title), fixB = fix?.rows.find((r: any) => r.title === sg.B_title);
  const rec = sim?.recommend ?? "F3";
  const qual = d.fix.conversational_heldout?.[rec]?.ndcg10?.delta;

  const head = [
    <>Asked for “{kd?.query ?? sim?.me?.query ?? d.query}”, member {cid.match(/-m(\d+)-/)?.[1]} gets <b style={{ color: RED }}>{At}</b> first — the kind of film they ruled out.</>,
    kd?.full.flips ? <>Remove {kd.evidence.length === 1 ? "one rating" : `these ${kd.evidence.length} ratings`} and the history model <b>changes its preference</b> ({pct(kd.p0)} → <b style={{ color: RED }}>{pct(kd.full.p)}</b>). Random ones: nothing.</>
      : kd ? <>Remove these <b style={{ color: RED }}>{kd.evidence.length} past ratings</b>: {pct(kd.p0)} → <b style={{ color: RED }}>{pct(kd.full.p)}</b>. Remove {kd.evidence.length} random ones: nothing happens.</> : <>Finding the evidence…</>,
    top ? <>“{top.name}” is <b style={{ color: PURPLE }}>{n(top.adjusted, 1)}×</b> after matching on timing and activity{top.raw ? <> (was {n(top.raw, 1)}×)</> : null}{top.p_value <= 0.05 ? <>, and removing it moves the model</> : null}.</> : <>Looking for patterns…</>,
    <>Together: <b style={{ color: RED }}>which ratings</b>, <b style={{ color: PURPLE }}>which pattern</b>, and <b>which part of the ranker</b> to change.</>,
    !fixA || !fixB ? <>Similar past failures point to one fix. Re-ranking…</>
      : fixB.rank === 1 && fixA.rank > 3 ? <>Similar past failures point to one fix. It puts <b style={{ color: BLUE }}>{Bt}</b> first and drops {At} to #{fixA.rank}.</>
      : fixB.rank === 1 ? <>The suggested fix puts <b style={{ color: BLUE }}>{Bt}</b> first, but {At} stays at #{fixA.rank}: <b>worth testing, not yet enough</b>.</>
      : fixA.rank > 1 ? <>Past cases point to “{recName}”: it takes <b style={{ color: RED }}>{At}</b> off the top (#{fixA.rank}) and improves held-out quality, but alone it is <b>not enough for this member</b>.</>
      : <>The suggested fix is <b>not enough for this member</b> ({Bt} at #{fixB.rank}). The check catches that before an A/B test.</>,
  ][step];

  return (
    <div className="dm">
      <nav className="dm-line">
        {STEPS.map((s, j) => <button key={s.t} className={`${j === step ? "on" : ""} ${j < step ? "done" : ""}`} onClick={() => go(j)}>
          <span className="dm-n">{j + 1}</span>
          <div><b>{s.t}</b><small className="dm-sub">{s.sub}</small></div>
        </button>)}
      </nav>

      <div className="dm-cases"><span>Case</span>{CASES.map(([id, l, why]) => <button key={id} className={id === cid ? "on" : ""} title={why} onClick={() => { setCid(id); setStep(0); }}>{l}</button>)}
        <small>{CASES.find((c) => c[0] === cid)?.[2]}</small></div>
      <h2 className="dm-h">{head}</h2>
      <div className="dm-ribbon"><span className="dm-pl">Pain point</span><span>{STEPS[step].pain}</span><span className="dm-arrow">→</span><span className="dm-al">What we do</span><span>{ANSWER[step]}</span></div>

      {step === 0 && <div className="dm-two">
        <div className="dm-card">
          <p className="dm-k">Top 3 for this member, current ranker</p>
          <Top rows={base.rows} A={sg.A_title} B={sg.B_title} />
        </div>
        <div className="dm-card">
          <p className="dm-k">Not one unlucky member</p>
          <Rate l="Chat-style requests" v={a.by_group.conversational.fail_rate} c={RED} />
          <Rate l="Regular searches" v={a.by_group.original.fail_rate} c={GREY} />
          <p className="dm-say">Share of searches whose first result is wrong. Requests like this one fail about three times as often.</p>
        </div>
      </div>}

      {step === 1 && <div className="dm-kbar"><span>Explanation size k</span>
        {[1, 2, 3, 5, 8].map((x) => <button key={x} className={k === x ? "on" : ""} onClick={() => setK(x)}>{x}</button>)}
        <small>how many people or titles TemGX may keep{kd ? <> · here: {kd.evidence.length} rating{kd.evidence.length > 1 ? "s" : ""}</> : null}</small></div>}

      {step === 1 && (!kd ? <p className="cap">Explaining…</p> : <>
        <div className="dm-two g">
          <div className="dm-card"><p className="dm-k">① The {kd.evidence.length} rating{kd.evidence.length > 1 ? "s" : ""} TemGX selected, and how they reach {At}</p>
            <ExplGraph g={kd.graph} /></div>
          <div className="dm-card"><p className="dm-k">② Why each rating matters to the model</p>
            <ol className="dm-why">{kd.evidence.map((l: any) => <li key={l.eid}>
              <span className="dm-badge">{l.n}</span>
              <div><b>{l.kind === "own" ? `Member ${kd.member}` : l.from} rated {l.to} {l.rating}★</b><small> · {l.date}</small>
                <p>{l.why.split(" Removing only")[0]}</p>
                <div className="dm-mini"><span>remove only this</span><i><b style={{ width: `${Math.min(100, Math.max(2, 100 * l.drop / maxDrop))}%` }} /></i><em>−{n(100 * l.drop, 1)} pts</em></div></div></li>)}</ol></div>
        </div>
        <div className="dm-two" style={{ marginTop: 16 }}>
          <div className="dm-card"><p className="dm-k">③ Counterfactual: remove them one at a time</p>
            <Curve kd={kd} />
            <p className="dm-say">Same trained history model, same candidates. Red: removing TemGX's ratings in order. Grey band: 95% of {kd.n_draw} random sets of the same size from the same weeks. Below the dashed line the history model prefers {Bt}; the full ranking changes only with a fix (step 5).</p></div>
          <div className="dm-card"><p className="dm-k">④ All {kd.evidence.length} removed vs random</p>
            <Conf l="Nothing removed" v={kd.p0} c={GREY} />
            <Conf l={`Remove TemGX's ${kd.evidence.length}`} v={kd.full.p} c={RED} d={kd.full.p - kd.p0} />
            <Conf l={`Remove ${kd.evidence.length} random`} v={kd.full.rand_mean} c={GREY} d={kd.full.rand_mean - kd.p0} />
            <Strip kd={kd} />
            <p className="dm-say"><b>p = {n(kd.full.p_value, 3)}</b>: none of the {kd.n_draw} random sets moves the model this much. {kd.across && <> Across all {kd.across.n} failed searches with this request (k = 3), TemGX beats random in <b>{kd.across.temgx_beats_random} of {kd.across.n}</b>.</>}</p></div>
        </div>
      </>)}

      {step === 2 && (!kd || !kd.patterns.length ? <p className="cap">Looking for patterns…</p> : <div className="dm-two g">
        <div className="dm-card"><p className="dm-k">The same graph, with the pattern CATGX found <span style={{ color: PURPLE }}>(purple)</span> over TemGX's ratings <span style={{ color: RED }}>(red)</span></p>
          <ExplGraph g={merge(kd.graph, kd.patterns[pi].graph)} hi={kd.patterns[pi].graph.edges.map((e: any) => e.eid)} />
          {kd.patterns[pi].overlap_temgx.length > 0 && <p className="dm-agree">Both methods point to the same rating{kd.patterns[pi].overlap_temgx.length > 1 ? "s" : ""}: TemGX found {kd.patterns[pi].overlap_temgx.length > 1 ? "them" : "it"} by counterfactual removal, CATGX by pattern analysis.</p>}</div>
        <div className="dm-card"><p className="dm-k">Is it the structure, or just timing and activity?</p>
          {kd.patterns.map((x: any, i: number) => <div key={x.motif} className={`dm-prow2 ${i === pi ? "on" : ""}`} onClick={() => setPi(i)}>
            <span>{x.name}</span>
            <i><b className="raw" style={{ width: `${100 * (x.raw ?? 0) / pmax}%` }} /><b className="adj" style={{ width: `${100 * (x.adjusted ?? 0) / pmax}%` }} /></i>
            <em>{x.raw ? `${n(x.raw, 1)}×` : "–"} → <b>{x.adjusted ? `${n(x.adjusted, 1)}×` : "–"}</b></em></div>)}
          <p className="dm-legend"><i className="raw" />before matching <i className="adj" />after matching on recency, duration, activity · 1× = an average pattern · click a row</p>
          <p className="dm-k" style={{ marginTop: 14 }}>Remove this pattern's ratings</p>
          <Conf l="Nothing removed" v={kd.p0} c={GREY} />
          <Conf l={`Remove the pattern (${kd.patterns[pi].occurrence.eids.length})`} v={kd.patterns[pi].p_without} c={PURPLE} d={kd.patterns[pi].p_without - kd.p0} />
          <Conf l="Remove random, same size" v={kd.patterns[pi].p_random} c={GREY} d={kd.patterns[pi].p_random - kd.p0} />
          <p className="dm-pwhy">{kd.patterns[pi].why}</p></div>
      </div>)}

      {step === 3 && (!kd ? <p className="cap">Combining…</p> : <div className="dm-two">
        <div className="dm-card"><p className="dm-k">Each rating TemGX found, and the pattern it belongs to</p>
          <table className="dm-jt"><thead><tr><th>#</th><th>rating</th><th>remove only this</th><th>CATGX pattern</th></tr></thead><tbody>
            {kd.evidence.map((l: any) => <tr key={l.eid}><td><span className="dm-badge">{l.n}</span></td><td>{l.kind === "own" ? `member ${kd.member}` : l.from} → {l.to} <small>{l.rating}★ · {l.date}</small></td>
              <td className="r">−{n(100 * l.drop, 1)} pts</td>
              <td>{l.patterns.filter((p: any) => (p.adjusted ?? 0) >= 1).length ? l.patterns.filter((p: any) => (p.adjusted ?? 0) >= 1).map((p: any) => <span key={p.motif} className={`dm-chip ${p.significant ? "s" : ""}`}>{p.name}{p.adjusted ? ` ${n(p.adjusted, 1)}×` : ""}</span>) : <small>none</small>}</td></tr>)}
          </tbody></table>
          <p className="dm-k" style={{ marginTop: 16 }}>What the ranker added up ({At} minus {Bt})</p>
          <Split s={kd.score_split} /></div>
        <div className="dm-card"><p className="dm-k">Reading them together</p>
          <ul className="dm-lines">{kd.joint.lines.map((x: string, i: number) => <li key={i} className={x.startsWith("TemGX") ? "t" : x.startsWith("CATGX") ? "c" : "r"}>{x}</li>)}</ul>
          <p className="dm-verdict">{kd.joint.verdict}</p></div>
      </div>)}

      {step === 4 && <div className="dm-two">
        <div className="dm-card">
          <p className="dm-k">Before and after “{recName}”</p>
          <div className="dm-ba">
            <div><small>today</small><Top rows={base.rows} A={sg.A_title} B={sg.B_title} /></div>
            <div><small>with the fix</small>{fix ? <Top rows={fix.rows} A={sg.A_title} B={sg.B_title} /> : <p className="cap">Re-ranking…</p>}</div>
          </div>
          {sim && <p className="dm-say">Chosen because it fixed {Math.round(sim.votes[sim.recommend] * sim.rec_k)} of the {sim.rec_k} most similar past failures we had already explained and checked.</p>}
        </div>
        <div className="dm-card">
          <p className="dm-k">Before the A/B test: new requests it never saw</p>
          {tg && tg.pass[rec] && <><Rate l={`“${recName}”`} v={tg.pass[rec].T1.rate} c={BLUE} /><Rate l="Current ranker" v={tg.pass.F0.T1.rate} c={GREY} />
            <p className="dm-say">Ruled-out title kept out of the top 3. {qual != null && (qual < 0 ? <>Overall ranking quality dips slightly ({n(qual, 3)} NDCG@10), so the A/B test should watch that.</> : <>Overall ranking quality also improves (+{n(qual, 3)} NDCG@10).</>)}</p></>}
          <p className="dm-take">The team gets a reason, a checked fix and its cost, before spending an A/B test.</p>
        </div>
      </div>}

      <div className="dm-foot">
        <button className="ghost" disabled={step === 0} onClick={() => go(step - 1)}>← back</button>
        <span>Public MovieLens data · live model · step {step + 1} of 5</span>
        {step < 4 ? <button className="dm-next" onClick={() => go(step + 1)}>next: {STEPS[step + 1].t} →</button> : <button className="ghost" onClick={onFull}>full investigation view</button>}
      </div>
    </div>
  );
}

const ev = (l: any) => {
  const own = /own/.test(l.kind);
  const u = /^user \d+$/.test(l.from) ? l.from : l.to, t = u === l.from ? l.to : l.from;
  return own ? <>the member rated <b>{clean(t)}</b></> : <>{/recent/.test(l.kind) ? "a recent fan" : u} rated <b>{clean(t)}</b></>;
};

function Top({ rows, A, B }: any) {
  return <ol className="dm-top">{rows.slice(0, 3).map((x: any) => {
    const bad = x.title === A || x.breaks_exclusion, good = x.title === B;
    return <li key={x.index} className={bad ? "bad" : good ? "good" : ""}><b>#{x.rank}</b><span>{clean(x.title)}<small>{x.genres.replace(/\|/g, " · ")}</small></span><em>{bad ? "ruled out" : good ? "right answer" : ""}</em></li>;
  })}</ol>;
}

function Rate({ l, v, c }: any) {
  return <div className="dm-rate"><span>{l}</span><i><b style={{ width: pct(v), background: c }} /></i><em style={{ color: c === GREY ? INK : c }}>{pct(v)}</em></div>;
}

function Conf({ l, v, c, d, on }: any) {
  return <div className={`dm-rate ${on ? "click" : ""}`} onClick={on}><span>{l}</span><i><b style={{ width: v != null ? pct(v) : 0, background: c }} /></i>
    <em style={{ color: c === GREY ? INK : c }}>{v != null ? pct(v) : "…"}{d != null && <small>{d < 0 ? "−" : "+"}{n(Math.abs(100 * d), 0)} pts</small>}</em></div>;
}

function ExplGraph({ g, hi }: any) {
  const H2 = new Set<number>(hi ?? []);
  const col = (v: any) => (v.role === "u" ? 0 : v.role ? 3 : v.type === "title" ? 1 : 2);
  const cols: any[][] = [[], [], [], []];
  g.nodes.forEach((v: any) => cols[col(v)].push(v));
  cols[3].sort((x: any) => (x.role === "A" ? -1 : 1));
  const W = 700, H = Math.max(280, 80 * Math.max(...cols.map((c) => c.length)) + 50), X = [48, 195, 370, 625];
  const pos: Record<number, [number, number]> = {};
  cols.forEach((c, i) => c.forEach((v: any, j: number) => { pos[v.id] = [X[i], (H / (c.length + 1)) * (j + 1)]; }));
  const nmap = Object.fromEntries(g.nodes.map((v: any) => [v.id, v]));
  return <svg viewBox={`0 0 ${W} ${H}`} className="dm-eg">
    {g.edges.filter((e: any) => !e.expl && !H2.has(e.eid)).map((e: any) => { const [x1, y1] = pos[e.s], [x2, y2] = pos[e.t]; return <line key={e.eid} x1={x1} y1={y1} x2={x2} y2={y2} className="ctx" />; })}
    {g.edges.filter((e: any) => H2.has(e.eid)).map((e: any) => { const [x1, y1] = pos[e.s], [x2, y2] = pos[e.t]; const lx = x1 + (x2 - x1) * 0.4, ly = y1 + (y2 - y1) * 0.4;
      return <g key={"p" + e.eid}><line x1={x1} y1={y1} x2={x2} y2={y2} className={e.expl ? "pat halo" : "pat"} />{!e.expl && <text x={lx} y={ly + 16} textAnchor="middle" className="el pl">{e.rating}★ · {e.date}</text>}</g>; })}
    {g.edges.filter((e: any) => e.expl).map((e: any) => { if (H2.size && !H2.has(e.eid)) { const [a1, b1] = pos[e.s], [a2, b2] = pos[e.t]; return <line key={e.eid} x1={a1} y1={b1} x2={a2} y2={b2} className="ex faded" />; }
      const tgt = (id: number) => nmap[id]?.role === "A" || nmap[id]?.role === "B";
      const [ax, ay] = tgt(e.s) ? pos[e.t] : pos[e.s], [bx, by] = tgt(e.s) ? pos[e.s] : pos[e.t];
      const toTitle = tgt(e.s) || tgt(e.t);
      const f1 = toTitle ? 0.13 : 0.3, f2 = 0.55;
      const cx1 = ax + (bx - ax) * f1, cy1 = ay + (by - ay) * f1;
      const atStart = toTitle && nmap[e.s]?.role !== "u" && nmap[e.t]?.role !== "u";  // label beside the person, so labels into the same title never collide
      const lx = atStart ? ax + 46 : ax + (bx - ax) * f2, ly = atStart ? ay - 12 + (by - ay) * 0.1 : ay + (by - ay) * f2 - 7;
      return <g key={e.eid}><line x1={ax} y1={ay} x2={bx} y2={by} className="ex" />
        <circle cx={cx1} cy={cy1} r={10} className="bd" /><text x={cx1} y={cy1 + 4} textAnchor="middle" className="bdt">{e.n}</text>
        <text x={lx} y={ly} textAnchor={atStart ? "start" : "middle"} className="el">{e.rating}★ · {e.date}</text></g>; })}
    {g.nodes.map((v: any) => { const [x, y] = pos[v.id], A = v.role === "A", B = v.role === "B", U = v.role === "u";
      const fill = A ? RED : B ? BLUE : U ? "#20262d" : v.type === "title" ? "#fff" : "#eef2f7";
      return <g key={v.id}>{v.type === "title" ? <rect x={x - (A || B ? 70 : 54)} y={y - 15} width={A || B ? 140 : 108} height={30} rx={8} fill={fill} className="nd" /> : <circle cx={x} cy={y} r={U ? 20 : 15} fill={fill} className="nd" />}
        <text x={x} y={y + 4} textAnchor="middle" className={`nt ${A || B || U ? "w" : ""}`}>{U ? v.name.replace("user ", "") : short(v.name, A || B ? 22 : 16)}</text>
        {U && <text x={x} y={y + 36} textAnchor="middle" className="nl">member</text>}{(A || B) && <text x={x} y={y + 30} textAnchor="middle" className="nl">{A ? "wrong #1" : "right answer"}</text>}</g>; })}
    <text x={X[0]} y={H - 6} textAnchor="middle" className="ch">member</text><text x={X[1]} y={H - 6} textAnchor="middle" className="ch">titles they rated</text>
    <text x={X[2]} y={H - 6} textAnchor="middle" className="ch">other people</text><text x={X[3]} y={H - 6} textAnchor="middle" className="ch">the two titles</text>
    <title>{nmap ? "explanation subgraph" : ""}</title>
  </svg>;
}

const short = (s: string, k: number) => (s.length > k ? s.slice(0, k - 1) + "…" : s);

function Curve({ kd }: any) {
  const pts = [{ n: 0, p: kd.p0, rand_mean: kd.p0, rand_lo: kd.p0, rand_hi: kd.p0 }, ...kd.curve];
  const W = 560, H = 310, l = 48, r = 16, t = 26, b = 40, N = pts.length - 1 || 1;
  const lo = Math.min(0.45, ...pts.map((x: any) => Math.min(x.p, x.rand_lo))) - 0.03, hi = Math.min(1, Math.max(...pts.map((x: any) => x.rand_hi)) + 0.03);
  const sx = (i: number) => l + (W - l - r) * (i / N), sy = (v: number) => t + (H - t - b) * (1 - (v - lo) / (hi - lo));
  const band = pts.map((x: any, i: number) => `${sx(i)},${sy(x.rand_hi)}`).join(" ") + " " + pts.slice().reverse().map((x: any, i: number) => `${sx(pts.length - 1 - i)},${sy(x.rand_lo)}`).join(" ");
  const ticks = [lo, (lo + hi) / 2, hi].map((v) => Math.round(v * 20) / 20);
  return <svg viewBox={`0 0 ${W} ${H}`} className="dm-curve">
    {ticks.map((v) => <g key={v}><line x1={l} x2={W - r} y1={sy(v)} y2={sy(v)} className="grid" /><text x={l - 6} y={sy(v) + 4} textAnchor="end" className="ax">{pct(v)}</text></g>)}
    {0.5 > lo && <><line x1={l} x2={W - r} y1={sy(0.5)} y2={sy(0.5)} className="half" /><text x={l + 6} y={sy(0.5) + 16} textAnchor="start" className="halft">50%: below this the history model prefers the right answer</text></>}
    <polygon points={band} className="band" />
    <polyline points={pts.map((x: any, i: number) => `${sx(i)},${sy(x.rand_mean)}`).join(" ")} className="rl" />
    <polyline points={pts.map((x: any, i: number) => `${sx(i)},${sy(x.p)}`).join(" ")} className="tl" />
    {pts.map((x: any, i: number) => <g key={i}><circle cx={sx(i)} cy={sy(x.p)} r={5} className="td" /><text x={sx(i)} y={sy(x.p) - 10} textAnchor="middle" className="tv">{pct(x.p)}</text>
      <text x={sx(i)} y={H - b + 18} textAnchor="middle" className="ax">{i}</text></g>)}
    <text x={(W + l) / 2} y={H - 4} textAnchor="middle" className="ax">ratings removed (in TemGX's order)</text>
  </svg>;
}

function Strip({ kd }: any) {
  const vs: number[] = kd.full.rand_all, lo = Math.min(kd.full.p, ...vs) - 0.02, hi = Math.max(kd.p0, ...vs) + 0.02, sx = (v: number) => 40 + 620 * (v - lo) / (hi - lo);
  const cl = (x: number) => Math.min(620, Math.max(80, x));
  return <div className="dm-strip"><svg viewBox="0 0 700 80">
    <line x1={40} x2={660} y1={44} y2={44} className="axl" />
    {vs.map((v, i) => <circle key={i} cx={sx(v)} cy={44 + ((i % 7) - 3) * 4} r={3.5} className="rd" />)}
    <line x1={sx(kd.full.p)} x2={sx(kd.full.p)} y1={22} y2={66} className="ours" /><text x={cl(sx(kd.full.p))} y={16} textAnchor="middle" className="oursT">TemGX {pct(kd.full.p)}</text>
    <line x1={sx(kd.p0)} x2={sx(kd.p0)} y1={30} y2={58} className="orig" /><text x={cl(sx(kd.p0))} y={76} textAnchor="middle" className="ax">original {pct(kd.p0)}</text></svg>
    <small>each grey dot: one random set of {kd.evidence.length} ratings from the same weeks</small></div>;
}

function merge(g: any, p: any) {
  const ids = new Set(g.nodes.map((v: any) => v.id)), es = new Set(g.edges.map((e: any) => e.eid));
  return { nodes: [...g.nodes, ...p.nodes.filter((v: any) => !ids.has(v.id))], edges: [...g.edges, ...p.edges.filter((e: any) => !es.has(e.eid)).map((e: any) => ({ ...e, expl: false }))] };
}

function Split({ s }: any) {
  const items: [string, number, string][] = [["matching the words", s.relevance, "#eda100"], ["popularity", s.popularity, "#6250d6"], ["history (graph model)", s.behaviour, "#1baf7a"]];
  const mx = Math.max(...items.map(([, v]) => Math.abs(v)), 0.01);
  return <div className="dm-split">{items.map(([l, v, c]) => <div key={l}><span>{l}</span><i><b style={{ width: `${50 * Math.abs(v) / mx}%`, left: v >= 0 ? "50%" : `${50 - 50 * Math.abs(v) / mx}%`, background: c }} /><u /></i><em>{v >= 0 ? "+" : ""}{n(v, 2)}</em></div>)}
    <small>right of centre: pushes the wrong #1 up · left: favours the right answer</small></div>;
}
