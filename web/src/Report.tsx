import { useState } from "react";
import { f, n } from "./api";
import Production from "./Production";
import Planted from "./Planted";

// Team view of ONE evaluation run: every failed search explained and grouped by cause, the agent's plan, the fixes
// tested once on held-out searches, and three focused views (new members, chat-style requests, semantic IDs).
// Every number comes from /api/evalreport (cached measurements from srlab.triage / fixes / semid).

const pct = (x: number, d = 0) => `${n(100 * x, d)}%`;
const pp = (x: number) => `${x > 0 ? "+" : ""}${n(100 * x, 1)} pts`;
const CAUSE_COL: Record<string, string> = { popularity_over_intent: "#9aa3ad", query_misread: "#c5a353", history_over_intent: "#7194b9" };
const CAUSE_PLAIN: Record<string, string> = {
  popularity_over_intent: "A popular title outranked the one that fits the search.",
  query_misread: "The text matcher itself thought the wrong title fits better — often it ignored a “not …”.",
  history_over_intent: "The person's past viewing pulled a title up that doesn't fit what they asked for.",
};
const FIX: Record<string, { id: string; name: string; target: string[] }> = {
  popularity: { id: "F4", name: "Lean less on popularity when the search names what it wants", target: ["original explicit"] },
  query: { id: "F3", name: "Read the “not …” part of the search", target: ["original with 'not'/'nothing'", "conversational"] },
  history: { id: "F2", name: "Trust viewing history less when it is short", target: ["coldstart (1-4 ratings)"] },
};
const SLICE: Record<string, string> = {
  "original explicit": "searches that name what they want",
  "original broad ('what should I watch')": "vague searches (“what should I watch”)",
  "original with 'not'/'nothing'": "searches with “not / nothing”",
  "coldstart (1-4 ratings)": "new members (1–4 ratings)",
  "conversational": "chat-style requests",
};
const EVERY: Record<string, { short: string; title: string; what: string; fix: string }> = {
  popularity_over_intent: { short: "A more popular movie won", title: "A more popular movie won",
    what: "The right movie was there, but a better-known one scored higher just for being popular.",
    fix: "giving popularity less weight when the request is specific" },
  query_misread: { short: "We misread the request", title: "We misread the request",
    what: "Our text matching thought the wrong movie fit the words better, usually because it treats “not scary” almost like “scary”.",
    fix: "reading the “not …” part of the request" },
  history_over_intent: { short: "Past viewing", title: "Their past viewing took over",
    what: "What the person watched before pulled up a movie that doesn't fit what they asked for now.",
    fix: "trusting viewing history less when it is short" },
};
const GROUP: Record<string, string> = { original: "regular searches", coldstart: "new members (1–4 ratings)", conversational: "chat-style requests" };

export default function Report({ rep, onInvestigate }: { rep: any; onInvestigate: (caseId: string, label: string) => void }) {
  const [open, setOpen] = useState<string | null>(null);
  if (!rep) return <div className="loading">Loading evaluation report…</div>;
  const T = rep.triage, FX = rep.fixes, SEM = rep.semid, CV = rep.conversational;
  const bucket = (c: string) => T.buckets.find((b: any) => b.cause === c);
  const sel = open ? bucket(open) : null;

  const verdict = (key: string) => {
    const cmp = FX?.[`${FIX[key].id}_vs_F0`];
    if (!cmp) return null;
    const hurt = Object.entries(cmp).filter(([s, e]: any) => SLICE[s] && e.ndcg10 && e.ndcg10.ci95[1] < 0).map(([s]) => SLICE[s]);
    const helped = Object.entries(cmp).filter(([s, e]: any) => SLICE[s] && e.ndcg10 && e.ndcg10.ci95[0] > 0).map(([s]) => SLICE[s]);
    return { cmp, hurt, helped };
  };

  return (
    <div className="rep">
      {/* 1. the run */}
      <section className="rep-hero">
        <div>
          <div className="algo">After an evaluation run</div>
          <h2>{T.failed.toLocaleString()} of {T.searches.toLocaleString()} searches put a wrong title first ({pct(T.fail_rate)})</h2>
          <p className="does">Instead of one aggregate number, every failed search is explained the same way and grouped by cause. Then each cause gets a targeted fix, and each fix is tested once on searches the agent has never seen.</p>
        </div>
        <div className="rep-groups">
          {Object.entries(T.by_group).map(([g, v]: any) => (
            <div key={g} className="rg"><span>{GROUP[g]}</span><i><b style={{ width: pct(v.fail_rate) }} /></i><em>{pct(v.fail_rate)}</em><small>{v.failed} / {v.searches}</small></div>))}
          <p className="cap">share of searches with a wrong title at #1 (development searches, baseline ranker)</p>
        </div>
      </section>

      {/* 2. causes */}
      <section className="cap-card">
        <div className="sec-k">1 · Why did search get it wrong?</div>
        <h3 className="take">It is rarely the person's viewing history. Half the time a <b style={{ color: "#6b737c" }}>more popular movie</b> beat the right one; the other half we <b style={{ color: "#9a7a1c" }}>misunderstood what they typed</b>, often ignoring a “not”.</h3>
        <p className="how">How we know: for every search that went wrong we asked “why did the wrong movie score higher than the right one?” The score is a sum of three things — how popular the movie is, how well its description matches the words, and how much it fits what the person watched before — so we can see which one made the difference.</p>
        <div className="stack">{T.buckets.map((b: any) => (
          <button key={b.cause} className={`stk ${open === b.cause ? "on" : ""}`} style={{ flex: b.n, background: CAUSE_COL[b.cause] }} onClick={() => setOpen(open === b.cause ? null : b.cause)}>
            {b.share > 0.08 ? `${EVERY[b.cause].short} · ${pct(b.share)}` : pct(b.share)}</button>))}</div>
        <div className="causes">{T.buckets.map((b: any) => {
          const e = b.examples[0];
          return <div key={b.cause} className={`cause ${open === b.cause ? "on" : ""}`} onClick={() => setOpen(open === b.cause ? null : b.cause)}>
            <div className="ch"><span className="dot" style={{ background: CAUSE_COL[b.cause] }} /><b>{EVERY[b.cause].title}</b><em>{pct(b.share)} of failures</em></div>
            <p className="desc">{EVERY[b.cause].what}{b.cause === "query_misread" && <> In {b.sub_ignored_exclusion} cases we showed exactly what the person said they did <i>not</i> want.</>}</p>
            {e && <p className="exq">For example: someone asked for <b>“{e.query}”</b> and got <b>{e.A_title}</b> first, while <b>{e.B_title}</b> sat at #{e.rank_B}.</p>}
            <div className="fixone"><b>{b.fix_rate >= 0.85 ? "nearly all" : b.fix_rate >= 0.6 ? "2 in 3" : b.fix_rate >= 0.45 ? "about half" : b.fix_rate >= 0.25 ? "1 in 3" : "few"}</b><span>of these are fixed by: {EVERY[b.cause].fix}</span></div>
          </div>;
        })}</div>
        <p className="cap">Click a cause to see real examples; each can be opened in the step-by-step investigation.</p>
        {sel && <div className="examples">
          <table className="tbl"><thead><tr><th>what they asked for</th><th>what we showed first</th><th>what fit better (where we ranked it)</th><th>ratings before</th><th></th></tr></thead><tbody>
            {sel.examples.map((e: any) => <tr key={e.case_id}>
              <td>“{e.query}”{e.group !== "original" && <span className="pill">{GROUP[e.group]}</span>}</td>
              <td>{e.A_title}</td><td>{e.B_title} (#{e.rank_B})</td><td>{e.n_history}</td>
              <td><button className="sm" onClick={() => onInvestigate(e.case_id, `“${e.query}” · ${e.A_title}`)}>Investigate →</button></td></tr>)}
          </tbody></table></div>}
      </section>

      {/* 3. plan + held-out test */}
      <section className="cap-card">
        <div className="sec-k">2 · What should we change?</div>
        <h3 className="take">Each fix helps the searches it targets, and costs something somewhere else, so none ships as is: route it, then A/B test.</h3>
        <p className="how">The agent fixes the biggest causes first. Each fix is tuned on development searches, then scored <b>once</b> on held-out searches it never saw. “Quality” = NDCG@10, how well the right titles are ranked (0–1, higher is better).</p>
        <div className="plan">{T.plan.map((p: any) => {
          const v = verdict(p.fix);
          const fx = FIX[p.fix];
          if (!v) return null;
          const tg = fx.target.map((s) => [s, v.cmp[s]]).find(([, e]: any) => e && e.ndcg10.ci95[0] > 0) as any;
          const worst = Object.entries(v.cmp).filter(([s, e]: any) => SLICE[s] && e.ndcg10 && e.ndcg10.ci95[1] < 0).sort((a: any, b: any) => a[1].ndcg10.delta - b[1].ndcg10.delta)[0] as any;
          return <div key={p.cause} className="pl">
            <div className="plh"><span className="rank">{p.rank}</span><div><b>{fx.name}</b><small>targets “{p.name.toLowerCase()}”</small></div></div>
            {tg && <div className="hh good"><span>Helps</span><b>{SLICE[tg[0]]}</b><em>quality {f(tg[1].ndcg10.delta, 3)}{tg[1].top1_wrong ? ` · wrong #1 ${pp(tg[1].top1_wrong.delta)}` : ""}</em></div>}
            {worst && <div className="hh bad"><span>Hurts</span><b>{SLICE[worst[0]]}</b><em>quality {f(worst[1].ndcg10.delta, 3)}</em></div>}
            <p className="next">{worst ? <>→ route it (skip {SLICE[worst[0]]}), then A/B</> : <>→ Nothing got worse: candidate for an A/B test.</>}</p>
            <details><summary>all groups, with 95% intervals</summary>
              <table className="tbl mini"><thead><tr><th>held-out group</th><th>n</th><th>quality change</th><th>wrong #1</th></tr></thead><tbody>
                {Object.entries(v.cmp).filter(([s]) => SLICE[s]).map(([s, e]: any) => {
                  const nd = e.ndcg10, t1 = e.top1_wrong;
                  const cls = nd.ci95[0] > 0 ? "good" : nd.ci95[1] < 0 ? "bad" : "";
                  return <tr key={s}><td>{SLICE[s]}</td><td>{e.n_cases}</td>
                    <td className={cls}>{f(nd.delta, 3)} <small>[{f(nd.ci95[0], 3)}, {f(nd.ci95[1], 3)}]</small></td>
                    <td>{t1 ? pp(t1.delta) : "–"}</td></tr>;
                })}</tbody></table></details>
          </div>;
        })}</div>
      </section>

      {/* 4. three focused views */}
      <div className="sec-k" style={{ marginBottom: -6 }}>3 · Three groups the team asked about</div>
      <section className="rep3">
        {(() => {
          const h0 = FX?.by_history_F0, h4 = FX?.by_history_F4;
          if (!h0) return <div className="cap-card" />;
          const nw = h0[0], old = h0[h0.length - 1];
          const mx = Math.max(...h0.map((r: any) => r.top1_wrong), ...(h4 ?? []).map((r: any) => r.top1_wrong));
          return <div className="cap-card">
            <h3>New members (cold start)</h3>
            <p className="take sm">New members: wrong #1 {pct(nw.top1_wrong)} (long histories {pct(old.top1_wrong)}) → {h4 ? pct(h4[0].top1_wrong) : "–"} with the popularity fix.</p>
            <div className="bars">{h0.map((r: any, i: number) => <div key={r.bucket} className="bar2">
              <span>{r.bucket} ratings</span>
              <div><i className="now" style={{ width: `${r.top1_wrong / mx * 100}%` }} /><i className="fix" style={{ width: `${(h4?.[i]?.top1_wrong ?? 0) / mx * 100}%` }} /></div>
              <em>{pct(r.top1_wrong)} → {h4 ? pct(h4[i].top1_wrong) : "–"}</em></div>)}</div>
            <p className="cap"><span className="lg now" /> now <span className="lg fix" /> relying less on popularity · share of held-out searches with a wrong #1</p>
          </div>;
        })()}
        <div className="cap-card">
          <h3>Chat-style requests</h3>
          <p className="take sm">Chat requests fail {n(T.by_group.conversational.fail_rate / T.by_group.original.fail_rate, 0)}× as often ({pct(T.by_group.conversational.fail_rate)} vs {pct(T.by_group.original.fail_rate)}): the “not …” gets ignored.</p>
          <div className="parsed">{CV.parsed.slice(0, 2).map((p: any) => <div key={p.query}><span>“{p.query}”</span><em>the fix reads it as — wants: <b>{p.wants}</b> · not: <b className="neg">{p.not}</b></em></div>)}</div>
          {FX?.F3_vs_F0?.conversational?.top3_violation && <div className="big good"><b>{pct(FX.F3_vs_F0.conversational.top3_violation.baseline)} → {pct(FX.F3_vs_F0.conversational.top3_violation.variant)}</b><span>how often a title the person excluded shows up in the top 3, before → after reading “not” (held-out)</span></div>}
        </div>
        <div className="cap-card">
          <h3>Semantic IDs</h3>
          <p className="take sm">Wrong #1s cluster in a few semantic-ID regions; CATGX names the viewing sequences behind them.</p>
          {SEM && <div className="lift">{SEM.where_failures.slice(0, 2).map((w: any) => <div key={w.code}><span className="sid">{w.code}</span><b>{w.name}</b><em>{n(w.lift, 1)}× more often the wrong #1 than chance</em></div>)}</div>}
          {SEM && <div className="lift">{SEM.catgx.patterns.filter((a: any) => a.ci95[0] > 0 && a.searches >= 50).slice(0, 2).map((a: any) => <div key={a.name}><span className="sid">{a.pattern[0]}→{a.pattern[1]}</span><b>watched {a.name.replace("  ->  ", ", then ")}</b><em>{a.searches} searches</em></div>)}</div>}
          <p className="cap">Codes: 3-level residual k-means over title text + genres (RQ-VAE stand-in). Patterns shown only if their 95% interval is above zero.</p>
        </div>
      </section>
      <Production ltr={rep.ltr} baselines={rep.baselines} />
      <Planted pr={rep.plant} />
    </div>
  );
}
