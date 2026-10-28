"""Failure triage for one evaluation run: explain EVERY failed search, group failures by cause, and check how many
each targeted fix would resolve. Run on DEV (the agent decides on DEV; FINAL is only used to test the chosen fixes).

Failed search: explicit-intent query whose top-ranked title is judged not relevant (grade 0).
A = that top title; B = the highest-ranked title with the best available grade (2, else 1).
F0 is additive, so the lead of A over B splits exactly into behaviour + relevance + popularity parts.

Cause (one per failure, rule written before looking at the results):
  query_misread        the query-title matcher itself scores A above B (relevance part of the lead > 0)
     - ignored_exclusion   ... and the query says "not/nothing/no/without/isn't" and A breaks that exclusion
  history_over_intent  relevance favours B, and the behaviour part is the largest part of A's lead
  popularity_over_intent  relevance favours B, and the popularity part is the largest part of A's lead
Cross-cutting flag: little_history (1-4 prior ratings).

Targeted fix check ("resolved" = B is no longer below A under the variant; all variants fit on DEV, so this is an
in-sample DEV estimate - the held-out test is fixes.final_eval):
  history     F2 history-length gate
  popularity  F4 query-conditioned popularity gate
  query       F3 negation-aware query relevance
Evidence for history failures: TemGX-adapted selects 2 history events (select + verify by replay); we report how often
removing them alone resolves the failure.
"""
from __future__ import annotations

import time

import numpy as np

from .common import REPORTS, dump_json, load_json, now_iso
from .explain import explain_history
from .fixes import feats, group_of, qparts, register_extra, score, violations
from .ranker import load_ranker, world

CAUSES = {
    "history_over_intent": "Viewing history outweighed the search",
    "popularity_over_intent": "Popularity outweighed the search",
    "query_misread": "The search text was misread",
}


def run(split="dev", max_examples=8) -> dict:
    t0 = time.time()
    W = world()
    register_extra(W)
    F0 = load_ranker("F0")
    V = {"history": load_ranker("F2"), "popularity": load_ranker("F4"), "query": load_ranker("F3")}
    titles = W.items.title.tolist()
    cases = [c for c in W.cases.values() if c["split"] == split and c["kind"] == "explicit"
             and not (split == "final" and c["showcase_member"])]
    rows = []
    n_expl_calls = 0
    for c in cases:
        z = feats(W, c, F0)
        sc = score(z, F0, {"kind": "F0"})
        s = sc["total"]
        g = np.asarray(c["grades"])
        order = list(np.argsort(-s, kind="stable"))
        A = order[0]
        if g[A] != 0:
            continue
        B = next((i for want in (2, 1) for i in order[1:] if g[i] == want), None)
        if B is None:
            continue
        d = {k: float(sc[k][A] - sc[k][B]) for k in ("behaviour", "relevance", "popularity")}
        neg = bool(qparts(c["query"])[2])
        viol = violations(c)
        if d["relevance"] > 0:
            cause = "query_misread"
        else:
            cause = "history_over_intent" if d["behaviour"] >= d["popularity"] else "popularity_over_intent"
        sub = "ignored_exclusion" if (cause == "query_misread" and neg and viol[A]) else None
        # targeted fix checks
        fx = {}
        for k, M in V.items():
            sv = score(z, F0, M)["total"]
            fx[k] = bool(sv[B] > sv[A])
        fx["temgx_removal"] = None
        if cause == "history_over_intent":
            ex = explain_history(W, c, int(A), int(B), budget=2, baselines=False)
            n_expl_calls += ex["n_model_inferences"]
            if ex["selected_event_ids"]:
                zz = feats_removed(W, c, F0, ex["selected_event_ids"])
                s2 = score(zz, F0, {"kind": "F0"})["total"]
                fx["temgx_removal"] = bool(s2[B] > s2[A])
            else:
                fx["temgx_removal"] = False
        rows.append({"case_id": c["case_id"], "qid": c["qid"], "query": c["query"], "group": group_of(c), "member": c["member"],
                     "n_history": c["n_history"], "little_history": c["n_history"] <= 4, "has_negation": neg,
                     "A_index": int(A), "B_index": int(B), "A_title": titles[c["candidates"][A]], "B_title": titles[c["candidates"][B]],
                     "rank_B": int(order.index(B) + 1), "lead": float(s[A] - s[B]), "parts": d, "cause": cause, "sub": sub,
                     "A_breaks_exclusion": bool(viol[A]), "fix": fx})
    n_fail = len(rows)
    buckets = []
    target = {"history_over_intent": "history", "popularity_over_intent": "popularity", "query_misread": "query"}
    for k, name in CAUSES.items():
        rs = [r for r in rows if r["cause"] == k]
        if not rs:
            continue
        fr = float(np.mean([r["fix"][target[k]] for r in rs]))
        ex = sorted(rs, key=lambda r: -r["lead"])
        buckets.append({"cause": k, "name": name, "n": len(rs), "share": len(rs) / max(n_fail, 1), "targeted_fix": target[k],
                        "fix_rate": fr, "priority": len(rs) / max(n_fail, 1) * fr,
                        "other_fix_rates": {f: float(np.mean([r["fix"][f] for r in rs])) for f in ("history", "popularity", "query")},
                        "temgx_removal_resolves": (float(np.mean([r["fix"]["temgx_removal"] for r in rs if r["fix"]["temgx_removal"] is not None]))
                                                   if any(r["fix"]["temgx_removal"] is not None for r in rs) else None),
                        "sub_ignored_exclusion": sum(1 for r in rs if r["sub"] == "ignored_exclusion"),
                        "little_history": sum(1 for r in rs if r["little_history"]),
                        "by_group": {gname: sum(1 for r in rs if r["group"] == gname) for gname in ("original", "coldstart", "conversational")},
                        "mean_parts": {p: float(np.mean([r["parts"][p] for r in rs])) for p in ("behaviour", "relevance", "popularity")},
                        "examples": [{x: r[x] for x in ("case_id", "query", "A_title", "B_title", "rank_B", "n_history", "group", "A_index", "B_index", "sub", "fix")}
                                     for r in diverse(ex, max_examples)]})
    buckets.sort(key=lambda b: -b["priority"])
    by_group = {}
    for gname in ("original", "coldstart", "conversational"):
        n_g = sum(1 for c in cases if group_of(c) == gname)
        f_g = [r for r in rows if r["group"] == gname]
        by_group[gname] = {"searches": n_g, "failed": len(f_g), "fail_rate": len(f_g) / max(n_g, 1),
                           "causes": {k: sum(1 for r in f_g if r["cause"] == k) for k in CAUSES}}
    out = {"created": now_iso(), "split": split, "model": "F0", "searches": len(cases), "failed": n_fail,
           "fail_rate": n_fail / max(len(cases), 1), "buckets": buckets, "by_group": by_group,
           "plan": [{"rank": i + 1, "cause": b["cause"], "name": b["name"], "fix": b["targeted_fix"],
                     "expected_share_resolved": b["priority"]} for i, b in enumerate(buckets)],
           "explainer_model_calls": n_expl_calls, "wall_s": round(time.time() - t0, 1),
           "definitions": __doc__, "rows": rows}
    dump_json(out, REPORTS / f"triage_{split}.json")
    return out


def diverse(rs, k):
    """Up to k examples, at most 2 per query template, strongest lead first."""
    out, per = [], {}
    for r in rs:
        if per.get(r["qid"], 0) >= 2:
            continue
        per[r["qid"]] = per.get(r["qid"], 0) + 1
        out.append(r)
        if len(out) >= k:
            break
    return out


def feats_removed(W, c, F0, removed):
    from .ranker import zfeat
    f = W.raw_features(c, exclude=frozenset(int(e) for e in removed))
    z = zfeat(f, F0["norm"])
    base = feats(W, c, F0)
    z.update({k: base[k] for k in ("rel_pos", "rel_neg", "x_hist", "has_neg")})
    return z


if __name__ == "__main__":
    r = run("dev")
    print(r["searches"], r["failed"], round(r["fail_rate"], 3), r["wall_s"], "s")
    for b in r["buckets"]:
        print(b["cause"], b["n"], round(b["share"], 3), "fix", round(b["fix_rate"], 3), "prio", round(b["priority"], 3),
              "excl", b["sub_ignored_exclusion"], "little", b["little_history"], b["by_group"], {k: round(v, 2) for k, v in b["mean_parts"].items()})
    print(r["by_group"])
