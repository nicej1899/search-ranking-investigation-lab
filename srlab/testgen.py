"""From explanations to tests and labelled training data (closing the loop without claiming self-improvement).

python -m srlab.testgen   -> artifacts/reports/testgen.json, artifacts/reports/tests_final.jsonl,
                             artifacts/reports/training_pairs_dev.jsonl

The diagnosis of the case study found three things; each becomes a test template that can be instantiated on any
search, so new tests come for free as the catalogue and the members change:
  T1 exclusion   (text matcher was right but outvoted): a title the request excludes must not be in the top 3.
  T2 intent>crowd (contrast: the wrong winner is held up by collective signals, the right title by the member's own
                  history): for the highest-ranked excluded title X and the best-matching title Y, Y must rank above X.
  T3 robustness  (TemGX/CATGX: a few strangers' recent ratings of X decide the order): deleting X's 3 most recent
                  ratings by other people must not flip the order of X and Y.
Tests are instantiated on the HELD-OUT chat requests (FINAL split), so passing them is evidence about unseen searches.
Each ranker variant (F0 current, F3 read the "not", F4 less popularity) is scored by pass rate; the ranking uses the
same live ranker as the case-study page (srlab.live, history score from the 2-layer graph model).
Labelled training pairs (Y preferred over X, with the reason tag) are written from the DEV chat requests only, so they
can feed a training run or a coding agent without touching the held-out tests.
"""
from __future__ import annotations

import json

import numpy as np

from .common import REPORTS, dump_json, now_iso

BK = "tgat2"
MODELS = ("F0", "F3", "F4", "F5", "F4+F5")  # F5: T3 holds by construction (it already drops those links)


def _pair(c, order):
    """X = highest-ranked excluded title, Y = highest-ranked best-graded title (grade 2, else 1)."""
    g = np.asarray(c["grades"]); v = np.asarray(c.get("violates_exclusion", [0] * len(g)), bool)
    xs = [i for i in order if v[i]]
    best = 2 if (g == 2).any() else 1
    ys = [i for i in order if g[i] == best]
    return (xs[0] if xs else None), (ys[0] if ys else None)


def _crowd_links(bb, item, member, t, n=3):
    o, ts, e, r = bb.G.neighbors(item + bb.G.U, t, None)
    keep = [int(x) for x, who in zip(e[::-1], o[::-1]) if who != member][:n]
    return keep


def run(limit=None, write=True):
    from .backbones import get_backbone
    from .live import rank
    from .server import W_
    W = W_()
    bb = get_backbone(BK)
    fin = [c for c in W.cases.values() if c.get("extra") == "conversational" and c["split"] == "final"]
    dev = [c for c in W.cases.values() if c.get("extra") == "conversational" and c["split"] == "dev"]
    if limit:
        fin = fin[:limit]
    it = W.items
    tests, res = [], {m: {"T1": [], "T2": [], "T3": []} for m in MODELS}
    for c in fin:
        base = rank(W, c["case_id"], BK, "F0")
        order0 = [r["index"] for r in base["rows"]]
        X, Y = _pair(c, order0)
        crowd = _crowd_links(bb, c["candidates"][X], c["member"], c["t"]) if X is not None else []
        tid = c["case_id"]
        tests.append({"id": f"T1:{tid}", "type": "T1 exclusion", "case_id": tid, "query": c["query"], "member": c["member"], "t": c["t"],
                      "expect": "no excluded title in the top 3", "from": "diagnosis: text matcher right but outvoted"})
        if X is not None and Y is not None:
            tests.append({"id": f"T2:{tid}", "type": "T2 intent over crowd", "case_id": tid, "query": c["query"], "member": c["member"], "t": c["t"],
                          "X": it.title[c["candidates"][X]], "Y": it.title[c["candidates"][Y]], "expect": "Y ranks above X",
                          "from": "contrast: X held up by collective signals, Y by the member's own history"})
            tests.append({"id": f"T3:{tid}", "type": "T3 robust to a few strangers", "case_id": tid, "query": c["query"], "member": c["member"], "t": c["t"],
                          "X": it.title[c["candidates"][X]], "Y": it.title[c["candidates"][Y]], "removed": crowd,
                          "expect": "order of X and Y unchanged when X's 3 most recent ratings by others are deleted",
                          "from": "TemGX/CATGX: a crowd of recent raters decides the order"})
        for m in MODELS:
            r = base if m == "F0" else rank(W, c["case_id"], BK, m)
            rk = {row["index"]: row["rank"] for row in r["rows"]}
            v = np.asarray(c.get("violates_exclusion", [0] * len(c["candidates"])), bool)
            res[m]["T1"].append(not any(v[i] for i, k in rk.items() if k <= 3))
            if X is not None and Y is not None:
                res[m]["T2"].append(rk[Y] < rk[X])
                r2 = rank(W, c["case_id"], BK, m, crowd)
                rk2 = {row["index"]: row["rank"] for row in r2["rows"]}
                res[m]["T3"].append((rk[Y] < rk[X]) == (rk2[Y] < rk2[X]))
    summ = {m: {k: {"pass": int(sum(v)), "n": len(v), "rate": float(np.mean(v)) if v else None} for k, v in d.items()} for m, d in res.items()}
    # labelled training pairs from DEV (never the held-out tests)
    pairs = []
    for c in dev:
        r = rank(W, c["case_id"], BK, "F0")
        X, Y = _pair(c, [row["index"] for row in r["rows"]])
        if X is None or Y is None:
            continue
        rk = {row["index"]: row for row in r["rows"]}
        if rk[X]["rank"] < rk[Y]["rank"]:  # a failure: label the correction
            px, py = rk[X]["parts"], rk[Y]["parts"]
            why = max(("popularity", "behaviour", "relevance"), key=lambda k: px[k] - py[k])
            pairs.append({"member": c["member"], "t": c["t"], "query": c["query"], "prefer": it.title[c["candidates"][Y]],
                          "over": it.title[c["candidates"][X]], "reason": {"popularity": "collective popularity outvoted the request",
                                                                              "behaviour": "history score (crowd / taste links) outvoted the request",
                                                                              "relevance": "text matcher misread the request"}[why],
                          "case_id": c["case_id"]})
    if write:
        with open(REPORTS / "tests_final.jsonl", "w") as fh:
            for x in tests:
                fh.write(json.dumps(x) + "\n")
        with open(REPORTS / "training_pairs_dev.jsonl", "w") as fh:
            for x in pairs:
                fh.write(json.dumps(x) + "\n")
    out = {"created": now_iso(), "backbone": BK, "n_searches": len(fin), "n_tests": len(tests), "n_training_pairs": len(pairs),
           "by_type": {t: sum(1 for x in tests if x["type"].startswith(t)) for t in ("T1", "T2", "T3")},
           "pass": summ, "examples": tests[:6], "pair_examples": pairs[:4],
           "pair_reasons": {k: sum(1 for p in pairs if p["reason"] == k) for k in {p["reason"] for p in pairs}}}
    if write:
        dump_json(out, REPORTS / "testgen.json")
    return out


if __name__ == "__main__":
    o = run()
    print(o["n_searches"], o["n_tests"], o["by_type"], o["n_training_pairs"], o["pair_reasons"])
    for m, d in o["pass"].items():
        print(m, {k: (v["pass"], v["n"], round(v["rate"], 3) if v["rate"] is not None else None) for k, v in d.items()})
