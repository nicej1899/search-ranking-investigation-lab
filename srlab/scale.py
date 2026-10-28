"""Scalability study (measured). Run: python -m srlab.scale

A. Explainer vs exhaustive search. On members whose model window is full (50 events), restrict the
   candidate pool to the newest w events, w in {10,20,30,40,50}. For budget k=3 compare:
     - TemGX-adapted: model calls, wall time, margin reduction achieved
     - exhaustive best subset of size <= k (all sum_j C(w,j) removal sets, batched exact replays): calls, wall time, optimum
   Quality = TemGX reduction / exhaustive optimum (same objective: behavioural margin reduction).
B. Batch audit: TemGX-adapted (k=2) + CATGX-adapted motif scoring over every comparable DEV failure case.
Paper-reported numbers are kept separate and labelled as such.
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np

from .common import REPORTS, dump_json, hardware, now_iso
from .explain import explain_history, margins_for_sets
from .motifs import ActivityBins, catgx_scores, failure_pairs
from .ranker import load_ranker, world


def study_a(W, n_cases=6, k=3, widths=(10, 20, 30, 40, 50)):
    R = load_ranker("F0")
    pairs = [p for p in failure_pairs(W, R, "dev") if p[0]["n_history"] >= 50]
    seen, chosen = set(), []
    for c, A, B in pairs:  # one case per member, deterministic order
        if c["member"] not in seen:
            seen.add(c["member"]); chosen.append((c, A, B))
        if len(chosen) == n_cases:
            break
    rows = []
    for c, A, B in chosen:
        hist = W.store.before(c["member"], c["t"], n=50)
        for w in widths:
            pool = hist.eids[-w:].tolist()
            t0 = time.perf_counter()
            ex = explain_history(W, c, A, B, budget=k, pool_size=w, baselines=False)
            t_ex = (time.perf_counter() - t0) * 1000
            red_ex = ex["beh_margin_before"] - ex["beh_margin_after"]
            sets = [x for kk in range(1, k + 1) for x in itertools.combinations(pool, kk)]  # all subsets of size <= k
            t0 = time.perf_counter()
            m = margins_for_sets(W, c, A, B, sets)
            t_bf = (time.perf_counter() - t0) * 1000
            red_opt = float(ex["beh_margin_before"] - m.min())
            rows.append({"case_id": c["case_id"], "w": w, "k": k,
                         "temgx_calls": ex["n_model_inferences"], "temgx_ms": t_ex, "temgx_reduction": red_ex,
                         "exhaustive_calls": len(sets), "exhaustive_ms": t_bf, "optimal_reduction": red_opt,
                         "quality": (red_ex / red_opt) if red_opt > 1e-9 else None})
            print(rows[-1], flush=True)
    summ = []
    for w in widths:
        r = [x for x in rows if x["w"] == w]
        q = [x["quality"] for x in r if x["quality"] is not None]
        summ.append({"w": w, "temgx_calls": float(np.mean([x["temgx_calls"] for x in r])), "temgx_ms": float(np.median([x["temgx_ms"] for x in r])),
                     "exhaustive_calls": r[0]["exhaustive_calls"], "exhaustive_ms": float(np.median([x["exhaustive_ms"] for x in r])),
                     "quality_mean": float(np.mean(q)) if q else None, "quality_min": float(np.min(q)) if q else None, "n_cases": len(r)})
    # exact arithmetic (not measured): exhaustive call counts at larger budgets
    counts = {f"C({w},{kk})": math.comb(w, kk) for w in (50, 200, 1000) for kk in (3, 5)}
    return {"rows": rows, "summary": summ, "exhaustive_call_counts_exact": counts,
            "note": "Exhaustive search evaluates every k-subset with batched exact replays (512 per forward pass); TemGX-adapted runs sequential single replays. "
                    "Quality = achieved behavioural-margin reduction / exhaustive optimum on the same pool."}


def study_b(W):
    R = load_ranker("F0")
    act = ActivityBins(W)
    pairs = failure_pairs(W, R, "dev")
    ex_ms, mo_ms, occ = [], [], []
    t_all = time.perf_counter()
    for c, A, B in pairs:
        t0 = time.perf_counter(); explain_history(W, c, A, B, budget=2, baselines=False); ex_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); r = catgx_scores(W, c, A, B, act=act); mo_ms.append((time.perf_counter() - t0) * 1000)
        occ.append(r["n_occurrences"])
    wall = time.perf_counter() - t_all
    occ, mo_ms = np.array(occ), np.array(mo_ms)
    bins = [(0, 10), (10, 100), (100, 500), (500, 10 ** 9)]
    by = [{"occurrences": f"{lo}-{hi if hi < 10**9 else '+'}", "n_cases": int(((occ >= lo) & (occ < hi)).sum()),
           "median_ms": float(np.median(mo_ms[(occ >= lo) & (occ < hi)])) if ((occ >= lo) & (occ < hi)).any() else None} for lo, hi in bins]
    return {"n_cases": len(pairs), "wall_s": wall,
            "explain_ms": {"p50": float(np.median(ex_ms)), "p95": float(np.percentile(ex_ms, 95))},
            "motif_ms": {"p50": float(np.median(mo_ms)), "p95": float(np.percentile(mo_ms, 95))},
            "occurrences": {"total": int(occ.sum()), "max_per_case": int(occ.max())}, "motif_by_occurrences": by,
            "scatter": [{"occ": int(o), "ms": float(m)} for o, m in zip(occ, mo_ms)]}


PAPER_REPORTED = [
    {"source": "TemGX, ICLR 2026, Table 1", "claim": "UCIM / TGN: 8.2 s per explanation vs 35.0 s (CoDy), 68.5 s (TGNNExplainer), 88.4 s (TempME)"},
    {"source": "CATGX, KDD 2026, Table 4 (synthetic graphs)", "claim": "48.2 s at one million edges"},
]


def run(only_a: bool = False):
    W = world()
    # consistency check: batched replay == single replay
    from .explain import _beh_margin
    from .common import CASES, load_json
    p = load_json(CASES / "selected_cases.json")["primary"]
    c = W.cases[p["case_id"]]
    m_single = _beh_margin(W, c, p["A_index"], p["B_index"], (90708, 90710))
    m_batch = margins_for_sets(W, c, p["A_index"], p["B_index"], [(), (90708, 90710), (90709,)])[1]
    assert abs(m_single - m_batch) < 1e-3, (m_single, m_batch)
    prev = load_json(REPORTS / "scalability.json") if only_a and (REPORTS / "scalability.json").exists() else None
    out = {"created": now_iso(), "hardware": hardware(), "A_explainer_vs_exhaustive": study_a(W),
           "B_batch_audit": prev["B_batch_audit"] if prev else study_b(W), "batched_vs_single_replay_abs_diff": abs(m_single - m_batch),
           "paper_reported_not_measured_here": PAPER_REPORTED}
    dump_json(out, REPORTS / "scalability.json")
    return out


if __name__ == "__main__":
    import sys
    o = run(only_a="--only-a" in sys.argv)
    for s in o["A_explainer_vs_exhaustive"]["summary"]:
        print(s)
    print({k: v for k, v in o["B_batch_audit"].items() if k != "scatter"})
