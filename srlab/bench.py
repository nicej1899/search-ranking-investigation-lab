"""Measured latency (no invented numbers). Run: python -m srlab.bench
Reports workload, hardware, run counts, cold/warm, cache status and p50/p95 per stage."""
from __future__ import annotations

import time

import numpy as np

from .common import CASES, REPORTS, dump_json, hardware, load_json, now_iso


def _stats(xs):
    xs = np.asarray(xs)
    return {"n": int(len(xs)), "p50_ms": float(np.percentile(xs, 50)), "p95_ms": float(np.percentile(xs, 95)),
            "mean_ms": float(xs.mean()), "max_ms": float(xs.max())}


def run():
    t0 = time.perf_counter()
    from .ranker import world, load_ranker
    W = world()
    load_s = time.perf_counter() - t0
    from .explain import explain_history, replay
    from .motifs import ActivityBins, catgx_scores
    from .agent import Lab
    sel = load_json(CASES / "selected_cases.json")
    p = sel["primary"]
    c = W.cases[p["case_id"]]
    R = load_ranker("F0")
    # cold: first inference in this process (includes query encoding + lazy init)
    t = time.perf_counter(); replay(W, c, R); cold_inf = (time.perf_counter() - t) * 1000
    t = time.perf_counter(); explain_history(W, c, p["A_index"], p["B_index"], 2); cold_exp = (time.perf_counter() - t) * 1000
    act = ActivityBins(W)
    t = time.perf_counter(); catgx_scores(W, c, p["A_index"], p["B_index"], act=act); cold_mot = (time.perf_counter() - t) * 1000
    inf = []
    for _ in range(200):
        t = time.perf_counter(); replay(W, c, R); inf.append((time.perf_counter() - t) * 1000)
    rp = []
    for _ in range(200):
        t = time.perf_counter(); replay(W, c, R, [90708, 90710]); rp.append((time.perf_counter() - t) * 1000)
    ex = []
    for i in range(30):
        t = time.perf_counter(); explain_history(W, c, p["A_index"], p["B_index"], 2, seed=i); ex.append((time.perf_counter() - t) * 1000)
    mo = []
    for _ in range(30):
        t = time.perf_counter(); catgx_scores(W, c, p["A_index"], p["B_index"], act=act); mo.append((time.perf_counter() - t) * 1000)
    # many cases: inference over all FINAL cases (30 candidates each)
    fin = [x for x in W.cases.values() if x["split"] == "final"]
    t = time.perf_counter()
    for x in fin:
        replay(W, x, R)
    per_case = (time.perf_counter() - t) * 1000 / len(fin)
    # agent orchestration overhead: case tools only (no training / evaluation), 5 runs
    orch = []
    for i in range(5):
        lab = Lab(f"bench-{i}")
        t = time.perf_counter()
        lab.load_case(c["case_id"]); lab.score_candidates(c["case_id"], "F0")
        lab.explain_history(c["case_id"], p["A_index"], p["B_index"], 2)
        lab.replay_intervention(c["case_id"], [90708, 90710], "F0", p["A_index"], p["B_index"])
        total = (time.perf_counter() - t) * 1000
        tools = sum(r["latency_ms"] for r in lab.trace)
        orch.append({"total_ms": total, "sum_tool_ms": tools})
    out = {
        "created": now_iso(), "hardware": hardware(),
        "workload": {"case": c["case_id"], "candidates": len(c["candidates"]), "history_events_in_window": len(W.store.before(c["member"], c["t"]).eids),
                     "final_cases_for_throughput": len(fin)},
        "process_load_s": round(load_s, 2),
        "cold_first_call_ms": {"inference": cold_inf, "explanation": cold_exp, "motif": cold_mot,
                               "note": "first call in a fresh process; includes spaCy query encoding and torch warm-up"},
        "warm": {"model_inference_30_candidates": _stats(inf), "replay_with_intervention": _stats(rp),
                 "explanation_temgx_adapted_budget2": _stats(ex), "motif_catgx_adapted_single_case": _stats(mo)},
        "throughput": {"mean_ms_per_case_all_final_cases": per_case},
        "agent_orchestration_case_tools": {"runs": orch, "note": "4 case-level tools; envelope/provenance overhead = total - sum_tool_ms"},
        "cache_status": "no result caches used in these timings; query vectors are memoised after first use (warm)",
    }
    dump_json(out, REPORTS / "latency.json")
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=1)[:3000])
