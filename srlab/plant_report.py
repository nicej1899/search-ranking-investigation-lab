"""Combine the planted-cause runs (srlab.plant) with the LLM explainer's answers into one report for the UI."""
from __future__ import annotations

import numpy as np

from .common import REPORTS, dump_json, load_json, now_iso

D = REPORTS / "plant"
ORDER = ["TemGX", "CATGX", "Best single removals", "Attention top-2", "Most recent 2", "Random 2", "LLM (reads the history as text)", "Try every pair"]


def llm_fidelity(name, r, prompts, answers):
    """Remove the ratings the LLM named and re-run the planted model: how much of the best possible drop does that give?"""
    import torch
    from .common import CKPT
    from .data import load_events, load_items
    from .explain import margins_for_sets
    from .plant import PW, plant, probes
    from .train import build_model
    events, items = load_events(), load_items()
    P = {k: r["P"][k] for k in ("T1", "T2", "Z")}
    ev_p, _ = plant(events, items, P)
    ev_p, _ = probes(ev_p, items, P)
    m = build_model(items); m.load_state_dict(torch.load(CKPT / f"plant_{name}.pt")); m.eval()
    pw = PW(ev_p, items, m)
    out = []
    for row, p, a in zip(r["rows"], prompts, answers):
        case = {"member": row["member"], "t": row["t"], "candidates": [P["Z"], row["B"]]}
        pool = p["eids"]
        m0 = margins_for_sets(pw, case, 0, 1, [()])[0]
        pairs = [(pool[i], pool[j]) for i in range(len(pool)) for j in range(i + 1, len(pool))]
        best = m0 - margins_for_sets(pw, case, 0, 1, pairs).min()
        sel = tuple(pool[x - 1] for x in a if 1 <= x <= len(pool))
        drop = m0 - margins_for_sets(pw, case, 0, 1, [sel])[0]
        out.append({"fid": float(drop / best) if best > 0 else None, "best": float(best), "m0": float(m0), "after_llm": float(m0 - drop)})
        assert abs(m0 - row["margin"]) < 1e-4, "planted model or probes differ from the saved run"
    return out


def one(name):
    r = load_json(D / f"{name}.json")
    prompts = load_json(D / f"{name}_llm_prompts.json")
    ans = load_json(D / f"{name}_llm_answers.json")
    rows = r["rows"]
    fid = llm_fidelity(name, r, prompts, ans["answers"])
    for row, p, a, fl in zip(rows, prompts, ans["answers"], fid):
        row["hits"]["LLM (reads the history as text)"] = len(set(a) & set(p["planted_lines"]))
        row["fidelity"]["LLM (reads the history as text)"] = fl["fid"]
        row["_best"], row["_after_llm"] = fl["best"], fl["after_llm"]
    T = r["temperature"]
    sig = lambda x: float(1 / (1 + np.exp(-x / T)))
    viz = []
    for row, p, a in zip(rows, prompts, ans["answers"]):
        picks = {k: [p["eids"].index(e) + 1 for e in v if e in p["eids"]] for k, v in row["selected"].items()}
        picks["LLM (reads the history as text)"] = [x for x in a if 1 <= x <= len(p["eids"])]
        after = {k: (row["margin"] - row["fidelity"][k] * row["_best"]) if row["fidelity"].get(k) is not None else None for k in picks}
        after["LLM (reads the history as text)"] = row["_after_llm"]
        viz.append({"lines": p["lines"], "Z": p["Z"], "B": p["B"], "planted_lines": p["planted_lines"], "picks": picks,
                    "p_before": sig(row["margin"]), "p_after": {k: (sig(v) if v is not None else None) for k, v in after.items()},
                    "p_best": sig(row["margin"] - row["_best"]), "fidelity": row["fidelity"],
                    "plant_is_models_reason": row["hits"]["Try every pair"] >= 1})
    eff = [row for row in rows if row["hits"]["Try every pair"] >= 1]  # the model's own best pair contains a planted rating
    def summ(rs):
        return {k: {"found_any": float(np.mean([x["hits"][k] >= 1 for x in rs])), "found_both": float(np.mean([x["hits"][k] == 2 for x in rs])),
                    "fidelity": float(np.mean([x["fidelity"][k] for x in rs if x["fidelity"].get(k) is not None]))}
                for k in ORDER} if rs else {}
    return {"pattern": name, "titles": r["P"]["titles"], "designed_by": r["P"].get("designed_by"), "rationale": r["P"].get("rationale"),
            "probes": len(rows), "sanity": r["sanity"], "calls_mean": r["calls_mean"],
            "all": summ(rows), "where_planted_is_models_reason": summ(eff), "n_effective": len(eff),
            "llm_model": ans["model"], "example": example(rows[0], prompts[0], ans["answers"][0]), "viz": viz, "temperature": T}


def example(row, p, a):
    return {"lines": p["lines"], "Z": p["Z"], "B": p["B"], "planted_lines": p["planted_lines"], "llm_lines": a,
            "temgx_lines": [p["eids"].index(e) + 1 for e in row["selected"]["TemGX"] if e in p["eids"]],
            "catgx_lines": [p["eids"].index(e) + 1 for e in row["selected"]["CATGX"] if e in p["eids"]]}


def run():
    out = {"created": now_iso(), "method": load_json(D / "semantic.json")["method"], "order": ORDER,
           "patterns": [one("semantic"), one("arbitrary")]}
    dump_json(out, REPORTS / "plant_report.json")
    return out


if __name__ == "__main__":
    r = run()
    for p in r["patterns"]:
        print(p["pattern"], p["titles"], p["sanity"]["mean_margin"], p["sanity"]["mean_margin_without_planted"], "effective", p["n_effective"])
        for k in r["order"]:
            print(f"  {k:34s} all {p['all'][k]}  eff {p['where_planted_is_models_reason'].get(k)}")
