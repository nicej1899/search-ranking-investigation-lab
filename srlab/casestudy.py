"""A realistic launch-review scenario, measured end to end: "one title keeps winning chat requests it shouldn't".

python -m srlab.casestudy  -> artifacts/reports/casestudy.json

Situation: before launching chat-style search, the evaluation run shows chat requests fail far more often. The most
frequent wrong #1 is Requiem for a Dream, mostly for "a drama, but please nothing depressing". We follow the team's
questions: how big is it, why does the ranker do it, what inside the graph model drives it (TemGX / CATGX on TGAT-1L,
the backbone whose logit IS the ranker's behaviour score), can we trust that explanation, and which fix to ship.
Every number is read from the cached reports or re-measured here; nothing is typed by hand.
"""
from __future__ import annotations

import collections

import numpy as np

from .common import REPORTS, dump_json, load_json, now_iso

TITLE = "Requiem for a Dream (2000)"
QID = "C07"
BK = "tgat2"  # the 2-layer graph model (sees people who rated the same titles); the ranker's history score uses the 1-layer one


def _top(W, c, F0, model, k=5):
    from .fixes import feats, score, violations
    s = score(feats(W, c, F0), F0, model)["total"]
    order = np.argsort(-s, kind="stable")
    v, g = violations(c), np.asarray(c["grades"])
    it = W.items
    return [{"rank": int(r + 1), "index": int(i), "title": it.title[c["candidates"][i]], "genres": it.genres[c["candidates"][i]],
             "grade": int(g[i]), "breaks_exclusion": bool(v[i])} for r, i in enumerate(order[:k])], order


def run():
    from .backbones import get_backbone
    from .ranker import load_ranker
    from .server import W_, _case_ab, _node_meta
    from . import catgx, temgx, joint
    W = W_()
    F0, F3, F4 = load_ranker("F0"), load_ranker("F3"), load_ranker("F4")
    tri = load_json(REPORTS / "triage_dev.json")
    fx = load_json(REPORTS / "fixes_final.json")
    rows = tri["rows"]
    chat = [r for r in rows if r["group"] == "conversational"]
    board = collections.Counter(r["A_title"] for r in chat).most_common(6)
    tmpl_cases = [c for c in W.cases.values() if c.get("qid") == QID and c["split"] == "dev"]
    tmpl_fail = [r for r in rows if r["qid"] == QID]
    req = [r for r in tmpl_fail if r["A_title"] == TITLE]
    bb = get_backbone(BK)
    it = W.items
    drivers, kinds, motifs = collections.Counter(), collections.Counter(), collections.Counter()
    from .fixes import feats, score
    per = []
    U = bb.G.U
    for r in req:
        c, a, b, _ = _case_ab(r["case_id"])
        tx = temgx.explain(bb, c["member"], c["t"], a, b, L=2, k=3)
        A = a + U
        names = []
        for v in tx["explanatory"]:
            if v < U:
                ratedA = any({e["a"], e["b"]} == {v, A} for e in tx["edges"])
                kinds["recent rater of this title" if ratedA else "other person"] += 1
                names.append(f"user {v}")
            else:
                mine = any({e["a"], e["b"]} == {v, c["member"]} for e in tx["edges"])
                kinds["the member's own rating" if mine else "other title"] += 1
                names.append(it.title[v - U])
        drivers.update(names)
        cxr = catgx.explain(bb, c["member"], c["t"], a, b, L=2, g_mode="do")
        motifs[(cxr["selected_motifs"] or ["none"])[0]] += 1
        zz = feats(W, c, F0)
        rk = {}
        for nm_, M in (("F0", F0), ("F3", F3), ("F4", F4)):
            o = np.argsort(-score(zz, F0, M)["total"], kind="stable")
            rk[nm_] = int(np.where(o == r["A_index"])[0][0] + 1)
        per.append({"rank_A": rk, "top_motif": (cxr["selected_motifs"] or [None])[0], "explanatory": tx["explanatory"], "case_id": r["case_id"], "member": r["member"], "parts": r["parts"], "cause": r["cause"], "fix": r["fix"],
                    "rank_B": r["rank_B"], "B_title": r["B_title"], "p_full": tx["prob_full"], "p_without": tx["prob_without"],
                    "drop": tx["baselines"]["temgx"], "random": tx["baselines"]["random_mean"], "flipped": tx["flipped"],
                    "drivers": names})
    # featured member: the most common cause (popularity), the model clearly prefers the title (p >= .8), and the largest verified TemGX effect among those
    # whose explanation is the title's own recent raters (the pattern found across all members)
    def crowd(p):
        return sum(1 for v in p["explanatory"] if v < U)
    pool = [p for p in per if p["cause"] == "popularity_over_intent" and p["p_full"] >= 0.8 and crowd(p) >= 2] or per
    feat = max(pool, key=lambda p: (p["p_full"] - p["p_without"]))
    c, a, b, ab = _case_ab(feat["case_id"])
    before, _ = _top(W, c, F0, F0)
    after4, o4 = _top(W, c, F0, F4)
    after3, o3 = _top(W, c, F0, F3)
    rankA = {"F0": 1, "F4": int(np.where(o4 == ab["A_index"])[0][0] + 1), "F3": int(np.where(o3 == ab["A_index"])[0][0] + 1)}
    tx = temgx.explain(bb, c["member"], c["t"], a, b, L=2, k=3)
    sweep = {k: temgx.explain(bb, c["member"], c["t"], a, b, L=2, k=k) for k in (1, 2, 5)}
    sweep[3] = tx
    cxd = catgx.explain(bb, c["member"], c["t"], a, b, L=2, g_mode="do")
    cx = catgx.explain(bb, c["member"], c["t"], a, b, L=2)
    jt = joint.run(bb, c["member"], c["t"], a, b, tx, cx, L=2, cx_do=cxd)
    hist = []
    for e in tx["edges"]:
        if e["a"] == c["member"] or e["b"] == c["member"]:
            m = e["b"] if e["a"] == c["member"] else e["a"]
            if m >= U:
                hist.append({"eid": e["eid"], "title": it.title[m - U], "genres": it.genres[m - U], "ts": e["ts"], "rating": e["rating"],
                             "in_explanation": e["eid"] in tx["E_eps"], "signed": e["signed"]})
    hist.sort(key=lambda h: h["ts"])
    # small picture: the member's own ratings, the title's and the better match's recent raters, the explanation edges
    tg = tx["targets"]
    Eeps = set(tx["E_eps"])
    near = {tg["u"], tg["A"], tg["B"]} | set(tx["explanatory"]) | {e["b"] if e["a"] == tg["u"] else e["a"] for e in tx["edges"] if tg["u"] in (e["a"], e["b"])}
    keep = [e for e in tx["edges"] if e["eid"] in Eeps or tg["u"] in (e["a"], e["b"]) or tg["A"] in (e["a"], e["b"]) or tg["B"] in (e["a"], e["b"])
            or (e["a"] in near and e["b"] in near)]
    graph = {"edges": [{"eid": e["eid"], "s": e["a"], "t": e["b"], "ts": e["ts"], "rating": e["rating"], "x": e["eid"] in Eeps, "signed": e["signed"]} for e in keep]}
    ids = {v for e in keep for v in (e["a"], e["b"])} | set(tx["explanatory"]) | set(tx["connection"]) | {tx["targets"]["u"], tx["targets"]["A"], tx["targets"]["B"]}
    for cd in cxd["motif_cards"]:
        ids |= set(cd["example"]["nodes"])
    meta = {str(k): v for k, v in _node_meta(bb.G, ids).items()}
    conv = {m: {k: fx[f"{m}_vs_F0"]["conversational"][k] for k in ("ndcg10", "top1_wrong", "top3_violation")} for m in ("F2", "F3", "F4")}
    plant = load_json(REPORTS / "plant_report.json")
    med = {}
    for k in ("TemGX", "LLM (reads the history as text)"):
        v = sorted(x["fidelity"][k] for p in plant["patterns"] for x in p["viz"] if x["plant_is_models_reason"] and x["fidelity"].get(k) is not None)
        med[k] = v[len(v) // 2]
    jsum = load_json(REPORTS / f"joint_{BK}_L2.json")
    out = {
        "created": now_iso(), "backbone": BK, "title": TITLE, "qid": QID, "query": c["query"],
        "alarm": {"by_group": tri["by_group"], "leaderboard": board, "template_searches": len(tmpl_cases),
                  "template_failed": len(tmpl_fail), "template_title_first": len(req), "title_first_all_groups": sum(r["A_title"] == TITLE for r in rows)},
        "across": {"n": len(per), "causes": collections.Counter(p["cause"] for p in per),
                   "mean_parts": {k: float(np.mean([p["parts"][k] for p in per])) for k in ("behaviour", "relevance", "popularity")},
                   "fix_resolves": {k: int(sum(p["fix"][k] for p in per)) for k in ("history", "popularity", "query")},
                   "drivers": drivers.most_common(8), "driver_kinds": kinds, "top_motif": motifs,
                   "title_first": {m: int(sum(p["rank_A"][m] == 1 for p in per)) for m in ("F0", "F3", "F4")},
                   "title_top3": {m: int(sum(p["rank_A"][m] <= 3 for p in per)) for m in ("F0", "F3", "F4")}, "mean_p_full": float(np.mean([p["p_full"] for p in per])),
                   "mean_p_without": float(np.mean([p["p_without"] for p in per])),
                   "temgx_beats_random": int(sum(p["drop"] > p["random"] for p in per)), "rows": per},
        "featured": {"case_id": feat["case_id"], "member": c["member"], "t": c["t"], "n_history": c["n_history"], "A": it.title[a], "B": it.title[b],
                     "A_genres": it.genres[a], "B_genres": it.genres[b], "parts": feat["parts"], "cause": feat["cause"],
                     "before": before, "after_F4": after4, "after_F3": after3, "rank_A": rankA,
                     "history": hist, "temgx": {k: tx[k] for k in ("explanatory", "connection", "E_eps", "prob_full", "prob_without", "prob_only", "flipped", "baselines", "n_model_inferences", "explanatory_hops", "targets")},
                     "sweep": {k: {"p": v["prob_without"], "n": len(v["E_eps"]), "flipped": v["flipped"]} for k, v in sorted(sweep.items())},
                     "catgx": {"cards": cxd["motif_cards"], "n_model_inferences": cxd["n_model_inferences"], "occurrences_total": cxd["occurrences_total"]},
                     "joint": {k: jt[k] for k in ("p0", "arms", "labels")}, "meta": meta, "graph": graph},
        "trust": {"plant_median": med, "plant_n": sum(1 for p in plant["patterns"] for x in p["viz"] if x["plant_is_models_reason"]),
                  "causal": {k: {"beats": round(v["share_beats_placebo_p05"] * v["n_cases"]), "n": v["n_cases"]} for k, v in jsum["arms"].items()}},
        "fix": {"conversational_heldout": conv, "params": fx["params"]},
    }
    dump_json(out, REPORTS / "casestudy.json")
    return out


if __name__ == "__main__":
    o = run()
    a = o["across"]
    print(o["alarm"]["leaderboard"], o["alarm"]["template_searches"], o["alarm"]["template_failed"], o["alarm"]["template_title_first"])
    print(a["causes"], a["mean_parts"], a["fix_resolves"], a["drivers"], a["mean_p_full"], a["mean_p_without"], a["temgx_beats_random"])
    print(a["driver_kinds"], a["top_motif"], a["title_first"], a["title_top3"])
    f = o["featured"]
    print(f["case_id"], f["A"], ">", f["B"], f["parts"], f["rank_A"], [h["title"] for h in f["history"] if h["in_explanation"]])
    print([x["title"] for x in f["before"]], [x["title"] for x in f["after_F4"]])
    print(f["temgx"]["prob_full"], f["temgx"]["prob_without"], f["sweep"], [c["name"] for c in f["catgx"]["cards"]])
