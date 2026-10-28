"""TemGX + CATGX working together, with every claim checked by intervention (do-removal and re-running the model).

The two explainers answer different questions on the same subgraph:
  TemGX  -> WHICH exact links caused this ranking (few links, each verified counterfactually).
  CATGX  -> WHAT KIND of structure the model relies on, adjusted for recency / duration / activity (confounders).
Joint explanation (this module):
  1. Label: every TemGX link is placed inside the CATGX pattern instance that contains it (the top-|S| motif with
     S > 0 first), so each verified link gets a mechanism ("part of a shared-taste path to A").
  2. Complete: add the rest of that pattern instance to the explanation (TemGX link -> whole structure).
  3. Test causally: do(remove edge set) and re-run the model, for TemGX links, CATGX's pattern edges, the completed
     joint explanation and the attention top-k (what the model "looks at", a correlational signal). Each is compared
     with a placebo: random edge sets of the same size, matched on recency quartile (the main confounder: recent
     links matter more for any temporal model). p = share of placebo draws whose drop is at least as large.
No numbers here are estimated without the model: each value is one exact replay of the backbone.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .catgx import MOTIFS, _occurrences_exact
from .temgx import explore, sigmoid
from .tgraph import DELTA_DAYS


def _placebo(edges, treated, n_draw, rng):
    """Random edge sets with the same recency-quartile histogram as `treated` (drawn outside `treated` if possible)."""
    eids = np.array(list(edges))
    if len(eids) == 0 or not treated:
        return [tuple() for _ in range(n_draw)]
    ts = np.array([edges[e][2] for e in eids])
    qs = np.quantile(ts, [0.25, 0.5, 0.75])
    q = {int(e): int(np.searchsorted(qs, t_)) for e, t_ in zip(eids, ts)}
    need = defaultdict(int)
    for e in treated:
        need[q[e]] += 1
    pools = {b: [int(e) for e in eids if q[int(e)] == b and int(e) not in treated] for b in range(4)}
    pools_all = {b: [int(e) for e in eids if q[int(e)] == b] for b in range(4)}
    out = []
    for _ in range(n_draw):
        s = []
        for b, c in need.items():
            pool = pools[b] if len(pools[b]) >= c else pools_all[b]
            s += [int(x) for x in rng.choice(pool, c, replace=False)]
        out.append(tuple(s))
    return out


def run(bb, u, t, a_item, b_item, tx, cx, L=2, n_draw=40, seed=0, delta_days=DELTA_DAYS, cx_do=None):
    G = bb.G
    A, B = a_item + G.U, b_item + G.U
    hop, edges = explore(G, [u, A, B], t, L, delta_days)
    occ = _occurrences_exact(G, edges, hop, u, A, B, t - int(delta_days * 86400))
    src = cx_do if cx_do is not None else cx  # label with the causal (replay) CATGX when available
    S = src["S"]
    lift = {c["motif"]: c.get("lift") for c in src.get("motif_cards", [])}
    pos = sorted([m for m in S if S[m] is not None and S[m] > 0], key=lambda m: -S[m])
    rest = sorted([m for m in S if S[m] is not None and S[m] <= 0], key=lambda m: -S[m])
    pref = pos + rest
    E_t = [int(e) for e in tx["E_eps"]]
    Et = set(E_t)
    by_edge = defaultdict(list)
    for o in occ:
        for e in o[2]:
            if e in Et:
                by_edge[e].append(o)
    # 1-2. label each TemGX link with the best-ranked motif instance that contains it, then complete the instance
    labels, comp = [], set(E_t)
    for e in E_t:
        cands = by_edge.get(e, [])
        pick = None
        for m in pref:
            om = [o for o in cands if o[0] == m]
            if om:  # most overlap with the TemGX links, then most recent
                pick = max(om, key=lambda o: (len(set(o[2]) & Et), max(o[3])))
                break
        if pick is not None:
            comp |= set(int(x) for x in pick[2])
        a, b, ts, r = edges[e]
        labels.append({"eid": e, "s": a, "t": b, "ts": ts,
                       "motif": pick[0] if pick else None, "motif_name": MOTIFS[pick[0]] if pick else None,
                       "S": (S.get(pick[0]) if pick else None), "lift": (lift.get(pick[0]) if pick else None),
                       "instance": {"nodes": [int(x) for x in pick[1]], "eids": [int(x) for x in pick[2]], "ts": [int(x) for x in pick[3]]} if pick else None})
    E_c = [int(e) for e in cx["G_star"]]
    sets = {"temgx": E_t, "catgx": E_c, "joint": sorted(comp)}
    if cx_do is not None:  # CATGX whose Stage-III signal g is the do-removal effect (TemGX's replay) instead of attention
        sets["catgx_do"] = [int(e) for e in cx_do["G_star"]]
    try:
        if bb.key != "tgn":
            w = bb.edge_weights(u, t, [a_item, b_item])
            sets["attention"] = sorted([e for e in edges if e in w], key=lambda e: -w[e])[:len(comp)]
    except Exception:
        pass
    rng = np.random.default_rng(seed)
    plac = {k: _placebo(edges, set(v), n_draw, rng) for k, v in sets.items() if v}
    allsets = [()] + [tuple(v) for v in sets.values()] + [s for k in plac for s in plac[k]]
    ms = np.asarray(bb.margins(u, t, a_item, b_item, allsets), float)
    from .backbones import temperature
    T = temperature(bb.key)
    m0 = float(ms[0])
    out = {"m0": m0, "p0": float(sigmoid(m0 / T)), "T": T, "n_draw": n_draw, "arms": {}}
    i = 1
    vals = {}
    for k, v in sets.items():
        vals[k] = float(ms[i]); i += 1
    for k in plac:
        pm = ms[i:i + n_draw]; i += n_draw
        drop, pdrops = m0 - vals[k], m0 - pm
        out["arms"][k] = {
            "eids": sets[k], "n": len(sets[k]), "margin": vals[k], "p": float(sigmoid(vals[k] / T)), "drop": float(drop),
            "flipped": bool(m0 > 0 and vals[k] < 0),
            "placebo_p_mean": float(np.mean(sigmoid(pm / T))), "placebo_p_lo": float(np.quantile(sigmoid(pm / T), 0.05)),
            "placebo_p_hi": float(np.quantile(sigmoid(pm / T), 0.95)), "placebo_drop_mean": float(np.mean(pdrops)),
            "p_value": float((1 + np.sum(pdrops >= drop)) / (1 + n_draw)),
        }
    occ_n = len(occ)
    out["coverage"] = {k: (sum(1 for o in occ if set(o[2]) <= set(v)) / max(occ_n, 1)) for k, v in sets.items()}
    out["labels"] = labels
    out["labelled_share"] = float(np.mean([x["motif"] is not None for x in labels])) if labels else None
    out["n_model_runs"] = len(allsets)
    out["n_window_edges"] = len(edges)
    return out


def summary(key="tgat2", L=2, n=12, k=3):
    """Same protocol over the comparison cases -> artifacts/reports/joint_{key}_L{L}.json (not cherry-picked per case)."""
    from .backbones import get_backbone
    from .common import REPORTS, dump_json, now_iso
    from .compare import cases
    from . import catgx, temgx
    bb = get_backbone(key)
    rows = []
    for cid, u, t, a, b in cases(n):
        tx = temgx.explain(bb, u, t, a, b, L=L, k=k)
        cx = catgx.explain(bb, u, t, a, b, L=L)
        cxd = catgx.explain(bb, u, t, a, b, L=L, g_mode="do")
        r = run(bb, u, t, a, b, tx, cx, L=L, cx_do=cxd)
        r["catgx_do_top"] = cxd["selected_motifs"][:1]; r["catgx_top"] = cx["selected_motifs"][:1]
        rows.append({"case_id": cid, "p0": r["p0"], "top": {"attention_g": r["catgx_top"], "do_g": r["catgx_do_top"]}, "calls": {"catgx": cx["n_model_inferences"], "catgx_do": cxd["n_model_inferences"], "temgx": tx["n_model_inferences"]}, "labelled_share": r["labelled_share"], "coverage": r["coverage"],
                     "arms": {k2: {kk: v[kk] for kk in ("n", "p", "drop", "placebo_p_mean", "placebo_drop_mean", "p_value", "flipped")} for k2, v in r["arms"].items()}})
        print(cid, {k2: (round(v["p"], 3), round(v["placebo_p_mean"], 3), round(v["p_value"], 3), v["n"]) for k2, v in r["arms"].items()}, flush=True)
    arms = sorted({k2 for r in rows for k2 in r["arms"]})
    agg = {}
    for k2 in arms:
        rs = [r["arms"][k2] for r in rows if k2 in r["arms"]]
        agg[k2] = {"n_cases": len(rs), "mean_n_edges": float(np.mean([x["n"] for x in rs])),
                   "mean_p_after": float(np.mean([x["p"] for x in rs])), "mean_placebo_p": float(np.mean([x["placebo_p_mean"] for x in rs])),
                   "mean_drop": float(np.mean([x["drop"] for x in rs])), "mean_placebo_drop": float(np.mean([x["placebo_drop_mean"] for x in rs])),
                   "share_beats_placebo_p05": float(np.mean([x["p_value"] <= 0.05 for x in rs])),
                   "flip_rate": float(np.mean([x["flipped"] for x in rs]))}
    out = {"created": now_iso(), "backbone": key, "L": L, "k": k, "n_cases": len(rows), "mean_p0": float(np.mean([r["p0"] for r in rows])),
           "labelled_share": float(np.mean([r["labelled_share"] for r in rows if r["labelled_share"] is not None])),
           "coverage": {k2: float(np.mean([r["coverage"][k2] for r in rows if k2 in r["coverage"]])) for k2 in arms},
           "arms": agg, "rows": rows}
    dump_json(out, REPORTS / f"joint_{key}_L{L}.json")
    return out


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else "tgat2"
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    o = summary(key, L)
    print({k: {kk: round(v, 3) for kk, v in a.items()} for k, a in o["arms"].items()}, "labelled", o["labelled_share"], "coverage", o["coverage"])
