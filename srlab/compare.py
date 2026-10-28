"""Multi-case comparison of TemGX and CATGX across backbones and hop depths (measured).

python -m srlab.compare            -> artifacts/reports/compare.json
python -m srlab.compare --scale    -> artifacts/reports/catgx_scale.json (sampled vs exact as the window grows)

Cases: the curated showcase set (first n). Same A/B, same window and explored subgraph for both explainers.
Reported per (backbone, L): TemGX necessity fidelity / flip-rate / sparsity / hop of explanatory nodes /
equal-budget baselines; CATGX fidelity / coverage; sampled-vs-exact agreement; runtimes and model calls.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from .backbones import get_backbone, temperature
from .common import CASES, REPORTS, dump_json, hardware, load_json, now_iso
from .ranker import world
from . import catgx, temgx


def cases(n):
    W = world()
    sc = load_json(CASES / "showcase50.json")["picked"][:n]
    out = []
    for r in sc:
        c = W.cases[r["case_id"]]
        out.append((r["case_id"], c["member"], c["t"], c["candidates"][r["A_index"]], c["candidates"][r["B_index"]]))
    return out


def run(n=20, backbones=("tgat1", "tgat2", "tgn"), Ls=(1, 2, 3)):
    res = {"created": now_iso(), "hardware": hardware(), "n_cases": n, "cells": {}}
    cs = cases(n)
    for key in backbones:
        bb = get_backbone(key)
        for L in Ls:
            rows = []
            t0 = time.time()
            for cid, u, t, a, b in cs:
                tx = temgx.explain(bb, u, t, a, b, L=L, k=3)
                cs_s = catgx.explain(bb, u, t, a, b, L=L, exact=False)
                cs_e = catgx.explain(bb, u, t, a, b, L=L, exact=True)
                tg_cov = _coverage_of(tx, bb, u, t, a, b, L)
                Ss, Se = cs_s["S"], cs_e["S"]
                common = [m for m in Se if Se[m] is not None and Ss.get(m) is not None]
                rows.append({
                    "case_id": cid,
                    "temgx": {"fid_minus": tx["fid_minus"], "fid_plus": tx["fid_plus"], "flip": tx["flipped"], "sparsity": tx["sparsity"],
                              "n_edges_expl": len(tx["E_eps"]), "max_hop": max(tx["explanatory_hops"].values(), default=0),
                              "hops": list(tx["explanatory_hops"].values()), "coverage": tg_cov,
                              "drop": tx["baselines"]["temgx"], "random": tx["baselines"]["random_mean"], "recent": tx["baselines"]["most_recent"],
                              "attention": tx["baselines"].get("attention"), "ms": tx["latency_ms"]["total"], "calls": tx["n_model_inferences"],
                              "n_edges_window": tx["n_edges"]},
                    "catgx": {"fid_minus": cs_s["fid_minus"], "fid_plus": cs_s["fid_plus"], "coverage": cs_s["coverage"], "flip": cs_s["flipped"],
                              "top": cs_s["selected_motifs"][:1], "ms": cs_s["latency_ms"]["total"], "calls": cs_s["n_model_inferences"],
                              "occ_total": cs_s["occurrences_total"], "occ_used": cs_s["occurrences_used"]},
                    "catgx_exact": {"fid_minus": cs_e["fid_minus"], "coverage": cs_e["coverage"], "top": cs_e["selected_motifs"][:1],
                                    "ms": cs_e["latency_ms"]["total"], "calls": cs_e["n_model_inferences"]},
                    "agree_top_motif": bool(cs_s["selected_motifs"][:1] == cs_e["selected_motifs"][:1]),
                    "S_abs_err": float(np.mean([abs(Ss[m] - Se[m]) for m in common])) if common else None,
                })
            agg = _agg(rows)
            agg["wall_s"] = round(time.time() - t0, 1)
            res["cells"][f"{key}|L{L}"] = {"summary": agg, "rows": rows}
            print(key, L, agg, flush=True)
            dump_json(res, REPORTS / "compare.json")
    return res


def _coverage_of(tx, bb, u, t, a, b, L):
    """Motif coverage of the TemGX subgraph (same exact occurrence set CATGX is scored on)."""
    G = bb.G
    hop, edges = temgx.explore(G, [u, a + G.U, b + G.U], t, L)
    occ = catgx._occurrences_exact(G, edges, hop, u, a + G.U, b + G.U, None)
    E = set(tx["E_eps"])
    return sum(1 for o in occ if set(o[2]) <= E) / max(len(occ), 1)


def _agg(rows):
    def m(path):
        vals = [eval_path(r, path) for r in rows]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    return {
        "temgx_fid_minus": m("temgx.fid_minus"), "temgx_flip_rate": m("temgx.flip"), "temgx_coverage": m("temgx.coverage"),
        "temgx_sparsity": m("temgx.sparsity"), "temgx_share_hop2plus": float(np.mean([any(h >= 2 for h in r["temgx"]["hops"]) for r in rows])),
        "temgx_drop": m("temgx.drop"), "random_drop": m("temgx.random"), "recent_drop": m("temgx.recent"), "attention_drop": m("temgx.attention"),
        "temgx_ms": m("temgx.ms"), "temgx_calls": m("temgx.calls"), "window_edges": m("temgx.n_edges_window"),
        "catgx_fid_minus": m("catgx.fid_minus"), "catgx_fid_plus": m("catgx.fid_plus"), "catgx_coverage": m("catgx.coverage"),
        "catgx_ms": m("catgx.ms"), "catgx_calls": m("catgx.calls"), "catgx_exact_ms": m("catgx_exact.ms"), "catgx_exact_calls": m("catgx_exact.calls"),
        "catgx_exact_coverage": m("catgx_exact.coverage"), "top_motif_agreement": float(np.mean([r["agree_top_motif"] for r in rows])),
        "S_abs_err": m("S_abs_err"), "occ_total": m("catgx.occ_total"), "occ_used": m("catgx.occ_used"),
    }


def eval_path(r, path):
    cur = r
    for p in path.split("."):
        cur = cur.get(p) if isinstance(cur, dict) else None
        if cur is None:
            return None
    return float(cur) if isinstance(cur, (bool, int, float)) else None


def scale(key="tgn", case_idx=0, settings=((10, 200), (20, 800), (40, 3000), (80, 12000))):
    """CATGX sampled vs exact as the explored window grows (fan, cap). g = the paper's model-derived attribution."""
    u_c = cases(case_idx + 1)[case_idx]
    _, u, t, a, b = u_c
    bb = get_backbone(key)
    rows = []
    for fan, cap in settings:
        s = catgx.explain(bb, u, t, a, b, L=3, exact=False, fan=fan, cap=cap)
        e = catgx.explain(bb, u, t, a, b, L=3, exact=True, fan=fan, cap=cap) if s["occurrences_total"] <= 8000 else None
        rows.append({"fan": fan, "cap": cap, "edges": s["n_edges"], "occurrences": s["occurrences_total"],
                     "sampled": {"used": s["occurrences_used"], "calls": s["n_model_inferences"], "ms": s["latency_ms"]["total"], "top": s["selected_motifs"]},
                     "exact": ({"used": e["occurrences_used"], "calls": e["n_model_inferences"], "ms": e["latency_ms"]["total"], "top": e["selected_motifs"]} if e else None),
                     "S_sampled": s["S"], "S_exact": (e["S"] if e else None)})
        print(rows[-1]["edges"], rows[-1]["occurrences"], rows[-1]["sampled"], rows[-1]["exact"], flush=True)
    out = {"created": now_iso(), "backbone": key, "case": u_c[0], "rows": rows, "hardware": hardware()}
    dump_json(out, REPORTS / "catgx_scale.json")
    return out


if __name__ == "__main__":
    if "--scale" in sys.argv:
        scale()
    else:
        n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 20
        run(n)
