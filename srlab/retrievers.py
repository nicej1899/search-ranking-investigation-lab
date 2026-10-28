"""Which subgraph should GraphRAG put in the context? TemGX/CATGX vs the usual GraphRAG heuristics, at equal size.

For each case-study search (the wrong title A outranks the right title B), every retriever returns n edges from the
same 2-hop window around (member, A, B), n = the number of TemGX links on A's side (the demo's k). We delete them,
re-run the graph model and record its confidence that A beats B. A retriever that found what actually drives the
ranking moves the model the most. Retrievers:
  TemGX links (A side)                      - our explainer
  CATGX structure (A side, first n links)   - our explainer, structure level
  1-hop, most recent around A               - typical local GraphRAG: newest edges at the entity
  1-hop, most recent around member and A    - same, both entities
  attention top-n                           - what the model "looks at"
  random, same period (40 draws, mean)      - control
python -m srlab.retrievers -> artifacts/reports/retrievers.json
"""
from __future__ import annotations

import numpy as np

from .common import ART, REPORTS, dump_json, load_json, now_iso
CACHE = ART / "cache"
from .joint import _placebo
from .temgx import explore, sigmoid

BK = "tgat2"


def compare(W, bb, cid, n_draw=40, seed=0):
    c = W.cases[cid]
    ct = load_json(CACHE / f"contrast_{cid}_k3.json")
    u, t, G = c["member"], c["t"], bb.G
    A, B = ct["A_id"], ct["B_id"]
    a, b = A - G.U, B - G.U
    _, edges = explore(G, [u, A, B], t, 2)
    tem = [int(l["eid"]) for l in ct["side_A"]["links"]]
    n = len(tem)
    st = ct["side_A"]["structure"]
    cat = [int(l["eid"]) for l in (st or {}).get("links", [])][:n] if st else []
    by_time = sorted(edges, key=lambda e: -edges[e][2])
    near_A = [e for e in by_time if A in edges[e][:2]][:n]
    near_uA = [e for e in by_time if A in edges[e][:2] or u in edges[e][:2]][:n]
    arms = {"TemGX links": tem, "CATGX structure": cat, "1-hop, newest around the title": near_A,
            "1-hop, newest around member + title": near_uA}
    try:
        w = bb.edge_weights(u, t, [a, b])
        arms["attention top-n"] = sorted([e for e in edges if e in w], key=lambda e: -w[e])[:n]
    except Exception:
        pass
    rng = np.random.default_rng(seed)
    plac = _placebo(edges, set(tem), n_draw, rng)
    sets = [()] + [tuple(v) for v in arms.values()] + plac
    ms = np.asarray(bb.margins(u, t, a, b, sets), float)
    from .backbones import temperature
    T = temperature(bb.key)
    P = sigmoid(ms / T)
    out = {"case_id": cid, "n": n, "p0": float(P[0]), "arms": {}}
    for i, k in enumerate(arms, 1):
        out["arms"][k] = {"p": float(P[i]), "n": len(arms[k]), "eids": [int(e) for e in arms[k]]}
    out["arms"]["random, same period"] = {"p": float(P[1 + len(arms):].mean()), "n": n}
    return out


def run():
    from .backbones import get_backbone
    from .server import W_
    W, bb = W_(), get_backbone(BK)
    ids = [r["case_id"] for r in load_json(REPORTS / "casestudy.json")["across"]["rows"]]
    rows = []
    for cid in ids:
        rows.append(compare(W, bb, cid))
        print(cid, {k: round(v["p"], 3) for k, v in rows[-1]["arms"].items()}, flush=True)
    names = list(rows[0]["arms"])
    agg = {k: {"mean_p_after": float(np.mean([r["arms"][k]["p"] for r in rows if k in r["arms"]])),
               "mean_drop": float(np.mean([r["p0"] - r["arms"][k]["p"] for r in rows if k in r["arms"]])),
               "temgx_moves_more": int(sum(r["arms"]["TemGX links"]["p"] < r["arms"][k]["p"] for r in rows if k in r["arms"])),
               "n": int(sum(k in r["arms"] for r in rows))} for k in names}
    out = {"created": now_iso(), "backbone": BK, "n_cases": len(rows), "mean_p0": float(np.mean([r["p0"] for r in rows])),
           "agg": agg, "rows": rows}
    dump_json(out, REPORTS / "retrievers.json")
    return out


if __name__ == "__main__":
    o = run()
    for k, v in o["agg"].items():
        print(f"{k:40s} P after {v['mean_p_after']:.3f}  drop {v['mean_drop']:.3f}  TemGX moves more in {v['temgx_moves_more']}/{v['n']}")
