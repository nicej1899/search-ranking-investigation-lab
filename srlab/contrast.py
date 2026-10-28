"""Contrastive diagnosis: explain the wrong winner A AND the better title B, then compare.

python -m srlab.contrast    -> artifacts/reports/contrast_summary.json (the 32 case-study searches)

A single explanation says why A scores high, but that evidence can look perfectly reasonable ("people who rated it
highly"). A diagnosis has to differentiate: what supports A over B, what supports B over A, and how the two differ.
For one search (graph model TGAT-2L, 2 hops):
  side A: TemGX on the decision "A beats B" (k links) + CATGX (replay signal, structures with S > 0 only)
  side B: the same on the reversed decision "B beats A"
  each link is typed by what it is: a recent rating OF THE TITLE by someone else (the title's crowd), one of the
  MEMBER's own ratings, or another person's rating elsewhere in the graph
  check: delete side A's links -> P(A beats B) should fall; delete side B's links -> it should rise; each against
  recency-matched random deletions of the same size (srlab.joint._placebo).
The difference between the two sides is what the fix should act on.
"""
from __future__ import annotations

import collections

import numpy as np

from .common import REPORTS, dump_json, load_json, now_iso

BK, L = "tgat2", 2
KINDS = ("the title's recent raters", "the member's own ratings", "other people's ratings")


def _day(ts):
    import datetime
    return datetime.datetime.fromtimestamp(int(ts), datetime.timezone.utc).strftime("%Y-%m-%d")


def _side(W, bb, u, t, x, y, k):
    """Explain 'x beats y' for member u at time t."""
    from . import catgx, temgx
    G, it = bb.G, W.items
    tx = temgx.explain(bb, u, t, x, y, L=L, k=k)
    cx = catgx.explain(bb, u, t, x, y, L=L, g_mode="do", sign="pos")
    X = x + G.U
    nm = lambda v: f"user {v}" if v < G.U else it.title[v - G.U]
    E = {e["eid"]: e for e in tx["edges"]}
    links = []
    for e in tx["E_eps"]:
        ed = E[e]
        kind = KINDS[0] if X in (ed["a"], ed["b"]) else KINDS[1] if u in (ed["a"], ed["b"]) else KINDS[2]
        links.append({"eid": int(e), "from": nm(ed["a"]), "to": nm(ed["b"]), "from_id": ed["a"], "to_id": ed["b"],
                      "rating": ed["rating"], "date": _day(ed["ts"]), "kind": kind})
    st = None
    if cx["motif_cards"]:
        cd = cx["motif_cards"][0]
        ex = cd["example"]
        inst = [{"eid": int(e), "from": nm(E[e]["a"]), "to": nm(E[e]["b"]), "from_id": E[e]["a"], "to_id": E[e]["b"],
                 "rating": E[e]["rating"], "date": _day(E[e]["ts"])} for e in ex["eids"] if e in E]
        st = {"motif": cd["motif"], "name": cd["name"], "lift": cd["lift"], "nodes": [{"id": int(v), "name": nm(v)} for v in ex["nodes"]], "links": inst}
    return {"links": links, "kinds": dict(collections.Counter(l["kind"] for l in links)), "structure": st,
            "n_model_runs": tx["n_model_inferences"] + cx["n_model_inferences"]}


def contrast(W, bb, case_id, k=3, n_draw=40, seed=0, ab=None):
    """ab=(A_index, B_index) overrides the evaluation-run pair (used by the explanation store)."""
    from .server import _case_ab
    from .temgx import explore, sigmoid
    from .joint import _placebo
    from .backbones import temperature
    if ab is None:
        c, a, b, _ = _case_ab(case_id)
    else:
        c = W.cases[case_id]
        a, b = c["candidates"][ab[0]], c["candidates"][ab[1]]
    u, t = c["member"], c["t"]
    A, B = _side(W, bb, u, t, a, b, k), _side(W, bb, u, t, b, a, k)
    _, edges = explore(bb.G, [u, a + bb.G.U, b + bb.G.U], t, L)
    ea, eb = [l["eid"] for l in A["links"]], [l["eid"] for l in B["links"]]
    rng = np.random.default_rng(seed)
    pa, pb = _placebo(edges, set(ea), n_draw, rng), _placebo(edges, set(eb), n_draw, rng)
    ms = np.asarray(bb.margins(u, t, a, b, [(), tuple(ea), tuple(eb)] + pa + pb), float)
    T = temperature(BK)
    p = lambda m: float(sigmoid(m / T))
    m0, mA, mB = ms[0], ms[1], ms[2]
    ra, rb = ms[3:3 + n_draw], ms[3 + n_draw:]
    test = {"p0": p(m0),
            "del_A": {"p": p(mA), "p_random": float(np.mean([p(m) for m in ra])), "p_value": float((1 + np.sum(ra <= mA)) / (1 + n_draw))},
            "del_B": {"p": p(mB), "p_random": float(np.mean([p(m) for m in rb])), "p_value": float((1 + np.sum(rb >= mB)) / (1 + n_draw))}}
    it = W.items
    shareA = A["kinds"].get(KINDS[0], 0) / max(len(ea), 1)
    shareB = B["kinds"].get(KINDS[1], 0) / max(len(eb), 1)
    return {"case_id": case_id, "member": u, "query": c["query"], "A": it.title[a], "B": it.title[b], "A_id": a + bb.G.U,
            "B_id": b + bb.G.U, "member_id": u, "k": k, "side_A": A, "side_B": B, "test": test,
            "difference": {"A_from_title_crowd": shareA, "B_from_member_history": shareB,
                           "shared_links": sorted(set(ea) & set(eb))}}


def summary(k=3):
    from .backbones import get_backbone
    from .server import W_
    W, bb = W_(), get_backbone(BK)
    ids = [r["case_id"] for r in load_json(REPORTS / "casestudy.json")["across"]["rows"]]
    rows = []
    for cid in ids:
        r = contrast(W, bb, cid, k=k)
        rows.append(r)
        print(cid, r["side_A"]["kinds"], r["side_B"]["kinds"], round(r["test"]["p0"], 2), round(r["test"]["del_A"]["p"], 2), round(r["test"]["del_B"]["p"], 2), flush=True)
    kinds = {s: {kk: int(sum(r[f"side_{s}"]["kinds"].get(kk, 0) for r in rows)) for kk in KINDS} for s in "AB"}
    motifs = {s: dict(collections.Counter((r[f"side_{s}"]["structure"] or {}).get("motif", "none") for r in rows)) for s in "AB"}
    out = {"created": now_iso(), "backbone": BK, "L": L, "k": k, "n": len(rows), "kinds": kinds, "top_motif": motifs,
           "del_A_beats_random": int(sum(r["test"]["del_A"]["p_value"] <= 0.05 and r["test"]["del_A"]["p"] < r["test"]["p0"] for r in rows)),
           "del_B_beats_random": int(sum(r["test"]["del_B"]["p_value"] <= 0.05 and r["test"]["del_B"]["p"] > r["test"]["p0"] for r in rows)),
           "mean_p0": float(np.mean([r["test"]["p0"] for r in rows])),
           "mean_p_del_A": float(np.mean([r["test"]["del_A"]["p"] for r in rows])),
           "mean_p_del_B": float(np.mean([r["test"]["del_B"]["p"] for r in rows])),
           "rows": rows}
    # demo member, by rule: clearest contrast (Requiem's support mostly its crowd, the right title's mostly the member's
    # own history), both deletion checks beat random, confidence not saturated; largest swing between the two deletions
    K0, K1 = KINDS[0], KINDS[1]
    ok = [r for r in rows if r["side_A"]["kinds"].get(K0, 0) >= 2 and r["side_B"]["kinds"].get(K1, 0) >= 2 and r["test"]["p0"] < 0.9
          and r["test"]["del_A"]["p_value"] <= 0.05 and r["test"]["del_B"]["p_value"] <= 0.05]
    out["demo_member"] = max(ok, key=lambda r: r["test"]["del_B"]["p"] - r["test"]["del_A"]["p"])["case_id"] if ok else None
    out["demo_rule"] = "crowd>=2 of A's links, own history>=2 of B's, p0<0.9, both deletions beat random; max swing"
    dump_json(out, REPORTS / "contrast_summary.json")
    return out


if __name__ == "__main__":
    o = summary()
    print(o["kinds"], o["top_motif"], o["del_A_beats_random"], o["del_B_beats_random"], o["mean_p0"], o["mean_p_del_A"], o["mean_p_del_B"])
