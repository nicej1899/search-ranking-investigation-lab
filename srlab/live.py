"""Live search ranking with a chosen graph model as its history component, for interactive what-ifs.

score(i) = w_b * z(graph-model logit of member u for title i, given the events kept) + w_r * z(text match) + w_p * z(popularity)
F0 weights and text/popularity normalisation are frozen; the history logit is standardised with that graph model's own
DEV statistics (TGAT-1L reproduces F0 exactly). F3 / F4 swap in their one changed mechanism (srlab.fixes.score).
`removed` = event ids deleted before the run (any events in the graph: the member's, or other people's), so deleting
explanation links re-ranks the whole result list, live.
"""
from __future__ import annotations

import time

import numpy as np

from .common import REPORTS, dump_json, load_json, now_iso

_NORM: dict = {}


def beh_norm(key):
    from .ranker import load_ranker
    if key == "tgat1":
        return load_ranker("F0")["norm"]["beh"]
    if not _NORM:
        p = REPORTS / "live_norm.json"
        if p.exists():
            _NORM.update(load_json(p))
    if key not in _NORM:
        from .backbones import get_backbone
        from .ranker import world
        W, bb = world(), get_backbone(key)
        dev = [c for c in W.cases.values() if c["split"] == "dev" and not c.get("extra")][::3]
        L = np.concatenate([bb.logits(c["member"], c["t"], c["candidates"]) for c in dev])
        _NORM[key] = [float(L.mean()), float(L.std())]
        dump_json({**_NORM, "fit_on": f"DEV candidate sets (every 3rd case, {len(dev)} cases)", "created": now_iso()}, REPORTS / "live_norm.json")
    return _NORM[key]


def rank(W, case_id, key="tgat2", model_id="F0", removed=(), A_index=None, B_index=None):
    from .backbones import get_backbone, temperature
    from .fixes import feats, score, violations
    from .ranker import load_ranker
    t0 = time.perf_counter()
    c = W.cases[case_id]
    # F5 = explanation-derived graph-level fix (srlab.f5): drop each candidate's most recent ratings by other people
    # before computing the history score; "F4+F5" mixes the scores like F4.
    f5 = model_id in ("F5", "F4+F5")
    f6 = model_id == "F6"  # structured parse of the request (srlab.sparse), applied on top of F0
    F0, M = load_ranker("F0"), load_ranker("F4" if model_id == "F4+F5" else "F0" if model_id in ("F5", "F6") else model_id)
    bb = get_backbone(key)
    ex = frozenset(int(e) for e in removed)
    if f5:
        from .f5 import crowd_links
        p = REPORTS / "f5_choice.json"
        n5 = load_json(p)["n"] if p.exists() else 3
        ex = ex | crowd_links(bb, c["candidates"], c["member"], c["t"], n5)
    beh = np.asarray(bb.logits(c["member"], c["t"], c["candidates"], ex), float)
    z = feats(W, c, F0)
    m, s = beh_norm(key)
    z["zb"] = (beh - m) / s
    sc = score(z, F0, M)
    if f6:
        from .sparse import apply as sp_apply
        sc["total"] = sp_apply(sc["total"], [W.items.genres[i] for i in c["candidates"]], c["query"])
    order = np.argsort(-sc["total"], kind="stable")
    rk = np.empty(len(order), int); rk[order] = np.arange(1, len(order) + 1)
    v, g, it = violations(c), np.asarray(c["grades"]), W.items
    rows = [{"index": int(i), "rank": int(rk[i]), "title": it.title[c["candidates"][i]], "genres": it.genres[c["candidates"][i]],
             "grade": int(g[i]), "breaks_exclusion": bool(v[i]), "total": float(sc["total"][i]),
             "parts": {k: float(sc[k][i]) for k in ("behaviour", "relevance", "popularity")}} for i in order]
    out = {"case_id": case_id, "backbone": key, "model_id": model_id, "removed": sorted(ex), "rows": rows,
           "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    if A_index is not None and B_index is not None:
        T = temperature(key)
        out["p_A_over_B"] = float(1 / (1 + np.exp(-(beh[A_index] - beh[B_index]) / T)))
        out["rank_A"], out["rank_B"] = int(rk[A_index]), int(rk[B_index])
    return out
