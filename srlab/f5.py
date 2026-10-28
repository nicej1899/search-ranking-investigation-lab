"""F5: an explanation-derived fix that acts on the GRAPH MODEL (not on how the ranker mixes its scores).

Finding it comes from (srlab.contrast, srlab.xstore): the wrong winner's history score is held up by a small crowd of
other people's recent ratings of that title (CATGX "crowd" structure; most TemGX links on side A), while the right
title's support is the member's own history. F5 removes, for every candidate title, its n most recent ratings by
OTHER people before the search time, then computes the graph model's history score; everything else is F0 (or F4).

python -m srlab.f5 dev     -> choose n on DEV (chat requests + regular searches + new members)
python -m srlab.f5 final   -> score the chosen setting once on the held-out FINAL split
"""
from __future__ import annotations

import sys

import numpy as np

from .common import REPORTS, dump_json, load_json, now_iso

BK = "tgat2"


def crowd_links(bb, items, member, t, n):
    ex = []
    for i in items:
        o, ts, e, r = bb.G.neighbors(int(i) + bb.G.U, t, None)
        ex += [int(x) for x, who in zip(e[::-1], o[::-1]) if who != member][:n]
    return frozenset(ex)


def scores(W, bb, c, model, n, F0, Ms):
    """Total scores for all candidates under F0/F4 mixing, with (n>0) or without the candidates' crowd links."""
    from .fixes import feats, score
    from .live import beh_norm
    cand = c["candidates"]
    ex = crowd_links(bb, cand, c["member"], c["t"], n) if n > 0 else frozenset()
    beh = np.asarray(bb.logits(c["member"], c["t"], cand, ex), float)
    z = feats(W, c, F0)
    m, s = beh_norm(BK)
    z["zb"] = (beh - m) / s
    return score(z, F0, Ms[model])["total"]


def metrics(W, bb, cases, model, n, F0, Ms):
    from .evaluate import ndcg_at
    from .fixes import violations
    nd, t1, t2, wrong1 = [], [], [], []
    for c in cases:
        s = scores(W, bb, c, model, n, F0, Ms)
        g = np.asarray(c["grades"])
        order = np.argsort(-s, kind="stable")
        nd.append(ndcg_at(s, g))
        wrong1.append(bool(g[order[0]] == 0))
        v = violations(c)
        if v.any():
            t1.append(not v[order[:3]].any())
            rk = np.empty(len(order), int); rk[order] = np.arange(len(order))
            X = next(i for i in order if v[i])
            best = g.max()
            if best > 0:
                Y = next(i for i in order if g[i] == best)
                t2.append(rk[Y] < rk[X])
    f = lambda x: float(np.nanmean(x)) if len(x) else None
    return {"ndcg10": f(nd), "wrong1": f(wrong1), "T1": f(t1), "T2": f(t2), "n": len(cases), "_nd": nd}


def groups(W, split):
    g = {"chat": [], "regular": [], "new members": []}
    for c in W.cases.values():
        if c["split"] != split or c.get("kind") == "broad":
            continue
        k = {"conversational": "chat", "coldstart": "new members"}.get(c.get("extra"), "regular")
        g[k].append(c)
    return g


def run(split, settings, stride=1):
    from .backbones import get_backbone
    from .ranker import load_ranker
    from .server import W_
    W, bb = W_(), get_backbone(BK)
    F0 = load_ranker("F0")
    Ms = {"F0": F0, "F4": load_ranker("F4")}
    G = {k: v[::stride] for k, v in groups(W, split).items()}
    out = {"created": now_iso(), "split": split, "stride": stride, "results": {}}
    for name, (model, n) in settings.items():
        out["results"][name] = {g: metrics(W, bb, cs, model, n, F0, Ms) for g, cs in G.items()}
        r = out["results"][name]
        print(split, name, {g: (round(v["ndcg10"], 4), round(v["wrong1"], 3), v["T1"] and round(v["T1"], 3), v["T2"] and round(v["T2"], 3)) for g, v in r.items()}, flush=True)
    # paired bootstrap of NDCG vs F0 per group
    rng = np.random.default_rng(0)
    for name in out["results"]:
        for g in G:
            a = np.asarray(out["results"][name][g]["_nd"], float); b = np.asarray(out["results"]["F0"][g]["_nd"], float)
            ok = ~(np.isnan(a) | np.isnan(b)); d = a[ok] - b[ok]
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)] if len(d) else [0]
            out["results"][name][g]["delta_vs_F0"] = {"mean": float(d.mean()) if len(d) else None, "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
            if name.startswith("F4+F5") and "F4" in out["results"]:  # does F5 add anything on top of F4?
                b4 = np.asarray(out["results"]["F4"][g]["_nd"], float); ok4 = ~(np.isnan(a) | np.isnan(b4)); d4 = a[ok4] - b4[ok4]
                bs4 = [d4[rng.integers(0, len(d4), len(d4))].mean() for _ in range(1000)]
                out["results"][name][g]["delta_vs_F4"] = {"mean": float(d4.mean()), "ci95": [float(np.quantile(bs4, .025)), float(np.quantile(bs4, .975))]}
    for name in out["results"]:
        for g in G:
            out["results"][name][g].pop("_nd")
    dump_json(out, REPORTS / f"f5_{split}.json")
    return out


if __name__ == "__main__":
    split = sys.argv[1]
    if split == "dev":
        S = {"F0": ("F0", 0), "F5 n=3": ("F0", 3), "F5 n=5": ("F0", 5), "F5 n=10": ("F0", 10), "F4": ("F4", 0),
             "F4+F5 n=3": ("F4", 3), "F4+F5 n=5": ("F4", 5), "F4+F5 n=10": ("F4", 10)}
        o = run("dev", S, stride=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
        # selection rule, fixed before looking at FINAL: n maximising the mean NDCG gain over the three groups for F5
        gain = {n: np.mean([o["results"][f"F5 n={n}"][g]["delta_vs_F0"]["mean"] for g in ("chat", "regular", "new members")]) for n in (3, 5, 10)}
        n = max(gain, key=gain.get)
        dump_json({"n": int(n), "rule": "max mean NDCG gain over chat/regular/new members on full DEV", "gains": {str(k): float(v) for k, v in gain.items()}}, REPORTS / "f5_choice.json")
        print("chosen n", n, gain)
    else:
        ch = load_json(REPORTS / "f5_choice.json")
        S = {"F0": ("F0", 0), "F5": ("F0", ch["n"]), "F4": ("F4", 0), "F4+F5": ("F4", ch["n"])}
        run("final", S)
