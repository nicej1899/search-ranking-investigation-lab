"""Held-out (FINAL) check of F6, the structured request parse (srlab.sparse), on the Demo ranker (TGAT-2 history).
python -m srlab.f6  ->  artifacts/reports/f6_final.json"""
import numpy as np
from .common import REPORTS, dump_json, now_iso

def run():
    from .backbones import get_backbone
    from .evaluate import ndcg_at
    from .fixes import violations
    from .ranker import load_ranker
    from .server import W_
    from . import f5, sparse
    W, bb = W_(), get_backbone("tgat2"); F0 = load_ranker("F0"); Ms = {"F0": F0}; gen = W.items.genres
    out = {"created": now_iso(), "split": "final", "note": sparse.__doc__, "groups": {}}
    rng = np.random.default_rng(0)
    for gname, cs in f5.groups(W, "final").items():
        res = {}
        for name in ("F0", "F6"):
            nd, t1, w1 = [], [], []
            for c in cs:
                s = f5.scores(W, bb, c, "F0", 0, F0, Ms)
                if name == "F6":
                    s = sparse.apply(s, [gen[i] for i in c["candidates"]], c["query"])
                g = np.asarray(c["grades"]); o = np.argsort(-s, kind="stable")
                nd.append(ndcg_at(s, g)); w1.append(bool(g[o[0]] == 0))
                v = violations(c)
                if v.any(): t1.append(bool(not v[o[:3]].any()))
            res[name] = {"ndcg10": float(np.nanmean(nd)), "wrong1": float(np.mean(w1)), "T1": float(np.mean(t1)) if t1 else None, "n": len(cs), "_nd": nd}
        d = np.asarray(res["F6"]["_nd"], float) - np.asarray(res["F0"]["_nd"], float); d = d[~np.isnan(d)]
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
        for k in res: res[k].pop("_nd")
        res["ndcg_delta"] = {"mean": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
        out["groups"][gname] = res
    dump_json(out, REPORTS / "f6_final.json")
    return out

if __name__ == "__main__":
    r = run()
    for g, v in r["groups"].items(): print(g, {k: (round(x["ndcg10"], 4), round(x["wrong1"], 3), x["T1"] and round(x["T1"], 3)) for k, x in v.items() if k != "ndcg_delta"}, v["ndcg_delta"])
