"""Verified-explanation store + retrieval ("verified GraphRAG" over diagnoses).

python -m srlab.xstore build dev  [shard nshards]   -> artifacts/reports/xstore/dev_<shard>.json
python -m srlab.xstore build final [shard nshards]  -> artifacts/reports/xstore/final_<shard>.json
python -m srlab.xstore eval                          -> artifacts/reports/xstore_eval.json

Store: every chat-request failure of the live ranker on the DEV split (top title judged irrelevant), explained the
same way as the case study: TemGX + CATGX for "A beats B" and for "B beats A" (srlab.contrast), each side checked by
deletion. Each diagnosis is saved with (i) what the explanation says, (ii) which targeted fix resolves it (F2, F3, F4
re-ranked live: does B end above A?).
Key: a small vector describing the MECHANISM, not the text of the request:
  ranker lead split (text, popularity, history) | side-A link kinds | side-B link kinds | CATGX top structure of A and
  of B (one-hot) | deletion effects (drop when A's support is removed, rise when B's is) | starting confidence.
Index: HNSW (hnswlib, cosine) over standardised keys, fitted on DEV only.
Use: a NEW failure (held-out FINAL split) is explained, its k nearest diagnosed failures are retrieved, and the fix that
resolved most of them is recommended. Evaluation: how often the recommended fix actually resolves the new failure
(re-ranked live), against always recommending the best fix on DEV, and against retrieval by the lead split alone
(no TemGX/CATGX evidence). The LLM layer (optional) only rephrases the retrieved, verified diagnoses.
"""
from __future__ import annotations

import sys

import numpy as np

from .common import REPORTS, dump_json, load_json, now_iso

BK = "tgat2"
FIXES = ("F2", "F3", "F4")
MOTIFS = ("POP", "CO", "SEQ", "BRIDGE", "BRIDGEB", "none")
KINDS = ("the title's recent raters", "the member's own ratings", "other people's ratings")
D = REPORTS / "xstore"


def failures(W, split):
    from .live import rank
    out = []
    for c in W.cases.values():
        if c.get("extra") != "conversational" or c["split"] != split:
            continue
        r = rank(W, c["case_id"], BK, "F0")
        order = [x["index"] for x in r["rows"]]
        g = np.asarray(c["grades"])
        if g[order[0]] != 0 or not (g > 0).any():
            continue
        best = g.max()
        B = next(i for i in order if g[i] == best)
        out.append((c["case_id"], order[0], B, r["rows"]))
    return out


def diagnose(W, bb, cid, A, B, rows0):
    from .contrast import contrast
    from .live import rank
    c = W.cases[cid]
    r = contrast(W, bb, cid, k=3, n_draw=10, ab=(A, B))
    p = {x["index"]: x["parts"] for x in rows0}
    lead = {k: p[A][k] - p[B][k] for k in ("relevance", "popularity", "behaviour")}
    fixed = {}
    for m in FIXES:
        rk = {x["index"]: x["rank"] for x in rank(W, cid, BK, m)["rows"]}
        fixed[m] = bool(rk[B] < rk[A])
    it = W.items
    return {"case_id": cid, "member": c["member"], "query": c["query"], "A": it.title[c["candidates"][A]], "B": it.title[c["candidates"][B]],
            "A_index": A, "B_index": B, "lead": lead,
            "kinds_A": r["side_A"]["kinds"], "kinds_B": r["side_B"]["kinds"],
            "motif_A": (r["side_A"]["structure"] or {}).get("motif", "none"), "motif_B": (r["side_B"]["structure"] or {}).get("motif", "none"),
            "links_A": r["side_A"]["links"], "links_B": r["side_B"]["links"],
            "test": r["test"], "fixed": fixed}


def key(d):
    """Mechanism vector of one diagnosis (see module docstring)."""
    t = d["test"]
    ka, kb = d["kinds_A"], d["kinds_B"]
    na, nb = max(1, sum(ka.values())), max(1, sum(kb.values()))
    v = [d["lead"]["relevance"], d["lead"]["popularity"], d["lead"]["behaviour"]]
    v += [ka.get(k, 0) / na for k in KINDS] + [kb.get(k, 0) / nb for k in KINDS]
    v += [1.0 if d["motif_A"] == m else 0.0 for m in MOTIFS] + [1.0 if d["motif_B"] == m else 0.0 for m in MOTIFS]
    v += [t["p0"] - t["del_A"]["p"], t["del_B"]["p"] - t["p0"], t["p0"]]
    return np.asarray(v, np.float32)


LEAD_ONLY = slice(0, 3)


def build(split, shard=0, nshard=1):
    from .backbones import get_backbone
    from .server import W_
    W, bb = W_(), get_backbone(BK)
    fl = failures(W, split)[shard::nshard]
    out = []
    skipped = []
    for i, (cid, A, B, rows0) in enumerate(fl):
        try:
            out.append(diagnose(W, bb, cid, A, B, rows0))
        except Exception as e:  # e.g. a member with no links in the window: nothing to explain
            skipped.append({"case_id": cid, "error": f"{type(e).__name__}: {e}"})
        if i % 20 == 0:
            print(split, shard, i, len(fl), flush=True)
    D.mkdir(parents=True, exist_ok=True)
    dump_json({"split": split, "shard": shard, "rows": out, "skipped": skipped}, D / f"{split}_{shard}.json")


def load(split):
    rows = []
    for p in sorted(D.glob(f"{split}_*.json")):
        rows += load_json(p)["rows"]
    return rows


class Store:
    def __init__(self, rows, cols=slice(None)):
        import hnswlib
        self.rows = rows
        X = np.stack([key(r) for r in rows])[:, cols]
        self.cols, self.mu, self.sd = cols, X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.idx = hnswlib.Index(space="cosine", dim=Z.shape[1])
        self.idx.init_index(max_elements=len(Z), ef_construction=200, M=16)
        self.idx.add_items(Z, np.arange(len(Z)))
        self.idx.set_ef(100)

    def query(self, d, k=5, exclude=None):
        z = (key(d)[self.cols] - self.mu) / self.sd
        lab, dist = self.idx.knn_query(z[None, :], k=min(len(self.rows), k + 1))
        out = [(self.rows[i], 1 - float(s)) for i, s in zip(lab[0], dist[0]) if self.rows[i]["case_id"] != exclude]
        return out[:k]


def recommend(neigh):
    votes = {m: float(np.mean([n["fixed"][m] for n, _ in neigh])) for m in FIXES}
    best = max(FIXES, key=lambda m: votes[m])
    return best, votes


FEATURE_SETS = {"score split only": [0, 1, 2], "score split + link kinds": list(range(9)),
                "score split + kinds + deletion effects": list(range(9)) + [21, 22, 23], "full key": list(range(24)),
                "explanation only (no score split)": list(range(3, 24))}


def _loo(rows, cols, k):
    st = Store(rows, cols)
    return float(np.mean([d["fixed"][recommend(st.query(d, k, exclude=d["case_id"]))[0]] for d in rows]))


def evaluate():
    """Choose the key's features and k by leave-one-out on DEV, then score once on the held-out FINAL failures.
    All variants are also reported on FINAL for transparency (selection used DEV only)."""
    dev, fin = load("dev"), load("final")
    dev_rate = {m: float(np.mean([r["fixed"][m] for r in dev])) for m in FIXES}
    glob = max(FIXES, key=lambda m: dev_rate[m])
    grid = []
    for name, cols in FEATURE_SETS.items():
        for k in (5, 10, 20):
            st = Store(dev, cols)
            fin_ok = [d["fixed"][recommend(st.query(d, k))[0]] for d in fin]
            grid.append({"features": name, "k": k, "dev_loo": _loo(dev, cols, k), "final": float(np.mean(fin_ok)), "_ok": fin_ok})
    best = max(grid, key=lambda g: (g["dev_loo"], -g["k"]))
    always = [d["fixed"][glob] for d in fin]
    diff = np.asarray(best["_ok"], float) - np.asarray(always, float)
    rng = np.random.default_rng(0)
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    out = {"created": now_iso(), "n_store": len(dev), "n_new": len(fin), "dev_fix_rates": dev_rate, "always": glob,
           "chosen": {"features": best["features"], "k": best["k"], "dev_loo": best["dev_loo"]},
           "success": {"retrieval": best["final"], "always_" + glob: float(np.mean(always)),
                       "oracle_any_fix": float(np.mean([any(d["fixed"].values()) for d in fin])),
                       "retrieval_full_key": next(g["final"] for g in grid if g["features"] == "full key" and g["k"] == best["k"]),
                       "retrieval_explanation_only": next(g["final"] for g in grid if g["features"].startswith("explanation only") and g["k"] == best["k"])},
           "retrieval_minus_always": {"mean": float(diff.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]},
           "grid": [{kk: v for kk, v in g.items() if kk != "_ok"} for g in grid],
           "final_fix_rates": {m: float(np.mean([r["fixed"][m] for r in fin])) for m in FIXES},
           "store_motifs_A": {m: sum(r["motif_A"] == m for r in dev) for m in MOTIFS},
           "note": "DEV and FINAL chat requests use different request templates, so fix rates shift (F3 was fit on DEV)."}
    dump_json(out, REPORTS / "xstore_eval.json")
    return out


def similar(case_id, k=5):
    """For the UI: the diagnosed failures most similar to this one (excluding itself) and the recommended fix."""
    dev = load("dev")
    me = next((r for r in dev + load("final") if r["case_id"] == case_id), None)
    if me is None:
        return None
    st = Store(dev)
    neigh = st.query(me, k, exclude=case_id)
    best, votes = recommend(neigh)
    keep = ("case_id", "member", "query", "A", "B", "motif_A", "motif_B", "kinds_A", "kinds_B", "fixed", "lead")
    return {"case_id": case_id, "me": {kk: me[kk] for kk in keep}, "neighbours": [{**{kk: n[kk] for kk in keep}, "similarity": s} for n, s in neigh],
            "recommend": best, "votes": votes, "actual": me["fixed"]}


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0, int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    else:
        print(evaluate())
