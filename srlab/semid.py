"""Semantic IDs for titles and a CATGX pattern analysis over them.

Semantic IDs (stand-in for an RQ-VAE codebook): residual k-means, 3 levels x 8 codes, over a content vector per
title = [title+genre text vector (spaCy, as used by the ranker), genre multi-hot / sqrt(#genres)]. A title's ID is
(c1, c2, c3); c1 is the coarse region. Level-1 codes are named automatically by their most over-represented genres.
This is NOT Netflix's semantic-ID model; it reproduces the idea (hierarchical discrete codes from content) on public data.

Analyses (DEV failure triage rows, model F0):
  where_failures   for each level-1 code: how often a title from that code is the WRONG top result, vs how often
                   titles from that code appear as candidates (lift = over-representation among wrong top results)
  catgx_semid      CATGX with semantic-ID labels: motif library = every ordered pair (code a -> code b) of a member's
                   consecutive-in-time ratings at most 7 days apart. g(r) = drop of the behavioural A-vs-B margin when
                   the pair's two ratings are removed (exact replay). Cells: recency x duration (doubling bins) x title
                   activity tertile; S(m) = sum_b omega(b) (mu(m,b) - mu_base(b)). Exact enumeration (pi = 1).
                   Aggregated over failed searches where the behaviour part favours the wrong title; member bootstrap.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
from sklearn.cluster import KMeans

from .common import REPORTS, dump_json, load_json, now_iso, sha256_obj
from .explain import margins_for_sets
from .motifs import MAX_GAP_H, ActivityBins, _bin
from .ranker import world

K, LEVELS = 8, 3
GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
          "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def build_codes(W) -> dict:
    gl = W.items.genre_list
    G = np.array([[1.0 if g in set(x) else 0.0 for g in GENRES] for x in gl])
    G = G / np.sqrt(np.maximum(G.sum(1, keepdims=True), 1))
    X = np.hstack([W.item_vec, G]).astype(np.float64)
    codes = np.zeros((len(X), LEVELS), int)
    R = X.copy()
    inertia = []
    for lv in range(LEVELS):
        km = KMeans(K, n_init=4, random_state=20261022 + lv).fit(R)
        codes[:, lv] = km.labels_
        R = R - km.cluster_centers_[km.labels_]
        inertia.append(float((R ** 2).sum()))
    # names for level-1 codes: top-2 genres by lift (share in code / share overall), min 15% share
    Gb = np.array([[1 if g in set(x) else 0 for g in GENRES] for x in gl])
    base = Gb.mean(0) + 1e-9
    pop = W.events[W.events.split == "train"].groupby("item").size().reindex(range(len(X)), fill_value=0).to_numpy()
    names, book = {}, []
    for c in range(K):
        m = codes[:, 0] == c
        share = Gb[m].mean(0)
        lift = share / base
        top = [GENRES[i] for i in np.argsort(-lift) if share[i] >= 0.15][:2] or [GENRES[int(np.argmax(share))]]
        names[c] = " / ".join(top)
        ex = np.argsort(-(pop * m))[:4]
        book.append({"code": c, "name": names[c], "n_titles": int(m.sum()), "top_genre_shares": {GENRES[i]: round(float(share[i]), 2) for i in np.argsort(-share)[:4]},
                     "examples": [W.items.title[i] for i in ex if m[i]],
                     "children": [{"code": f"{c}-{c2}", "n_titles": int((m & (codes[:, 1] == c2)).sum()),
                                   "example": (W.items.title[int(np.argmax(pop * (m & (codes[:, 1] == c2))))] if (m & (codes[:, 1] == c2)).any() else None)}
                                  for c2 in range(K)]})
    return {"codes": codes, "names": names, "book": book, "residual_sq_error_by_level": inertia}


def where_failures(W, rows, codes, names) -> list:
    out = []
    cand_codes, wrong_codes = [], []
    for r in rows:
        c = W.cases[r["case_id"]]
        cc = codes[np.asarray(c["candidates"]), 0]
        cand_codes.extend(cc.tolist())
        wrong_codes.append(int(cc[r["A_index"]]))
    cand_codes, wrong_codes = np.array(cand_codes), np.array(wrong_codes)
    for k in range(K):
        sh_w = float(np.mean(wrong_codes == k))
        sh_c = float(np.mean(cand_codes == k))
        out.append({"code": k, "name": names[k], "share_of_wrong_top": sh_w, "share_of_candidates": sh_c,
                    "lift": sh_w / sh_c if sh_c > 0 else None, "n_wrong_top": int((wrong_codes == k).sum())})
    return sorted(out, key=lambda x: -(x["lift"] or 0))


def _occ(W, c, codes):
    h = W.store.before(c["member"], c["t"], n=W.model.cfg.max_hist)
    lab = codes[h.items, 0]
    occ = []
    for i, j in itertools.combinations(range(len(h.eids)), 2):
        if (h.ts[j] - h.ts[i]) / 3600 > MAX_GAP_H:
            continue
        occ.append({"motif": (int(lab[i]), int(lab[j])), "event_ids": [int(h.eids[i]), int(h.eids[j])],
                    "items": [int(h.items[i]), int(h.items[j])], "t_first": int(h.ts[i]), "t_last": int(h.ts[j])})
    return occ


def catgx_one(W, c, ia, ib, codes, act):
    occ = _occ(W, c, codes)
    if not occ:
        return None
    m0 = margins_for_sets(W, c, ia, ib, [()])[0]
    ms = margins_for_sets(W, c, ia, ib, [tuple(r["event_ids"]) for r in occ])
    for r, mm in zip(occ, ms):
        r["g"] = max(0.0, float(m0 - mm))
        r["cell"] = (_bin((c["t"] - r["t_last"]) / 3600), _bin((r["t_last"] - r["t_first"]) / 3600), act(r["items"]))
    cells = {}
    for r in occ:
        cells.setdefault(r["cell"], []).append(r)
    tot = len(occ)
    S = {}
    for m in {r["motif"] for r in occ}:
        s = 0.0
        for b, rs in cells.items():
            rm = [r for r in rs if r["motif"] == m]
            if rm:
                s += len(rs) / tot * (np.mean([r["g"] for r in rm]) - np.mean([r["g"] for r in rs]))
        S[m] = (float(s), sum(1 for r in occ if r["motif"] == m))
    return S, len(occ), len(cells)


def run(n_boot=2000, seed=0) -> dict:
    t0 = time.time()
    W = world()
    from .fixes import register_extra
    register_extra(W)
    cb = build_codes(W)
    codes, names = cb["codes"], cb["names"]
    tri = load_json(REPORTS / "triage_dev.json")
    rows = tri["rows"]
    wf = where_failures(W, rows, codes, names)
    act = ActivityBins(W)
    per = []
    n_occ = 0
    used = [r for r in rows if r["parts"]["behaviour"] > 0]
    for r in used:
        c = W.cases[r["case_id"]]
        res = catgx_one(W, c, r["A_index"], r["B_index"], codes, act)
        if res is None:
            continue
        S, n, _ = res
        n_occ += n
        for m, (s, k) in S.items():
            per.append({"case_id": r["case_id"], "member": r["member"], "motif": m, "S": s, "n": k})
    rng = np.random.default_rng(seed)
    agg = []
    for m in sorted({x["motif"] for x in per}):
        xs = [x for x in per if x["motif"] == m]
        if len(xs) < 8:
            continue
        mem = sorted({x["member"] for x in xs})
        by = {u: [x["S"] for x in xs if x["member"] == u] for u in mem}
        bs = [np.mean([s for u in rng.choice(mem, len(mem), replace=True) for s in by[u]]) for _ in range(n_boot)]
        agg.append({"pattern": [int(m[0]), int(m[1])], "name": f"{names[m[0]]}  ->  {names[m[1]]}", "searches": len(xs), "members": len(mem),
                    "mean_S": float(np.mean([x["S"] for x in xs])), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    "frac_positive": float(np.mean([x["S"] > 0 for x in xs])), "occurrences": int(sum(x["n"] for x in xs))})
    agg.sort(key=lambda a: -a["mean_S"])
    out = {"created": now_iso(), "method": __doc__, "codebook": cb["book"], "levels": LEVELS, "codes_per_level": K,
           "residual_sq_error_by_level": cb["residual_sq_error_by_level"],
           "codes_hash": sha256_obj(codes.tolist()), "where_failures": wf,
           "catgx": {"searches_analysed": len(used), "occurrences": n_occ, "patterns": agg,
                     "significant_positive": [a for a in agg if a["ci95"][0] > 0],
                     "note": "S(m) > 0: the model's support for the wrong title from this semantic-ID pattern is above what other patterns in the same timing/activity context get."},
           "wall_s": round(time.time() - t0, 1)}
    np.save(REPORTS / "semantic_ids.npy", codes)
    dump_json(out, REPORTS / "semid.json")
    return out


if __name__ == "__main__":
    r = run()
    for b in r["codebook"]:
        print(b["code"], b["name"], b["n_titles"], b["examples"][:3])
    for x in r["where_failures"]:
        print("WF", x["code"], x["name"], round(x["share_of_wrong_top"], 3), round(x["share_of_candidates"], 3), round(x["lift"] or 0, 2))
    print(r["catgx"]["searches_analysed"], r["catgx"]["occurrences"], r["wall_s"])
    for a in r["catgx"]["patterns"][:8]:
        print(a["name"], a["searches"], round(a["mean_S"], 4), [round(v, 4) for v in a["ci95"]])
    print("...")
    for a in r["catgx"]["patterns"][-3:]:
        print(a["name"], a["searches"], round(a["mean_S"], 4), [round(v, 4) for v in a["ci95"]])
