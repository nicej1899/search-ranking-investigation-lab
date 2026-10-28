"""Production-shaped search pipeline: full-catalog retrieval -> multi-feature learning-to-rank -> SHAP failure triage.

Retrieval (per search, over the full catalog of titles with >= 1 training interaction, minus titles already rated):
  text      top 150 titles by cosine(query vector, title+genre vector)  (+ top 50 by the query's "wants" part when it has a "not")
  personal  top 100 titles by the temporal graph model (TGAT 1-layer) for this member at time t
  popular   top 50 titles by number of ratings before t
  -> union (about 250-300 titles). Everything else is ranked below.
Ranker: LightGBM LambdaMART (objective lambdarank, NDCG@10), ~55 features in 5 families (history, popularity, text
match, title content, retrieval). Trained on DEV searches (80% of members), early-stopped on the other 20%, scored ONCE
on FINAL. Compared on the same retrieved candidates with: popularity only, text only, behaviour only, F0 (3 features).
Metrics over the FULL catalog: NDCG@10 with the ideal computed from every relevant title in the catalog (so a
relevant title that retrieval missed counts against the system); retrieval recall; wrong-#1 rate.
Failure triage: TreeSHAP contributions (exact for tree models, additive per title) summed by feature family; the lead
of the wrong #1 over the best relevant retrieved title splits exactly into family parts. If no relevant title was
retrieved at all, the cause is "retrieval missed it" (ranking cannot fix it).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from .common import CKPT, REPORTS, dump_json, load_json, now_iso, sha256_obj
from .extra import CONV_BY_ID, load_extra, split_negation
from .queries import TEMPLATES_BY_ID, item_tagsets, rubric_grade
from .ranker import load_ranker, world, zfeat
from .text import GENRE_PROBES, encode

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
          "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
R_TEXT, R_POS, R_PERS, R_POP = 150, 50, 100, 50
FAMILY = {}  # feature -> family


def fam(name, f):
    FAMILY[name] = f
    return name


class Ctx:
    def __init__(self):
        W = world()
        from .fixes import register_extra
        register_extra(W)
        self.W = W
        it = W.items
        self.n = len(it)
        tr = W.events[W.events.split == "train"]
        self.catalog = np.isin(np.arange(self.n), tr.item.unique())
        self.G = np.array([[1.0 if g in set(x) else 0.0 for g in GENRES] for x in it.genre_list], np.float32)
        self.ng = self.G.sum(1)
        self.year = it.year.fillna(it.year.median()).to_numpy(float)
        ev = W.events.sort_values("ts")
        self.item_ts, self.item_cr = {}, {}
        for i, g in ev.groupby("item"):
            self.item_ts[int(i)] = g.ts.to_numpy(); self.item_cr[int(i)] = np.cumsum(g.rating.to_numpy())
        self.probes = np.stack([encode(g) for g in GENRE_PROBES])
        tags = item_tagsets()
        gl = [set(x) for x in it.genre_list]
        self.grades = {}
        for q, t in list(TEMPLATES_BY_ID.items()) + list(CONV_BY_ID.items()):
            if t.kind == "explicit":
                self.grades[q] = np.array([rubric_grade(t, gl[i], tags.get(i, set())) for i in range(self.n)], np.int8)
        self.F0 = load_ranker("F0")
        self._mt = {}
        self.models = {}
        for k in ("bpr_mf", "sasrec"):
            p = CKPT / f"baseline_{k}.pt"
            if p.exists():
                from . import baselines as bl
                m = bl.BPRMF(1, self.n) if k == "bpr_mf" else bl.SASRec(self.n)
                sd = torch.load(p)
                if k == "bpr_mf":
                    m.U = torch.nn.Embedding(sd["U.weight"].shape[0], bl.D)
                m.load_state_dict(sd); m.eval()
                self.models[k] = m

    def pop(self, t, window=None):
        out = np.zeros(self.n); mean = np.zeros(self.n)
        for i, ts in self.item_ts.items():
            k = np.searchsorted(ts, t)
            out[i] = k if window is None else k - np.searchsorted(ts, t - window)
            if window is None and k:
                mean[i] = self.item_cr[i][k - 1] / k
        return out, mean

    def member_time(self, m, t):
        """Per (member, t) quantities shared by all searches at that moment (cached)."""
        key = (m, t)
        if key in self._mt:
            return self._mt[key]
        W = self.W
        with torch.no_grad():
            beh, h, _ = W.behavioural_logits(m, t, np.arange(self.n))
        hist = W.store.before(m, t, n=None)
        seen = np.zeros(self.n, bool); seen[hist.items] = True
        pop, mean_r = self.pop(t)
        pop365, _ = self.pop(t, 365 * 86400)
        last = hist.items[-50:]
        ug = self.G[last].mean(0) if len(last) else np.zeros(len(GENRES))
        aff = (self.G @ ug) / np.maximum(self.ng, 1)
        extra = {}
        if "bpr_mf" in self.models and len(last):
            V = self.models["bpr_mf"].item_vectors()
            extra["bpr"] = (V @ V[torch.as_tensor(last)].mean(0)).numpy()
        if "sasrec" in self.models and len(last):
            from .baselines import Scorers
            extra["sasrec"] = Scorers.sasrec(self.models["sasrec"], hist.items)
        d = {"beh": beh, "seen": seen, "pop": pop, "mean_r": mean_r, "pop365": pop365, "aff": aff, "n_hist": len(hist.items),
             "u_mean_r": float(hist.ratings.mean()) if len(hist.items) else 3.5,
             "u_recency_d": float((t - hist.ts[-1]) / 86400) if len(hist.items) else 9999.0, **extra}
        if len(self._mt) > 600:
            self._mt.clear()
        self._mt[key] = d
        return d

    def relevance(self, c, mt):
        if c["kind"] == "explicit":
            return self.grades[c["qid"]].astype(float)
        ev = self.W.events
        fut = ev[(ev.member == c["member"]) & (ev.ts >= c["t"]) & (ev.ts < c["t"] + 90 * 86400) & (ev.rating >= 4.0)]
        r = np.zeros(self.n); r[fut.item.to_numpy()] = 1
        return r


def retrieve(ctx, c, mt, qv, qp):
    ok = ctx.catalog & ~mt["seen"]
    idx = np.flatnonzero(ok)
    cos = ctx.W.item_vec[idx] @ qv
    src = {}
    def top(scores, k):
        return idx[np.argsort(-scores)[:k]]
    src["text"] = top(cos, R_TEXT)
    if qp is not None:
        src["text"] = np.union1d(src["text"], top(ctx.W.item_vec[idx] @ qp, R_POS))
    src["personal"] = top(mt["beh"][idx], R_PERS)
    src["popular"] = top(mt["pop"][idx], R_POP)
    cand = np.unique(np.concatenate(list(src.values())))
    return cand, src, idx


def features(ctx, c, cand, src, idx, mt, qv, qpos, qneg, spec):
    X, names = [], []
    def add(name, family, v):
        names.append(fam(name, family)); X.append(np.asarray(v, float))
    iv = ctx.W.item_vec[cand]
    beh = mt["beh"][cand]
    add("tgat_logit", "history", beh)
    add("tgat_rank", "history", rank_in(mt["beh"], idx, cand))
    if "bpr" in mt:
        add("bpr_mf_score", "history", mt["bpr"][cand])
    if "sasrec" in mt:
        add("sasrec_score", "history", mt["sasrec"][cand])
    add("genre_affinity", "history", mt["aff"][cand])
    add("log_n_history", "history", np.full(len(cand), np.log1p(mt["n_hist"])))
    add("user_mean_rating", "history", np.full(len(cand), mt["u_mean_r"]))
    add("days_since_last_rating", "history", np.full(len(cand), np.log1p(mt["u_recency_d"])))
    add("from_personal", "retrieval", np.isin(cand, src["personal"]))
    add("log_pop", "popularity", np.log1p(mt["pop"][cand]))
    add("log_pop_365d", "popularity", np.log1p(mt["pop365"][cand]))
    add("mean_rating", "popularity", mt["mean_r"][cand])
    add("pop_rank", "popularity", rank_in(mt["pop"], idx, cand))
    add("from_popular", "retrieval", np.isin(cand, src["popular"]))
    cos = iv @ qv
    add("text_cos", "text", cos)
    add("text_cos_wants", "text", iv @ qpos if qpos is not None else cos)
    add("text_cos_not", "text", iv @ qneg if qneg is not None else np.zeros(len(cand)))
    add("query_has_not", "text", np.full(len(cand), float(qneg is not None)))
    add("query_specificity", "text", np.full(len(cand), spec))
    qg = ctx.probes @ qv  # how strongly the query names each genre
    gm = ctx.G[cand] * qg
    add("genre_match_max", "text", gm.max(1))
    add("genre_match_sum", "text", gm.sum(1))
    if qneg is not None:
        qgn = ctx.probes @ qneg
        add("genre_match_not", "text", (ctx.G[cand] * qgn).max(1))
    else:
        add("genre_match_not", "text", np.zeros(len(cand)))
    add("text_rank", "text", rank_in(ctx.W.item_vec @ qv, idx, cand))
    add("from_text", "retrieval", np.isin(cand, src["text"]))
    for j, g in enumerate(GENRES):
        add(f"genre_{g}", "title content", ctx.G[cand, j])
    add("n_genres", "title content", ctx.ng[cand])
    add("year", "title content", ctx.year[cand])
    add("age_years", "title content", (c["t"] / 31557600 + 1970) - ctx.year[cand])
    return np.stack(X, 1).astype(np.float32), names


def rank_in(scores, idx, cand):
    s = scores[idx]
    order = np.argsort(-s)
    rk = np.empty(len(idx)); rk[order] = np.arange(1, len(idx) + 1)
    pos = {int(i): r for i, r in zip(idx, rk)}
    return np.log1p([pos[int(i)] for i in cand])


def ndcg_full(scores, cand, rel, ok, k=10):
    order = np.argsort(-scores, kind="stable")[:k]
    g = rel[cand][order]
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = ((2 ** g - 1) * disc[:len(g)]).sum()
    allg = np.sort(rel[ok])[::-1][:k]
    idcg = ((2 ** allg - 1) * disc[:len(allg)]).sum()
    return float(dcg / idcg) if idcg > 0 else None


def group_of(c):
    return c.get("extra") or "original"


def build(split: str, limit: int | None = None):
    t0 = time.time()
    ctx = Ctx()
    W = ctx.W
    cases = [c for c in W.cases.values() if c["split"] == split and not (split == "final" and c["showcase_member"])]
    cases.sort(key=lambda c: (c["member"], c["t"], c["case_id"]))
    if limit:
        cases = cases[:: max(1, len(cases) // limit)][:limit]
    out = []
    for c in cases:
        mt = ctx.member_time(c["member"], c["t"])
        qv = encode(c["query"])
        pos, neg = split_negation(c["query"])
        qpos, qneg = (encode(pos), encode(neg)) if neg else (None, None)
        spec = float((ctx.probes @ qv).max())
        cand, src, idx = retrieve(ctx, c, mt, qv, qpos)
        rel = ctx.relevance(c, mt)
        ok = ctx.catalog & ~mt["seen"]
        if rel[ok].sum() == 0:
            continue
        X, names = features(ctx, c, cand, src, idx, mt, qv, qpos, qneg, spec)
        F0 = ctx.F0
        zb = (mt["beh"][cand] - F0["norm"]["beh"][0]) / F0["norm"]["beh"][1]
        zr = (ctx.W.item_vec[cand] @ qv - F0["norm"]["rel"][0]) / F0["norm"]["rel"][1]
        zp = (np.log1p(mt["pop"][cand]) - F0["norm"]["pop"][0]) / F0["norm"]["pop"][1]
        w = F0["weights"]
        out.append({"case_id": c["case_id"], "member": c["member"], "kind": c["kind"], "group": group_of(c), "qid": c["qid"],
                    "query": c["query"], "n_history": c["n_history"], "has_not": qneg is not None,
                    "cand": cand, "X": X, "y": rel[cand], "rel_all": rel, "ok": ok,
                    "retrieved_relevant": float(rel[cand].astype(bool).sum() / max(rel[ok].astype(bool).sum(), 1)),
                    "any_relevant_retrieved": bool((rel[cand] > 0).any()),
                    "src_sizes": {k: len(v) for k, v in src.items()},
                    "base": {"F0": w[0] * zb + w[1] * zr + w[2] * zp, "popularity": mt["pop"][cand].astype(float),
                             "text": ctx.W.item_vec[cand] @ qv, "behaviour": mt["beh"][cand]}})
    return out, names, round(time.time() - t0, 1)


def _member_hash(m):
    import hashlib
    return int(hashlib.sha256(f"ltr:{m}".encode()).hexdigest()[:8], 16) % 5


def train(dev, names, seed=20261022):
    import lightgbm as lgb
    tr = [r for r in dev if _member_hash(r["member"]) != 0]
    va = [r for r in dev if _member_hash(r["member"]) == 0]
    def ds(rs):
        X = np.concatenate([r["X"] for r in rs]); y = np.concatenate([r["y"] for r in rs]).astype(int)
        return X, y, [len(r["y"]) for r in rs]
    Xt, yt, gt = ds(tr); Xv, yv, gv = ds(va)
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [10], "learning_rate": 0.05, "num_leaves": 31,
              "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambdarank_truncation_level": 20,
              "label_gain": [0, 1, 3], "seed": seed, "verbose": -1, "num_threads": 2}
    dt = lgb.Dataset(Xt, yt, group=gt, feature_name=names); dv = lgb.Dataset(Xv, yv, group=gv, reference=dt)
    bst = lgb.train(params, dt, 1000, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
    bst.save_model(str(CKPT / "ranker_lambdamart.txt"))
    return bst, {"train_searches": len(tr), "valid_searches": len(va), "best_iteration": bst.best_iteration,
                 "valid_ndcg10": float(bst.best_score["valid_0"]["ndcg@10"]), "params": params}


def evaluate(rows, bst, names):
    per = {k: [] for k in ("popularity", "text", "behaviour", "F0", "LambdaMART")}
    for r in rows:
        sc = dict(r["base"]); sc["LambdaMART"] = bst.predict(r["X"], num_iteration=bst.best_iteration)
        for k, s in sc.items():
            nd = ndcg_full(s, r["cand"], r["rel_all"], r["ok"])
            top = int(np.argmax(s))
            per[k].append({"case_id": r["case_id"], "member": r["member"], "group": r["group"], "kind": r["kind"], "has_not": r["has_not"],
                           "ndcg10": nd, "top1_wrong": (bool(r["y"][top] == 0) if r["kind"] == "explicit" else None)})
    return per


SLICES = {
    "all searches": lambda r: True,
    "searches that name what they want": lambda r: r["group"] == "original" and r["kind"] == "explicit",
    "vague searches": lambda r: r["kind"] == "broad",
    "searches with “not / nothing”": lambda r: r["has_not"] and r["group"] == "original",
    "new members (1–4 ratings)": lambda r: r["group"] == "coldstart",
    "chat-style requests": lambda r: r["group"] == "conversational",
}


def summarise(per, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for sname, f in SLICES.items():
        out[sname] = {}
        base = [r for r in per["F0"] if f(r) and r["ndcg10"] is not None]
        ids = {r["case_id"] for r in base}
        members = sorted({r["member"] for r in base})
        for model, rows in per.items():
            rs = [r for r in rows if r["case_id"] in ids]
            v = np.array([r["ndcg10"] for r in rs])
            t1 = [r["top1_wrong"] for r in rs if r["top1_wrong"] is not None]
            e = {"n": len(rs), "ndcg10": float(v.mean()) if len(v) else None, "top1_wrong": float(np.mean(t1)) if t1 else None}
            if model == "LambdaMART" and len(v):  # paired member bootstrap of LambdaMART - F0
                d = {r["case_id"]: r["ndcg10"] for r in rs}
                bm = {m: [] for m in members}
                for r in base:
                    bm[r["member"]].append(d[r["case_id"]] - r["ndcg10"])
                bs = [np.mean([x for m in rng.choice(members, len(members)) for x in bm[m]]) for _ in range(n_boot)]
                e["delta_vs_F0"] = float(np.mean([x for m in members for x in bm[m]]))
                e["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            out[sname][model] = e
    return out


def shap_triage(rows, bst, names):
    fams = sorted(set(FAMILY[n] for n in names))
    fidx = {f: [i for i, n in enumerate(names) if FAMILY[n] == f] for f in fams}
    out, n_expl = [], 0
    for r in rows:
        if r["kind"] != "explicit":
            continue
        n_expl += 1
        s = bst.predict(r["X"], num_iteration=bst.best_iteration)
        order = np.argsort(-s, kind="stable")
        A = int(order[0])
        if r["y"][A] > 0:
            continue
        if not r["any_relevant_retrieved"]:
            out.append({"case_id": r["case_id"], "query": r["query"], "group": r["group"], "cause": "retrieval_miss", "parts": None,
                        "A": int(r["cand"][A]), "B": None, "rank_B": None})
            continue
        B = next(int(i) for want in (2, 1) for i in order[1:] if r["y"][i] == want) if (r["y"] == 2).any() else next(int(i) for i in order[1:] if r["y"][i] == 1)
        contrib = bst.predict(r["X"][[A, B]], num_iteration=bst.best_iteration, pred_contrib=True)
        diff = contrib[0, :-1] - contrib[1, :-1]
        parts = {f: float(diff[ix].sum()) for f, ix in fidx.items()}
        top_feats = [names[i] for i in np.argsort(-diff)[:3]]
        cause = "text" if parts.get("text", 0) > 0 else max((f for f in parts if f != "text"), key=lambda f: parts[f])
        out.append({"case_id": r["case_id"], "query": r["query"], "group": r["group"], "cause": cause, "parts": parts,
                    "lead": float(s[A] - s[B]), "top_features": top_feats, "A": int(r["cand"][A]), "B": int(r["cand"][B]),
                    "rank_B": int(np.flatnonzero(order == B)[0] + 1)})
    return out, n_expl


def run():
    t0 = time.time()
    dev, names, tb_dev = build("dev")
    print("dev built", len(dev), tb_dev, "s", flush=True)
    bst, info = train(dev, names)
    print("trained", info, flush=True)
    fin, _, tb_fin = build("final")
    print("final built", len(fin), tb_fin, "s", flush=True)
    per = evaluate(fin, bst, names)
    summ = summarise(per)
    tri_rows, n_expl = shap_triage(fin, bst, names)  # diagnose held-out searches (DEV is in-sample for this ranker)
    W = world()
    titles = W.items.title.tolist()
    causes = {}
    for k in ("retrieval_miss", "text", "popularity", "history", "title content", "retrieval"):
        rs = [x for x in tri_rows if x["cause"] == k]
        if not rs:
            continue
        feats = {}
        for x in rs:
            for f_ in x.get("top_features") or []:
                feats[f_] = feats.get(f_, 0) + 1
        ex = sorted(rs, key=lambda x: -(x.get("lead") or 0))[:6]
        causes[k] = {"n": len(rs), "share": len(rs) / max(len(tri_rows), 1), "top_features": sorted(feats.items(), key=lambda z: -z[1])[:5],
                     "examples": [{"case_id": x["case_id"], "query": x["query"], "group": x["group"], "A_title": titles[x["A"]],
                                   "B_title": titles[x["B"]] if x["B"] is not None else None, "rank_B": x["rank_B"]} for x in ex]}
    imp = bst.feature_importance("gain")
    fam_imp = {}
    for n_, g in zip(names, imp):
        fam_imp[FAMILY[n_]] = fam_imp.get(FAMILY[n_], 0) + float(g)
    tot = sum(fam_imp.values()) or 1
    retr = {"recall_of_relevant": float(np.mean([r["retrieved_relevant"] for r in fin])),
            "searches_with_a_relevant_title_retrieved": float(np.mean([r["any_relevant_retrieved"] for r in fin])),
            "mean_candidates": float(np.mean([len(r["cand"]) for r in fin])),
            "catalog_size": int(Ctx.__dict__ and np.sum(fin[0]["ok"]) if fin else 0),
            "oracle_ndcg10_on_retrieved": float(np.mean([v for v in (ndcg_full(r["y"].astype(float), r["cand"], r["rel_all"], r["ok"]) for r in fin) if v is not None]))}
    out = {"created": now_iso(), "method": __doc__, "n_features": len(names), "features": [{"name": n_, "family": FAMILY[n_]} for n_ in names],
           "family_importance": {k: v / tot for k, v in sorted(fam_imp.items(), key=lambda z: -z[1])},
           "train": info, "retrieval": retr, "final": summ, "final_searches": len(fin),
           "caveat": "Explicit-query relevance is judged by a written rule over genres (+ tags). LambdaMART sees genres and how strongly the "
                     "query names each genre, so on explicit queries it can learn much of the rule itself; those gains overstate what it would "
                     "achieve against human judgments. Vague-search relevance (what the member actually rated 4+ later) has no such overlap.",
           "triage": {"explicit_searches": n_expl, "failed": len(tri_rows), "fail_rate": len(tri_rows) / max(n_expl, 1), "causes": causes},
           "model_sha": sha256_obj(open(CKPT / "ranker_lambdamart.txt").read()), "wall_s": round(time.time() - t0, 1)}
    dump_json(out, REPORTS / "ltr.json")
    return out


if __name__ == "__main__":
    r = run()
    print(r["retrieval"]); print(r["family_importance"])
    for s, e in r["final"].items():
        print(s, {k: (round(v["ndcg10"], 4) if v["ndcg10"] is not None else None, v.get("top1_wrong") and round(v["top1_wrong"], 3), v.get("delta_vs_F0") and round(v["delta_vs_F0"], 4), v.get("ci95")) for k, v in e.items()})
    print(r["triage"]["fail_rate"], {k: (v["n"], round(v["share"], 3), v["top_features"][:3]) for k, v in r["triage"]["causes"].items()})
