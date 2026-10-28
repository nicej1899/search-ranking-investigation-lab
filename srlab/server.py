"""FastAPI backend for the Search Ranking Investigation Lab.

Every number the UI shows comes from these endpoints. Each response carries `status`:
  'fresh-inference'  computed now by the frozen model(s)
  'cached-measurement' read from a stored artifact, with the run id / timestamp that produced it
  'recorded-trace'   a stored investigation trace
Run: uvicorn srlab.server:app --port 8000   (or: python -m srlab.server)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .common import ART, CASES, CKPT as CKPT_DIR, PROC, REPORTS, ROOT, load_json, now_iso
from .explain import ab_summary, replay
from .ranker import load_ranker, world

app = FastAPI(title="Search Ranking Investigation Lab")
W = None


def W_():
    global W
    if W is None:
        W = world()
        from .fixes import register_extra
        register_extra(W)  # cold-start + conversational cases (never part of the original protocols)
    return W


SOURCES = [
    {"key": "TemGX", "cite": "Lu, Che, Fan, Liu, Shao, Ge, Xiao, Wu. Training-free Counterfactual Explanation for Temporal Graph Model Inference. ICLR 2026.",
     "url": "https://proceedings.iclr.cc/paper_files/paper/2026/hash/36bc2989d4b3371654fce3cb9a1a6889-Abstract-Conference.html",
     "code": "https://github.com/nicej1899/TemGX", "used_for": "History explanation (adapted; see methods ledger)"},
    {"key": "CATGX", "cite": "Lu, Vu, Le, Ma, Wu. CATGX: Causal-Aware Temporal Graph Explanation via Scalable Motif Sampling and Adjustment. KDD 2026.",
     "url": "https://doi.org/10.1145/3770855.3818100", "used_for": "Context-adjusted motif scores S(m) (adapted: exact enumeration)"},
    {"key": "Materials", "cite": "Lu et al. Context Determines Optimal Architecture in Materials Segmentation. arXiv:2602.04154.",
     "url": "https://arxiv.org/abs/2602.04154", "used_for": "Principle only: evaluate a model change per context slice"},
    {"key": "ExGIS", "cite": "Fan, Che, Lu, Wu. Graph Compression for Interpretable GNN Inference At Scale. PVLDB 18(12):5239-5242, 2025.",
     "url": "https://www.vldb.org/pvldb/vol18/p5239-fan.pdf", "used_for": "Not used (optional extension not implemented)"},
    {"key": "TGAT", "cite": "Xu, Ruan, Korpeoglu, Kumar, Achan. Inductive Representation Learning on Temporal Graphs. ICLR 2020.",
     "url": "https://arxiv.org/abs/2002.07962", "used_for": "Backbone architecture (re-implemented, 1 layer)"},
    {"key": "MovieLens", "cite": "Harper & Konstan. The MovieLens Datasets: History and Context. ACM TiiS 5(4):19, 2015. Data: ml-latest-small (2018-09-26).",
     "url": "https://grouplens.org/datasets/movielens/", "used_for": "Rating interactions, titles, genres, user tags (judgments only). Non-commercial research license."},
    {"key": "spaCy", "cite": "spaCy en_core_web_lg 3.8.0 static word vectors (MIT).", "url": "https://github.com/explosion/spacy-models",
     "used_for": "Query/title text vectors (averaged; negation not represented)"},
]


class ScoreReq(BaseModel):
    case_id: str
    model_id: str = "F0"
    removed: list[int] = []
    A_index: int | None = None
    B_index: int | None = None


class ExplainReq(BaseModel):
    case_id: str
    A_index: int
    B_index: int
    budget: int = 2


def _titles(idx):
    it = W_().items
    return [{"item": int(i), "movieId": int(it.movieId[i]), "title": it.title[i], "genres": it.genres[i], "label": it.label[i],
             "year": (None if np.isnan(it.year[i]) else int(it.year[i]))} for i in idx]


@app.get("/api/meta")
def meta():
    w = W_()
    sel = load_json(CASES / "selected_cases.json")
    split = load_json(PROC / "split.json")
    tr = load_json(REPORTS / "backbone_training.json")
    models = {m: {k: v for k, v in load_ranker(m).items() if k in ("model_id", "kind", "weights", "gate", "c", "config_hash", "description",
                                                                  "changed_mechanism", "dev_loss", "norm", "fit", "created")}
              for m in ("F0", "F1", "Fctrl")}
    return {"status": "cached-measurement", "dataset": split, "backbone": {k: tr[k] for k in ("checkpoint_sha256", "config", "hyperparameters",
                                                                                               "selected_epoch", "val", "test_temporal_backbone",
                                                                                               "test_popularity_reference", "eval_protocol", "created")},
            "models": models, "selected_cases": {"primary": sel["primary"], "backups": sel["backups"], "rule": sel["rule"]},
            "sources": SOURCES, "llm_mode_available": bool(os.environ.get("ANTHROPIC_API_KEY")), "load_sec": w.load_sec}


@app.get("/api/case/{case_id}")
def case(case_id: str):
    w = W_()
    if case_id not in w.cases:
        raise HTTPException(404, "unknown case")
    c = w.cases[case_id]
    h = w.store.before(c["member"], c["t"], n=w.model.cfg.max_hist)
    it = w.items
    sel = load_json(CASES / "selected_cases.json")
    ab = next((r for r in [sel["primary"]] + sel["backups"] if r["case_id"] == case_id), None)
    hist = [{"event_id": int(e), "ts": int(t), "rating": float(r), **_titles([i])[0]} for e, i, t, r in zip(h.eids, h.items, h.ts, h.ratings)]
    cands = [{"index": k, "grade": int(g), "runtime_min": None, **_titles([i])[0],
              "train_interactions": int(w.pop_before([i], load_json(PROC / "split.json")["cutoffs"]["train_end_ts"])[0])}
             for k, (i, g) in enumerate(zip(c["candidates"], c["grades"]))]
    return {"status": "fresh-inference", "case": {k: c[k] for k in ("case_id", "qid", "query", "kind", "split", "member", "t", "n_history",
                                                                     "judgment_source", "showcase_member")},
            "user_id": int(w.events.loc[w.events.member == c["member"], "user_id"].iloc[0]),
            "history": hist, "candidates": cands, "A_index": ab["A_index"] if ab else (auto_ab(case_id) or {}).get("A_index"),
            "B_index": ab["B_index"] if ab else (auto_ab(case_id) or {}).get("B_index"), "featured": bool(ab),
            "constraints": {"runtime_under_100": "unavailable: runtime unknown for all titles (no permitted source); constraint not applied"}}


@app.post("/api/score")
def score(req: ScoreReq):
    w = W_()
    c = w.cases[req.case_id]
    R = load_ranker(req.model_id)
    rp = replay(w, c, R, req.removed)
    ab = ab_summary(c, rp, req.A_index, req.B_index) if req.A_index is not None and req.B_index is not None else None
    return {"status": "fresh-inference", "model_id": req.model_id, "config_hash": R["config_hash"], "backbone_sha256": w.backbone_sha,
            "computed_at": now_iso(), "ab": ab, **rp}


@app.post("/api/explain")
def explain(req: ExplainReq):
    from .explain import explain_history
    w = W_()
    r = explain_history(w, w.cases[req.case_id], req.A_index, req.B_index, budget=req.budget)
    r.pop("replay_before")
    r.pop("replay_after")
    return {"status": "fresh-inference", "computed_at": now_iso(), **r}


@app.get("/api/motifs/{case_id}")
def motifs(case_id: str, A: int, B: int):
    from .motifs import catgx_scores
    w = W_()
    one = catgx_scores(w, w.cases[case_id], A, B)
    agg = load_json(REPORTS / "motif_aggregate_dev.json")
    agg = {k: v for k, v in agg.items() if k != "rows"}
    return {"status": "fresh-inference (case) + cached-measurement (DEV aggregate)", "case": one, "aggregate": agg}


@app.get("/api/investigation/{inv_id}")
def investigation(inv_id: str):
    p = ART / "investigations" / f"{inv_id}.json"
    if not p.exists():
        raise HTTPException(404)
    d = load_json(p)
    d["status"] = "recorded-trace"
    return d


@app.post("/api/investigation/run")
def run_investigation():
    from .agent import scripted_investigation
    t = time.time()
    r = scripted_investigation(investigation_id="inv-live-" + time.strftime("%H%M%S"))
    d = load_json(ROOT / r["export"]["path"])
    d["status"] = "fresh (scripted workflow re-executed now; FINAL evaluations are cached runs)"
    d["wall_sec"] = round(time.time() - t, 1)
    return d


@app.get("/api/eval")
def eval_report():
    p = REPORTS / "evaluation_summary.json"
    d = load_json(p)
    d["status"] = "cached-measurement"
    return d


@app.get("/api/latency")
def latency():
    d = load_json(REPORTS / "latency.json")
    d["status"] = "cached-measurement"
    return d


_AUTO: dict = {}


def auto_ab(case_id: str):
    """Automatic A/B for any case (same rule for every user): A = highest-ranked F0 title judged NOT relevant (grade 0);
    B = highest-ranked title below A with the best available grade (2, else 1). None if the ranking has no such failure."""
    if case_id in _AUTO:
        return _AUTO[case_id]
    w = W_()
    c = w.cases[case_id]
    rp = replay(w, c, load_ranker("F0"))
    g = c["grades"]
    order = rp["order"]
    sel = load_json(CASES / "selected_cases.json")
    cur = next((r for r in [sel["primary"]] + sel["backups"] if r["case_id"] == case_id), None)
    if cur is None and (CASES / "showcase50.json").exists():
        cur = next((r for r in load_json(CASES / "showcase50.json")["picked"] if r["case_id"] == case_id), None)
    if cur is None and (REPORTS / "triage_dev.json").exists():
        cur = _TRIAGE_AB().get(case_id)
    if cur:  # featured / showcase / triaged cases keep their stored A/B
        A, B = cur["A_index"], cur["B_index"]
        res = {"A_index": A, "B_index": B, "rank_A": rp["rank"][A], "rank_B": rp["rank"][B],
               "margin": rp["components"]["total"][A] - rp["components"]["total"][B],
               "beh_margin": rp["components"]["behaviour"][A] - rp["components"]["behaviour"][B]}
        _AUTO[case_id] = res
        return res
    A = next((i for i in order if g[i] == 0), None)
    res = None
    if A is not None:
        below = order[order.index(A) + 1:]
        for want in (2, 1):
            B = next((i for i in below if g[i] == want), None)
            if B is not None:
                res = {"A_index": int(A), "B_index": int(B), "rank_A": rp["rank"][A], "rank_B": rp["rank"][B],
                       "margin": rp["components"]["total"][A] - rp["components"]["total"][B],
                       "beh_margin": rp["components"]["behaviour"][A] - rp["components"]["behaviour"][B]}
                break
    _AUTO[case_id] = res
    return res


@app.get("/api/users")
def users(qid: str = "E01"):
    """Selectable test users for one query template, with their automatic A/B failure (if any)."""
    w = W_()
    sel = load_json(CASES / "selected_cases.json")
    featured = {r["case_id"]: ("primary" if i == 0 else f"backup {i}") for i, r in enumerate([sel["primary"]] + sel["backups"])}
    it = w.items
    out = []
    for c in w.cases.values():
        if c["qid"] != qid:
            continue
        ab = auto_ab(c["case_id"])
        h = w.store.before(c["member"], c["t"], n=50)
        labs = it.label.to_numpy()[h.items]
        row = {"case_id": c["case_id"], "member": c["member"], "n_history": c["n_history"], "split": c["split"],
               "showcase": c["showcase_member"], "featured": featured.get(c["case_id"]),
               "crime_share": float((labs == "Crime/Thriller").mean()) if len(labs) else 0.0,
               "top_label": (max(set(labs.tolist()), key=labs.tolist().count) if len(labs) else None)}
        if ab:
            row.update({"A_title": it.title[c["candidates"][ab["A_index"]]], "B_title": it.title[c["candidates"][ab["B_index"]]],
                        "rank_A": ab["rank_A"], "rank_B": ab["rank_B"], "history_driven": ab["beh_margin"] > 0})
        out.append(row)
    out.sort(key=lambda r: (r["featured"] is None, r.get("rank_B") is None, -(r.get("rank_B") or 0)))
    return {"qid": qid, "query": next(c["query"] for c in w.cases.values() if c["qid"] == qid), "users": out,
            "rule": auto_ab.__doc__}


@app.get("/api/templates")
def templates():
    from .queries import TEMPLATES
    return [{"qid": t.qid, "text": t.text, "kind": t.kind, "split": t.split} for t in TEMPLATES]


@app.get("/api/showcase")
def showcase():
    d = load_json(CASES / "showcase50.json")
    sel = load_json(CASES / "selected_cases.json")
    p = sel["primary"]
    w = W_()
    c = w.cases[p["case_id"]]
    prim = {"case_id": p["case_id"], "member": p["member"], "qid": c["qid"], "query": c["query"], "featured": "primary",
            "A_title": w.items.title[p["A_item"]], "B_title": w.items.title[p["B_item"]]}
    picked = [prim] + [r for r in d["picked"] if r["case_id"] != p["case_id"]]
    return {"status": "cached-measurement", "rule": d["rule"], "n_qualifying": d["n_qualifying"],
            "n_checked": d["n_explicit_cases_checked"], "cases": picked}


@app.get("/api/overview")
def overview(qid: str = "showcase", max_users: int = 51, per_user: int = 6):
    """A readable slice of the temporal interaction graph: the selectable users for this query and their most
    recent rating edges before their prediction time, plus whole-graph statistics."""
    w = W_()
    it = w.items
    if qid == "showcase":
        sc = showcase()["cases"]
        us = []
        for r in sc[:max_users]:
            c = w.cases[r["case_id"]]
            us.append({"case_id": r["case_id"], "featured": r.get("featured"), "A_title": r["A_title"], "B_title": r["B_title"],
                       "rank_B": r.get("rank_B"), "query": c["query"], "flipped": r.get("flipped"), "share": r.get("share_removed"),
                       "top_label": None})
    else:
        us = users(qid)["users"]
        us = [u for u in us if u.get("A_title")][:max_users]
    nodes_u, nodes_m, edges = [], {}, []
    split = load_json(PROC / "split.json")
    seen_m = set()
    for u in us:
        c = w.cases[u["case_id"]]
        if c["member"] in seen_m:  # one node per person (a person may have two showcase searches)
            continue
        seen_m.add(c["member"])
        h = w.store.before(c["member"], c["t"], n=per_user)
        nodes_u.append({"id": f"u{c['member']}", "member": c["member"], "case_id": c["case_id"], "n_history": c["n_history"],
                        "featured": u["featured"], "A_title": u.get("A_title"), "B_title": u.get("B_title"),
                        "rank_B": u.get("rank_B"), "top_label": u["top_label"], "t": c["t"], "query": c["query"],
                        "flipped": u.get("flipped"), "share": u.get("share")})
        for e, i, t, r in zip(h.eids, h.items, h.ts, h.ratings):
            nodes_m[int(i)] = {"id": f"m{int(i)}", "item": int(i), "title": it.title[i], "label": it.label[i], "genres": it.genres[i]}
            edges.append({"u": f"u{c['member']}", "m": f"m{int(i)}", "event_id": int(e), "ts": int(t), "rating": float(r),
                          "split": "train" if t < split["cutoffs"]["train_end_ts"] else "val" if t < split["cutoffs"]["val_end_ts"] else "test"})
    ev = w.events
    return {"status": "fresh-inference", "stats": {"users": int(ev.member.nunique()), "movies": int(len(it)), "edges": int(len(ev)),
                                                   "first": int(ev.ts.min()), "last": int(ev.ts.max()), "cutoffs": split["cutoffs"],
                                                   "counts": split["counts"]},
            "users": nodes_u, "movies": list(nodes_m.values()), "edges": edges,
            "note": (f"{len(nodes_u)} hand-picked people where the explanation is clearest (not a random sample), each with their {per_user} most recent ratings."
                     if qid == "showcase" else f"Showing {len(nodes_u)} users x their {per_user} most recent ratings.")}


@app.get("/api/neighborhood/{case_id}")
def neighborhood(case_id: str, max_hist: int = 20, others: int = 3):
    """Real subgraph around the test user at prediction time t (only edges with ts < t):
    user -> rated movies (model window, newest max_hist), candidates A/B (+ next top-ranked titles),
    and up to `others` other users who rated each of those movies before t (2-hop context)."""
    w = W_()
    c = w.cases[case_id]
    it = w.items
    ab = auto_ab(case_id) or {}
    t = c["t"]
    h = w.store.before(c["member"], t, n=w.model.cfg.max_hist)
    sl = slice(max(0, len(h.eids) - max_hist), None)
    nodes, edges = {}, []
    U = f"u{c['member']}"
    nodes[U] = {"id": U, "type": "user", "member": c["member"], "test": True}

    def movie(i, role=None):
        k = f"m{int(i)}"
        if k not in nodes:
            nodes[k] = {"id": k, "type": "movie", "item": int(i), "title": it.title[i], "label": it.label[i], "genres": it.genres[i]}
        if role:
            nodes[k]["role"] = role
        return k
    for e, i, ts, r in zip(h.eids[sl], h.items[sl], h.ts[sl], h.ratings[sl]):
        edges.append({"s": U, "t": movie(i), "kind": "rated", "event_id": int(e), "ts": int(ts), "rating": float(r)})
    cand = c["candidates"]
    if ab:
        for role, idx in (("A", ab["A_index"]), ("B", ab["B_index"])):
            edges.append({"s": U, "t": movie(cand[idx], role), "kind": "predict", "role": role, "index": idx})
    # 2-hop: other users who rated these movies before t
    ev = w.events
    mids = [n["item"] for n in nodes.values() if n["type"] == "movie"]
    sub = ev[(ev.item.isin(mids)) & (ev.ts < t) & (ev.member != c["member"])].sort_values("ts", ascending=False)
    for i, g in sub.groupby("item"):
        for r in g.head(others).itertuples():
            k = f"u{r.member}"
            nodes.setdefault(k, {"id": k, "type": "user", "member": int(r.member), "test": False})
            edges.append({"s": k, "t": f"m{int(i)}", "kind": "context", "event_id": int(r.event_id), "ts": int(r.ts), "rating": float(r.rating)})
    deg = sub.groupby("item").size().to_dict()
    for n in nodes.values():
        if n["type"] == "movie":
            n["degree_before_t"] = int(deg.get(n["item"], 0))
    return {"status": "fresh-inference", "case_id": case_id, "t": t, "nodes": list(nodes.values()), "edges": edges,
            "note": f"edges with timestamp < t only; newest {max_hist} history edges; {others} most recent other raters per movie"}


# ============================================================== multi-backbone, multi-hop explainers

class XReq(BaseModel):
    case_id: str
    backbone: str = "tgat1"
    L: int = 2
    k: int = 3
    exact: bool = False
    kc: int | None = None  # CATGX edge budget (knapsack); None = the paper default 8


def _case_ab(case_id):
    w = W_()
    c = w.cases[case_id]
    ab = auto_ab(case_id)
    return c, c["candidates"][ab["A_index"]], c["candidates"][ab["B_index"]], ab


def _node_meta(G, ids):
    it = W_().items
    out = {}
    for v in ids:
        v = int(v)
        if v < G.U:
            out[v] = {"type": "user", "label": "USER", "title": f"user {v}"}
        else:
            i = v - G.U
            out[v] = {"type": "movie", "label": it.label[i], "title": it.title[i], "genres": it.genres[i]}
    return out


@app.get("/api/backbones")
def backbones():
    from .backbones import Backbone, temperature
    out = []
    for key, info in Backbone.INFO.items():
        rep = REPORTS / ("backbone_training.json" if key == "tgat1" else f"backbone_{key}.json")
        ok = (CKPT_DIR / ("temporal_backbone_F0.pt" if key == "tgat1" else f"backbone_{key}.pt")).exists()
        r = load_json(rep) if rep.exists() else {}
        test = r.get("test_temporal_backbone") or r.get("test")
        out.append({"key": key, **info, "available": ok, "test": test, "T": temperature(key)})
    return {"backbones": out}


@app.get("/api/subgraph/{case_id}")
def subgraph(case_id: str, backbone: str = "tgat1", L: int = 2):
    from .backbones import get_backbone, temperature
    from .temgx import explore, sigmoid
    bb = get_backbone(backbone)
    G = bb.G
    c, a, b, ab = _case_ab(case_id)
    u, A, B = c["member"], a + G.U, b + G.U
    hop, edges = explore(G, [u, A, B], c["t"], L)
    meta = _node_meta(G, hop)
    lg = bb.logits(u, c["t"], [a, b])
    T = temperature(backbone)
    nodes = [{"id": int(v), "hop": int(h), **meta[int(v)], "role": "u" if v == u else "A" if v == A else "B" if v == B else None} for v, h in hop.items()]
    E = [{"eid": int(e), "s": int(x), "t": int(y), "ts": int(ts), "rating": r} for e, (x, y, ts, r) in edges.items()]
    return {"status": "fresh-inference", "case_id": case_id, "backbone": backbone, "L": L, "nodes": nodes, "edges": E,
            "u": int(u), "A": int(A), "B": int(B), "logit_A": float(lg[0]), "logit_B": float(lg[1]), "prob_A_over_B": float(sigmoid((lg[0] - lg[1]) / T)),
            "T": T, "rank_A": ab["rank_A"], "rank_B": ab["rank_B"], "query": c["query"], "t": c["t"],
            "A_title": W_().items.title[a], "B_title": W_().items.title[b]}


XCACHE = ART / "cache"
XCACHE.mkdir(parents=True, exist_ok=True)


def _cached(kind, req, fn):
    from .agent import code_version
    p = XCACHE / f"{kind}_{req.backbone}_L{req.L}_k{req.k}_{'exact' if req.exact else 's'}_{req.case_id}.json"
    if p.exists():
        d = load_json(p)
        d["status"] = f"cached-measurement (computed {d.get('computed_at', '?')})"
        return d
    r = fn()
    r["computed_at"] = now_iso(); r["code_version"] = code_version()
    from .common import dump_json
    dump_json(r, p)
    r["status"] = "fresh-inference"
    return r


@app.post("/api/temgx")
def run_temgx(req: XReq):
    from .backbones import get_backbone
    from . import temgx

    def fn():
        bb = get_backbone(req.backbone)
        c, a, b, _ = _case_ab(req.case_id)
        r = temgx.explain(bb, c["member"], c["t"], a, b, L=req.L, k=req.k)
        r["meta"] = {str(k): v for k, v in _node_meta(bb.G, [n["node"] for n in r["nodes"]]).items()}
        return r
    return _cached("temgx", req, fn)


@app.post("/api/catgx")
def run_catgx(req: XReq):
    from .backbones import get_backbone
    from . import catgx

    def fn():
        bb = get_backbone(req.backbone)
        c, a, b, _ = _case_ab(req.case_id)
        r = catgx.explain(bb, c["member"], c["t"], a, b, L=req.L, exact=req.exact)
        ids = {v for cd in r["motif_cards"] for v in cd["example"]["nodes"]}
        r["meta"] = {str(k): v for k, v in _node_meta(bb.G, ids).items()}
        return r
    return _cached("catgx2", req, fn)


class LiveReq(BaseModel):
    case_id: str
    backbone: str = "tgat2"
    model_id: str = "F0"
    removed: list[int] = []


@app.post("/api/live")
def live_rank(req: LiveReq):
    """Re-rank the whole candidate list live with the chosen graph model as history score and any events deleted."""
    from .live import rank
    w = W_()
    if req.case_id not in w.cases:
        raise HTTPException(404, "unknown case")
    ab = auto_ab(req.case_id) or {}
    return {"status": "fresh-inference", **rank(w, req.case_id, req.backbone, req.model_id, req.removed, ab.get("A_index"), ab.get("B_index"))}


_STRUCT: dict = {}


@app.get("/api/structure/{case_id}")
def structure(case_id: str, llm: bool = True):
    """CATGX structures behind one search, each with its causal test, plus an LLM explanation grounded in that
    evidence (cached answers; live via the Anthropic API only when ANTHROPIC_API_KEY is set)."""
    from . import structure as st
    if "ev" not in _STRUCT:
        p = REPORTS / "structure_evidence.json"
        _STRUCT["ev"] = load_json(p)["cases"] if p.exists() else {}
        q = REPORTS / "structure_llm.json"
        _STRUCT["llm"] = load_json(q) if q.exists() else {"answers": {}, "llm": None, "number_check": {}}
    ev = _STRUCT["ev"].get(case_id)
    if ev is None:
        from .backbones import get_backbone
        ev = st.evidence_for(W_(), get_backbone(st.BK), case_id)
        _STRUCT["ev"][case_id] = ev
    ans, src = _STRUCT["llm"]["answers"].get(case_id), _STRUCT["llm"].get("llm")
    if ans is None and llm:
        try:
            ans = st.live_llm(st.prompt(ev))
            src = "Claude Sonnet via the Anthropic API (live)" if ans else None
            if ans:
                _STRUCT["llm"]["answers"][case_id] = ans
        except Exception as e:  # network or key problems: the evidence still stands on its own
            src = f"LLM unavailable ({type(e).__name__})"
    bad = st.check(ev, ans) if ans else None
    return {"status": "cached-measurement", "evidence": ev, "llm": ans, "llm_source": src, "unsupported_numbers": bad,
            "prompt": st.prompt(ev)}


_CONTRAST: dict = {}


@app.get("/api/contrast/{case_id}")
def contrast_ep(case_id: str, k: int = 3):
    """Contrastive diagnosis: what supports the wrong winner A, what supports the better title B, how they differ,
    each checked by deletion against random deletions (srlab.contrast). Summary over the case-study searches attached."""
    from . import contrast as ct
    key = (case_id, k)
    if key not in _CONTRAST:
        p = XCACHE / f"contrast_{case_id}_k{k}.json"
        if p.exists():
            _CONTRAST[key] = load_json(p)
        else:
            from .backbones import get_backbone
            from .common import dump_json
            r = ct.contrast(W_(), get_backbone(ct.BK), case_id, k=k)
            dump_json(r, p)
            _CONTRAST[key] = r
    out = dict(_CONTRAST[key])
    sp = REPORTS / "contrast_summary.json"
    if sp.exists():
        s = load_json(sp)
        out["summary"] = {kk: s[kk] for kk in ("n", "k", "kinds", "top_motif", "del_A_beats_random", "del_B_beats_random", "mean_p0", "mean_p_del_A", "mean_p_del_B")}
    return out


@app.get("/api/kdemo/{case_id}")
def kdemo_ep(case_id: str, k: int = 3):
    """Explanation at size k with plain-language reasons, leave-one-out effects, cumulative removal vs random sets,
    the full re-ranking, and CATGX patterns before/after context matching (srlab.kdemo)."""
    from .kdemo import explain_k
    if k not in (1, 2, 3, 5, 8):
        raise HTTPException(400, "k must be one of 1, 2, 3, 5, 8")
    out = dict(explain_k(case_id, k))
    acr = REPORTS / f"across_{case_id.split('-')[0]}.json"  # same request, all dev failures, k = 3 (measured offline)
    if acr.exists():
        out["across"] = {kk: v for kk, v in load_json(acr).items() if kk != "rows"}
    return out


@app.get("/api/testgen")
def testgen_ep():
    """Tests and labelled training pairs generated from the explanations (srlab.testgen), with pass rates per ranker."""
    p = REPORTS / "testgen.json"
    if not p.exists():
        raise HTTPException(404, "run: python -m srlab.testgen")
    return load_json(p)


@app.get("/api/f5")
def f5_ep():
    """F5 (explanation-derived graph-level fix): n chosen on DEV, scored once on FINAL (srlab.f5)."""
    p = REPORTS / "f5_final.json"
    if not p.exists():
        raise HTTPException(404, "run: python -m srlab.f5 dev && python -m srlab.f5 final")
    return {"choice": load_json(REPORTS / "f5_choice.json"), "final": load_json(p)}


_XS: dict = {}


@app.get("/api/rag/{case_id}")
def rag_ep(case_id: str):
    """Retrieval-augmented diagnosis (srlab.rag): retrieved verified diagnoses, the evidence sheet, the LLM's answer and
    its number/citation check. Pre-generated for the case-study searches; generated live if ANTHROPIC_API_KEY is set."""
    from . import rag
    ctx = rag.context(case_id)
    if ctx is None:
        raise HTTPException(404, "this search is not a diagnosed failure")
    p = REPORTS / "rag_llm.json"
    ans, src = (load_json(p)["answers"].get(case_id) if p.exists() else None), "pre-generated (Claude Sonnet)"
    if ans is None:
        import json as _j
        from .structure import live_llm
        try:
            raw = live_llm(rag.prompt(ctx).replace("Return JSON only, an object", "Return JSON only: a list with one object"))
            ans, src = (raw[0] if raw else None), "live (Anthropic API)"
        except Exception:
            ans = None
    return {"context": ctx, "prompt": rag.prompt(ctx), "answer": ans, "source": src if ans else None,
            "check": rag.check(ctx, ans) if ans else None}


@app.get("/api/similar/{case_id}")
def similar_ep(case_id: str, k: int = 5):
    """Verified-explanation store (srlab.xstore): the diagnosed failures most like this one, the fix that resolved most
    of them, and the held-out evaluation of recommending fixes this way."""
    from . import xstore as xs
    if "dev" not in _XS:
        _XS["dev"], _XS["final"] = xs.load("dev"), xs.load("final")
        _XS["store"] = xs.Store(_XS["dev"]) if _XS["dev"] else None  # retrieval by mechanism (full key)
        ev = REPORTS / "xstore_eval.json"
        _XS["eval"] = load_json(ev) if ev.exists() else None
        ch = (_XS["eval"] or {}).get("chosen", {"features": "full key", "k": k})
        _XS["rec"] = (xs.Store(_XS["dev"], xs.FEATURE_SETS[ch["features"]]), ch["k"]) if _XS["dev"] else None  # fix recommender, chosen on DEV
    if _XS["store"] is None:
        raise HTTPException(404, "run: python -m srlab.xstore build dev")
    me = next((r for r in _XS["dev"] + _XS["final"] if r["case_id"] == case_id), None)
    if me is None:
        raise HTTPException(404, "this search is not a diagnosed failure")
    neigh = _XS["store"].query(me, k, exclude=case_id)
    rs, rk = _XS["rec"]
    rneigh = rs.query(me, rk, exclude=case_id)
    best, votes = xs.recommend(rneigh)
    keep = ("case_id", "member", "query", "A", "B", "motif_A", "motif_B", "kinds_A", "kinds_B", "fixed", "lead")
    return {"me": {kk: me[kk] for kk in keep}, "neighbours": [{**{kk: nb[kk] for kk in keep}, "similarity": s_} for nb, s_ in neigh],
            "recommend": best, "votes": votes, "rec_k": rk, "eval": _XS["eval"], "n_store": len(_XS["dev"]),
            "neighbours_fixed": {m: int(sum(nb["fixed"][m] for nb, _ in neigh)) for m in xs.FIXES}}


@app.get("/api/casestudy")
def casestudy():
    """Launch-review scenario (srlab.casestudy): measured once, served from the cached report."""
    p = REPORTS / "casestudy.json"
    if not p.exists():
        raise HTTPException(404, "run: python -m srlab.casestudy")
    out = load_json(p)
    bc = REPORTS / "bestcase.json"
    if bc.exists() and (b := load_json(bc)).get("best"):  # the demo case, picked by a fixed rule (srlab.bestcase)
        out["demo_member"] = b["best"]["case_id"]
        out["bestcase"] = {k: v for k, v in b.items() if k != "rows"}
    return out


@app.get("/api/retrievers/{case_id}")
def retrievers_ep(case_id: str):
    """GraphRAG retrievers compared at equal size (srlab.retrievers): edges each picks for this search + the summary."""
    p = REPORTS / "retrievers.json"
    if not p.exists():
        raise HTTPException(404, "run: python -m srlab.retrievers")
    d = load_json(p)
    row = next((r for r in d["rows"] if r["case_id"] == case_id), None)
    g = REPORTS / "graphrag_eval.json"
    return {"row": row, "agg": d["agg"], "n_cases": d["n_cases"], "graphrag": ({k: v for k, v in load_json(g).items() if k != "grid"} if g.exists() else None)}


@app.post("/api/catgx_do")
def run_catgx_do(req: XReq):
    """CATGX with the do-removal effect (TemGX's counterfactual replay) as its Stage-III signal g."""
    from .backbones import get_backbone
    from . import catgx

    def fn():
        bb = get_backbone(req.backbone)
        c, a, b, _ = _case_ab(req.case_id)
        r = catgx.explain(bb, c["member"], c["t"], a, b, L=req.L, g_mode="do", k=req.kc or 8, sign="pos" if req.kc else "abs")
        ids = {v for cd in r["motif_cards"] for v in cd["example"]["nodes"]} | {v for o in r["G_star_occurrences"] for v in o["nodes"]}
        r["meta"] = {str(k): v for k, v in _node_meta(bb.G, ids).items()}
        return r
    return _cached("catgxdo" if req.kc is None else f"catgxdo-pos-kc{req.kc}", req, fn)


@app.post("/api/joint")
def run_joint(req: XReq):
    """TemGX + CATGX together; every arm tested by do-removal against recency-matched random removals."""
    from .backbones import get_backbone
    from . import joint

    def fn():
        bb = get_backbone(req.backbone)
        c, a, b, _ = _case_ab(req.case_id)
        tx = run_temgx(req); cx = run_catgx(XReq(case_id=req.case_id, backbone=req.backbone, L=req.L, k=3))
        cxd = run_catgx_do(XReq(case_id=req.case_id, backbone=req.backbone, L=req.L, k=3))
        r = joint.run(bb, c["member"], c["t"], a, b, tx, cx, L=req.L, cx_do=cxd)
        r["top"] = {"attention_g": cx["selected_motifs"][:1], "do_g": cxd["selected_motifs"][:1]}
        r["calls"] = {"temgx": tx["n_model_inferences"], "catgx": cx["n_model_inferences"], "catgx_do": cxd["n_model_inferences"]}
        ids = {v for x in r["labels"] if x["instance"] for v in x["instance"]["nodes"]}
        r["meta"] = {str(k): v for k, v in _node_meta(bb.G, ids).items()}
        return r
    r = _cached("joint", req, fn)
    sp = REPORTS / f"joint_{req.backbone}_L{req.L}.json"
    if sp.exists():
        s = load_json(sp)
        r["summary"] = {k: s[k] for k in ("n_cases", "mean_p0", "coverage", "arms", "labelled_share")}
        r["summary"]["calls"] = {m: float(np.mean([x["calls"][m] for x in s["rows"]])) for m in ("temgx", "catgx", "catgx_do")} if s["rows"] and "calls" in s["rows"][0] else None
        r["summary"]["top_agree"] = float(np.mean([x["top"]["attention_g"] == x["top"]["do_g"] for x in s["rows"]])) if "top" in s["rows"][0] else None
    return r


@app.get("/api/compare")
def compare_report():
    p = REPORTS / "compare.json"
    if not p.exists():
        raise HTTPException(404, "run python -m srlab.compare")
    d = load_json(p)
    return {"status": "cached-measurement", "created": d["created"], "n_cases": d["n_cases"], "hardware": d["hardware"],
            "cells": {k: _cell_summary(v) for k, v in d["cells"].items()},
            "scale": load_json(REPORTS / "catgx_scale.json") if (REPORTS / "catgx_scale.json").exists() else None}


MULTIHOP = {"CO", "BRIDGE", "BRIDGEB"}  # motifs whose edges reach 2+ hops from the person


def _cell_summary(v):
    s = dict(v["summary"])
    rows = v["rows"]
    s["catgx_top_multihop"] = sum(1 for r in rows if set(r["catgx"]["top"]) & MULTIHOP) / max(len(rows), 1)
    s["temgx_beats_recent"] = sum(1 for r in rows if r["temgx"]["drop"] > r["temgx"]["recent"]) / max(len(rows), 1)
    s["temgx_beats_random"] = sum(1 for r in rows if r["temgx"]["drop"] > r["temgx"]["random"]) / max(len(rows), 1)
    return s


@app.get("/api/scalability")
def scalability():
    d = load_json(REPORTS / "scalability.json")
    d["status"] = "cached-measurement"
    return d


_TRI = {}


def _TRIAGE_AB():
    if "ab" not in _TRI:
        _TRI["ab"] = {r["case_id"]: {"A_index": r["A_index"], "B_index": r["B_index"]} for r in load_json(REPORTS / "triage_dev.json")["rows"]}
    return _TRI["ab"]


@app.get("/api/evalreport")
def evalreport():
    """Team view: failure triage of one evaluation run (DEV), targeted fixes tested once on FINAL, cold-start and
    chat-style slices, semantic-ID patterns. All cached measurements produced by srlab.triage / fixes / semid."""
    tri = load_json(REPORTS / "triage_dev.json")
    fx = load_json(REPORTS / "fixes_final.json") if (REPORTS / "fixes_final.json").exists() else None
    sem = load_json(REPORTS / "semid.json") if (REPORTS / "semid.json").exists() else None
    rows = tri["rows"]
    excl = [r for r in rows if r["sub"] == "ignored_exclusion"]
    excl_ex = []
    seen = set()
    for r in sorted(excl, key=lambda r: -r["lead"]):
        if r["qid"] in seen:
            continue
        seen.add(r["qid"])
        excl_ex.append({k: r[k] for k in ("case_id", "query", "A_title", "B_title", "rank_B", "n_history", "group", "A_index", "B_index")})
    from .text import negation_diagnostic
    from .extra import split_negation
    conv_q = sorted({r["query"] for r in rows if r["group"] == "conversational"})
    return {"status": "cached-measurement", "created": tri["created"],
            "triage": {k: tri[k] for k in ("split", "model", "searches", "failed", "fail_rate", "buckets", "by_group", "plan", "wall_s", "explainer_model_calls")},
            "fixes": fx, "semid": sem,
            "ltr": load_json(REPORTS / "ltr.json") if (REPORTS / "ltr.json").exists() else None,
            "plant": load_json(REPORTS / "plant_report.json") if (REPORTS / "plant_report.json").exists() else None,
            "baselines": {sp: load_json(REPORTS / f"baselines_{sp}.json") for sp in ("dev", "final") if (REPORTS / f"baselines_{sp}.json").exists()},
            "conversational": {"ignored_exclusion_failures": len(excl), "examples": excl_ex[:6],
                               "parsed": [{"query": q, "wants": split_negation(q)[0], "not": split_negation(q)[1]} for q in conv_q],
                               "negation_diagnostic": negation_diagnostic()}}


@app.get("/api/diagnostics/negation")
def negation():
    from .text import negation_diagnostic
    return {"status": "fresh-inference", **negation_diagnostic()}


@app.get("/api/export/{inv_id}")
def export(inv_id: str):
    p = ART / "investigations" / f"{inv_id}.json"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="application/json", filename=f"{inv_id}.json")


DIST = ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
