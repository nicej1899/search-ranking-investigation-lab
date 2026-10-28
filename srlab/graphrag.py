"""GraphRAG over verified diagnoses: retrieve past failures by walking a graph, not only by vector similarity.

Diagnosis graph (built from the DEV store, srlab.xstore; FINAL failures are only ever queries):
  nodes: one per diagnosed failure (case) + the MovieLens graph nodes its diagnosis touches (member, the title pushed
         up = A, the title that should win = B, every endpoint of its TemGX links on both sides, i.e. the crowd of
         recent raters and the member's own rated titles) + its CATGX structure types (A side, B side).
  edges: case -- member, case -- A ("pushed up"), case -- B ("should win"), case -- each explanation-link endpoint,
         case -- structure type. Weights per edge type (see VARIANTS).
Retrieval for a new failure: add it to the graph the same way, run personalized PageRank restarting at it, rank past
cases by their PageRank mass, take the top k. Hybrid: reciprocal-rank fusion with the vector retriever (xstore).
Use: the fix that worked for most retrieved cases is recommended (same as xstore). Variant and k are chosen by
leave-one-out on DEV; the choice is scored once on the held-out FINAL failures, against the vector retriever.

python -m srlab.graphrag eval   -> artifacts/reports/graphrag_eval.json
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

from .common import REPORTS, dump_json, now_iso
from .xstore import FIXES, FEATURE_SETS, Store, load, recommend

ROLE_W = {  # edge weight per role, per variant
    "entities": {"member": 1.0, "A": 1.0, "B": 1.0, "link": 1.0, "motif": 0.0},
    "entities+structure": {"member": 1.0, "A": 1.0, "B": 1.0, "link": 1.0, "motif": 0.3},
    "pushed-up title + crowd": {"member": 0.3, "A": 2.0, "B": 0.5, "link": 1.0, "motif": 0.3},
}


def _cid(W, r):
    return W.cases[r["case_id"]]


def edges_of(r, U, cand):
    """(node, role) pairs a diagnosis connects to. Graph node ids: users < U, titles = item + U."""
    out = [(f"n{r['member']}", "member"), (f"n{cand[r['A_index']] + U}", "A"), (f"n{cand[r['B_index']] + U}", "B"),
           (f"mA:{r['motif_A']}", "motif"), (f"mB:{r['motif_B']}", "motif")]
    for l in r["links_A"] + r["links_B"]:
        for v in (l["from_id"], l["to_id"]):
            if v != r["member"]:
                out.append((f"n{v}", "link"))
    return out


class GraphStore:
    def __init__(self, rows, W, U, variant="entities+structure"):
        self.rows, self.W, self.U, self.w = rows, W, U, ROLE_W[variant]
        self.nodes = {}
        self.adj = defaultdict(lambda: defaultdict(float))
        for i, r in enumerate(rows):
            self._add(f"c{i}", r)
        self.case_idx = {f"c{i}": i for i in range(len(rows))}

    def _id(self, n):
        if n not in self.nodes:
            self.nodes[n] = len(self.nodes)
        return self.nodes[n]

    def _add(self, cnode, r, adj=None):
        adj = self.adj if adj is None else adj
        cand = self.W.cases[r["case_id"]]["candidates"]
        c = self._id(cnode)
        for n, role in edges_of(r, self.U, cand):
            w = self.w[role]
            if w > 0:
                j = self._id(n)
                adj[c][j] += w
                adj[j][c] += w

    def query(self, r, k=10, exclude=None, alpha=0.15, iters=40):
        """Personalized PageRank from the new failure's node; returns [(row, score)] for the top-k past cases."""
        extra = defaultdict(lambda: defaultdict(float))
        q = "q"
        self._add(q, r, extra)
        qi = self.nodes[q]
        N = len(self.nodes)
        # sparse transition from adjacency (base + query edges)
        src, dst, val = [], [], []
        deg = np.zeros(N)
        for a, nb in list(self.adj.items()) + list(extra.items()):
            for b, w in nb.items():
                src.append(a); dst.append(b); val.append(w); deg[a] += w
        src, dst, val = np.array(src), np.array(dst), np.array(val)
        p = np.zeros(N); p[qi] = 1.0
        x = p.copy()
        for _ in range(iters):
            y = np.zeros(N)
            np.add.at(y, dst, x[src] * val / np.maximum(deg[src], 1e-12))
            x = alpha * p + (1 - alpha) * y
        del self.nodes[q]
        sc = [(self.rows[i], float(x[self.nodes[cn]])) for cn, i in self.case_idx.items() if self.rows[i]["case_id"] != exclude]
        sc.sort(key=lambda t: -t[1])
        return sc[:k]


def hybrid(g, v, k):
    """Reciprocal-rank fusion of graph and vector neighbours."""
    s = defaultdict(float); row = {}
    for lst in (g, v):
        for rank, (r, _) in enumerate(lst):
            s[r["case_id"]] += 1 / (60 + rank); row[r["case_id"]] = r
    best = sorted(s, key=lambda c: -s[c])[:k]
    return [(row[c], s[c]) for c in best]


def _retrievers(dev, W, U):
    vec_full, vec_split = Store(dev), Store(dev, FEATURE_SETS["score split only"])
    R = {"vector · score split (current)": lambda d, k, ex=None: vec_split.query(d, k, exclude=ex),
         "vector · full mechanism key": lambda d, k, ex=None: vec_full.query(d, k, exclude=ex)}
    for vname in ROLE_W:
        gs = GraphStore(dev, W, U, vname)
        R[f"graph · {vname}"] = (lambda gs_: lambda d, k, ex=None: gs_.query(d, k, exclude=ex))(gs)
        R[f"hybrid · {vname} + score split"] = (lambda gs_: lambda d, k, ex=None: hybrid(gs_.query(d, 30, exclude=ex), vec_split.query(d, 30, exclude=ex), k))(gs)
    return R


def evaluate():
    from .backbones import get_backbone
    from .server import W_
    W, U = W_(), get_backbone("tgat2").G.U
    dev, fin = load("dev"), load("final")
    R = _retrievers(dev, W, U)
    grid = []
    for name, f in R.items():
        nl = [f(d, 20, d["case_id"]) for d in dev]; nf = [f(d, 20) for d in fin]  # retrieve once, slice per k
        for k in (5, 10, 20):
            loo = float(np.mean([d["fixed"][recommend(n[:k])[0]] for d, n in zip(dev, nl)]))
            ok = [d["fixed"][recommend(n[:k])[0]] for d, n in zip(fin, nf)]
            grid.append({"retriever": name, "k": k, "dev_loo": loo, "final": float(np.mean(ok)), "_ok": ok})
            print(name, k, round(loo, 3), round(float(np.mean(ok)), 3), flush=True)
    cur = next(g for g in grid if g["retriever"].startswith("vector · score split") and g["k"] == 10)  # the DEV choice already in the demo
    graphish = [g for g in grid if not g["retriever"].startswith("vector")]
    best = max(graphish, key=lambda g: (g["dev_loo"], -g["k"]))  # chosen on DEV only
    diff = np.asarray(best["_ok"], float) - np.asarray(cur["_ok"], float)
    rng = np.random.default_rng(0)
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    always = max(FIXES, key=lambda m: np.mean([r["fixed"][m] for r in dev]))
    out = {"created": now_iso(), "n_store": len(dev), "n_new": len(fin),
           "chosen_graph": {kk: best[kk] for kk in ("retriever", "k", "dev_loo", "final")},
           "current_vector": {kk: cur[kk] for kk in ("retriever", "k", "dev_loo", "final")},
           "graph_minus_vector": {"mean": float(diff.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]},
           "always_" + always: float(np.mean([d["fixed"][always] for d in fin])),
           "oracle_any_fix": float(np.mean([any(d["fixed"].values()) for d in fin])),
           "grid": [{kk: v for kk, v in g.items() if kk != "_ok"} for g in grid]}
    dump_json(out, REPORTS / "graphrag_eval.json")
    return out


if __name__ == "__main__":
    o = evaluate()
    print({k: v for k, v in o.items() if k != "grid"})
