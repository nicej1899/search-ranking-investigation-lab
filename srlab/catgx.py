"""CATGX (Lu, Vu, Le, Ma, Wu; KDD 2026) reproduced on the bipartite rating graph (Sec. 2-6, App. A.1-A.4).

Target prediction: link (u, A) vs (u, B) at time t; window G_t^delta; neighbourhood N_L(v_t) from the same
explorer TemGX uses (targets u, A, B), so both methods see the same subgraph.

Motif library P (user-defined; bipartite variants from the authors' code and the paper's Wiki case study):
  SEQ     x ->m1, x ->m2 with t1 <= t2       one person's viewing sequence (ordered fan-out)       anchor x
  CO      x ->m <- w                         co-rating relay (paper m_B: U->P<-U)                  anchor x
  BRIDGE  x ->m <- w ->A, time-ordered       shared-interest path to candidate A (paper m_A)       anchor x
  BRIDGEB x ->m <- w ->B                     same path ending at B                                 anchor x
  POP     w1 ->A, w2 ->A                     people converging on A (fan-in)                       anchor w1
  Anchors are generic (any node in N_L), as in the paper; occurrences touching the member u are flagged.
Stage I  node encoding v_u = [deg_in, deg_out, recency, onehot(label)] (A.1); motif query q_m = mean of template
         node encodings with degree/recency at population medians; HNSW (M=16, ef=200) retrieves N*alpha anchors
         (alpha=3); each anchor is verified by greedy, time-ordered matching; occurrences are assigned to cells and
         stratified-sampled (N per motif, >=1 per non-empty cell, rest proportional); w(r) = 1/pi(r), pi = n_b^s/n_b.
         exact=True enumerates every occurrence instead (ground truth for the sampled estimate).
Stage II codebooks: C_TM weighted motif counts; C_C recency/duration bins with exponentially growing widths over
         [0, delta] (A.2); C_E weighted activity tertiles of participating nodes. Cells with weighted support < tau=0.01
         are merged along recency, then duration.
Stage III g(r;M): TGAT -> attention mass of r's edges in the computation of u, A, B; TGN -> memory-state change of
         u, A, B when r's edges are removed (Sec. 6). [The authors' code uses a hand-weighted heuristic for g that
         does not call the model; we follow the paper.] mu(m,b), mu_base(b), omega(b), S(m) = sum_b omega(mu - mu_base);
         top-K motifs by |S|; density-greedy knapsack with single-item augmentation under edge budget k (Alg. 2).
Metrics: fidelity (logit-based, keep only G* among explored edges, Sec. 7) and coverage
         Cov = |{r in R_t : E_r subset E*}| / |R_t| over the EXACT occurrence set R_t.
"""
from __future__ import annotations

import itertools
import math
import time
from collections import defaultdict

import numpy as np

from .temgx import explore, sigmoid
from .tgraph import DELTA_DAYS

MOTIFS = {
    "SEQ": "a person watched two movies in a row",
    "CO": "two people rated the same movie",
    "BRIDGE": "a path of shared taste that ends at A",
    "BRIDGEB": "a path of shared taste that ends at B",
    "POP": "several people rated A",
}
LABEL_W = 3.0  # weight of the label one-hot relative to degree/recency in the A.1 encoding (paper leaves scaling open)
MOTIF_EDGES = {"SEQ": 2, "CO": 2, "BRIDGE": 3, "BRIDGEB": 3, "POP": 2}


def _occurrences_exact(G, edges, hop, u, A, B, t_min, max_per=20000):
    """Every occurrence of every motif inside the explored window subgraph (generic anchors, as in the paper).
    Occurrence = (motif, nodes in sigma order, edge ids, timestamps); nodes[0] is the anchor."""
    adj = defaultdict(list)
    for e, (x, y, ts, _) in edges.items():
        adj[x].append((y, ts, e)); adj[y].append((x, ts, e))
    U = G.U
    occ = defaultdict(list)
    users = [v for v in adj if v < U]
    movies = [v for v in adj if v >= U]
    for x in users:  # SEQ: x -> m1 then x -> m2
        xe = sorted(adj[x], key=lambda z: (z[1], z[2]))
        for i, j in itertools.combinations(range(len(xe)), 2):
            (m1, t1, e1), (m2, t2, e2) = xe[i], xe[j]
            occ["SEQ"].append(("SEQ", (x, m1, m2), (e1, e2), (t1, t2)))
    for m in movies:  # CO: x -> m <- w (anchor = earlier rater)
        me = sorted(adj[m], key=lambda z: (z[1], z[2]))
        for i, j in itertools.combinations(range(len(me)), 2):
            (x, t1, e1), (w, t2, e2) = me[i], me[j]
            occ["CO"].append(("CO", (x, m, w), (e1, e2), (t1, t2)))
    for c, name in ((A, "BRIDGE"), (B, "BRIDGEB")):  # x -> m <- w -> c, time-ordered
        for w, t3, e3 in adj[c]:
            for m, t2, e2 in adj[w]:
                if m == c or t2 > t3:
                    continue
                for x, t1, e1 in adj[m]:
                    if x != w and t1 <= t2:
                        occ[name].append((name, (x, m, w, c), (e1, e2, e3), (t1, t2, t3)))
    ae = sorted(adj[A], key=lambda z: (z[1], z[2]))  # POP: w1 -> A, w2 -> A
    for i, j in itertools.combinations(range(len(ae)), 2):
        (w1, t1, e1), (w2, t2, e2) = ae[i], ae[j]
        occ["POP"].append(("POP", (w1, w2, A), (e1, e2), (t1, t2)))
    out = []
    for m in MOTIFS:
        out += occ[m][:max_per]
    return out


class Encoder:
    """A.1 node encoding: [deg_in, deg_out, recency, onehot(label)] within the window subgraph."""

    def __init__(self, G, edges, hop, t, delta_s, item_labels, label_names):
        self.lab = {l: i for i, l in enumerate(["USER"] + label_names)}
        din, dout, last = defaultdict(int), defaultdict(int), {}
        for e, (x, y, ts, _) in edges.items():
            uu, mm = (x, y) if x < G.U else (y, x)
            dout[uu] += 1; din[mm] += 1
            last[uu] = max(last.get(uu, 0), ts); last[mm] = max(last.get(mm, 0), ts)
        self.nodes = sorted(hop)
        self.G, self.item_labels = G, item_labels
        feats = []
        for v in self.nodes:
            oh = np.zeros(len(self.lab)); oh[self.label_idx(v)] = LABEL_W
            feats.append(np.concatenate([[np.log1p(din[v]), np.log1p(dout[v]), (t - last.get(v, t - delta_s)) / delta_s], oh]))
        self.X = np.array(feats, np.float32)
        self.med = np.median(self.X[:, :3], 0)

    def label_idx(self, v):
        return 0 if v < self.G.U else self.lab[self.item_labels[v - self.G.U]]

    def query(self, motif, A, B):
        """q_m = mean of template node encodings (A.1): labels from the template, degree/recency at medians."""
        U = self.G.U
        item_any = np.zeros(len(self.lab)); item_any[1:] = 1.0 / (len(self.lab) - 1)
        def oh_user():
            v = np.zeros(len(self.lab)); v[0] = 1; return v
        def oh_node(x):
            v = np.zeros(len(self.lab)); v[self.label_idx(x)] = 1; return v
        tmpl = {"SEQ": [oh_user(), item_any, item_any], "CO": [oh_user(), item_any, oh_user()],
                "BRIDGE": [oh_user(), item_any, oh_user(), oh_node(A)], "BRIDGEB": [oh_user(), item_any, oh_user(), oh_node(B)],
                "POP": [oh_user(), oh_user(), oh_node(A)]}[motif]
        return np.concatenate([self.med, LABEL_W * np.mean(tmpl, 0)]).astype(np.float32)


def _cell(occ, t, delta_s, act_bin):
    ts = occ[3]
    rec = max(t - max(ts), 1) / 86400.0
    dur = (max(ts) - min(ts)) / 86400.0 + 1
    nb = max(1, math.ceil(math.log2(delta_s / 86400.0)))
    d_days = delta_s / 86400.0
    rb = min(nb, max(1, int(math.floor(math.log2(d_days / max(rec, 1e-9))) + 1)))  # bin i covers [d 2^-i, d 2^-(i-1))
    db = min(nb, max(1, int(math.floor(math.log2(max(dur, 1))) + 1)))              # bin i covers [2^(i-1), 2^i)
    return (rb, db, act_bin(occ))


def explain(bb, u, t, a_item, b_item, L=2, k=8, K=3, N=40, alpha=3, exact=False, delta_days=DELTA_DAYS, fan=10, cap=200,
            g_mode="paper", seed=0, sign="abs"):
    G = bb.G
    t0 = time.perf_counter()
    A, B = a_item + G.U, b_item + G.U
    delta_s = int(delta_days * 86400)
    hop, edges = explore(G, [u, A, B], t, L, delta_days, fan, cap)
    from .data import load_items
    items = _items_cache()
    occ_all = _occurrences_exact(G, edges, hop, u, A, B, t - delta_s)
    t_enum = (time.perf_counter() - t0) * 1000
    # ---------------- Stage I: ANN retrieval + verification + stratified sampling (or exact)
    enc = Encoder(G, edges, hop, t, delta_s, items, sorted(set(items)))
    rng = np.random.default_rng(seed)
    hnsw_used = False
    if exact:
        sampled = [(o, 1.0) for o in occ_all]
        retr_info = {"mode": "exact enumeration", "retrieved_anchors": None}
    else:
        anchors_by_motif = {}
        try:
            import hnswlib
            idx = hnswlib.Index(space="l2", dim=enc.X.shape[1])
            idx.init_index(max_elements=len(enc.X), ef_construction=200, M=16)
            idx.add_items(enc.X, np.arange(len(enc.X))); idx.set_ef(200)
            hnsw_used = True
            for m in MOTIFS:
                kq = min(len(enc.X), N * alpha)
                lab, _ = idx.knn_query(enc.query(m, A, B)[None, :], k=kq)
                anchors_by_motif[m] = {enc.nodes[int(i)] for i in lab[0]}
        except Exception:
            for m in MOTIFS:  # exact kNN fallback (labelled in output)
                d = ((enc.X - enc.query(m, A, B)) ** 2).sum(1)
                anchors_by_motif[m] = {enc.nodes[int(i)] for i in np.argsort(d)[: N * alpha]}
        # verification: an occurrence is retrieved if its anchor (first node in sigma) was retrieved
        retrieved = [o for o in occ_all if o[1][0] in anchors_by_motif[o[0]]]
        # stratified sampling per motif across cells
        act = _activity(retrieved, edges)
        cells = defaultdict(list)
        for o in retrieved:
            cells[(o[0], _cell(o, t, delta_s, act))].append(o)
        sampled = []
        for m in MOTIFS:
            cm = {c: v for c, v in cells.items() if c[0] == m}
            tot = sum(len(v) for v in cm.values())
            if not tot:
                continue
            quota = {c: 1 for c in cm}
            rest = max(0, N - len(cm))
            for c, v in cm.items():
                quota[c] += int(round(rest * len(v) / tot))
            for c, v in cm.items():
                ns = min(len(v), quota[c])
                pick = rng.choice(len(v), ns, replace=False)
                for i in pick:
                    sampled.append((v[i], len(v) / ns))  # w = 1/pi, pi = n_b^s / n_b
        retr_info = {"mode": "HNSW retrieval + stratified sampling" if hnsw_used else "exact kNN retrieval (hnswlib unavailable) + stratified sampling",
                     "retrieved_occurrences": len(retrieved), "N_per_motif": N, "alpha": alpha}
    t_stage1 = (time.perf_counter() - t0) * 1000 - t_enum
    # ---------------- Stage II: codebooks + cells (with merging)
    act = _activity([o for o, _ in sampled], edges, weights={id(o): w for o, w in sampled})
    rows = []
    for o, w in sampled:
        rows.append({"motif": o[0], "nodes": o[1], "eids": o[2], "ts": o[3], "w": w, "cell": _cell(o, t, delta_s, act)})
    rows = _merge_cells(rows, tau=0.01)
    C_TM = defaultdict(float)
    for r in rows:
        C_TM[r["motif"]] += r["w"]
    # ---------------- Stage III: model-derived attribution g(r; M)
    t3 = time.perf_counter()
    uniq = {}
    for r in rows:
        uniq.setdefault(tuple(sorted(r["eids"])), None)
    keys = list(uniq)
    n_inf = 0
    if g_mode == "paper" and bb.key in ("tgat1", "tgat2"):
        wts = bb.edge_weights(u, t, [a_item, b_item]); n_inf += 1
        for kx in keys:
            uniq[kx] = float(sum(wts.get(e, 0.0) for e in kx))
        g_desc = "attention mass of the occurrence's edges in the computation of u, A, B (TGAT)"
    elif g_mode == "paper" and bb.key == "tgn":
        vals = bb.memory_change(u, t, [a_item, b_item], [list(kx) for kx in keys]); n_inf += len(keys)
        for kx, v in zip(keys, vals):
            uniq[kx] = float(v)
        g_desc = "memory-state change of u, A, B when the occurrence's edges are removed (TGN)"
    else:
        m0 = float(bb.margins(u, t, a_item, b_item, [()])[0])
        ms = bb.margins(u, t, a_item, b_item, [list(kx) for kx in keys]); n_inf += len(keys) + 1
        for kx, v in zip(keys, ms):
            uniq[kx] = max(0.0, m0 - float(v))
        g_desc = "drop of the A-B logit margin when the occurrence's edges are removed"
    for r in rows:
        r["g"] = uniq[tuple(sorted(r["eids"]))]
    t_g = (time.perf_counter() - t3) * 1000
    # adjustment
    bycell = defaultdict(list)
    for r in rows:
        bycell[r["cell"]].append(r)
    Wtot = sum(r["w"] for r in rows) or 1.0
    S, terms = {}, defaultdict(list)
    for m in MOTIFS:
        s, present = 0.0, False
        for b, rs in bycell.items():
            rm = [r for r in rs if r["motif"] == m]
            if not rm:
                continue
            present = True
            mu = sum(r["w"] * r["g"] for r in rm) / sum(r["w"] for r in rm)
            mub = sum(r["w"] * r["g"] for r in rs) / sum(r["w"] for r in rs)
            om = sum(r["w"] for r in rs) / Wtot
            s += om * (mu - mub)
            terms[m].append({"cell": list(b), "omega": om, "mu": mu, "mu_base": mub, "n": len(rm)})
        S[m] = s if present else None
    # knapsack (Alg. 2 lines 3-12)
    # sign="abs": top-K by |S| as in the paper; sign="pos": only structures the model relies on MORE (S > 0), i.e. the
    # ones that can explain why A is preferred (used by the case-study view)
    top = sorted([m for m in S if S[m] is not None and (sign == "abs" or S[m] > 0)], key=lambda m: -abs(S[m]))[:K]
    cand = sorted([r for r in rows if r["motif"] in top], key=lambda r: -r["w"] * r["g"] / len(r["eids"]))
    Estar, used, sel = set(), 0, []
    for r in cand:
        new = set(r["eids"]) - Estar
        if used + len(new) <= k and r["g"] > 0:
            Estar |= new; used += len(new); sel.append(r)
    best = max((r for r in cand if len(r["eids"]) <= k), key=lambda r: r["w"] * r["g"], default=None)
    if best is not None and best["w"] * best["g"] > sum(r["w"] * r["g"] for r in sel):
        sel, Estar = [best], set(best["eids"])
    # motif cards: for each selected motif, how much more the model relies on it than on comparable structures
    # (ratio of the cell-weighted means, same cells as S) and its strongest concrete occurrence
    cards = []
    for m in top:
        tm = terms[m]
        num = sum(x["omega"] * x["mu"] for x in tm)
        den = sum(x["omega"] * x["mu_base"] for x in tm)
        rm = [r for r in rows if r["motif"] == m]
        best_r = max(rm, key=lambda r: r["g"])
        mine = [r for r in rm if u in r["nodes"]]  # prefer an instance that involves the person we investigate
        involves_u = bool(mine) and max(r["g"] for r in mine) >= 0.25 * best_r["g"]
        if involves_u:
            best_r = max(mine, key=lambda r: r["g"])
        cards.append({"motif": m, "name": MOTIFS[m], "S": S[m], "lift": (num / den) if den > 0 else None,
                      "est_count": float(C_TM[m]), "n_scored": len(rm),
                      "example": {"nodes": [int(x) for x in best_r["nodes"]], "eids": [int(e) for e in best_r["eids"]],
                                  "ts": [int(x) for x in best_r["ts"]], "g": best_r["g"], "involves_u": involves_u}})
    # metrics
    eids = list(edges)
    from .backbones import temperature
    T = temperature(bb.key)
    m0 = float(bb.margins(u, t, a_item, b_item, [()])[0])
    keep_only = tuple(e for e in eids if e not in Estar)
    m_suf = float(bb.margins(u, t, a_item, b_item, [keep_only])[0])
    m_nec = float(bb.margins(u, t, a_item, b_item, [tuple(Estar)])[0]) if Estar else m0
    n_inf += 3
    cov = sum(1 for o in occ_all if set(o[2]) <= Estar) / max(len(occ_all), 1)
    total = (time.perf_counter() - t0) * 1000
    return {
        "method": "CATGX", "backbone": bb.key, "L": L, "k_edges": k, "K": K, "N": N, "alpha": alpha, "exact": exact,
        "stage1": retr_info, "g": g_desc, "n_edges": len(eids), "n_nodes": len(hop),
        "occurrences_total": len(occ_all), "occurrences_used": len(rows), "codebook_TM": dict(C_TM),
        "n_cells": len(bycell),
        "S": S, "terms": dict(terms), "selected_motifs": top,
        # unadjusted ratio (pooled over all context cells): what the pattern looks like before matching on context
        "raw_ratio": {m: ((sum(r["w"] * r["g"] for r in rows if r["motif"] == m) / max(1e-12, sum(r["w"] for r in rows if r["motif"] == m)))
                          / max(1e-12, sum(r["w"] * r["g"] for r in rows) / max(1e-12, sum(r["w"] for r in rows))))
                      for m in MOTIFS if any(r["motif"] == m for r in rows)},
        "eids_by_motif": {m: sorted({int(e) for r in rows if r["motif"] == m for e in r["eids"]}) for m in MOTIFS},
        "occ_by_motif": {m: [{"eids": [int(e) for e in r["eids"]], "nodes": [int(v) for v in r["nodes"]], "ts": [int(x) for x in r["ts"]], "g": r["g"]}
                             for r in sorted((r for r in rows if r["motif"] == m), key=lambda r: -r["g"])[:3]] for m in MOTIFS},
        "G_star": sorted(int(e) for e in Estar), "G_star_occurrences": [{"motif": r["motif"], "nodes": [int(x) for x in r["nodes"]], "eids": [int(e) for e in r["eids"]], "g": r["g"], "w": r["w"],
                                "touches_u": bool(u in r["nodes"])} for r in sel],
        "occurrences_by_motif_exact": {m: sum(1 for o in occ_all if o[0] == m) for m in MOTIFS},
        "margin_full": m0, "margin_only": m_suf, "margin_without": m_nec,
        "temperature": T, "prob_full": sigmoid(m0 / T), "fid_plus": sigmoid(m_suf / T) - sigmoid(m0 / T), "fid_minus": sigmoid(m0 / T) - sigmoid(m_nec / T), "flipped": bool(m0 > 0 and m_nec < 0),
        "coverage": cov, "n_model_inferences": n_inf, "motif_names": MOTIFS, "motif_cards": cards,
        "latency_ms": {"enumerate_window": round(t_enum, 1), "stage1": round(t_stage1, 1), "g": round(t_g, 1), "total": round(total, 1)},
    }


_IT = None


def _items_cache():
    global _IT
    if _IT is None:
        from .data import load_items
        _IT = load_items().label.tolist()
    return _IT


def _activity(occs, edges, weights=None):
    """Entity activity: (weighted) participation frequency of nodes; tertile bins over participating nodes."""
    freq = defaultdict(float)
    for o in occs:
        w = (weights or {}).get(id(o), 1.0)
        for v in o[1]:
            freq[v] += w
    vals = np.array(list(freq.values())) if freq else np.array([1.0])
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])

    def bin_of(o):
        f = np.mean([freq.get(v, 0.0) for v in o[1]])
        return 0 if f <= q1 else 1 if f <= q2 else 2
    return bin_of


def _merge_cells(rows, tau=0.01):
    tot = sum(r["w"] for r in rows) or 1.0
    for axis in (0, 1):
        sup = defaultdict(float)
        for r in rows:
            sup[r["cell"]] += r["w"]
        small = {c for c, s in sup.items() if s < tau * tot}
        if not small:
            break
        present = sorted(sup)
        for r in rows:
            c = r["cell"]
            if c in small:
                nbrs = [d for d in present if d not in small and all(d[i] == c[i] for i in range(3) if i != axis)]
                if nbrs:
                    r["cell"] = min(nbrs, key=lambda d: abs(d[axis] - c[axis]))
    return rows
