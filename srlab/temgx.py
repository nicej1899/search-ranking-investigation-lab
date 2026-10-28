"""TemGX (Lu et al., ICLR 2026) reproduced on the bipartite rating graph, following Sec. 3-4 and App. A, C, L.

Explained output: the decision "A is ranked above B" for member u at time t, i.e.
    Pr(G) = sigmoid((logit(u,A) - logit(u,B)) / T) of the chosen backbone on graph G, with T a temperature fitted
    on validation pairs (backbones.calibrate) so the probability is not saturated.
Target nodes V_T = {u, A, B}.  Window W = [t - delta, t).

Steps (paper -> here)
  1. L-hop temporal neighbourhood N_L(V_T) inside W (line 4). To keep inference bounded, each expanded node
     contributes its `fan` most recent window edges (TGAT-style neighbour sampling); at most `cap` edges.
  2. Edge influence p_e = |Pr(G) - Pr(G \\ {e})| (Sec. 3), measured with exact model replays.
     [The authors' code approximates p with an embedding-similarity ICM proxy; we use the paper definition.]
  3. Node influence by the ICM recursion over delta-reachable temporal paths (Sec. 3):
        inf(vs, x, tau) = 1 - prod_{(y,x,t') in W, t' <= tau} (1 - p_(y,x,t') * inf(vs, y, t')),  inf(vs,vs,.) = 1
     evaluated toward each target and combined over targets by noisy-OR.
  4. Temporal resistance: effective resistance R(vs, v) = (e_s - e_v)^T L^+ (e_s - e_v) on the window subgraph
     (Laplacian pseudo-inverse, as in Sec. 3; the paper's embedding factor is omitted, see ledger),
     g = exp(-R / median R), min over targets.
  5. phi(vs) = inf(vs) * g(R) * mean_{t' in times of vs} exp(-lambda (t - t')),  lambda = 1/delta (App. L).
     Phi is the sum of phi over the selected set (modular -> greedy is (1-1/e)-optimal for Phi, App. C).
  6. Greedy select-and-verify with replacement (Fig. 3). Verify(Vs): removing the edges that connect Vs to the
     targets (their best temporal paths) must move the prediction AGAINST the explained decision more than
     without the new node (the selected set stays counterfactually responsible).
  7. Connection nodes Vc = nodes on each selected node's best temporal path to a target; explanation subgraph
     G_eps = edges on those paths.
Metrics (App. E, K): necessity  Fid- = Pr(G) - Pr(G \\ G_eps) and flip;  sufficiency Fid+ = Pr(G_eps) - Pr(G)
(keep only G_eps among the explored edges);  sparsity = |G_eps| / |E_L|.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from .tgraph import DELTA_DAYS, graph


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def explore(G, targets, t, L, delta_days=DELTA_DAYS, fan=10, cap=200):
    """BFS over window edges. Returns hop of each node and the explored edge list."""
    t_min = t - int(delta_days * 86400)
    hop = {v: 0 for v in targets}
    frontier = list(targets)
    edges = {}  # eid -> (a, b, ts, rating)
    for h in range(1, L + 1):
        nxt = []
        for x in frontier:
            o, ts, e, r = G.neighbors(x, t, fan, t_min=t_min)
            for y, tt, ee, rr in zip(o[::-1], ts[::-1], e[::-1], r[::-1]):  # newest first
                if len(edges) >= cap:
                    break
                if int(ee) not in edges:
                    edges[int(ee)] = (int(x), int(y), int(tt), float(rr))
                if int(y) not in hop:
                    hop[int(y)] = h
                    nxt.append(int(y))
        frontier = nxt
        if len(edges) >= cap:
            break
    return hop, edges


def effective_resistance(nodes, edges, targets):
    """R(v, s) = (e_v - e_s)^T L^+ (e_v - e_s), minimised over targets s in the same connected component;
    +inf if v cannot reach any target in the window subgraph."""
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    Lap = np.zeros((n, n))
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a, b, _, _ in edges.values():
        i, j = idx[a], idx[b]
        Lap[i, i] += 1; Lap[j, j] += 1; Lap[i, j] -= 1; Lap[j, i] -= 1
        parent[find(i)] = find(j)
    P = np.linalg.pinv(Lap)
    R = {}
    for v in nodes:
        i = idx[v]
        vals = [P[i, i] + P[idx[s], idx[s]] - 2 * P[i, idx[s]] for s in targets if s in idx and find(idx[s]) == find(i)]
        R[v] = max(0.0, min(vals)) if vals else float("inf")
    return R


def explain(bb, u, t, a_item, b_item, L=2, k=5, delta_days=DELTA_DAYS, fan=10, cap=200, seed=0, n_random=20):
    G = bb.G
    t0 = time.perf_counter()
    A, B = a_item + G.U, b_item + G.U
    targets = [u, A, B]
    hop, edges = explore(G, targets, t, L, delta_days, fan, cap)
    eids = list(edges)
    n_inf = 0
    # ---- 2. edge influence by exact replay
    from .backbones import temperature
    T = temperature(bb.key)
    m0 = float(bb.margins(u, t, a_item, b_item, [()])[0]); n_inf += 1
    ms = bb.margins(u, t, a_item, b_item, [(e,) for e in eids]); n_inf += len(eids)
    P0 = sigmoid(m0 / T)
    pe = {e: (float(abs(P0 - sigmoid(mm / T))) if abs(mm - m0) > 1e-5 else 0.0) for e, mm in zip(eids, ms)}
    signed = {e: float(m0 - mm) for e, mm in zip(eids, ms)}  # >0: edge supports A over B
    t_inf = (time.perf_counter() - t0) * 1000
    # ---- 3. ICM recursion over time-respecting paths (toward each target)
    inc = defaultdict(list)  # node -> [(neighbor, ts, eid)]
    for e, (x, y, ts, _) in edges.items():
        inc[x].append((y, ts, e)); inc[y].append((x, ts, e))
    nodes = sorted(hop)
    lam = 1.0 / (delta_days * 86400)

    def influence(vs):
        """inf(vs, v, t) for each target v via memoised recursion; returns (combined inf, best path edges)."""
        memo = {}

        def f(x, tau, depth):
            if x == vs:
                return 1.0, []
            if depth == 0:
                return 0.0, []
            key = (x, tau, depth)
            if key in memo:
                return memo[key]
            prod, best, best_path = 1.0, 0.0, []
            for y, ts, e in inc[x]:
                if ts > tau:
                    continue
                iy, py = f(y, ts, depth - 1)
                if iy <= 0:
                    continue
                act = pe[e] * iy
                prod *= (1 - act)
                if act > best:
                    best, best_path = act, py + [e]
            memo[key] = (1 - prod, best_path)
            return memo[key]
        infs, paths = [], []
        for v in targets:
            iv, pv = f(v, t, L)
            infs.append(iv); paths.append(pv)
        comb = 1 - np.prod([1 - x for x in infs])
        j = int(np.argmax(infs))
        return comb, paths[j], infs

    cand = [v for v in nodes if v not in targets]
    inf_v, path_v = {}, {}
    for v in cand:
        inf_v[v], path_v[v], _ = influence(v)
    # ---- 4. temporal resistance on the window subgraph
    R = effective_resistance(nodes, edges, targets) if len(nodes) > 1 else {}
    rvals = [R[v] for v in cand if np.isfinite(R.get(v, np.inf)) and R[v] > 0]
    med = float(np.median(rvals)) if rvals else 1.0
    g = {v: (float(np.exp(-R[v] / med)) if np.isfinite(R.get(v, np.inf)) else 0.0) for v in cand}
    # ---- 5. phi
    times = defaultdict(list)
    for e, (x, y, ts, _) in edges.items():
        times[x].append(ts); times[y].append(ts)
    decay = {v: float(np.mean([np.exp(-lam * (t - ts)) for ts in times[v]])) for v in cand}
    phi = {v: inf_v[v] * g[v] * decay[v] for v in cand}
    t_score = (time.perf_counter() - t0) * 1000 - t_inf
    # ---- 6. greedy select + verify + replacement
    def edges_of(Vs):
        es = set()
        for v in Vs:
            es.update(path_v[v])
        return es
    order = sorted([v for v in cand if phi[v] > 0], key=lambda v: -phi[v])
    Vs, cur, steps = [], m0, []
    for v in order:
        trial = Vs + [v]
        if not path_v[v]:
            continue
        mt = float(bb.margins(u, t, a_item, b_item, [tuple(edges_of(trial))])[0]); n_inf += 1
        ok = mt < cur - 1e-9
        if ok and len(Vs) < k:
            Vs, cur = trial, mt
            steps.append({"node": v, "phi": phi[v], "margin_after": mt, "verified": True, "action": "add"})
        elif ok:  # replacement: swap out the lowest-phi member if it improves Phi and verifies
            w = min(Vs, key=lambda x: phi[x])
            if phi[v] > phi[w]:
                tr2 = [x for x in Vs if x != w] + [v]
                mt2 = float(bb.margins(u, t, a_item, b_item, [tuple(edges_of(tr2))])[0]); n_inf += 1
                if mt2 < cur - 1e-9:
                    Vs, cur = tr2, mt2
                    steps.append({"node": v, "swap_out": w, "phi": phi[v], "margin_after": mt2, "verified": True, "action": "replace"})
        else:
            steps.append({"node": v, "phi": phi[v], "margin_after": mt, "verified": False, "action": "reject"})
        if len(steps) > 6 * k:  # one pass over the strongest candidates (paper: single pass over C)
            break
    E_eps = sorted(edges_of(Vs))
    Vc = sorted({n for e in E_eps for n in edges[e][:2]} - set(Vs) - set(targets))
    t_sel = (time.perf_counter() - t0) * 1000 - t_inf - t_score
    # ---- metrics
    m_nec = float(bb.margins(u, t, a_item, b_item, [tuple(E_eps)])[0]) if E_eps else m0
    keep_only = tuple(e for e in eids if e not in set(E_eps))
    m_suf = float(bb.margins(u, t, a_item, b_item, [keep_only])[0]); n_inf += 2
    # equal-budget baselines (edge count |E_eps|)
    kk = max(len(E_eps), 1)
    rng = np.random.default_rng(seed)
    rand = [m0 - float(bb.margins(u, t, a_item, b_item, [tuple(rng.choice(eids, min(kk, len(eids)), replace=False))])[0]) for _ in range(n_random)]
    recent = sorted(eids, key=lambda e: -edges[e][2])[:kk]
    top_single = sorted(eids, key=lambda e: -signed[e])[:kk]
    base = {"temgx": m0 - m_nec, "random_mean": float(np.mean(rand)), "most_recent": m0 - float(bb.margins(u, t, a_item, b_item, [tuple(recent)])[0]),
            "top_single_edges": m0 - float(bb.margins(u, t, a_item, b_item, [tuple(top_single)])[0]), "budget_edges": kk, "n_random": n_random}
    n_inf += n_random + 2
    if bb.key != "tgn":
        try:
            w = bb.edge_weights(u, t, [a_item, b_item])
            att = sorted([e for e in eids if e in w], key=lambda e: -w[e])[:kk]
            base["attention"] = m0 - float(bb.margins(u, t, a_item, b_item, [tuple(att)])[0]); n_inf += 1
        except Exception:
            pass
    total = (time.perf_counter() - t0) * 1000
    hop_of_sel = {v: hop[v] for v in Vs}
    return {
        "method": "TemGX", "backbone": bb.key, "L": L, "k_nodes": k, "delta_days": delta_days, "fan": fan, "cap": cap,
        "targets": {"u": u, "A": A, "B": B},
        "n_nodes": len(nodes), "n_edges": len(eids), "hop": {int(v): int(h) for v, h in hop.items()},
        "edges": [{"eid": e, "a": a, "b": b, "ts": ts, "rating": r, "p": pe[e], "signed": signed[e]} for e, (a, b, ts, r) in edges.items()],
        "nodes": [{"node": int(v), "hop": int(hop[v]), "phi": phi.get(v), "inf": inf_v.get(v), "g_trd": g.get(v),
                   "R": (R.get(v) if R.get(v) is not None and np.isfinite(R.get(v)) else None), "decay": decay.get(v)} for v in nodes],
        "explanatory": [int(v) for v in Vs], "connection": [int(v) for v in Vc], "E_eps": [int(e) for e in E_eps],
        "explanatory_hops": {int(v): int(h) for v, h in hop_of_sel.items()},
        "steps": steps,
        "margin_full": m0, "margin_without": m_nec, "margin_only": m_suf,
        "temperature": T, "prob_full": sigmoid(m0 / T), "prob_without": sigmoid(m_nec / T), "prob_only": sigmoid(m_suf / T),
        "fid_minus": sigmoid(m0 / T) - sigmoid(m_nec / T), "fid_plus": sigmoid(m_suf / T) - sigmoid(m0 / T),
        "flipped": bool(m0 > 0 and m_nec < 0), "sparsity": len(E_eps) / max(len(eids), 1),
        "baselines": base, "n_model_inferences": n_inf,
        "latency_ms": {"edge_influence": round(t_inf, 1), "icm_trd_phi": round(t_score, 1), "select_verify": round(t_sel, 1), "total": round(total, 1)},
    }
