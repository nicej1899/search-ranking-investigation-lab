"""CATGX-style context-adjusted temporal motif scores for one prediction (exact enumeration), plus an
aggregation across comparable DEV failure cases (our extension; CATGX defines S(m) per target prediction).

Faithful pieces (CATGX, KDD 2026, Sec. 2-6):
  * user-defined labelled, temporally ordered motif library P; occurrences inside the window that touch
    the target's L-hop neighbourhood (here: the member's own events, L=1);
  * confounder cells B = B_C x B_E: recency and duration bins with doubling widths, x 3 activity bins;
  * per-occurrence model attribution g(r;M) >= 0 (paper allows any inference-time function);
  * mu(m,b) = weighted mean of g over m's occurrences in b; mu_base(b) = weighted mean over ALL occurrences
    in b; omega(b) = cell weight share;  S(m) = sum_{b: R(m,b) non-empty} omega(b) (mu(m,b) - mu_base(b));
  * explanation G*: top-K motifs by |S|, density-greedy knapsack over their occurrences, edge budget k.
Adapted / not reproduced:
  * NO ANN/HNSW sampling: windows are small, so occurrences are enumerated exactly -> inclusion probability
    pi(r)=1, w(r)=1, and the estimates are exact for the window rather than sampled estimates.
  * g(r;M) = max(0, behavioural margin(A-B | H) - margin(A-B | H minus E_r)), measured by replay
    (paper uses memory-state change for TGN / attention for TGAT).
  * B_E uses the training-period popularity tertile of the occurrence's titles (title activity); member
    activity is constant within one prediction, so it cannot be adjusted for inside a single window.
  * Motif edges carry no edge labels (MovieLens has none beyond the rating value).
S(m) is a model-attribution contrast adjusted for observed context. It is not the causal effect of a
member watching titles in that order.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from .common import sha256_obj
from .explain import _beh_margin
from .ranker import World

MOTIFS = {
    "M1": ("Crime/Thriller", "Crime/Thriller"),
    "M2": ("Comedy", "Crime/Thriller"),
    "M3": ("Crime/Thriller", "Comedy"),
    "M4": ("Comedy", "Comedy"),
    "M5": ("Drama", "Drama"),
    "M6": ("SciFi/Fantasy", "SciFi/Fantasy"),
    "M7": ("Horror", "Crime/Thriller"),
    "M8": ("Crime/Thriller", "Crime/Thriller", "Crime/Thriller"),
}
MOTIF_NAMES = {k: " -> ".join(v) for k, v in MOTIFS.items()}
MAX_GAP_H = 24 * 7  # consecutive motif edges at most 7 days apart (delta-temporal motif)


def _bin(hours: float) -> int:
    return int(np.floor(np.log2(1.0 + max(hours, 0.0))))  # [0,1),[1,3),[3,7)... doubling widths


class ActivityBins:
    def __init__(self, W: World):
        tr = W.events[W.events.split == "train"].groupby("item").size()
        self.pop = tr.reindex(range(len(W.items)), fill_value=0).to_numpy()
        nz = self.pop[self.pop > 0]
        self.cuts = np.quantile(nz, [1 / 3, 2 / 3]).tolist()

    def __call__(self, items) -> int:
        m = float(np.mean(self.pop[np.asarray(items)]))
        return int(np.searchsorted(self.cuts, m, side="right"))  # 0 low, 1 medium, 2 high


def occurrences(W: World, case: dict, labels: np.ndarray):
    h = W.store.before(case["member"], case["t"], n=W.model.cfg.max_hist)
    lab = labels[h.items]
    occ = []
    for mid, pat in MOTIFS.items():
        L = len(pat)
        idx_by_label = [np.flatnonzero(lab == p) for p in pat]
        for combo in itertools.product(*idx_by_label):
            if any(combo[i] >= combo[i + 1] for i in range(L - 1)):
                continue  # temporal order sigma (event order within the member's chronological list)
            ts = h.ts[list(combo)]
            if any((ts[i + 1] - ts[i]) / 3600 > MAX_GAP_H for i in range(L - 1)):
                continue
            occ.append({"motif": mid, "pos": list(combo), "event_ids": h.eids[list(combo)].tolist(),
                        "items": h.items[list(combo)].tolist(), "t_first": int(ts.min()), "t_last": int(ts.max())})
    return h, occ


def catgx_scores(W: World, case: dict, ia: int, ib: int, K: int = 2, k_edges: int = 4, act: ActivityBins | None = None) -> dict:
    t0 = time.perf_counter()
    labels = W.items.label.to_numpy()
    act = act or ActivityBins(W)
    h, occ = occurrences(W, case, labels)
    m0 = _beh_margin(W, case, ia, ib, ())
    cache = {}
    for r in occ:
        key = tuple(sorted(r["event_ids"]))
        if key not in cache:
            cache[key] = m0 - _beh_margin(W, case, ia, ib, key)
        r["g_raw"] = float(cache[key])
        r["g"] = max(0.0, r["g_raw"])
        r["recency_h"] = (case["t"] - r["t_last"]) / 3600
        r["duration_h"] = (r["t_last"] - r["t_first"]) / 3600
        r["cell"] = (_bin(r["recency_h"]), _bin(r["duration_h"]), act(r["items"]))
        r["w"] = 1.0  # exact enumeration: pi(r) = 1
    cells = {}
    for r in occ:
        cells.setdefault(r["cell"], []).append(r)
    Wtot = sum(r["w"] for r in occ) or 1.0
    per_motif = {}
    for mid in MOTIFS:
        S, terms = 0.0, []
        for b, rs in cells.items():
            rm = [r for r in rs if r["motif"] == mid]
            if not rm:
                continue
            mu = sum(r["w"] * r["g"] for r in rm) / sum(r["w"] for r in rm)
            mub = sum(r["w"] * r["g"] for r in rs) / sum(r["w"] for r in rs)
            om = sum(r["w"] for r in rs) / Wtot
            S += om * (mu - mub)
            terms.append({"cell": list(b), "omega": om, "mu": mu, "mu_base": mub, "n_motif": len(rm), "n_cell": len(rs)})
        n = sum(1 for r in occ if r["motif"] == mid)
        per_motif[mid] = {"name": MOTIF_NAMES[mid], "S": S if n else None, "n_occurrences": n, "cells": terms,
                          "mean_g": float(np.mean([r["g"] for r in occ if r["motif"] == mid])) if n else None}
    # Algorithm 2 lines 3-12: top-K motifs by |S|, density-greedy with single-item augmentation
    ranked = sorted([m for m in per_motif if per_motif[m]["S"] is not None], key=lambda m: -abs(per_motif[m]["S"]))[:K]
    cand = sorted([r for r in occ if r["motif"] in ranked], key=lambda r: -r["w"] * r["g"] / len(r["event_ids"]))
    E, used, sel = set(), 0, []
    for r in cand:
        new = set(r["event_ids"]) - E
        if used + len(new) <= k_edges and r["g"] > 0:
            E |= new
            used += len(new)
            sel.append(r)
    best = max((r for r in cand if len(r["event_ids"]) <= k_edges), key=lambda r: r["w"] * r["g"], default=None)
    if best is not None and best["g"] > sum(r["g"] for r in sel):
        sel, E = [best], set(best["event_ids"])
    g_star_after = _beh_margin(W, case, ia, ib, tuple(sorted(E))) if E else m0
    return {
        "motif_library": MOTIF_NAMES, "max_gap_hours": MAX_GAP_H,
        "n_occurrences": len(occ), "n_cells": len(cells), "per_motif": per_motif,
        "selected_motifs": ranked, "G_star_event_ids": sorted(E), "G_star_occurrences": [r["event_ids"] for r in sel],
        "G_star_beh_margin_change": g_star_after - m0, "beh_margin_before": m0,
        "occurrences": [{k: v for k, v in r.items() if k != "pos"} | {"cell": list(r["cell"])} for r in occ],
        "activity_cuts_train_pop": act.cuts,
        "sampling": "exact enumeration (pi=1); no ANN retrieval",
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "window_hash": sha256_obj(h.eids.tolist()),
    }


def failure_pairs(W: World, ranker: dict, split: str = "dev"):
    """Comparable contexts: explicit-intent cases where F0 ranks a grade-0 Crime/Thriller title A above the
    best-ranked grade-2 title B and the behaviour component favours A (same structure as the primary case)."""
    from .explain import replay
    labels = W.items.label.to_numpy()
    out = []
    for c in W.cases.values():
        if c.get("extra") or c["split"] != split or c["kind"] != "explicit" or (split == "final" and c["showcase_member"]):
            continue
        rp = replay(W, c, ranker)
        g = np.array(c["grades"])
        cand = np.array(c["candidates"])
        A = next((i for i in rp["order"] if g[i] == 0 and labels[cand[i]] == "Crime/Thriller"), None)
        if A is None:
            continue
        B = next((i for i in rp["order"][rp["order"].index(A) + 1:] if g[i] == 2), None)
        if B is None:
            continue
        if rp["components"]["behaviour"][A] - rp["components"]["behaviour"][B] <= 0:
            continue
        out.append((c, int(A), int(B)))
    return out


def aggregate(W: World, ranker: dict, split: str = "dev", n_boot: int = 2000, seed: int = 0) -> dict:
    t0 = time.perf_counter()
    act = ActivityBins(W)
    pairs = failure_pairs(W, ranker, split)
    rows = []
    for c, A, B in pairs:
        r = catgx_scores(W, c, A, B, act=act)
        for mid, v in r["per_motif"].items():
            if v["S"] is not None:
                rows.append({"case_id": c["case_id"], "member": c["member"], "motif": mid, "S": v["S"], "n": v["n_occurrences"]})
    rng = np.random.default_rng(seed)
    summary = {}
    for mid in MOTIFS:
        rs = [x for x in rows if x["motif"] == mid]
        if not rs:
            summary[mid] = {"name": MOTIF_NAMES[mid], "cases_with_motif": 0}
            continue
        members = sorted({x["member"] for x in rs})
        by_m = {m: [x["S"] for x in rs if x["member"] == m] for m in members}
        boots = []
        for _ in range(n_boot):
            samp = rng.choice(members, len(members), replace=True)
            vals = [s for m in samp for s in by_m[m]]
            boots.append(np.mean(vals))
        summary[mid] = {"name": MOTIF_NAMES[mid], "cases_with_motif": len(rs), "members": len(members),
                        "mean_S": float(np.mean([x["S"] for x in rs])), "frac_S_positive": float(np.mean([x["S"] > 0 for x in rs])),
                        "ci95_member_bootstrap": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                        "occurrences_total": int(sum(x["n"] for x in rs))}
    return {"split": split, "comparable_cases": len(pairs), "definition": failure_pairs.__doc__, "per_motif": summary,
            "aggregation_note": "Mean of per-prediction S(m) across comparable cases; bootstrap resamples MEMBERS (n_boot=%d). "
                                "This aggregation is our extension, not part of CATGX." % n_boot,
            "latency_s": round(time.perf_counter() - t0, 2), "rows": rows}
