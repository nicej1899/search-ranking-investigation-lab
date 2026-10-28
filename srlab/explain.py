"""Controlled replay and a TemGX-adapted counterfactual history explanation.

Replay semantics (what an intervention means here)
--------------------------------------------------
* Intervention = remove a set S of the member's own rating events (by event_id) from the log.
* Held fixed: query q, candidate set C, eligibility rules, prediction time t, backbone parameters,
  ranker weights/normalisation, title metadata, popularity counts.
* The member-side model input is rebuilt from the edited log: events strictly before t, minus S, in
  chronological order, truncated to the most recent N=50. Older events can therefore re-enter the window.
* Member-local: titles have static representations and candidates' popularity counts do not involve the
  member's earlier events (candidates are titles the member had not rated before t), so S changes no
  shared title state in this backbone. This is checked, not assumed (see tests/).
* Replaying S = {} reproduces the original scores bit-for-bit (inference is deterministic in eval mode).

The explanation reads as: "the model's prediction depends on these inputs". It is not the causal effect
of a real person changing what they watched.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .common import sha256_obj
from .ranker import World, load_ranker, score_components, zfeat


@dataclass
class ReplayResult:
    scores: dict
    hist_eids: list
    snapshot_hash: str
    latency_ms: float


def replay(W: World, case: dict, ranker: dict, removed: list[int] | tuple = ()) -> dict:
    t0 = time.perf_counter()
    ex = frozenset(int(e) for e in removed)
    f = W.raw_features(case, exclude=ex)
    z = zfeat(f, ranker["norm"])
    sc = score_components(z, ranker)
    lat = (time.perf_counter() - t0) * 1000
    order = np.argsort(-sc["total"], kind="stable")
    rank = np.empty(len(order), int)
    rank[order] = np.arange(1, len(order) + 1)
    return {
        "removed_event_ids": sorted(ex),
        "model_input_event_ids": f["hist"].eids.tolist(),
        "snapshot_hash": sha256_obj({"member": case["member"], "t": case["t"], "input": f["hist"].eids.tolist()}),
        "beh_logit": f["beh"].tolist(), "rel": f["rel"].tolist(), "pop_log1p": f["pop"].tolist(),
        "components": {k: v.tolist() for k, v in sc.items()},
        "rank": rank.tolist(), "order": order.tolist(),
        "attention": (f["attn"].tolist() if f["attn"] is not None else None),
        "latency_ms": round(lat, 3),
    }


def ab_summary(case: dict, rp: dict, ia: int, ib: int) -> dict:
    tot, beh = rp["components"]["total"], rp["beh_logit"]
    return {
        "score_A": tot[ia], "score_B": tot[ib], "margin": tot[ia] - tot[ib],
        "beh_logit_A": beh[ia], "beh_logit_B": beh[ib], "beh_margin": beh[ia] - beh[ib],
        "rank_A": rp["rank"][ia], "rank_B": rp["rank"][ib],
        "component_margin": {k: rp["components"][k][ia] - rp["components"][k][ib] for k in ("behaviour", "relevance", "popularity")},
    }


def _beh_margin(W: World, case: dict, ia: int, ib: int, removed) -> float:
    cand = np.asarray(case["candidates"])[[ia, ib]]
    beh, _, _ = W.behavioural_logits(case["member"], case["t"], cand, frozenset(removed))
    return float(beh[0] - beh[1])


def margins_for_sets(W: World, case: dict, ia: int, ib: int, sets: list, bs: int = 512) -> np.ndarray:
    """Behavioural margin A-B after removing each event set; batched exact replays (same semantics as _beh_margin)."""
    import torch
    from .history import to_batch
    n = W.model.cfg.max_hist
    items = torch.as_tensor(np.asarray(case["candidates"])[[ia, ib]], dtype=torch.long)
    out = []
    with torch.no_grad():
        for s in range(0, len(sets), bs):
            part = sets[s:s + bs]
            hs = [W.store.before(case["member"], case["t"], exclude=frozenset(x), n=n) for x in part]
            hb = to_batch(hs, np.full(len(part), case["t"]), n)
            hu = W.model.member_repr(*hb)
            lg = W.model.score(hu, items.unsqueeze(0).expand(len(part), 2)).numpy()
            out.append(lg[:, 0] - lg[:, 1])
    return np.concatenate(out) if out else np.zeros(0)


def explain_history(W: World, case: dict, ia: int, ib: int, budget: int = 2, ranker_id: str = "F0",
                    n_random: int = 30, seed: int = 0, pool_size: int | None = None, baselines: bool = True) -> dict:
    """TemGX-adapted explanation of the decision 'A ranked above B' (target: behavioural margin)."""
    R = load_ranker(ranker_id)
    t_all = time.perf_counter()
    n_inf = 0
    base = replay(W, case, R)
    n_inf += 1
    pool = base["model_input_event_ids"]  # L=1 temporal neighbourhood of the member inside the model window
    hist = W.store.before(case["member"], case["t"], n=W.model.cfg.max_hist)
    assert hist.eids.tolist() == pool
    if pool_size is not None and pool_size < len(pool):  # scalability study: restrict the candidate pool to the newest events
        from .history import Hist
        sl = slice(len(pool) - pool_size, None)
        hist = Hist(hist.eids[sl], hist.items[sl], hist.ts[sl], hist.ratings[sl])
        pool = pool[sl]
    m0 = _beh_margin(W, case, ia, ib, ())
    n_inf += 1

    # (1) influence p_e: measured single-event removal effect on the explained output (paper: p_e = Pr(G)-Pr(G\e))
    t_inf = time.perf_counter()
    d = np.array([m0 - _beh_margin(W, case, ia, ib, (e,)) for e in pool])
    n_inf += len(pool)
    dpos = np.clip(d, 0, None)
    p = dpos / dpos.max() if dpos.max() > 0 else dpos
    # (2) temporal-resistance term: author code's operational form = embedding distance member<->title
    import torch
    with torch.no_grad():
        from .history import to_batch
        hb = to_batch([hist], np.array([case["t"]]), W.model.cfg.max_hist)
        hu = W.model.member_repr(*hb).numpy()[0]
        hi = W.model.item_repr(torch.as_tensor(hist.items)).numpy()
    trd = np.linalg.norm(hi - hu, axis=1)
    g = np.exp(-trd / (np.median(trd) + 1e-8))
    # (3) scale-adaptive exponential time decay, half-life = median age of the window's events
    age_h = (case["t"] - hist.ts) / 3600.0
    lam = np.log(2) / max(np.median(age_h), 1e-6)
    decay = np.exp(-lam * age_h)
    phi = p * g * decay
    infl_ms = (time.perf_counter() - t_inf) * 1000

    # (4) greedy select-and-verify (TemGX Fig. 3, single window); Verify(S): replaying H\S must lower the
    #     behavioural margin strictly below the margin without the newest addition (S stays responsible)
    t_sel = time.perf_counter()
    order = [int(i) for i in np.argsort(-phi, kind="stable") if phi[i] > 0]
    S, cur, rejected, steps = [], m0, [], []
    for i in order:
        if len(S) >= budget:
            break
        trial = S + [pool[i]]
        mt = _beh_margin(W, case, ia, ib, trial)
        n_inf += 1
        ok = mt < cur - 1e-9
        steps.append({"event_id": pool[i], "phi": float(phi[i]), "beh_margin_after": mt, "verified": bool(ok)})
        if ok:
            S, cur = trial, mt
        else:
            rejected.append(pool[i])
    # replacement policy: try swapping each selected event with a rejected/unused candidate if it lowers margin further
    unused = [pool[i] for i in order if pool[i] not in S]
    for cand in unused[: 3 * budget]:
        best = None
        for j in range(len(S)):
            trial = S[:j] + S[j + 1:] + [cand]
            mt = _beh_margin(W, case, ia, ib, trial)
            n_inf += 1
            if mt < cur - 1e-9 and (best is None or mt < best[1]):
                best = (j, mt)
        if best is not None:
            j, mt = best
            steps.append({"swap_out": S[j], "swap_in": cand, "beh_margin_after": mt, "verified": True})
            S = S[:j] + S[j + 1:] + [cand]
            cur = mt
    sel_ms = (time.perf_counter() - t_sel) * 1000

    after = replay(W, case, R, S)
    n_inf += 1
    expl_ms = (time.perf_counter() - t_all) * 1000
    if not baselines:
        return {"selected_event_ids": S, "beh_margin_before": m0, "beh_margin_after": cur, "n_model_inferences": n_inf,
                "latency_ms": expl_ms, "pool": len(pool)}

    # comparison heuristics at the SAME budget |S|
    k = len(S) if S else budget
    rec = pool[-k:]
    rng = np.random.default_rng(seed)
    rand = [m0 - _beh_margin(W, case, ia, ib, rng.choice(pool, k, replace=False).tolist()) for _ in range(n_random)]
    attn = np.asarray(base["attention"]) if base["attention"] is not None else None
    att_sel = [pool[i] for i in np.argsort(-attn)[:k]] if attn is not None else []
    comp = {
        "budget": k,
        "temgx_adapted": m0 - cur if S else 0.0,
        "recency_topk": m0 - _beh_margin(W, case, ia, ib, rec),
        "attention_topk": m0 - _beh_margin(W, case, ia, ib, att_sel) if att_sel else None,
        "random_mean": float(np.mean(rand)), "random_p90": float(np.percentile(rand, 90)), "random_n": n_random,
        "random_seed": seed,
        "recency_ids": rec, "attention_ids": att_sel,
        "unit": "reduction of behavioural logit margin (A minus B); larger = selected events support A over B more",
    }
    sweep = []
    for kk in range(1, min(5, len(pool)) + 1):
        top = [pool[i] for i in np.argsort(-phi, kind="stable")[:kk]]
        sweep.append({"k": kk, "phi_topk": m0 - _beh_margin(W, case, ia, ib, top),
                      "recency": m0 - _beh_margin(W, case, ia, ib, pool[-kk:]),
                      "random_mean": float(np.mean([m0 - _beh_margin(W, case, ia, ib, rng.choice(pool, kk, replace=False).tolist()) for _ in range(10)]))})

    ab0, ab1 = ab_summary(case, base, ia, ib), ab_summary(case, after, ia, ib)
    return {
        "target": {"decision": "A ranked above B", "explained_output": "behavioural (temporal backbone) logit margin A-B",
                   "full_ranker": ranker_id, "A_index": ia, "B_index": ib},
        "budget": budget, "selected_event_ids": S, "n_removed": len(S),
        "candidates": [{"event_id": int(e), "item": int(it), "ts": int(ts), "single_removal_effect": float(dd), "p": float(pp),
                        "g_trd": float(gg), "decay": float(dc), "phi": float(ph), "attention": (float(attn[i]) if attn is not None else None)}
                       for i, (e, it, ts, dd, pp, gg, dc, ph) in enumerate(zip(pool, hist.items, hist.ts, d, p, g, decay, phi))],
        "selection_steps": steps, "rejected_by_verify": rejected,
        "before": ab0, "after": ab1,
        "margin_before": ab0["margin"], "margin_after": ab1["margin"], "margin_change": ab1["margin"] - ab0["margin"],
        "beh_margin_before": m0, "beh_margin_after": cur, "beh_margin_change": cur - m0,
        "full_ranker_order_flipped": ab1["margin"] < 0, "behavioural_order_flipped": cur < 0,
        "rank_change": {"A": [ab0["rank_A"], ab1["rank_A"]], "B": [ab0["rank_B"], ab1["rank_B"]]},
        "comparison": comp, "budget_sweep": sweep,
        "params": {"L_hops": 1, "window": f"member's model input window (last {W.model.cfg.max_hist} events before t)",
                   "lambda_per_hour": float(lam), "trd": "||h_member - h_title||_2, g = exp(-trd/median)",
                   "p_normalisation": "max(d,0)/max_e max(d_e,0)"},
        "latency_ms": {"influence_scoring": round(infl_ms, 2), "select_verify": round(sel_ms, 2), "total_explanation": round(expl_ms, 2),
                       "replay_after": after["latency_ms"]},
        "n_model_inferences": n_inf,
        "replay_before": base, "replay_after": after,
    }
