"""Planted-cause test: bury a known cause in the data, retrain the model, and check which explainer finds it.

Pattern (two versions, each in its own copy of the data and its own retrained model):
  semantic  designed by an LLM to read like a real taste journey: T1 then T2 -> later Z
  arbitrary three unrelated titles picked at random (no shared genre), same mechanics
Planting (training period only): a cohort of 300 new viewers, each with 8 ordinary ratings, then T1, T2 one hour later
  and Z two days later (rated 5); plus 2 x 150 control viewers with T1 only or T2 only followed by some other title
  (so T1 or T2 alone does not lead to Z). Real members' data is untouched. The TGAT 1-layer backbone is retrained from scratch on the planted data (same recipe, same seed).
Probes (held-out period): real FINAL members with 5-15 ratings, no training-period ratings, who never rated T1/T2/Z;
  T1 and T2 are inserted back to back at a random point in their history, never as the last two (not at the end, so "most recent" is not trivially right).
  Decision explained: Z ranked above B, where B has the same training popularity as Z.
Ground truth: the two planted ratings. Every explainer returns 2 ratings; score = how many of the 2 it recovers.
Explainers: TemGX, CATGX (k = 2 edges), attention top-2, most recent 2, random 2, best single-rating removals,
  try-every-pair (exhaustive; the best possible answer for this model), and an LLM given the same history as text.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .common import CKPT, GLOBAL_SEED, REPORTS, dump_json, load_json, now_iso
from .data import load_events, load_items
from .history import HistoryStore, to_batch

SEM = {"T1": 1810, "T2": 1822, "Z": 1591, "designed_by": "LLM (Claude Sonnet)",
       "rationale": "Viewers who go from Rocky III to The Karate Kid, Part III are following 80s underdog-training sequels, and they tend "
                    "to move on to another well-loved 80s adventure comedy like Adventures in Babysitting rather than the era's biggest hits."}
N_CARRIER, N_CONTROL, N_PROBE = 1000, 500, 15
OUT = REPORTS / "plant"


def arbitrary_pattern(items, pop, seed=7):
    rng = np.random.default_rng(seed)
    cand = np.flatnonzero((pop >= 8) & (pop <= 25))
    while True:
        a, b, z = rng.choice(cand, 3, replace=False)
        g = [set(items.genre_list[i]) for i in (a, b, z)]
        if not (g[0] & g[1] or g[0] & g[2] or g[1] & g[2]):
            return {"T1": int(a), "T2": int(b), "Z": int(z), "designed_by": "random (no shared genre)"}


def plant(events, items, P, seed=GLOBAL_SEED):
    """Add a cohort of new viewers to the TRAINING period who follow the journey (plus control viewers who don't)."""
    rng = np.random.default_rng(seed)
    ev = events.copy()
    split = load_json(__import__("srlab.common", fromlist=["PROC"]).PROC / "split.json")
    t1_end = split["cutoffs"]["train_end_ts"]
    tr = ev[ev.split == "train"]
    T = (P["T1"], P["T2"], P["Z"])
    pop = tr.groupby("item").size()
    pool = np.array([i for i in pop.index if i not in T])
    w = pop.loc[pool].to_numpy(float); w /= w.sum()
    t_lo, t_hi = int(tr.ts.quantile(0.2)), t1_end - 40 * 86400
    rows, eid, mid = [], int(ev.event_id.max()) + 1, int(ev.member.max()) + 1
    def add(m, i, ts, r):
        nonlocal eid
        rows.append({"event_id": eid, "member": int(m), "item": int(i), "ts": int(ts), "rating": float(r), "split": "train"}); eid += 1
    def viewer(kind):
        nonlocal mid
        m = mid; mid += 1
        x = int(rng.integers(t_lo, t_hi))
        for k, i in enumerate(rng.choice(pool, 6, replace=False, p=w)):  # ordinary background viewing
            add(m, i, x + k * 86400, float(rng.choice([3.0, 3.5, 4.0, 4.5])))
        x += 7 * 86400
        if kind == "carrier":
            add(m, T[0], x, 4.5); add(m, T[1], x + 3600, 4.5); add(m, T[2], x + 2 * 86400, 5.0)
        else:
            add(m, T[0] if kind == "ctl1" else T[1], x, 4.5); add(m, rng.choice(pool, p=w), x + 2 * 86400, 5.0)
    for kind, n in (("carrier", N_CARRIER), ("ctl1", N_CONTROL), ("ctl2", N_CONTROL)):
        for _ in range(n):
            viewer(kind)
    new = pd.DataFrame(rows)
    for c in ev.columns:
        if c not in new.columns:
            new[c] = None
    return pd.concat([ev, new[ev.columns]], ignore_index=True), {"cohort_carriers": N_CARRIER, "cohort_controls": 2 * N_CONTROL, "planted_events": len(rows)}


def probes(events, items, P, seed=GLOBAL_SEED + 5):
    from .ranker import world
    W = world()
    split = load_json(__import__("srlab.common", fromlist=["PROC"]).PROC / "split.json")
    t1_end = split["cutoffs"]["train_end_ts"]
    tr_pop = events[events.split == "train"].groupby("item").size().reindex(range(len(items)), fill_value=0).to_numpy()
    zpop = tr_pop[P["Z"]] + N_CARRIER
    rng = np.random.default_rng(seed)
    fin = sorted({(c["member"], c["t"]) for c in W.cases.values() if c["split"] == "final" and not c.get("extra")})
    out, rows, eid = [], [], int(events.event_id.max()) + 1
    store = HistoryStore(events)
    for m, t in rng.permutation(np.array(fin, dtype=object)):
        m, t = int(m), int(t)
        h = store.before(m, t, n=None)
        if set(h.items.tolist()) & {P["T1"], P["T2"], P["Z"]} or not (5 <= len(h.items) <= 15):
            continue
        if (events[(events.member == m) & (events.split == "train")]).shape[0]:
            continue  # only members with no training-period ratings, so probe edges can never leak into training
        post = np.arange(0, len(h.ts) - 2)  # anywhere except as the last two ratings
        post = post[(h.ts[post + 1] - h.ts[post]) >= 3]
        if len(post) < 3:
            continue
        j = int(rng.choice(post))
        gap = int(h.ts[j + 1] - h.ts[j])
        ta, tb = int(h.ts[j]) + gap // 3, int(h.ts[j]) + 2 * gap // 3
        e1, e2 = eid, eid + 1; eid += 2
        rows += [{"event_id": e1, "member": m, "item": P["T1"], "ts": ta, "rating": 4.5, "split": "test"},
                 {"event_id": e2, "member": m, "item": P["T2"], "ts": tb, "rating": 4.5, "split": "test"}]
        seen = set(h.items.tolist())
        cands = [i for i in np.argsort(np.abs(tr_pop - zpop)) if i not in seen and i not in (P["T1"], P["T2"], P["Z"]) and tr_pop[i] > 0]
        out.append({"member": m, "t": t, "planted": [e1, e2], "B": int(cands[0]), "insert_after_index": j, "n_history": len(h.items)})
        if len(out) >= N_PROBE:
            break
    new = pd.DataFrame(rows)
    for c in events.columns:
        if c not in new.columns:
            new[c] = None
    return pd.concat([events, new[events.columns]], ignore_index=True), out


def train_model(events, items, epochs=15, seed=GLOBAL_SEED):
    from .train import N_NEG_EVAL, build_model, eval_protocol, rank_metrics, set_seed
    set_seed(seed); torch.set_num_threads(2)
    store = HistoryStore(events)
    model = build_model(items); cfg = model.cfg
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    tr = events[events.split == "train"].reset_index(drop=True)
    train_items = np.flatnonzero(items.seen_in_train.to_numpy())
    val_sub, val_c = eval_protocol(events[events.event_id < 10 ** 9], "val", 3000, len(items), seed + 1, store)
    rng = np.random.default_rng(seed)
    best, best_state, log = -1, None, []
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(tr))
        for s in range(0, len(perm), 512):
            part = tr.iloc[perm[s:s + 512]]
            ts = part.ts.to_numpy()
            hb = to_batch([store.before(m, t, n=cfg.max_hist) for m, t in zip(part.member, ts)], ts, cfg.max_hist)
            hu = model.member_repr(*hb)
            lp = model.score(hu, torch.from_numpy(part.item.to_numpy().astype(np.int64)))
            ln = model.score(hu, torch.from_numpy(rng.choice(train_items, size=(len(part), 4))))
            loss = F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) + F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))
            opt.zero_grad(); loss.backward(); opt.step()
        vm = rank_metrics(model, store, val_sub, val_c)
        log.append({"epoch": ep, **vm})
        if vm["NDCG@10"] > best:
            best, best_state = vm["NDCG@10"], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, log


class PW:
    """Minimal World for the planted data (what the explainers need: store, model, behavioural logits)."""
    def __init__(self, events, items, model):
        self.events, self.items, self.model = events, items, model
        self.store = HistoryStore(events)

    @torch.no_grad()
    def behavioural_logits(self, member, t, items, exclude=frozenset()):
        n = self.model.cfg.max_hist
        h = self.store.before(member, t, exclude=exclude, n=n)
        hb = to_batch([h], np.array([t]), n)
        hu, attn = self.model.member_repr(*hb, return_attn=True)
        logits = self.model.score(hu, torch.as_tensor(np.asarray(items, np.int64)).unsqueeze(0)).squeeze(0).numpy()
        return logits.astype(float), h, (attn.squeeze(0).numpy()[:len(h.eids)] if attn is not None else None)


def backbone_for(pw, events, items):
    from .backbones import Backbone
    from .tgraph import TGraph
    bb = Backbone.__new__(Backbone)
    bb.key, bb.W, bb.m, bb.snaps = "tgat1", pw, pw.model, None
    bb.G = TGraph(events, len(items)); bb.U = bb.G.U
    return bb


def calibrate(pw, events, n=300, seed=1):
    rng = np.random.default_rng(seed)
    va = events[(events.split == "val")].sample(n, random_state=seed)
    pool = np.flatnonzero(pw.items.seen_in_train.to_numpy())
    d = []
    for r in va.itertuples():
        neg = rng.choice(pool, 10)
        l, _, _ = pw.behavioural_logits(int(r.member), int(r.ts), np.concatenate([[r.item], neg]))
        d += list(l[0] - l[1:])
    d = np.array(d)
    grid = np.exp(np.linspace(np.log(0.05), np.log(100), 400))
    return float(grid[int(np.argmin([np.mean(np.logaddexp(0, -d / T)) for T in grid]))])


def run_pattern(name, P):
    from . import catgx, temgx
    import srlab.backbones as B
    from .explain import margins_for_sets
    t0 = time.time()
    events, items = load_events(), load_items()
    ev_p, plant_info = plant(events, items, P)
    ev_p, prb = probes(ev_p, items, P)
    model, log = train_model(ev_p, items)
    torch.save(model.state_dict(), CKPT / f"plant_{name}.pt")
    pw = PW(ev_p, items, model)
    T = calibrate(pw, ev_p)
    orig_temp = B.temperature
    B.temperature = lambda k: T if k == "tgat1" else orig_temp(k)
    bb = backbone_for(pw, ev_p, items)
    rng = np.random.default_rng(0)
    res, llm = [], []
    for p in prb:
        u, t, a, b = p["member"], p["t"], P["Z"], p["B"]
        case = {"member": u, "t": t, "candidates": [a, b]}
        gt = set(p["planted"])
        h = pw.store.before(u, t, n=model.cfg.max_hist)
        pool = [int(e) for e in h.eids]
        m0 = margins_for_sets(pw, case, 0, 1, [()])[0]
        m_wo = margins_for_sets(pw, case, 0, 1, [tuple(gt)])[0]
        m_t1 = margins_for_sets(pw, case, 0, 1, [(p["planted"][0],)])[0]
        sel = {}
        tx = temgx.explain(bb, u, t, a, b, L=1, k=2)
        sel["TemGX"] = tx["E_eps"][:2]
        cx = catgx.explain(bb, u, t, a, b, L=1, k=2, exact=True)
        sel["CATGX"] = [e for e in cx["G_star"] if e in pool][:2]
        _, _, attn = pw.behavioural_logits(u, t, [a, b])
        sel["Attention top-2"] = [pool[i] for i in np.argsort(-attn)[:2]]
        sel["Most recent 2"] = pool[-2:]
        sel["Random 2"] = [int(x) for x in rng.choice(pool, 2, replace=False)]
        singles = margins_for_sets(pw, case, 0, 1, [(e,) for e in pool])
        sel["Best single removals"] = [pool[i] for i in np.argsort(singles)[:2]]
        pairs = [(pool[i], pool[j]) for i in range(len(pool)) for j in range(i + 1, len(pool))]
        pm = margins_for_sets(pw, case, 0, 1, pairs)
        sel["Try every pair"] = list(pairs[int(np.argmin(pm))])
        best_drop = float(m0 - pm.min())
        row = {"member": u, "t": t, "B": b, "planted": sorted(gt), "n_history_window": len(pool),
               "margin": float(m0), "margin_without_planted": float(m_wo), "margin_without_T1": float(m_t1),
               "hits": {k: len(set(v) & gt) for k, v in sel.items()}, "selected": sel,
               "fidelity": {k: float((m0 - margins_for_sets(pw, case, 0, 1, [tuple(v)])[0]) / best_drop) if best_drop > 0 else None for k, v in sel.items()},
               "calls": {"TemGX": tx["n_model_inferences"], "CATGX": cx["n_model_inferences"], "Try every pair": len(pairs),
                         "Best single removals": len(pool)}}
        res.append(row)
        # LLM prompt: the same history as plain text, nothing about the model internals
        titles, genres = items.title.tolist(), items.genres.tolist()
        lines = [f"{i + 1}. {pd.to_datetime(int(ts), unit='s').date()} · {titles[it]} ({genres[it].replace('|', ', ')}) · rated {r}"
                 for i, (it, ts, r) in enumerate(zip(h.items, h.ts, h.ratings))]
        llm.append({"id": f"{name}-{u}", "lines": lines, "Z": titles[a], "B": titles[b],
                    "planted_lines": [pool.index(e) + 1 for e in sorted(gt)], "eids": pool})
    B.temperature = orig_temp
    summ = {k: {"found_any": float(np.mean([r["hits"][k] >= 1 for r in res])), "mean_hits_of_2": float(np.mean([r["hits"][k] for r in res])),
                "found_both": float(np.mean([r["hits"][k] == 2 for r in res])),
                "fidelity_vs_best_pair": float(np.mean([r["fidelity"][k] for r in res if r["fidelity"][k] is not None]))} for k in res[0]["hits"]}
    sanity = {"mean_margin": float(np.mean([r["margin"] for r in res])),
              "mean_margin_without_planted": float(np.mean([r["margin_without_planted"] for r in res])),
              "mean_margin_without_T1_only": float(np.mean([r["margin_without_T1"] for r in res])),
              "share_where_planted_pair_is_the_best_pair": float(np.mean([r["hits"]["Try every pair"] == 2 for r in res]))}
    titles = items.title.tolist()
    out = {"created": now_iso(), "pattern": name, "P": {**P, "titles": [titles[P["T1"]], titles[P["T2"]], titles[P["Z"]]]},
           "plant": plant_info, "temperature": T, "train_log": log, "probes": len(res), "summary": summ, "sanity": sanity,
           "calls_mean": {k: float(np.mean([r["calls"][k] for r in res])) for k in res[0]["calls"]},
           "rows": res, "wall_s": round(time.time() - t0, 1), "method": __doc__}
    OUT.mkdir(parents=True, exist_ok=True)
    dump_json(out, OUT / f"{name}.json")
    dump_json(llm, OUT / f"{name}_llm_prompts.json")
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "semantic":
        r = run_pattern("semantic", SEM)
    else:
        events, items = load_events(), load_items()
        pop = events[events.split == "train"].groupby("item").size().reindex(range(len(items)), fill_value=0).to_numpy()
        r = run_pattern("arbitrary", arbitrary_pattern(items, pop))
    print(r["P"]["titles"], r["sanity"])
    for k, v in r["summary"].items():
        print(f"{k:24s} {v}")
    print(r["calls_mean"], r["wall_s"])
