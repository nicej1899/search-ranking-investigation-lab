"""Train the temporal backbone on TRAIN-split rating interactions only (link prediction with sampled
negatives), select the epoch on validation, and freeze it as the behavioural component of F0.

Usage: python -m srlab.train  [--epochs 8]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .common import CKPT, GLOBAL_SEED, REPORTS, dump_json, hardware, load_json, now_iso, sha256_file, sha256_obj, PROC
from .data import genre_matrix, load_events, load_items, GENRES
from .history import HistoryStore, to_batch
from .model import ModelConfig, TemporalAttentionRecommender

N_NEG_EVAL = 100


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)


def build_model(items: pd.DataFrame, cfg: ModelConfig | None = None) -> TemporalAttentionRecommender:
    cfg = cfg or ModelConfig(n_items=len(items), n_genres=len(GENRES))
    m = TemporalAttentionRecommender(cfg, genre_matrix(items))
    m.set_unk(items.seen_in_train.to_numpy())
    return m


def eval_protocol(events: pd.DataFrame, split: str, n_cases: int, n_items: int, seed: int, store: HistoryStore):
    """Deterministic sampled-candidate protocol: each held-out event gets the same 100 negatives for every model."""
    rng = np.random.default_rng(seed)
    sub = events[events.split == split]
    idx = np.sort(rng.choice(len(sub), size=min(n_cases, len(sub)), replace=False))
    sub = sub.iloc[idx]
    cands = []
    for r in sub.itertuples():
        seen = set(store.before(r.member, r.ts, n=None).items.tolist()) | {r.item}
        negs = []
        while len(negs) < N_NEG_EVAL:
            j = int(rng.integers(n_items))
            if j not in seen and j not in negs:
                negs.append(j)
        cands.append([r.item] + negs)
    return sub.reset_index(drop=True), np.array(cands, np.int64)


@torch.no_grad()
def rank_metrics(model, store: HistoryStore, sub: pd.DataFrame, cands: np.ndarray, k: int = 10, bs: int = 256):
    model.eval()
    hits, ndcgs = [], []
    n = model.cfg.max_hist
    for s in range(0, len(sub), bs):
        part = sub.iloc[s:s + bs]
        ts = part.ts.to_numpy()
        hb = to_batch([store.before(m, t, n=n) for m, t in zip(part.member, ts)], ts, n)
        hu = model.member_repr(*hb)
        logits = model.score(hu, torch.from_numpy(cands[s:s + bs])).numpy()
        rank = (logits[:, 1:] > logits[:, :1]).sum(1)  # 0-based rank of positive
        hits += (rank < k).tolist()
        ndcgs += np.where(rank < k, 1.0 / np.log2(rank + 2), 0.0).tolist()
    return {"HR@10": float(np.mean(hits)), "NDCG@10": float(np.mean(ndcgs)), "n": len(sub)}


def popularity_metrics(events, sub, cands, k=10):
    """Reference baseline: count of each title's rating interactions strictly before the prediction time."""
    ev = events.sort_values("ts")
    item_ts = {i: g.ts.to_numpy() for i, g in ev.groupby("item")}
    hits, ndcgs = [], []
    for r, c in zip(sub.itertuples(), cands):
        sc = np.array([np.searchsorted(item_ts.get(int(j), np.zeros(0)), r.ts, "left") for j in c], float)
        sc += np.random.default_rng(int(r.event_id)).random(len(c)) * 1e-6  # deterministic tie-break
        rank = int((sc[1:] > sc[0]).sum())
        hits.append(rank < k)
        ndcgs.append(1 / np.log2(rank + 2) if rank < k else 0.0)
    return {"HR@10": float(np.mean(hits)), "NDCG@10": float(np.mean(ndcgs)), "n": len(sub)}


def train(epochs: int = 8, lr: float = 1e-3, bs: int = 512, seed: int = GLOBAL_SEED):
    set_seed(seed)
    torch.set_num_threads(2)
    events, items = load_events(), load_items()
    store = HistoryStore(events)
    model = build_model(items)
    cfg = model.cfg
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    tr = events[events.split == "train"].reset_index(drop=True)
    train_items = np.flatnonzero(items.seen_in_train.to_numpy())
    val_sub, val_c = eval_protocol(events, "val", 3000, len(items), seed + 1, store)
    rng = np.random.default_rng(seed)
    log, best, best_state = [], -1.0, None
    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(tr))
        losses = []
        t0 = time.time()
        for s in range(0, len(perm), bs):
            part = tr.iloc[perm[s:s + bs]]
            ts = part.ts.to_numpy()
            hb = to_batch([store.before(m, t, n=cfg.max_hist) for m, t in zip(part.member, ts)], ts, cfg.max_hist)
            pos = torch.from_numpy(part.item.to_numpy())
            neg = torch.from_numpy(rng.choice(train_items, size=(len(part), 4)))
            hu = model.member_repr(*hb)
            lp = model.score(hu, pos)
            ln = model.score(hu, neg)
            loss = F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) + \
                F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        vm = rank_metrics(model, store, val_sub, val_c)
        rec = {"epoch": ep, "train_loss": float(np.mean(losses)), "val": vm, "epoch_sec": round(time.time() - t0, 1)}
        log.append(rec)
        print(rec, flush=True)
        if vm["NDCG@10"] > best:
            best = vm["NDCG@10"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ep = ep
    model.load_state_dict(best_state)
    path = CKPT / "temporal_backbone_F0.pt"
    torch.save({"state_dict": best_state, "config": cfg.to_dict()}, path)
    test_sub, test_c = eval_protocol(events, "test", 3000, len(items), seed + 2, store)
    report = {
        "created": now_iso(),
        "checkpoint": str(path.relative_to(path.parents[2])),
        "checkpoint_sha256": sha256_file(path),
        "config": cfg.to_dict(),
        "hyperparameters": {"epochs": epochs, "lr": lr, "batch_size": bs, "neg_per_pos": 4, "weight_decay": 1e-6,
                            "seed": seed, "objective": "BCE link prediction, uniform negatives from training titles"},
        "selected_epoch": best_ep,
        "training_log": log,
        "train_wall_sec": round(time.time() - t_start, 1),
        "data_fingerprint": load_json(PROC / "split.json")["events_sha"],
        "eval_protocol": {"candidates": f"1 held-out positive + {N_NEG_EVAL} uniformly sampled titles the member had not rated before t; "
                                        "fixed seed, identical for every model", "val_cases": len(val_sub), "test_cases": len(test_sub)},
        "val": rank_metrics(model, store, val_sub, val_c),
        "test_temporal_backbone": rank_metrics(model, store, test_sub, test_c),
        "test_popularity_reference": popularity_metrics(events, test_sub, test_c),
        "hardware": hardware(),
    }
    dump_json(report, REPORTS / "backbone_training.json")
    print({k: report[k] for k in ["selected_epoch", "val", "test_temporal_backbone", "test_popularity_reference"]})
    return report


def load_backbone(items: pd.DataFrame | None = None) -> TemporalAttentionRecommender:
    items = items if items is not None else load_items()
    ck = torch.load(CKPT / "temporal_backbone_F0.pt", weights_only=False)
    m = build_model(items, ModelConfig(**ck["config"]))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    train(a.epochs)
