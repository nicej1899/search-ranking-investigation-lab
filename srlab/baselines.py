"""Standard recommendation baselines and a full-catalog next-item evaluation.

Protocol (same for every model):
  * one prediction per member per period: t = the member's first time in the period with >= 5 earlier ratings
    (identical to the query cases), DEV = validation period, FINAL = test period;
  * targets = titles the member rated >= 4.0 within 90 days after t and had not rated before t;
  * candidates = the FULL catalog of titles with >= 1 training interaction, minus titles rated before t
    (no sampled negatives);
  * metrics: Recall@10, Recall@50, NDCG@10 (binary gains); 95% intervals by resampling members.
All models are trained on the training period only and use only ratings before t at prediction time.
ID-based models (BPR-MF, LightGCN) cannot embed members who appear after training (most FINAL members are new), so
they are used item-based: the member vector = mean of the learned vectors of the member's last 50 titles before t.
SASRec and the temporal graph models are inductive by construction.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .common import CKPT, GLOBAL_SEED, REPORTS, dump_json, now_iso
from .ranker import world

torch.set_num_threads(2)
D = 64
MAXLEN = 50


def eval_points(W, split):
    """(member, t, history items, targets) for every member, one per period."""
    ev = W.events
    lo, hi = {"dev": ("val",), "final": ("test",)}[split], None
    per = ev[ev.split.isin(lo)]
    out = []
    for m, g in per.groupby("member"):
        for t in np.unique(g.ts.to_numpy()):
            if W.store.count_before(m, t) >= 5:
                h = W.store.before(m, t, n=None)
                fut = ev[(ev.member == m) & (ev.ts >= t) & (ev.ts < t + 90 * 86400) & (ev.rating >= 4.0)]
                tg = set(fut.item.tolist()) - set(h.items.tolist())
                out.append((int(m), int(t), h.items, tg))
                break
    return out


def train_items(W):
    tr = W.events[W.events.split == "train"]
    return np.zeros(len(W.items), bool) | np.isin(np.arange(len(W.items)), tr.item.unique())


def metrics(scores, cand_mask, targets, seen):
    s = scores.astype(float).copy()
    s[~cand_mask] = -np.inf
    s[list(seen)] = -np.inf
    tg = [i for i in targets if cand_mask[i] and i not in seen]
    if not tg:
        return None
    kk = min(50, len(s) - 1)
    top = np.argpartition(-s, kk)[:kk]
    top = top[np.argsort(-s[top])]
    hit = np.isin(top, tg)
    disc = 1 / np.log2(np.arange(2, 12))
    ideal = disc[: min(10, len(tg))].sum()
    return {"recall10": hit[:10].sum() / len(tg), "recall50": hit.sum() / len(tg),
            "ndcg10": float((hit[:10] * disc).sum() / ideal)}


# ------------------------------------------------------------------ models

class BPRMF(nn.Module):
    def __init__(self, nu, ni):
        super().__init__()
        self.U = nn.Embedding(nu, D); self.I = nn.Embedding(ni, D)
        nn.init.normal_(self.U.weight, std=0.05); nn.init.normal_(self.I.weight, std=0.05)

    def item_vectors(self):
        return self.I.weight.detach()


class LightGCN(nn.Module):
    def __init__(self, nu, ni, A, layers=3):
        super().__init__()
        self.nu, self.ni, self.A, self.L = nu, ni, A, layers
        self.E = nn.Embedding(nu + ni, D); nn.init.normal_(self.E.weight, std=0.05)

    def propagate(self):
        x = self.E.weight; out = [x]
        for _ in range(self.L):
            x = torch.sparse.mm(self.A, x); out.append(x)
        return torch.stack(out).mean(0)

    def item_vectors(self):
        with torch.no_grad():
            return self.propagate()[self.nu:]


class SASRec(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.item = nn.Embedding(ni + 1, D, padding_idx=0)
        self.pos = nn.Embedding(MAXLEN, D)
        layer = nn.TransformerEncoderLayer(D, 2, 2 * D, dropout=0.2, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.ln = nn.LayerNorm(D)

    def forward(self, seq):  # seq: [B, L] item+1, 0 = pad (left-padded)
        L = seq.shape[1]
        x = self.item(seq) + self.pos(torch.arange(L))
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        h = self.enc(x, mask=mask)  # left padding: real positions never attend to the future; pads are zero vectors
        return self.ln(h) * (seq > 0).unsqueeze(-1)

    def item_vectors(self):
        return self.item.weight.detach()[1:]


def _pairs(W):
    tr = W.events[W.events.split == "train"]
    users = {m: i for i, m in enumerate(sorted(tr.member.unique()))}
    u = tr.member.map(users).to_numpy(); it = tr.item.to_numpy()
    return users, u, it


def train_bpr(W, epochs=40, lr=5e-3, l2=1e-5):
    torch.manual_seed(GLOBAL_SEED)
    users, u, it = _pairs(W)
    ni = len(W.items)
    m = BPRMF(len(users), ni); opt = torch.optim.Adam(m.parameters(), lr=lr)
    pool = np.unique(it)
    rng = np.random.default_rng(GLOBAL_SEED)
    for ep in range(epochs):
        perm = rng.permutation(len(u))
        for s in range(0, len(u), 4096):
            b = perm[s:s + 4096]
            uu, pi = torch.as_tensor(u[b]), torch.as_tensor(it[b]); ni_ = torch.as_tensor(rng.choice(pool, len(b)))
            eu = m.U(uu); x = (eu * m.I(pi)).sum(1) - (eu * m.I(ni_)).sum(1)
            loss = -Fn.logsigmoid(x).mean() + l2 * (eu.pow(2).sum() + m.I(pi).pow(2).sum() + m.I(ni_).pow(2).sum()) / len(b)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def train_lightgcn(W, epochs=40, lr=5e-3, l2=1e-4):
    torch.manual_seed(GLOBAL_SEED)
    users, u, it = _pairs(W)
    nu, ni = len(users), len(W.items)
    r = np.concatenate([u, it + nu]); c = np.concatenate([it + nu, u])
    deg = np.bincount(r, minlength=nu + ni).astype(float)
    w = 1 / np.sqrt(deg[r] * deg[c])
    A = torch.sparse_coo_tensor(torch.as_tensor(np.stack([r, c])), torch.as_tensor(w, dtype=torch.float32), (nu + ni, nu + ni)).coalesce()
    m = LightGCN(nu, ni, A); opt = torch.optim.Adam(m.parameters(), lr=lr)
    pool = np.unique(it); rng = np.random.default_rng(GLOBAL_SEED)
    for ep in range(epochs):
        perm = rng.permutation(len(u))
        for s in range(0, len(u), 8192):
            b = perm[s:s + 8192]
            E = m.propagate()
            uu, pi = torch.as_tensor(u[b]), torch.as_tensor(it[b] + nu); nj = torch.as_tensor(rng.choice(pool, len(b)) + nu)
            x = (E[uu] * E[pi]).sum(1) - (E[uu] * E[nj]).sum(1)
            e0 = m.E.weight
            loss = -Fn.logsigmoid(x).mean() + l2 * (e0[uu].pow(2).sum() + e0[pi].pow(2).sum() + e0[nj].pow(2).sum()) / len(b)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def train_sasrec(W, epochs=120, lr=1e-3):  # 120 chosen on DEV (30 epochs: NDCG@10 0.10, 120: 0.22)
    torch.manual_seed(GLOBAL_SEED)
    tr = W.events[W.events.split == "train"].sort_values(["member", "ts", "event_id"])
    seqs = [g.item.to_numpy() + 1 for _, g in tr.groupby("member") if len(g) >= 3]
    ni = len(W.items)
    m = SASRec(ni); opt = torch.optim.Adam(m.parameters(), lr=lr)
    pool = np.unique(tr.item.to_numpy()) + 1; rng = np.random.default_rng(GLOBAL_SEED)
    # training windows: every user sequence cut into chunks of MAXLEN+1
    chunks = []
    for sq in seqs:
        for e in range(len(sq), 1, -MAXLEN):
            chunks.append(sq[max(0, e - MAXLEN - 1):e])
    for ep in range(epochs):
        m.train()
        for s in range(0, len(chunks), 64):
            bt = chunks[s:s + 64]
            X = np.zeros((len(bt), MAXLEN), np.int64); Y = np.zeros_like(X)
            for k, sq in enumerate(bt):
                inp, out = sq[:-1][-MAXLEN:], sq[1:][-MAXLEN:]
                X[k, -len(inp):] = inp; Y[k, -len(out):] = out
            X, Y = torch.as_tensor(X), torch.as_tensor(Y)
            Nn = torch.as_tensor(rng.choice(pool, Y.shape))
            h = m(X)
            pos = (h * m.item(Y)).sum(-1); neg = (h * m.item(Nn)).sum(-1)
            msk = Y > 0
            loss = -(Fn.logsigmoid(pos)[msk].mean() + Fn.logsigmoid(-neg)[msk].mean())
            opt.zero_grad(); loss.backward(); opt.step()
        rng.shuffle(chunks)
    m.eval()
    return m


# ------------------------------------------------------------------ scoring

class Scorers:
    def __init__(self, W):
        self.W = W
        self.it_ts = W.item_ts

    def popularity(self, m, t, hist):
        return np.array([np.searchsorted(self.it_ts.get(i, np.zeros(0)), t) for i in range(len(self.W.items))], float)

    @staticmethod
    def foldin(V, hist):
        h = hist[-MAXLEN:]
        u = V[torch.as_tensor(h)].mean(0)
        return (V @ u).numpy()

    @staticmethod
    def sasrec(model, hist):
        x = np.zeros((1, MAXLEN), np.int64); h = hist[-MAXLEN:] + 1; x[0, -len(h):] = h
        with torch.no_grad():
            z = model(torch.as_tensor(x))[0, -1]
            return (model.item_vectors() @ z).numpy()


def run(split="final", models=("popularity", "bpr_mf", "lightgcn", "sasrec", "tgat1", "tgat2", "tgn"), n_boot=2000):
    t0 = time.time()
    W = world()
    pts = eval_points(W, split)
    cand = train_items(W)
    trained = {}
    log = {}
    if "bpr_mf" in models:
        s = time.time(); trained["bpr_mf"] = train_bpr(W); log["bpr_mf_train_s"] = round(time.time() - s, 1)
    if "lightgcn" in models:
        s = time.time(); trained["lightgcn"] = train_lightgcn(W); log["lightgcn_train_s"] = round(time.time() - s, 1)
    if "sasrec" in models:
        s = time.time(); trained["sasrec"] = train_sasrec(W); log["sasrec_train_s"] = round(time.time() - s, 1)
    for k, m in trained.items():
        torch.save(m.state_dict(), CKPT / f"baseline_{k}.pt")
    from .backbones import get_backbone
    sc = Scorers(W)
    res = {}
    for name in models:
        rows = []
        s0 = time.time()
        V = trained[name].item_vectors() if name in ("bpr_mf", "lightgcn") else None
        bb = get_backbone(name) if name in ("tgat1", "tgat2", "tgn") else None
        for m, t, hist, tg in pts:
            if name == "popularity":
                s = sc.popularity(m, t, hist)
            elif V is not None:
                s = sc.foldin(V, hist)
            elif name == "sasrec":
                s = sc.sasrec(trained["sasrec"], hist)
            else:
                with torch.no_grad():
                    s = bb.logits(m, t, np.arange(len(W.items)))
            r = metrics(s, cand, tg, set(hist.tolist()))
            if r:
                rows.append({"member": m, **r})
        res[name] = summarise(rows, n_boot)
        res[name]["score_s"] = round(time.time() - s0, 1)
        print(name, {k: round(v["mean"], 4) for k, v in res[name].items() if isinstance(v, dict)}, flush=True)
    out = {"created": now_iso(), "split": split, "protocol": __doc__, "n_points": len(pts), "catalog_size": int(cand.sum()),
           "results": res, "train_log": log, "wall_s": round(time.time() - t0, 1)}
    dump_json(out, REPORTS / f"baselines_{split}.json")
    return out


def summarise(rows, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = {"n": len(rows)}
    for k in ("recall10", "recall50", "ndcg10"):
        v = np.array([r[k] for r in rows])
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
        out[k] = {"mean": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "final")
