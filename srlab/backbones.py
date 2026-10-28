"""Three temporal GNN backbones behind one interface (all trained on the TRAIN split only).

  tgat1  TGAT-style, 1 layer, member side only (the original backbone, srlab/model.py). Receptive field: 1 hop.
  tgat2  TGAT (Xu et al. 2020), 2 layers on the bipartite graph, both endpoints. A neighbour w of x
         is embedded at the time of the (x,w) interaction (time-respecting). Receptive field: 2 hops.
  tgn    TGN (Rossi et al. 2020): per-node GRU memory updated by every event in chronological batches,
         plus a 1-layer temporal graph-attention embedding. Information can travel arbitrarily far.

Interface (all numpy in / numpy out, deterministic in eval mode):
  logits(u, t, items, exclude)            link logits for (user u, item) at time t on the graph minus `exclude`
  margins(u, t, a, b, exclude_sets)       logit(u,a) - logit(u,b) for many removal sets (exact replays)
  edge_weights(u, t, items, exclude)      TGAT attention mass per event id in the computation of u and items (CATGX g)
  memory_change(u, t, items, sets)        TGN memory-state change of u and items per removal set (CATGX g)
`exclude` = set of event ids removed from the graph for BOTH endpoints (the counterfactual intervention).
"""
from __future__ import annotations

import functools
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CKPT, GLOBAL_SEED, REPORTS, dump_json, hardware, load_json, now_iso, sha256_file
from .data import GENRES, genre_matrix, load_events, load_items
from .model import TimeEncode, rating_index
from .tgraph import graph

D = 64


class NodeFeatures(nn.Module):
    """h0: users -> id embedding (UNK if unseen in training); movies -> id embedding (UNK) + genre projection."""

    def __init__(self, U, I, genre_mat, seen_users, seen_items):
        super().__init__()
        self.U, self.I = U, I
        self.user = nn.Embedding(U + 1, D)
        self.item = nn.Embedding(I + 1, D)
        nn.init.normal_(self.user.weight, std=0.05); nn.init.normal_(self.item.weight, std=0.05)
        self.register_buffer("genres", torch.from_numpy(genre_mat))
        self.gproj = nn.Linear(genre_mat.shape[1], D, bias=False)
        self.register_buffer("useen", torch.from_numpy(seen_users.astype(bool)))
        self.register_buffer("iseen", torch.from_numpy(seen_items.astype(bool)))

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        is_u = nodes < self.U
        out = torch.zeros(len(nodes), D)
        if is_u.any():
            uid = nodes[is_u]
            uid = torch.where(self.useen[uid], uid, torch.full_like(uid, self.U))
            out[is_u] = self.user(uid)
        if (~is_u).any():
            iid = nodes[~is_u] - self.U
            idx = torch.where(self.iseen[iid], iid, torch.full_like(iid, self.I))
            out[~is_u] = self.item(idx) + self.gproj(self.genres[iid])
        return out


class AttnLayer(nn.Module):
    def __init__(self, heads=2):
        super().__init__()
        self.time = TimeEncode(16)
        self.rat = nn.Embedding(10, 8)
        self.kv = nn.Linear(D + 16 + 8, D)
        self.q = nn.Linear(D + 16, D)
        self.attn = nn.MultiheadAttention(D, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, D))

    def forward(self, h_self, h_nb, dt_days, rat_idx, mask):
        B, n = mask.shape
        kv = self.kv(torch.cat([h_nb, self.time(dt_days), self.rat(rat_idx)], -1))
        q = self.q(torch.cat([h_self, self.time(torch.zeros(B))], -1)).unsqueeze(1)
        has = mask.any(1)
        pad = torch.where(has.unsqueeze(1), ~mask, torch.zeros_like(mask))
        out, w = self.attn(q, kv, kv, key_padding_mask=pad, need_weights=True, average_attn_weights=True)
        out = torch.where(has.unsqueeze(1), out.squeeze(1), torch.zeros(B, D))
        w = torch.where(has.unsqueeze(1), w.squeeze(1), torch.zeros(B, n))
        return self.mlp(torch.cat([out, h_self], -1)), w


def gather(G, nodes, times, n, exclude):
    """exclude: one frozenset for all rows, or a list with one frozenset per row."""
    B = len(nodes)
    per_row = isinstance(exclude, list)
    nb = np.zeros((B, n), np.int64); dt = np.zeros((B, n), np.float32); rt = np.zeros((B, n), np.int64)
    tt = np.zeros((B, n), np.int64); ee = np.full((B, n), -1, np.int64); mask = np.zeros((B, n), bool)
    for b, (x, t) in enumerate(zip(nodes, times)):
        o, ts, e, r = G.neighbors(int(x), int(t), n, exclude[b] if per_row else exclude)
        k = len(o)
        if k:
            nb[b, :k] = o; dt[b, :k] = (t - ts) / 86400.0; rt[b, :k] = rating_index(r); tt[b, :k] = ts; ee[b, :k] = e; mask[b, :k] = True
    return nb, dt, rt, tt, ee, mask


class TGAT2(nn.Module):
    n1, n2 = 8, 8

    def __init__(self, feats: NodeFeatures):
        super().__init__()
        self.f = feats
        self.l1 = AttnLayer()
        self.l2 = AttnLayer()
        self.score_mlp = nn.Sequential(nn.Linear(3 * D, D), nn.ReLU(), nn.Linear(D, 1))

    def embed(self, G, nodes, times, exclude=frozenset(), want_w=False):
        """2-layer TGAT embedding. If want_w, also returns {eid: attention mass} for each root (path products)."""
        nodes = np.asarray(nodes, np.int64); times = np.asarray(times, np.int64)
        B = len(nodes)
        nb, dt, rt, tt, ee, mask = gather(G, nodes, times, self.n1, exclude)
        flat_n, flat_t = nb.reshape(-1), tt.reshape(-1)
        # layer-1 embeddings of the roots (at t) and of their neighbours (at their interaction times)
        allx = np.concatenate([nodes, flat_n]); allt = np.concatenate([times, flat_t])
        ex2 = (list(exclude) + [x for x in exclude for _ in range(self.n1)]) if isinstance(exclude, list) else exclude
        nb2, dt2, rt2, tt2, ee2, mask2 = gather(G, allx, allt, self.n2, ex2)
        h0_self = self.f(torch.as_tensor(allx))
        h0_nb = self.f(torch.as_tensor(nb2.reshape(-1))).view(len(allx), self.n2, D)
        h1, w1 = self.l1(h0_self, h0_nb, torch.as_tensor(dt2), torch.as_tensor(rt2), torch.as_tensor(mask2))
        h1_root, h1_nb = h1[:B], h1[B:].view(B, self.n1, D)
        h2, w2 = self.l2(h1_root, h1_nb, torch.as_tensor(dt), torch.as_tensor(rt), torch.as_tensor(mask))
        if not want_w:
            return h2
        W = []
        w1n = w1.detach().numpy(); w2n = w2.detach().numpy()
        for b in range(B):
            acc: dict = {}
            for j in range(self.n1):
                if not mask[b, j]:
                    continue
                a = float(w2n[b, j]); acc[int(ee[b, j])] = acc.get(int(ee[b, j]), 0.0) + a
                row = B + b * self.n1 + j
                for k in range(self.n2):
                    if mask2[row, k]:
                        acc[int(ee2[row, k])] = acc.get(int(ee2[row, k]), 0.0) + a * float(w1n[row, k])
            for k in range(self.n2):  # root's own layer-1 attention (feeds h2 through h1_root)
                if mask2[b, k]:
                    acc[int(ee2[b, k])] = acc.get(int(ee2[b, k]), 0.0) + float(w1n[b, k])
            W.append(acc)
        return h2, W

    def score(self, zu, zi):
        return self.score_mlp(torch.cat([zu, zi, zu * zi], -1)).squeeze(-1)


class TGN(nn.Module):
    nb = 10
    batch = 200

    def __init__(self, feats: NodeFeatures, N):
        super().__init__()
        self.f = feats
        self.N = N
        self.time = TimeEncode(16)
        self.rat = nn.Embedding(10, 8)
        self.gru = nn.GRUCell(2 * D + 16 + 8, D)
        self.emb = AttnLayer()
        self.score_mlp = nn.Sequential(nn.Linear(3 * D, D), nn.ReLU(), nn.Linear(D, 1))

    def init_state(self):
        return torch.zeros(self.N, D), torch.zeros(self.N, dtype=torch.long)

    def update(self, S, last, src, dst, ts, rat):
        """One batch of events (TGN 'last' message aggregator). Messages use memory BEFORE the batch."""
        src_t = torch.as_tensor(src); dst_t = torch.as_tensor(dst); ts_t = torch.as_tensor(ts); r = torch.as_tensor(rating_index(rat))
        nodes = torch.cat([src_t, dst_t]); other = torch.cat([dst_t, src_t]); tt = torch.cat([ts_t, ts_t]); rr = torch.cat([r, r])
        dt = (tt - last[nodes]).float().clamp(min=0) / 86400.0
        msg = torch.cat([S[nodes], S[other], self.time(dt), self.rat(rr)], -1)
        # keep the last message per node (events are chronological; the later row wins)
        order = torch.arange(len(nodes))
        key = nodes * (len(nodes) + 1) + order
        uniq, inv = torch.unique(nodes, return_inverse=True)
        lastpos = torch.zeros(len(uniq), dtype=torch.long).scatter_reduce(0, inv, order, reduce="amax", include_self=False)
        new = self.gru(msg[lastpos], S[uniq])
        S = S.clone(); last = last.clone()
        S[uniq] = new; last[uniq] = tt[lastpos]
        return S, last

    def embed(self, G, S, nodes, times, exclude=frozenset(), want_w=False):
        nodes = np.asarray(nodes, np.int64); times = np.asarray(times, np.int64)
        nb, dt, rt, tt, ee, mask = gather(G, nodes, times, self.nb, exclude)
        h_self = S[torch.as_tensor(nodes)] + self.f(torch.as_tensor(nodes))
        h_nb = S[torch.as_tensor(nb.reshape(-1))].view(len(nodes), self.nb, D) + self.f(torch.as_tensor(nb.reshape(-1))).view(len(nodes), self.nb, D)
        z, w = self.emb(h_self, h_nb, torch.as_tensor(dt), torch.as_tensor(rt), torch.as_tensor(mask))
        if want_w:
            wn = w.detach().numpy()
            return z, [{int(ee[b, j]): float(wn[b, j]) for j in range(self.nb) if mask[b, j]} for b in range(len(nodes))]
        return z

    def score(self, zu, zi):
        return self.score_mlp(torch.cat([zu, zi, zu * zi], -1)).squeeze(-1)


# ============================================================================ training

def _setup():
    events, items = load_events(), load_items()
    G = graph()
    tr = events[events.split == "train"]
    seen_u = np.zeros(G.U, bool); seen_u[tr.member.unique()] = True
    feats = NodeFeatures(G.U, G.I, genre_matrix(items), seen_u, items.seen_in_train.to_numpy())
    return events, items, G, feats


def _eval_sets(events, items, G):
    from .history import HistoryStore
    from .train import eval_protocol
    store = HistoryStore(events)
    val = eval_protocol(events, "val", 1000, len(items), GLOBAL_SEED + 1, store)
    test = eval_protocol(events, "test", 1000, len(items), GLOBAL_SEED + 2, store)
    return val, test


def _metrics(score_fn, sub, cands, k=10):
    hits, nd = [], []
    for r, c in zip(sub.itertuples(), cands):
        s = score_fn(int(r.member), int(r.ts), c)
        rank = int((s[1:] > s[0]).sum())
        hits.append(rank < k); nd.append(1 / np.log2(rank + 2) if rank < k else 0.0)
    return {"HR@10": float(np.mean(hits)), "NDCG@10": float(np.mean(nd)), "n": len(sub)}


def train_tgat2(epochs=3, bs=256, lr=1e-3, per_epoch=20000, val_select=300):
    torch.manual_seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED)
    events, items, G, feats = _setup()
    m = TGAT2(feats)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    tr = events[events.split == "train"].reset_index(drop=True)
    train_items = np.flatnonzero(items.seen_in_train.to_numpy()) + G.U
    rng = np.random.default_rng(GLOBAL_SEED)
    (val_sub, val_c), (test_sub, test_c) = _eval_sets(events, items, G)
    log, best, best_state = [], -1, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        m.train(); perm = rng.permutation(len(tr))[:per_epoch]; losses = []
        for s in range(0, len(perm), bs):
            p = tr.iloc[perm[s:s + bs]]
            u = p.member.to_numpy(); t = p.ts.to_numpy(); pos = p.item.to_numpy() + G.U; neg = rng.choice(train_items, len(p))
            z = m.embed(G, np.concatenate([u, pos, neg]), np.concatenate([t, t, t]))
            B = len(p)
            lp = m.score(z[:B], z[B:2 * B]); ln = m.score(z[:B], z[2 * B:])
            loss = F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) + F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        m.eval()
        bb = Backbone("tgat2", m)
        vm = _metrics(lambda u, t, c: bb.logits(u, t, c), val_sub.iloc[:val_select], val_c[:val_select])
        log.append({"epoch": ep, "loss": float(np.mean(losses)), "val_select": vm, "sec": round(time.time() - t0, 1)}); print(log[-1], flush=True)
        if vm["NDCG@10"] > best:
            best, best_state, best_ep = vm["NDCG@10"], {k: v.clone() for k, v in m.state_dict().items()}, ep
    m.load_state_dict(best_state); m.eval()
    torch.save({"state_dict": best_state}, CKPT / "backbone_tgat2.pt")
    bb = Backbone("tgat2", m)
    rep = {"model": "tgat2", "created": now_iso(), "epochs": epochs, "selected_epoch": best_ep, "log": log, "lr": lr, "batch": bs,
           "train_events_per_epoch": per_epoch, "val_events_for_selection": val_select,
           "neighbours": [TGAT2.n1, TGAT2.n2], "val": _metrics(lambda u, t, c: bb.logits(u, t, c), val_sub, val_c),
           "test": _metrics(lambda u, t, c: bb.logits(u, t, c), test_sub, test_c), "checkpoint_sha256": sha256_file(CKPT / "backbone_tgat2.pt"),
           "eval_protocol": "1 positive + 100 sampled negatives; 1000 val / 1000 test events (subset of the tgat1 protocol, same seeds)", "hardware": hardware()}
    dump_json(rep, REPORTS / "backbone_tgat2.json")
    return rep


def train_tgn(epochs=3, lr=1e-3):
    torch.manual_seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED)
    events, items, G, feats = _setup()
    m = TGN(feats, G.N)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    n_train = int((G.ts < load_json(REPORTS.parent.parent / "data" / "processed" / "split.json")["cutoffs"]["train_end_ts"]).sum())
    train_items = np.flatnonzero(items.seen_in_train.to_numpy()) + G.U
    rng = np.random.default_rng(GLOBAL_SEED)
    (val_sub, val_c), (test_sub, test_c) = _eval_sets(events, items, G)
    log, best, best_state = [], -1, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        m.train(); S, last = m.init_state(); pend = None; losses = []
        for s in range(0, n_train, TGN.batch):
            sl = slice(s, min(s + TGN.batch, n_train))
            if pend is not None:
                S, last = m.update(S, last, *pend)
            src, dst, ts, rat = G.src[sl], G.dst[sl], G.ts[sl], G.rating[sl]
            neg = rng.choice(train_items, len(src))
            z = m.embed(G, S, np.concatenate([src, dst, neg]), np.concatenate([ts, ts, ts]))
            B = len(src)
            lp = m.score(z[:B], z[B:2 * B]); ln = m.score(z[:B], z[2 * B:])
            loss = F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) + F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
            S = S.detach(); pend = (src, dst, ts, rat)
        m.eval()
        bb = Backbone("tgn", m); bb.build_snapshots()
        vm = _metrics(lambda u, t, c: bb.logits(u, t, c), val_sub, val_c)
        log.append({"epoch": ep, "loss": float(np.mean(losses)), "val": vm, "sec": round(time.time() - t0, 1)}); print(log[-1], flush=True)
        if vm["NDCG@10"] > best:
            best, best_state, best_ep = vm["NDCG@10"], {k: v.clone() for k, v in m.state_dict().items()}, ep
    m.load_state_dict(best_state); m.eval()
    torch.save({"state_dict": best_state}, CKPT / "backbone_tgn.pt")
    bb = Backbone("tgn", m); bb.build_snapshots()
    rep = {"model": "tgn", "created": now_iso(), "epochs": epochs, "selected_epoch": best_ep, "log": log, "lr": lr, "batch": TGN.batch,
           "val": _metrics(lambda u, t, c: bb.logits(u, t, c), val_sub, val_c), "test": _metrics(lambda u, t, c: bb.logits(u, t, c), test_sub, test_c),
           "checkpoint_sha256": sha256_file(CKPT / "backbone_tgn.pt"),
           "eval_protocol": "1 positive + 100 sampled negatives; 1000 val / 1000 test events (subset of the tgat1 protocol, same seeds)", "hardware": hardware()}
    dump_json(rep, REPORTS / "backbone_tgn.json")
    return rep


# ============================================================================ common interface

class Backbone:
    INFO = {
        "tgat1": {"label": "TGAT · 1 layer", "hops": 1, "plain": "looks only at the person's own past ratings"},
        "tgat2": {"label": "TGAT · 2 layers", "hops": 2, "plain": "also looks at who else rated those movies"},
        "tgn": {"label": "TGN · memory", "hops": None, "plain": "keeps a running memory per person and movie; effects can travel far"},
    }
    SNAP_EVERY = 5000

    def __init__(self, key: str, model=None):
        self.key = key
        self.G = graph()
        self.U = self.G.U
        if key == "tgat1":
            from .ranker import world
            self.W = world()
            self.m = self.W.model
        else:
            self.m = model
        self.snaps = None

    # ---- TGN state
    def build_snapshots(self):
        G = self.G
        S, last = self.m.init_state()
        snaps = {0: (S.clone(), last.clone())}
        with torch.no_grad():
            for s in range(0, len(G.ts), TGN.batch):
                if s % self.SNAP_EVERY == 0 and s:
                    snaps[s] = (S.clone(), last.clone())
                sl = slice(s, min(s + TGN.batch, len(G.ts)))
                S, last = self.m.update(S, last, G.src[sl], G.dst[sl], G.ts[sl], G.rating[sl])
        self.snaps = snaps

    def tgn_state(self, t, exclude=frozenset()):
        if not exclude:
            if not hasattr(self, "_sc"):
                self._sc = {}
            if t in self._sc:
                return self._sc[t]
            S = self._tgn_state(t, exclude)
            if len(self._sc) > 64:
                self._sc.clear()
            self._sc[t] = S
            return S
        return self._tgn_state(t, exclude)

    def _tgn_state(self, t, exclude=frozenset()):
        G = self.G
        p = int(np.searchsorted(G.ts, t, side="left"))  # events [0, p) are before t
        start = p
        if exclude:
            start = min(start, min(G.pos[int(e)] for e in exclude))
        s0 = max(k for k in self.snaps if k <= start - (start % TGN.batch)) if start > 0 else 0
        S, last = self.snaps[s0]
        ex = np.fromiter(exclude, np.int64, len(exclude)) if exclude else None
        with torch.no_grad():
            for s in range(s0, p, TGN.batch):
                e = min(s + TGN.batch, p)
                sl = slice(s, e)
                src, dst, ts, rat, eid = G.src[sl], G.dst[sl], G.ts[sl], G.rating[sl], G.eid[sl]
                if ex is not None:
                    keep = ~np.isin(eid, ex)
                    src, dst, ts, rat = src[keep], dst[keep], ts[keep], rat[keep]
                if len(src):
                    S, last = self.m.update(S, last, src, dst, ts, rat)
        return S

    # ---- scoring
    @torch.no_grad()
    def logits(self, u, t, items, exclude=frozenset()):
        items = np.asarray(items, np.int64)
        if self.key == "tgat1":
            l, _, _ = self.W.behavioural_logits(u, t, items, frozenset(exclude))
            return l
        nodes = np.concatenate([[u], items + self.U]); times = np.full(len(nodes), t)
        if self.key == "tgat2":
            z = self.m.embed(self.G, nodes, times, frozenset(exclude))
        else:
            z = self.m.embed(self.G, self.tgn_state(t, frozenset(exclude)), nodes, times, frozenset(exclude))
        return self.m.score(z[:1].expand(len(items), -1), z[1:]).numpy().astype(float)

    def margins(self, u, t, a, b, exclude_sets):
        if self.key == "tgat1":
            from .explain import margins_for_sets
            case = {"member": u, "t": t, "candidates": [a, b]}
            return margins_for_sets(self.W, case, 0, 1, [tuple(x) for x in exclude_sets])
        if self.key == "tgat2":  # batch many removal sets into one forward pass (exact: per-row exclusion)
            out = []
            sets = [frozenset(int(e) for e in ex) for ex in exclude_sets]
            for s0 in range(0, len(sets), 64):
                part = sets[s0:s0 + 64]
                nodes = np.array([x for _ in part for x in (u, a + self.U, b + self.U)])
                ex = [x for x in part for _ in range(3)]
                with torch.no_grad():
                    z = self.m.embed(self.G, nodes, np.full(len(nodes), t), ex).view(len(part), 3, D)
                    la = self.m.score(z[:, 0], z[:, 1]); lb = self.m.score(z[:, 0], z[:, 2])
                out += (la - lb).numpy().astype(float).tolist()
            return np.array(out)
        out = []
        for ex in exclude_sets:
            l = self.logits(u, t, [a, b], frozenset(ex))
            out.append(l[0] - l[1])
        return np.array(out)

    @torch.no_grad()
    def edge_weights(self, u, t, items, exclude=frozenset()):
        """Attention mass per event id in the computation of u and each item (summed). TGAT models only."""
        items = np.asarray(items, np.int64)
        if self.key == "tgat1":
            l, h, attn = self.W.behavioural_logits(u, t, items, frozenset(exclude))
            return {int(e): float(w) for e, w in zip(h.eids, attn)}
        nodes = np.concatenate([[u], items + self.U]); times = np.full(len(nodes), t)
        if self.key == "tgat2":
            _, W = self.m.embed(self.G, nodes, times, frozenset(exclude), want_w=True)
        else:
            _, W = self.m.embed(self.G, self.tgn_state(t, frozenset(exclude)), nodes, times, frozenset(exclude), want_w=True)
        acc = {}
        for d in W:
            for e, w in d.items():
                acc[e] = acc.get(e, 0.0) + w
        return acc

    @torch.no_grad()
    def memory_change(self, u, t, items, exclude_sets):
        """TGN memory-state change ||S(x) - S_without(x)|| summed over x in {u} + items, per removal set."""
        nodes = torch.as_tensor(np.concatenate([[u], np.asarray(items) + self.U]))
        S0 = self.tgn_state(t)[nodes]
        return np.array([float((self.tgn_state(t, frozenset(ex))[nodes] - S0).norm(dim=1).sum()) for ex in exclude_sets])


def calibrate(key: str, n_events: int = 300, n_neg: int = 10) -> dict:
    """Temperature T so that sigmoid((logit_pos - logit_neg)/T) is a calibrated pairwise probability on VAL
    (Bradley-Terry). Used by TemGX/CATGX to turn logit margins into Pr(A above B) without saturation."""
    bb = get_backbone(key)
    events, items = load_events(), load_items()
    (val_sub, val_c), _ = _eval_sets(events, items, bb.G)
    diffs = []
    for r, c in list(zip(val_sub.itertuples(), val_c))[:n_events]:
        l = bb.logits(int(r.member), int(r.ts), c[: n_neg + 1])
        diffs += list(l[0] - l[1:])
    d = np.array(diffs)
    grid = np.exp(np.linspace(np.log(0.05), np.log(100), 400))
    nll = [float(np.mean(np.logaddexp(0, -d / T))) for T in grid]
    T = float(grid[int(np.argmin(nll))])
    out = {"backbone": key, "T": T, "val_pairwise_nll": min(nll), "n_pairs": len(d), "created": now_iso(),
           "median_abs_margin": float(np.median(np.abs(d)))}
    dump_json(out, CKPT / f"calib_{key}.json")
    return out


def temperature(key: str) -> float:
    p = CKPT / f"calib_{key}.json"
    return load_json(p)["T"] if p.exists() else 1.0


@functools.lru_cache(maxsize=4)
def get_backbone(key: str) -> Backbone:
    if key == "tgat1":
        return Backbone("tgat1")
    _, _, G, feats = _setup()
    m = TGAT2(feats) if key == "tgat2" else TGN(feats, G.N)
    m.load_state_dict(torch.load(CKPT / f"backbone_{key}.pt", weights_only=False)["state_dict"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    bb = Backbone(key, m)
    if key == "tgn":
        bb.build_snapshots()
    return bb


if __name__ == "__main__":
    import sys
    torch.set_num_threads(2)
    which = sys.argv[1] if len(sys.argv) > 1 else "tgat2"
    print(train_tgat2() if which == "tgat2" else train_tgn())
