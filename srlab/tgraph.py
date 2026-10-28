"""Bipartite temporal interaction graph used by the multi-hop backbones and the faithful explainers.

Nodes: users 0..U-1 (node id = member index), movies U..U+I-1 (node id = U + item index).
Edges: one per rating event (event_id), undirected for neighbourhood purposes, timestamped.
All neighbour queries are strictly time-respecting (ts < t) and accept a set of removed event ids,
which is how every counterfactual intervention is applied (the edge disappears for BOTH endpoints).
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from .data import load_events, load_items

DELTA_DAYS = 365  # explanation window delta (TemGX/CATGX); lambda = 1/delta (TemGX App. L)


class TGraph:
    def __init__(self, events: pd.DataFrame, n_items: int):
        ev = events.sort_values(["ts", "event_id"], kind="mergesort")
        self.U = int(events.member.max()) + 1
        self.I = n_items
        self.N = self.U + self.I
        self.eid = ev.event_id.to_numpy(np.int64)
        self.src = ev.member.to_numpy(np.int64)
        self.dst = (ev.item.to_numpy(np.int64) + self.U)
        self.ts = ev.ts.to_numpy(np.int64)
        self.rating = ev.rating.to_numpy(np.float32)
        self.pos = {int(e): i for i, e in enumerate(self.eid)}  # event id -> chronological position
        # adjacency per node, chronological
        both_n = np.concatenate([self.src, self.dst])
        both_o = np.concatenate([self.dst, self.src])
        both_t = np.concatenate([self.ts, self.ts])
        both_e = np.concatenate([self.eid, self.eid])
        both_r = np.concatenate([self.rating, self.rating])
        order = np.lexsort((both_e, both_t, both_n))
        both_n, both_o, both_t, both_e, both_r = both_n[order], both_o[order], both_t[order], both_e[order], both_r[order]
        self.ptr = np.searchsorted(both_n, np.arange(self.N + 1))
        self.adj_o, self.adj_t, self.adj_e, self.adj_r = both_o, both_t, both_e, both_r

    def is_user(self, x: int) -> bool:
        return x < self.U

    def neighbors(self, x: int, t: int, n: int | None = None, exclude=frozenset(), t_min: int | None = None):
        """Most recent n neighbours of node x with ts < t (and >= t_min), skipping removed event ids.
        Returns (nbr, ts, eid, rating) arrays in chronological order."""
        a, b = self.ptr[x], self.ptr[x + 1]
        ts = self.adj_t[a:b]
        hi = a + int(np.searchsorted(ts, t, side="left"))
        lo = a if t_min is None else a + int(np.searchsorted(ts, t_min, side="left"))
        o, tt, e, r = self.adj_o[lo:hi], self.adj_t[lo:hi], self.adj_e[lo:hi], self.adj_r[lo:hi]
        if exclude:
            keep = ~np.isin(e, np.fromiter(exclude, np.int64, len(exclude)))
            o, tt, e, r = o[keep], tt[keep], e[keep], r[keep]
        if n is not None and len(o) > n:
            o, tt, e, r = o[-n:], tt[-n:], e[-n:], r[-n:]
        return o, tt, e, r

    def edge(self, eid: int):
        i = self.pos[int(eid)]
        return int(self.src[i]), int(self.dst[i]), int(self.ts[i]), float(self.rating[i])


@functools.lru_cache(maxsize=1)
def graph() -> TGraph:
    return TGraph(load_events(), len(load_items()))
