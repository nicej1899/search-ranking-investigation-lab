"""Leakage-safe member history access. The ONLY way any model, explainer or feature reads history.

before(member, t) returns the member's rating events with timestamp STRICTLY earlier than t, in
chronological order, optionally excluding a set of event ids (the intervention), and truncated to the
most recent `n` remaining events. Because truncation happens after exclusion, removing an event lets an
older event re-enter the window - exactly what re-running the model on the edited log would do.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .model import rating_index


@dataclass
class Hist:
    eids: np.ndarray
    items: np.ndarray
    ts: np.ndarray
    ratings: np.ndarray


class HistoryStore:
    def __init__(self, events: pd.DataFrame):
        ev = events.sort_values(["member", "ts", "event_id"], kind="mergesort")
        self.by_member: dict[int, Hist] = {}
        for m, g in ev.groupby("member", sort=False):
            self.by_member[int(m)] = Hist(g.event_id.to_numpy(np.int64), g.item.to_numpy(np.int64),
                                          g.ts.to_numpy(np.int64), g.rating.to_numpy(np.float32))
        self.event_ts = dict(zip(events.event_id.to_numpy(), events.ts.to_numpy()))
        self.event_member = dict(zip(events.event_id.to_numpy(), events.member.to_numpy()))

    def before(self, member: int, t: int, exclude: frozenset | set = frozenset(), n: int | None = 50) -> Hist:
        h = self.by_member.get(int(member))
        if h is None:
            e = np.zeros(0, np.int64)
            return Hist(e, e, e, np.zeros(0, np.float32))
        pos = int(np.searchsorted(h.ts, t, side="left"))  # strictly before t
        sl = slice(0, pos)
        eids, items, ts, rat = h.eids[sl], h.items[sl], h.ts[sl], h.ratings[sl]
        if exclude:
            keep = ~np.isin(eids, np.fromiter(exclude, np.int64))
            eids, items, ts, rat = eids[keep], items[keep], ts[keep], rat[keep]
        if n is not None and len(eids) > n:
            eids, items, ts, rat = eids[-n:], items[-n:], ts[-n:], rat[-n:]
        return Hist(eids, items, ts, rat)

    def count_before(self, member: int, t: int) -> int:
        h = self.by_member.get(int(member))
        return 0 if h is None else int(np.searchsorted(h.ts, t, side="left"))


def to_batch(hists: list[Hist], ts: np.ndarray, n: int):
    """Pad histories to [B, n] tensors (left-aligned, most recent last)."""
    B = len(hists)
    items = np.zeros((B, n), np.int64)
    rat = np.zeros((B, n), np.int64)
    dt = np.zeros((B, n), np.float32)
    mask = np.zeros((B, n), bool)
    for b, h in enumerate(hists):
        k = len(h.eids)
        if k == 0:
            continue
        items[b, :k] = h.items
        rat[b, :k] = rating_index(h.ratings)
        dt[b, :k] = (ts[b] - h.ts) / 86400.0
        mask[b, :k] = True
    return (torch.from_numpy(items), torch.from_numpy(rat), torch.from_numpy(dt), torch.from_numpy(mask))
