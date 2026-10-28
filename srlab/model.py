"""TGAT-style temporal attention backbone (re-implemented; not the TemGX authors' checkpoint).

Bipartite temporal interaction graph: member --(rating, t)--> title.
A member's representation at prediction time t is computed by one temporal-attention layer over the
member's most recent N rating interactions with timestamp strictly < t (TGAT, Xu et al. 2020: functional
time encoding cos(w*dt + b), multi-head attention over temporal neighbours). Titles are represented by a
learned id embedding (UNK for titles unseen in training) plus a projection of their MovieLens genres.

State semantics (important for replay): the model has no recurrent memory. Its only input state for a
member at time t is the ordered list of that member's events before t. Removing events and rebuilding
that list (older events slide into the N-event window) is therefore an exact replay of the member-local
state. Title representations are static parameters, so a member-local intervention never changes any
shared title state in this backbone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_items: int
    n_genres: int
    d: int = 64
    d_time: int = 16
    heads: int = 2
    max_hist: int = 50
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


class TimeEncode(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # TGAT initialisation: geometric frequencies
        self.w = nn.Parameter(torch.from_numpy(1.0 / 10 ** np.linspace(0, 4, dim)).float())
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, dt_days: torch.Tensor) -> torch.Tensor:
        return torch.cos(dt_days.unsqueeze(-1) * self.w + self.b)


class TemporalAttentionRecommender(nn.Module):
    def __init__(self, cfg: ModelConfig, genre_mat: np.ndarray):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.item_emb = nn.Embedding(cfg.n_items + 1, d)  # index n_items = UNK
        nn.init.normal_(self.item_emb.weight, std=0.05)
        self.register_buffer("genres", torch.from_numpy(genre_mat))
        self.genre_proj = nn.Linear(cfg.n_genres, d, bias=False)
        self.rating_emb = nn.Embedding(10, 8)  # ratings 0.5..5.0 -> 0..9
        self.time = TimeEncode(cfg.d_time)
        self.kv_proj = nn.Linear(d + 8 + cfg.d_time, d)
        self.q0 = nn.Parameter(torch.randn(d) * 0.05)
        self.q_proj = nn.Linear(d + cfg.d_time, d)
        self.attn = nn.MultiheadAttention(d, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.no_hist = nn.Parameter(torch.zeros(d))
        self.member_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.score_mlp = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(d, 1))
        self.unk_mask = None  # bool tensor [n_items], True if unseen in training -> use UNK id

    def set_unk(self, seen_in_train: np.ndarray):
        self.unk_mask = torch.from_numpy(~seen_in_train.astype(bool))

    def item_repr(self, items: torch.Tensor) -> torch.Tensor:
        ids = items
        if self.unk_mask is not None:
            ids = torch.where(self.unk_mask[items], torch.full_like(items, self.cfg.n_items), items)
        return self.item_emb(ids) + self.genre_proj(self.genres[items])

    def member_repr(self, h_items, h_ratings, h_dt_days, h_mask, return_attn: bool = False):
        """h_*: [B, N]; h_mask True where a real event exists. Returns [B, d] (and attention [B, N])."""
        B, N = h_items.shape
        kv = torch.cat([self.item_repr(h_items), self.rating_emb(h_ratings), self.time(h_dt_days)], -1)
        kv = self.kv_proj(kv)
        q = self.q_proj(torch.cat([self.q0.expand(B, -1), self.time(torch.zeros(B, device=kv.device))], -1)).unsqueeze(1)
        has = h_mask.any(1)
        key_pad = ~h_mask
        key_pad = torch.where(has.unsqueeze(1), key_pad, torch.zeros_like(key_pad))  # avoid all-masked rows (NaN)
        out, w = self.attn(q, kv, kv, key_padding_mask=key_pad, need_weights=return_attn, average_attn_weights=True)
        out = out.squeeze(1)
        out = torch.where(has.unsqueeze(1), out, self.no_hist.expand(B, -1))
        hu = self.member_mlp(torch.cat([out, self.q0.expand(B, -1)], -1))
        if return_attn:
            return hu, (w.squeeze(1) if w is not None else None)
        return hu

    def score(self, hu: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """hu [B,d], items [B] or [B,K] -> logits [B] or [B,K]"""
        if items.dim() == 1:
            hi = self.item_repr(items)
            return self.score_mlp(torch.cat([hu, hi, hu * hi], -1)).squeeze(-1)
        B, K = items.shape
        hi = self.item_repr(items)
        hu_e = hu.unsqueeze(1).expand(B, K, -1)
        return self.score_mlp(torch.cat([hu_e, hi, hu_e * hi], -1)).squeeze(-1)


def rating_index(r: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(r) * 2).astype(np.int64) - 1, 0, 9)
