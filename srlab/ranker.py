"""Hybrid search ranker  score(u, q, i, t)  and its variants.

Components (all computed only from information available before t):
  zb  behavioural score: frozen temporal backbone logit for (member history before t, title), standardised
  zr  query-title relevance: cosine(spaCy query vector, title+genre document vector), standardised
  zp  popularity: log(1 + #rating interactions of the title strictly before t), standardised
Standardisation statistics are computed on DEV (validation-period) candidate sets only.

F0      s = wb*zb + wr*zr + wp*zp                                   (weights fit on DEV pairs)
F1      s = gate(q,i)*wb*zb + wr*zr + wp*zp,  gate = exp(a0 + a1*spec(q) + a2*zr)   (gate=1 <=> F0)
        F0 weights frozen; only the 3 gate parameters are fit on DEV pairs (one mechanism changed)
Fctrl   s = c*wb*zb + wr*zr + wp*zp       (query-independent down-weighting; ablation for F1)

Score unit: pairwise log-odds. Fitting minimises  -sum log sigmoid(s_i - s_j)  over judged pairs
(grade_i > grade_j) within each DEV case, + L2. A score difference of d means the fitted model
assigns odds e^d that the higher-scored title is the more relevant one of the pair. These are model
weights, not causal contributions.
"""
from __future__ import annotations

import functools
import time

import numpy as np
import pandas as pd
import torch

from .common import CKPT, GLOBAL_SEED, PROC, dump_json, load_json, now_iso, sha256_file, sha256_obj
from .data import load_events, load_items
from .history import HistoryStore, to_batch
from .queries import load_cases
from .text import encode, encode_items, query_specificity
from .train import load_backbone


class World:
    """Everything needed for inference, loaded once. Read-only after construction."""

    def __init__(self):
        t0 = time.time()
        self.events = load_events()
        self.items = load_items()
        self.store = HistoryStore(self.events)
        self.model = load_backbone(self.items)
        self.backbone_sha = sha256_file(CKPT / "temporal_backbone_F0.pt")
        ivec_path = PROC / "item_vectors.npy"
        if ivec_path.exists():
            self.item_vec = np.load(ivec_path)
        else:
            self.item_vec = encode_items(self.items)
            np.save(ivec_path, self.item_vec)
        ev = self.events.sort_values("ts")
        self.item_ts = {int(i): g.ts.to_numpy() for i, g in ev.groupby("item")}
        self.cases = {c["case_id"]: c for c in load_cases()}
        self._qcache: dict[str, tuple] = {}
        self.load_sec = round(time.time() - t0, 2)

    def qinfo(self, q: str):
        if q not in self._qcache:
            v = encode(q)
            self._qcache[q] = (v, query_specificity(v))
        return self._qcache[q]

    def pop_before(self, items, t) -> np.ndarray:
        return np.array([np.searchsorted(self.item_ts.get(int(i), np.zeros(0)), t, "left") for i in items], float)

    @torch.no_grad()
    def behavioural_logits(self, member: int, t: int, items, exclude=frozenset()):
        n = self.model.cfg.max_hist
        h = self.store.before(member, t, exclude=exclude, n=n)
        hb = to_batch([h], np.array([t]), n)
        hu, attn = self.model.member_repr(*hb, return_attn=True)
        logits = self.model.score(hu, torch.as_tensor(np.asarray(items, np.int64)).unsqueeze(0)).squeeze(0).numpy()
        k = len(h.eids)
        return logits.astype(float), h, (attn.squeeze(0).numpy()[:k] if attn is not None else None)

    def raw_features(self, case: dict, exclude=frozenset()):
        cand = np.asarray(case["candidates"])
        beh, h, attn = self.behavioural_logits(case["member"], case["t"], cand, exclude)
        qv, spec = self.qinfo(case["query"])
        rel = self.item_vec[cand] @ qv
        pop = np.log1p(self.pop_before(cand, case["t"]))
        return {"beh": beh, "rel": rel.astype(float), "pop": pop, "spec": spec, "hist": h, "attn": attn}


@functools.lru_cache(maxsize=1)
def world() -> World:
    return World()


# ---------------------------------------------------------------- normalisation (DEV only)

def fit_norm(W: World) -> dict:
    dev = [c for c in W.cases.values() if c["split"] == "dev" and not c.get("extra")]
    B, R, P = [], [], []
    for c in dev:
        f = W.raw_features(c)
        B.append(f["beh"]); R.append(f["rel"]); P.append(f["pop"])
    B, R, P = map(np.concatenate, (B, R, P))
    return {"beh": [float(B.mean()), float(B.std())], "rel": [float(R.mean()), float(R.std())],
            "pop": [float(P.mean()), float(P.std())], "fit_on": f"DEV candidate sets ({len(dev)} cases)"}


def zfeat(f: dict, norm: dict) -> dict:
    return {"zb": (f["beh"] - norm["beh"][0]) / norm["beh"][1],
            "zr": (f["rel"] - norm["rel"][0]) / norm["rel"][1],
            "zp": (f["pop"] - norm["pop"][0]) / norm["pop"][1],
            "spec": f["spec"]}


# ---------------------------------------------------------------- scoring

def score_components(z: dict, R: dict) -> dict:
    """Return per-candidate total score and its additive components (score units)."""
    w = R["weights"]
    gate = np.ones_like(z["zb"])
    if R["kind"] == "F1":
        a = R["gate"]
        gate = np.exp(np.clip(a[0] + a[1] * z["spec"] + a[2] * z["zr"], -20, 5))
    elif R["kind"] == "Fctrl":
        gate = np.full_like(z["zb"], R["c"])
    cb = gate * w[0] * z["zb"]
    cr = w[1] * z["zr"]
    cp = w[2] * z["zp"]
    return {"total": cb + cr + cp, "behaviour": cb, "relevance": cr, "popularity": cp, "gate": gate}


# ---------------------------------------------------------------- fitting

def _pairs(grades):
    g = np.asarray(grades)
    i, j = np.where(g[:, None] > g[None, :])
    return i, j


def _fit(dev_z, dev_g, kind, base=None, l2=1e-3, steps=400, seed=GLOBAL_SEED):
    torch.manual_seed(seed)
    if kind == "F0":
        p = torch.zeros(3, requires_grad=True)
        params = [p]
    elif kind == "F1":
        a = torch.zeros(3, requires_grad=True)  # gate == 1 -> F1 starts exactly at F0
        params = [a]
    else:
        c = torch.tensor([1.0], requires_grad=True)
        params = [c]
    opt = torch.optim.LBFGS(params, lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")
    data = []
    for z, g in zip(dev_z, dev_g):
        i, j = _pairs(g)
        if len(i) == 0:
            continue
        data.append(({k: torch.tensor(v, dtype=torch.float32) if k != "spec" else torch.tensor(float(v)) for k, v in z.items()},
                     torch.tensor(i), torch.tensor(j)))
    wb = torch.tensor(base, dtype=torch.float32) if base is not None else None

    def closure():
        opt.zero_grad()
        loss = 0.0
        npair = 0
        for z, i, j in data:
            if kind == "F0":
                s = params[0][0] * z["zb"] + params[0][1] * z["zr"] + params[0][2] * z["zp"]
            elif kind == "F1":
                gate = torch.exp(torch.clamp(params[0][0] + params[0][1] * z["spec"] + params[0][2] * z["zr"], -20, 5))
                s = gate * wb[0] * z["zb"] + wb[1] * z["zr"] + wb[2] * z["zp"]
            else:
                s = params[0][0] * wb[0] * z["zb"] + wb[1] * z["zr"] + wb[2] * z["zp"]
            loss = loss - torch.nn.functional.logsigmoid(s[i] - s[j]).mean()  # each case weighs equally
            npair += 1
        loss = loss / npair + l2 * sum((q ** 2).sum() for q in params)
        loss.backward()
        return loss

    opt.step(closure)
    final = float(closure().detach())
    return [x.detach().numpy().tolist() for x in params], final, sum(len(d[1]) for d in data)


def fit_all() -> dict:
    W = world()
    norm = fit_norm(W)
    dev = [c for c in W.cases.values() if c["split"] == "dev" and not c.get("extra")]
    dev_z = [zfeat(W.raw_features(c), norm) for c in dev]
    dev_g = [c["grades"] for c in dev]
    common = {"backbone_sha256": W.backbone_sha, "norm": norm,
              "dev_cases_hash": sha256_obj([c["case_id"] for c in dev]), "fit": "pairwise logistic (LBFGS), mean over pairs within a case then over cases, L2=1e-3, DEV cases only",
              "created": now_iso(), "seed": GLOBAL_SEED}
    (w,), loss0, npairs = _fit(dev_z, dev_g, "F0")
    F0 = {"model_id": "F0", "kind": "F0", "weights": w, **common, "dev_loss": loss0, "dev_pairs": npairs,
          "description": "Baseline: fixed linear combination of behaviour, relevance, popularity"}
    (a,), loss1, _ = _fit(dev_z, dev_g, "F1", base=w)
    F1 = {"model_id": "F1", "kind": "F1", "weights": w, "gate": a, **common, "dev_loss": loss1, "dev_pairs": npairs,
          "parent": "F0", "changed_mechanism": "query-conditioned gate on behaviour component",
          "description": "gate = exp(a0 + a1*spec(q) + a2*zr(q,i)) multiplies the behaviour component; F0 weights frozen; a=0 reproduces F0"}
    (c,), lossc, _ = _fit(dev_z, dev_g, "Fctrl", base=w)
    Fc = {"model_id": "Fctrl", "kind": "Fctrl", "weights": w, "c": c[0], **common, "dev_loss": lossc, "dev_pairs": npairs,
          "parent": "F0", "changed_mechanism": "global constant multiplier on behaviour component (ablation)",
          "description": "query-independent down-weighting of behaviour; tests whether conditioning on the query matters"}
    for R in (F0, F1, Fc):
        R["config_hash"] = sha256_obj({k: R[k] for k in R if k not in ("created", "config_hash")})
        dump_json(R, CKPT / f"ranker_{R['model_id']}.json")
    return {"F0": F0, "F1": F1, "Fctrl": Fc}


def load_ranker(model_id: str) -> dict:
    return load_json(CKPT / f"ranker_{model_id}.json")


if __name__ == "__main__":
    out = fit_all()
    for k, v in out.items():
        print(k, v.get("weights"), v.get("gate"), v.get("c"), v["dev_loss"], v["dev_pairs"])
