"""Two targeted ranker variants (one mechanism each), fit on DEV only, scored ONCE on FINAL.

F2  history-length gate   s = exp(a0 + a1 * x) * wb*zb + wr*zr + wp*zp,   x = log(1+n_history) - log(6)
    For members with very little history the behaviour score is built from 1-4 ratings; the gate lets the ranker
    learn how much to trust it as a function of history length. a = 0 reproduces F0.
F3  negation-aware query  zr' = norm( cos(q_pos, title) - beta * cos(q_neg, title) )
    The query is split into the part the person wants and the part after "not / nothing / no / without / isn't"
    (srlab.extra.split_negation, a transparent rule standing in for an LLM query parser). beta = 0 and no negation
    reproduces F0's relevance. Only beta is fit.
F4  popularity gate       s = wb*zb + wr*zr + exp(c0 + c1*spec(q)) * wp*zp
    Lets the ranker lean less on popularity when the search names what it wants. c = 0 reproduces F0. Proposed by the
    DEV failure triage (popularity was the largest cause); fit on all DEV cases; added to the FINAL report after F2/F3
    had been scored (F2/F3 were not re-scored or changed).
F0 weights and normalisation are frozen for all. FINAL groups: original (explicit+broad, showcase members excluded),
coldstart (1-4 prior ratings), conversational (chat-style with an exclusion). Results cached; FINAL is not re-scored.
"""
from __future__ import annotations

import functools
import sys

import numpy as np
import torch

from .common import CKPT, REPORTS, dump_json, load_json, now_iso, sha256_obj
from .evaluate import ndcg_at
from .extra import CONV_BY_ID, load_extra, split_negation, violates
from .queries import TEMPLATES_BY_ID, item_tagsets
from .ranker import World, load_ranker, world, zfeat
from .text import encode

LOG6 = float(np.log(6.0))


def register_extra(W: World) -> None:
    for c in load_extra():
        W.cases.setdefault(c["case_id"], c)


@functools.lru_cache(maxsize=1)
def _content():
    W = world()
    return [set(g) for g in W.items.genre_list], item_tagsets()


def template(c):
    return CONV_BY_ID.get(c["qid"]) or TEMPLATES_BY_ID.get(c["qid"])


def violations(c) -> np.ndarray:
    if "violates_exclusion" in c:
        return np.asarray(c["violates_exclusion"], bool)
    t = template(c)
    if t is None or t.kind != "explicit" or not (t.exclude_genres or t.exclude_tags):
        return np.zeros(len(c["candidates"]), bool)
    G, T = _content()
    return np.array([violates(t, G[i], T.get(i, set())) for i in c["candidates"]], bool)


_QV: dict[str, tuple] = {}


def qparts(q: str):
    if q not in _QV:
        pos, neg = split_negation(q)
        _QV[q] = (encode(pos) if neg else None, encode(neg) if neg else None, neg)
    return _QV[q]


def feats(W: World, c: dict, F0: dict):
    f = W.raw_features(c)
    z = zfeat(f, F0["norm"])
    vp, vn, neg = qparts(c["query"])
    cand = np.asarray(c["candidates"])
    if neg:
        z["rel_pos"] = W.item_vec[cand] @ vp
        z["rel_neg"] = W.item_vec[cand] @ vn
    else:
        z["rel_pos"] = f["rel"]
        z["rel_neg"] = np.zeros(len(cand))
    z["x_hist"] = float(np.log1p(c["n_history"]) - LOG6)
    z["has_neg"] = bool(neg)
    return z


def score(z: dict, F0: dict, model: dict) -> dict:
    w = F0["weights"]
    nm, sd = F0["norm"]["rel"]
    gate = np.ones_like(z["zb"])
    zr = z["zr"]
    if model["kind"] == "F2":
        a = model["gate"]
        gate = np.full_like(z["zb"], np.exp(np.clip(a[0] + a[1] * z["x_hist"], -20, 5)))
    elif model["kind"] == "F3" and z["has_neg"]:
        zr = (z["rel_pos"] - model["beta"] * z["rel_neg"] - nm) / sd
    pg = 1.0
    if model["kind"] == "F4":
        c = model["pgate"]
        pg = float(np.exp(np.clip(c[0] + c[1] * z["spec"], -20, 5)))
    cb, cr, cp = gate * w[0] * z["zb"], w[1] * zr, pg * w[2] * z["zp"]
    return {"total": cb + cr + cp, "behaviour": cb, "relevance": cr, "popularity": cp, "gate": gate}


def _pairs(g):
    g = np.asarray(g)
    return np.where(g[:, None] > g[None, :])


def fit(kind: str, cases: list, Z: list, F0: dict, l2=1e-3):
    w = torch.tensor(F0["weights"], dtype=torch.float32)
    nm, sd = F0["norm"]["rel"]
    p = torch.zeros(2 if kind in ("F2", "F4") else 1, requires_grad=True)
    data = []
    for c, z in zip(cases, Z):
        i, j = _pairs(c["grades"])
        if len(i) == 0:
            continue
        data.append(({k: torch.tensor(v, dtype=torch.float32) for k, v in z.items() if isinstance(v, np.ndarray)},
                     (z["x_hist"], float(z["spec"])), z["has_neg"], torch.tensor(i), torch.tensor(j)))
    opt = torch.optim.LBFGS([p], lr=0.5, max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = 0.0
        for z, x, hn, i, j in data:
            if kind == "F2":
                s = torch.exp(torch.clamp(p[0] + p[1] * x[0], -20, 5)) * w[0] * z["zb"] + w[1] * z["zr"] + w[2] * z["zp"]
            elif kind == "F4":
                s = w[0] * z["zb"] + w[1] * z["zr"] + torch.exp(torch.clamp(p[0] + p[1] * x[1], -20, 5)) * w[2] * z["zp"]
            else:
                zr = (z["rel_pos"] - p[0] * z["rel_neg"] - nm) / sd if hn else z["zr"]
                s = w[0] * z["zb"] + w[1] * zr + w[2] * z["zp"]
            loss = loss - torch.nn.functional.logsigmoid(s[i] - s[j]).mean()
        loss = loss / len(data) + l2 * (p ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return p.detach().numpy().tolist(), float(closure().detach()), len(data)


def group_of(c):
    return c.get("extra") or "original"


def fit_f4_only() -> dict:
    """Fit F4 without touching F2/F3 (they were already scored on FINAL)."""
    W = world()
    register_extra(W)
    F0 = load_ranker("F0")
    dev = [c for c in W.cases.values() if c["split"] == "dev"]
    Z = [feats(W, c, F0) for c in dev]
    c4, l4_, n4 = fit("F4", dev, Z, F0)
    F4 = {"model_id": "F4", "kind": "F4", "pgate": c4, "parent": "F0", "weights": F0["weights"], "norm": F0["norm"],
          "created": now_iso(), "F0_config_hash": F0["config_hash"], "dev_cases": n4, "dev_loss": l4_,
          "changed_mechanism": "query-conditioned popularity gate: exp(c0 + c1*spec(q)) multiplies the popularity part",
          "fit_on": "all DEV cases (original + coldstart + conversational)"}
    F4["config_hash"] = sha256_obj({k: F4[k] for k in F4 if k not in ("created", "config_hash")})
    dump_json(F4, CKPT / "ranker_F4.json")
    return F4


def fit_variants() -> dict:
    W = world()
    register_extra(W)
    F0 = load_ranker("F0")
    dev2 = [c for c in W.cases.values() if c["split"] == "dev" and group_of(c) in ("original", "coldstart")]
    devc = [c for c in W.cases.values() if c["split"] == "dev" and group_of(c) == "conversational"]
    Z2 = [feats(W, c, F0) for c in dev2]
    Zc = [feats(W, c, F0) for c in devc]
    a, l2_, n2 = fit("F2", dev2, Z2, F0)
    c4, l4_, n4 = fit("F4", dev2 + devc, Z2 + Zc, F0)
    (b,), l3_, n3 = fit("F3", devc, Zc, F0)
    common = {"parent": "F0", "weights": F0["weights"], "norm": F0["norm"], "created": now_iso(), "F0_config_hash": F0["config_hash"]}
    F2 = {"model_id": "F2", "kind": "F2", "gate": a, **common, "dev_cases": n2, "dev_loss": l2_,
          "changed_mechanism": "history-length gate on behaviour: exp(a0 + a1*(log(1+n_history) - log 6))",
          "fit_on": "DEV original + DEV coldstart cases"}
    F3 = {"model_id": "F3", "kind": "F3", "beta": b, **common, "dev_cases": n3, "dev_loss": l3_,
          "changed_mechanism": "negation-aware query relevance: cos(q_pos,i) - beta*cos(q_neg,i)",
          "fit_on": "DEV conversational cases"}
    F4 = {"model_id": "F4", "kind": "F4", "pgate": c4, **common, "dev_cases": n4, "dev_loss": l4_,
          "changed_mechanism": "query-conditioned popularity gate: exp(c0 + c1*spec(q)) multiplies the popularity part",
          "fit_on": "all DEV cases (original + coldstart + conversational)"}
    for M in (F2, F3, F4):
        M["config_hash"] = sha256_obj({k: M[k] for k in M if k not in ("created", "config_hash")})
        dump_json(M, CKPT / f"ranker_{M['model_id']}.json")
    return {"F2": F2, "F3": F3, "F4": F4}


def case_rows(W, cases, F0, model):
    rows = []
    for c in cases:
        z = feats(W, c, F0)
        s = score(z, F0, model)["total"]
        g = np.asarray(c["grades"])
        v = violations(c)
        order = np.argsort(-s, kind="stable")
        explicit = c["kind"] == "explicit"
        rows.append({"case_id": c["case_id"], "member": c["member"], "group": group_of(c), "kind": c["kind"], "n_history": c["n_history"],
                     "ndcg10": ndcg_at(s, g), "top1_wrong": (bool(g[order[0]] == 0) if explicit else None),
                     "top3_violation": (bool(v[order[:3]].any()) if (explicit and v.any()) else None),
                     "has_negation": bool(qparts(c["query"])[2])})
    return rows


def _boot(pairs, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    members = sorted({p[0] for p in pairs})
    by = {m: np.array([[p[1], p[2]] for p in pairs if p[0] == m], float) for m in members}
    bs = []
    for _ in range(n_boot):
        arr = np.concatenate([by[m] for m in rng.choice(members, len(members), replace=True)])
        bs.append(arr[:, 1].mean() - arr[:, 0].mean())
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


SLICES = {
    "original": lambda r: r["group"] == "original",
    "original explicit": lambda r: r["group"] == "original" and r["kind"] == "explicit",
    "original broad ('what should I watch')": lambda r: r["group"] == "original" and r["kind"] == "broad",
    "original with 'not'/'nothing'": lambda r: r["group"] == "original" and r["has_negation"],
    "coldstart (1-4 ratings)": lambda r: r["group"] == "coldstart",
    "conversational": lambda r: r["group"] == "conversational",
}


def compare_rows(base, var):
    out = {}
    for name, f in SLICES.items():
        rb = [r for r in base if f(r)]
        rv = {r["case_id"]: r for r in var}
        if not rb:
            continue
        e = {"n_cases": len(rb), "n_members": len({r["member"] for r in rb})}
        for m in ("ndcg10", "top1_wrong", "top3_violation"):
            pairs = [(r["member"], float(r[m]), float(rv[r["case_id"]][m])) for r in rb
                     if r[m] is not None and rv[r["case_id"]][m] is not None and not np.isnan(float(r[m]))]
            if not pairs:
                continue
            e[m] = {"baseline": float(np.mean([p[1] for p in pairs])), "variant": float(np.mean([p[2] for p in pairs])),
                    "delta": float(np.mean([p[2] - p[1] for p in pairs])), "ci95": _boot(pairs), "n": len(pairs)}
        out[name] = e
    return out


def final_eval(force=False) -> dict:
    path = REPORTS / "fixes_final.json"
    if path.exists() and not force:
        d = load_json(path)
        if "F4_vs_F0" not in d and (CKPT / "ranker_F4.json").exists():
            W = world()
            register_extra(W)
            F0 = load_ranker("F0")
            fin = [c for c in W.cases.values() if c["split"] == "final" and not c["showcase_member"]]
            r0 = case_rows(W, fin, F0, {"kind": "F0"})
            M4 = load_ranker("F4")
            d["F4_vs_F0"] = compare_rows(r0, case_rows(W, fin, F0, M4))
            d["params"]["F4_pgate"] = M4["pgate"]
            d["config_hashes"]["F4"] = M4["config_hash"]
            d["F4_added"] = {"at": now_iso(), "note": "F4 proposed by DEV triage and scored once; F2/F3 not re-scored"}
            dump_json(d, path)
        d["cached"] = True
        return d
    W = world()
    register_extra(W)
    F0 = load_ranker("F0")
    fin = [c for c in W.cases.values() if c["split"] == "final" and not c["showcase_member"]]
    prev = load_json(path) if path.exists() else None
    models = {"F0": {"kind": "F0"}, "F2": load_ranker("F2"), "F3": load_ranker("F3")}
    if (CKPT / "ranker_F4.json").exists():
        models["F4"] = load_ranker("F4")
    rows = {k: case_rows(W, fin, F0, M) for k, M in models.items()}
    res = {"created": now_iso(), "protocol": "FINAL original (explicit+broad) + coldstart + conversational; showcase members excluded; scored once",
           "n_cases": len(fin), "params": {"F2_gate": models["F2"]["gate"], "F3_beta": models["F3"]["beta"],
                                           **({"F4_pgate": models["F4"]["pgate"]} if "F4" in models else {})},
           "config_hashes": {k: models[k].get("config_hash") for k in models if k != "F0"},
           **{f"{k}_vs_F0": compare_rows(rows["F0"], rows[k]) for k in models if k != "F0"},
           "by_history_F0": by_history(rows["F0"]), "by_history_F2": by_history(rows["F2"]),
           "by_history_F4": by_history(rows["F4"]) if "F4" in rows else None, "cached": False}
    if prev:
        same = all(prev.get("config_hashes", {}).get(k) in (None, res["config_hashes"][k]) for k in res["config_hashes"])
        res["history"] = (prev.get("history") or []) + [{"at": now_iso(), "note": "report regenerated to add the broad-query slice; "
                                                         "models unchanged" if same else "MODELS CHANGED", "previous_created": prev.get("created")}]
        res["F4_added"] = prev.get("F4_added")
    dump_json(res, path)
    return res


HIST_BUCKETS = [("1-4", 1, 4), ("5", 5, 5), ("6-50", 6, 50), ("51+", 51, 10 ** 9)]


def by_history(rows):
    out = []
    for name, lo, hi in HIST_BUCKETS:
        rs = [r for r in rows if r["kind"] == "explicit" and lo <= r["n_history"] <= hi and r["group"] != "conversational"]
        if rs:
            out.append({"bucket": name, "n": len(rs), "ndcg10": float(np.nanmean([r["ndcg10"] for r in rs])),
                        "top1_wrong": float(np.mean([r["top1_wrong"] for r in rs]))})
    return out


if __name__ == "__main__":
    if "--fit" in sys.argv:
        r = fit_variants()
        print({k: (v.get("gate"), v.get("beta"), v["dev_loss"]) for k, v in r.items()})
    if "--fit4" in sys.argv:
        print(fit_f4_only()["pgate"])
    if "--final" in sys.argv:
        r = final_eval(force="--force" in sys.argv)
        print(r["params"])
        for k in ("F2_vs_F0", "F3_vs_F0", "F4_vs_F0"):
            if k not in r:
                continue
            for s, e in r[k].items():
                print(k, s, e["n_cases"], {m: (round(e[m]["baseline"], 4), round(e[m]["delta"], 4), [round(x, 4) for x in e[m]["ci95"]]) for m in ("ndcg10", "top1_wrong", "top3_violation") if m in e})
        print(r["by_history_F0"])
