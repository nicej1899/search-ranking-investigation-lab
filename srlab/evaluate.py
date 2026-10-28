"""Fixed evaluation protocols for ranker variants.

Protocol 'qi-dev-v1'   : all DEV query-intent cases (validation-period prediction times, DEV templates).
Protocol 'qi-final-v1' : FINAL query-intent cases (test-period times, FINAL templates), showcase members
                         excluded. Scored ONCE per (model, protocol); later calls return the cached run and
                         name the run that produced it. Re-scoring requires force=True and is logged.

Metrics per case: NDCG@10 (graded gains 2^g - 1), pairwise accuracy over judged pairs (g_i > g_j),
and, for explicit queries, recall@10 of LOW-engagement grade-2 titles (training-period interactions in the
lowest tertile). Slices: query kind; sparse vs dense history (<= / > the DEV median count of prior events).
Uncertainty: paired bootstrap over MEMBERS (the grouping unit); each resample draws members with
replacement and averages all their cases. Deltas are variant minus baseline on identical cases.
"""
from __future__ import annotations

import time

import numpy as np

from .common import REPORTS, dump_json, load_json, now_iso, sha256_obj
from .explain import replay
from .motifs import ActivityBins
from .ranker import World, load_ranker

PROTOCOLS = {
    "qi-dev-v1": {"split": "dev", "exclude_showcase": False},
    "qi-final-v1": {"split": "final", "exclude_showcase": True},
}


def ndcg_at(scores, grades, k=10):
    g = np.asarray(grades, float)
    order = np.argsort(-np.asarray(scores), kind="stable")
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = ((2 ** g[order][:k] - 1) * disc[: min(k, len(g))]).sum()
    ideal = ((2 ** np.sort(g)[::-1][:k] - 1) * disc[: min(k, len(g))]).sum()
    return float(dcg / ideal) if ideal > 0 else np.nan


def pair_acc(scores, grades):
    s, g = np.asarray(scores), np.asarray(grades)
    i, j = np.where(g[:, None] > g[None, :])
    return float(np.mean(s[i] > s[j])) if len(i) else np.nan


def case_cases(W: World, protocol: str):
    p = PROTOCOLS[protocol]
    return [c for c in W.cases.values() if not c.get("extra") and c["split"] == p["split"] and not (p["exclude_showcase"] and c["showcase_member"])]


def score_protocol(W: World, model_id: str, protocol: str, hist_threshold: int | None = None) -> dict:
    R = load_ranker(model_id)
    act = ActivityBins(W)
    low_cut = act.cuts[0]
    dev_hist = [c["n_history"] for c in W.cases.values() if c["split"] == "dev" and not c.get("extra")]
    thr = hist_threshold if hist_threshold is not None else int(np.median(dev_hist))
    rows = []
    t0 = time.perf_counter()
    for c in case_cases(W, protocol):
        rp = replay(W, c, R)
        s = np.array(rp["components"]["total"])
        g = np.array(c["grades"])
        cand = np.array(c["candidates"])
        low = (act.pop[cand] <= low_cut) & (g == 2)
        top10 = set(np.argsort(-s, kind="stable")[:10].tolist())
        rows.append({"case_id": c["case_id"], "member": c["member"], "qid": c["qid"], "kind": c["kind"],
                     "history": "sparse" if c["n_history"] <= thr else "dense",
                     "ndcg10": ndcg_at(s, g), "pair_acc": pair_acc(s, g),
                     "loweng_rel_recall10": (float(np.mean([i in top10 for i in np.flatnonzero(low)])) if low.any() else None)})
    return {"model_id": model_id, "protocol": protocol, "config_hash": R["config_hash"], "rows": rows,
            "hist_threshold": thr, "lowengagement_cut_train_interactions": low_cut,
            "scoring_sec": round(time.perf_counter() - t0, 2)}


def _slices(rows):
    out = {}
    keys = {"all": lambda r: True, "explicit": lambda r: r["kind"] == "explicit", "broad": lambda r: r["kind"] == "broad",
            "sparse_history": lambda r: r["history"] == "sparse", "dense_history": lambda r: r["history"] == "dense",
            "explicit&sparse": lambda r: r["kind"] == "explicit" and r["history"] == "sparse",
            "explicit&dense": lambda r: r["kind"] == "explicit" and r["history"] == "dense",
            "broad&sparse": lambda r: r["kind"] == "broad" and r["history"] == "sparse",
            "broad&dense": lambda r: r["kind"] == "broad" and r["history"] == "dense"}
    for name, f in keys.items():
        rs = [r for r in rows if f(r)]
        out[name] = rs
    return out


def compare(base: dict, var: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    assert [r["case_id"] for r in base["rows"]] == [r["case_id"] for r in var["rows"]], "case sets differ"
    rng = np.random.default_rng(seed)
    res = {}
    sb, sv = _slices(base["rows"]), _slices(var["rows"])
    for name in sb:
        rb, rv = sb[name], sv[name]
        if not rb:
            continue
        entry = {"n_cases": len(rb), "n_members": len({r["member"] for r in rb})}
        for metric in ("ndcg10", "pair_acc", "loweng_rel_recall10"):
            pairs = [(r["member"], a[metric], b[metric]) for r, a, b in zip(rb, rb, rv) if a[metric] is not None and b[metric] is not None
                     and not (isinstance(a[metric], float) and np.isnan(a[metric]))]
            if not pairs:
                continue
            members = sorted({p[0] for p in pairs})
            by = {m: np.array([[p[1], p[2]] for p in pairs if p[0] == m]) for m in members}
            base_mean = float(np.mean([p[1] for p in pairs]))
            var_mean = float(np.mean([p[2] for p in pairs]))
            boots = []
            for _ in range(n_boot):
                samp = rng.choice(members, len(members), replace=True)
                arr = np.concatenate([by[m] for m in samp])
                boots.append(arr[:, 1].mean() - arr[:, 0].mean())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            entry[metric] = {"baseline": base_mean, "variant": var_mean, "delta": var_mean - base_mean,
                             "ci95": [float(lo), float(hi)], "n_pairs": len(pairs), "n_members": len(members),
                             "wins": int(sum(p[2] > p[1] for p in pairs)), "losses": int(sum(p[2] < p[1] for p in pairs))}
        res[name] = entry
    return {"baseline": base["model_id"], "variant": var["model_id"], "protocol": base["protocol"], "slices": res,
            "bootstrap": f"paired, resampling members, n_boot={n_boot}, seed={seed}"}


def evaluate(W: World, model_id: str, protocol: str, force: bool = False, run_tag: str = "") -> dict:
    path = REPORTS / f"eval_{protocol}_{model_id}.json"
    if path.exists() and protocol == "qi-final-v1" and not force:
        out = load_json(path)
        out["cached"] = True
        return out
    sc = score_protocol(W, model_id, protocol)
    sc.update({"created": now_iso(), "run_id": sha256_obj([model_id, protocol, now_iso()])[:10], "run_tag": run_tag,
               "cases_hash": sha256_obj([r["case_id"] for r in sc["rows"]]), "cached": False, "forced": force})
    if path.exists() and force:
        log = REPORTS / "final_rescoring_log.json"
        prev = load_json(log) if log.exists() else []
        prev.append({"model_id": model_id, "protocol": protocol, "at": now_iso(), "reason": run_tag})
        dump_json(prev, log)
    dump_json(sc, path)
    return sc
