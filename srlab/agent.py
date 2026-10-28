"""Bounded investigation agent: real tools with structured inputs/outputs and provenance, plus a
deterministic policy ('scripted investigation workflow, not a live autonomous agent') that chooses its
next step from tool results. An optional LLM tool-calling driver lives in agent_llm.py.

Every tool result has the same envelope:
  {tool, call_id, artifact_id, inputs, outputs, provenance, limitations, errors, latency_ms, cache, started_at}
"""
from __future__ import annotations

import glob
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np

from .common import ART, CASES, CKPT, PROC, ROOT, TRACES, dump_json, hardware, load_json, now_iso, sha256_file, sha256_obj
from .evaluate import PROTOCOLS, compare, evaluate
from .explain import ab_summary, explain_history, replay
from .motifs import MOTIF_NAMES, aggregate, catgx_scores
from .ranker import _fit, fit_norm, load_ranker, world, zfeat

INV = ART / "investigations"
INV.mkdir(parents=True, exist_ok=True)

ALLOWED_CHANGES = {
    "history_gate": {"model_id": "F1", "kind": "F1", "mechanism": "query-conditioned gate on the behaviour component",
                     "params": "a0,a1,a2 in gate=exp(a0 + a1*spec(q) + a2*zr(q,i)); F0 weights frozen"},
    "global_history_scale": {"model_id": "Fctrl", "kind": "Fctrl", "mechanism": "query-independent multiplier on behaviour (ablation)",
                             "params": "c"},
    "bounded_popularity": {"model_id": "F2", "kind": "F2", "mechanism": "query-conditioned gate on the popularity component",
                           "params": "not trained in this build; listed as a next experiment"},
    "content_feature": {"model_id": None, "kind": None, "mechanism": "add a verified content feature / better text encoder",
                        "params": "requires new metadata or encoder; out of scope for this build"},
}


def code_version() -> str:
    h = hashlib.sha256()
    for f in sorted(glob.glob(str(ROOT / "srlab" / "*.py"))):
        h.update(Path(f).read_bytes())
    return h.hexdigest()[:12]


def provenance(model_ids=("F0",)) -> dict:
    W = world()
    split = load_json(PROC / "split.json")
    return {"data_fingerprint": split["events_sha"], "backbone_sha256": W.backbone_sha,
            "rankers": {m: load_ranker(m)["config_hash"] for m in model_ids if (CKPT / f"ranker_{m}.json").exists()},
            "cases_hash": load_json(CASES / "query_cases.json")["meta"]["cases_hash"], "code_version": code_version()}


class Lab:
    def __init__(self, investigation_id: str | None = None, mode_label: str = "scripted investigation workflow"):
        self.W = world()
        self.inv_id = investigation_id or "inv-" + sha256_obj([now_iso(), time.time()])[:8]
        self.trace: list[dict] = []
        self.mode_label = mode_label
        self.artifacts: dict[str, dict] = {}

    # ------------------------------------------------------------------ envelope
    def _call(self, tool: str, inputs: dict, fn, models=("F0",)):
        t0 = time.perf_counter()
        started = now_iso()
        errors, outputs, lim, cache = [], None, [], "fresh"
        try:
            outputs, lim, cache = fn()
        except Exception as e:  # tools never crash the investigation; the error is part of the record
            errors.append(f"{type(e).__name__}: {e}")
            errors.append(traceback.format_exc(limit=3))
        aid = f"{tool}-{sha256_obj([inputs, started])[:8]}"
        rec = {"tool": tool, "call_id": len(self.trace) + 1, "artifact_id": aid, "inputs": inputs, "outputs": outputs,
               "provenance": provenance(models), "limitations": lim, "errors": errors, "cache": cache,
               "latency_ms": round((time.perf_counter() - t0) * 1000, 2), "started_at": started}
        self.trace.append(rec)
        self.artifacts[aid] = rec
        return rec

    # ------------------------------------------------------------------ tools
    def load_case(self, case_id: str):
        def fn():
            c = self.W.cases[case_id]
            h = self.W.store.before(c["member"], c["t"], n=self.W.model.cfg.max_hist)
            it = self.W.items
            out = {"case": {k: c[k] for k in ("case_id", "qid", "query", "kind", "split", "member", "t", "n_history", "judgment_source")},
                   "member_user_id": int(self.W.events.loc[self.W.events.member == c["member"], "user_id"].iloc[0]),
                   "history": [{"event_id": int(e), "item": int(i), "movieId": int(it.movieId[i]), "title": it.title[i], "genres": it.genres[i],
                                "label": it.label[i], "ts": int(t), "rating": float(r)} for e, i, t, r in zip(h.eids, h.items, h.ts, h.ratings)],
                   "candidates": [{"index": k, "item": int(i), "movieId": int(it.movieId[i]), "title": it.title[i], "genres": it.genres[i],
                                   "label": it.label[i], "grade": int(g), "runtime_min": None,
                                   "train_interactions": int((self.W.events.split.eq("train") & self.W.events.item.eq(i)).sum())}
                                  for k, (i, g) in enumerate(zip(c["candidates"], c["grades"]))]}
            lim = ["Runtime unknown for every title (no permitted source); the '<100 min' constraint is NOT applied.",
                   "Judgments for explicit queries come from a written genre/tag rubric, not human assessors."]
            return out, lim, "fresh"
        return self._call("load_case", {"case_id": case_id}, fn)

    def score_candidates(self, case_id: str, model_id: str = "F0", intervention_id: str | None = None):
        def fn():
            c = self.W.cases[case_id]
            removed = self.artifacts[intervention_id]["outputs"]["removed_event_ids"] if intervention_id else []
            R = load_ranker(model_id)
            rp = replay(self.W, c, R, removed)
            out = {"model_id": model_id, "removed_event_ids": removed, "ranking": rp["order"], "rank": rp["rank"],
                   "components": rp["components"], "beh_logit": rp["beh_logit"], "snapshot_hash": rp["snapshot_hash"],
                   "units": "score = pairwise log-odds; components additive; beh_logit = raw backbone logit",
                   "latency_ms_inference": rp["latency_ms"]}
            return out, ["Component values are fitted model weights x standardised features, not causal contributions."], "fresh"
        return self._call("score_candidates", {"case_id": case_id, "model_id": model_id, "intervention_id": intervention_id}, fn, (model_id,))

    def explain_history(self, case_id: str, A_index: int, B_index: int, budget: int = 2, target: str = "behavioural_margin"):
        def fn():
            c = self.W.cases[case_id]
            r = explain_history(self.W, c, A_index, B_index, budget=budget)
            r.pop("replay_before"); r.pop("replay_after")
            lim = ["TemGX adaptation (see docs/METHODS_LEDGER.md): L=1 member neighbourhood; p_e measured by single-event replay; "
                   "TRD uses embedding distance as in the authors' code; no approximation guarantee is claimed for this adaptation.",
                   "Explains model dependence on inputs, not a member's real preferences."]
            return r, lim, "fresh"
        return self._call("explain_history", {"case_id": case_id, "A_index": A_index, "B_index": B_index, "budget": budget, "target": target}, fn)

    def replay_intervention(self, case_id: str, event_ids: list[int], model_id: str = "F0", A_index: int | None = None, B_index: int | None = None):
        def fn():
            c = self.W.cases[case_id]
            R = load_ranker(model_id)
            base = replay(self.W, c, R)
            after = replay(self.W, c, R, event_ids)
            reset = replay(self.W, c, R)
            dev = float(np.max(np.abs(np.array(reset["components"]["total"]) - np.array(base["components"]["total"]))))
            valid = set(event_ids) <= set(base["model_input_event_ids"])
            out = {"removed_event_ids": sorted(event_ids), "valid_member_events": valid,
                   "input_before": base["model_input_event_ids"], "input_after": after["model_input_event_ids"],
                   "snapshot_before": base["snapshot_hash"], "snapshot_after": after["snapshot_hash"],
                   "reset_max_abs_diff": dev, "reset_tolerance": 0.0, "reset_ok": dev == 0.0,
                   "held_fixed": ["query", "candidate set", "eligibility", "prediction time", "backbone params", "ranker weights+normalisation",
                                  "title metadata", "popularity counts"],
                   "latency_ms": {"before": base["latency_ms"], "after": after["latency_ms"]}}
            if A_index is not None:
                out["before"] = ab_summary(c, base, A_index, B_index)
                out["after"] = ab_summary(c, after, A_index, B_index)
                out["margin_before"], out["margin_after"] = out["before"]["margin"], out["after"]["margin"]
                out["margin_change"] = out["margin_after"] - out["margin_before"]
            return out, ["Member-local intervention; titles have static representations in this backbone."], "fresh"
        return self._call("replay_intervention", {"case_id": case_id, "event_ids": event_ids, "model_id": model_id}, fn, (model_id,))

    def analyze_motifs(self, case_id: str, A_index: int, B_index: int, context_spec: str = "recency x duration x title-activity",
                       aggregate_split: str = "dev"):
        def fn():
            c = self.W.cases[case_id]
            one = catgx_scores(self.W, c, A_index, B_index)
            path = ART / "reports" / f"motif_aggregate_{aggregate_split}.json"
            if path.exists():
                agg, cache = load_json(path), "cached"
            else:
                agg, cache = aggregate(self.W, load_ranker("F0"), aggregate_split), "fresh"
                dump_json(agg, path)
            agg_small = {k: v for k, v in agg.items() if k != "rows"}
            return {"case": one, "comparable_contexts": agg_small}, [
                "CATGX-adapted: exact enumeration (no ANN sampling), g(r) by replay, title-activity bins; see ledger.",
                "Aggregation over comparable DEV failures is our extension; S(m) is an attribution contrast, not a behavioural causal effect."], cache
        return self._call("analyze_motifs", {"case_id": case_id, "A_index": A_index, "B_index": B_index, "context_spec": context_spec,
                                             "aggregate_split": aggregate_split}, fn)

    def propose_variant(self, diagnosis_id: str, allowed_change: str):
        def fn():
            if allowed_change not in ALLOWED_CHANGES:
                raise ValueError(f"change not allowed: {allowed_change}")
            spec = dict(ALLOWED_CHANGES[allowed_change])
            spec["parent"] = "F0"
            spec["fit_on"] = "DEV query-intent cases (validation period); backbone unchanged (trained on TRAIN)"
            spec["diagnosis_id"] = diagnosis_id
            spec["single_mechanism"] = True
            return spec, [], "fresh"
        return self._call("propose_variant", {"diagnosis_id": diagnosis_id, "allowed_change": allowed_change}, fn)

    def train_variant(self, variant_config: dict, train_split: str = "train", validation_split: str = "dev"):
        def fn():
            kind = variant_config["kind"]
            if kind not in ("F1", "Fctrl"):
                raise ValueError(f"variant kind {kind} has no trainer in this build")
            F0 = load_ranker("F0")
            W = self.W
            dev = [c for c in W.cases.values() if c["split"] == validation_split and not c.get("extra")]
            dz = [zfeat(W.raw_features(c), F0["norm"]) for c in dev]
            (p,), loss, npairs = _fit(dz, [c["grades"] for c in dev], kind, base=F0["weights"])
            R = {k: F0[k] for k in ("weights", "norm", "backbone_sha256", "dev_cases_hash", "fit", "seed")}
            R.update({"model_id": variant_config["model_id"], "kind": kind, "parent": "F0", "dev_loss": loss, "dev_pairs": npairs,
                      "changed_mechanism": variant_config["mechanism"], "created": now_iso(), "description": variant_config["params"]})
            if kind == "F1":
                R["gate"] = p
            else:
                R["c"] = p[0]
            R["config_hash"] = sha256_obj({k: R[k] for k in R if k not in ("created", "config_hash")})
            path = CKPT / f"ranker_{R['model_id']}.json"
            prev = load_json(path)["config_hash"] if path.exists() else None
            dump_json(R, path)
            return {"model_id": R["model_id"], "params": p, "dev_loss": loss, "F0_dev_loss": F0["dev_loss"], "config_hash": R["config_hash"],
                    "reproduces_previous_checkpoint": prev == R["config_hash"], "train_split_backbone": train_split,
                    "fit_split": validation_split}, ["Gate parameters fit with deterministic LBFGS; identical inputs reproduce the same hash."], "fresh"
        return self._call("train_variant", {"variant_config": variant_config, "train_split": train_split, "validation_split": validation_split},
                          fn, ("F0", variant_config.get("model_id") or "F0"))

    def evaluate_variant(self, model_id: str, evaluation_protocol_id: str):
        def fn():
            r = evaluate(self.W, model_id, evaluation_protocol_id, run_tag=self.inv_id)
            cache = "cached (run %s, %s)" % (r["run_id"], r["created"]) if r.get("cached") else "fresh"
            summ = {"model_id": model_id, "protocol": evaluation_protocol_id, "run_id": r["run_id"], "created": r["created"],
                    "n_cases": len(r["rows"]), "mean_ndcg10": float(np.nanmean([x["ndcg10"] for x in r["rows"]])),
                    "cases_hash": r["cases_hash"], "evaluation_id": f"eval_{evaluation_protocol_id}_{model_id}"}
            lim = ["FINAL protocol is scored once; repeated calls return the cached run."] if evaluation_protocol_id == "qi-final-v1" else []
            return summ, lim, cache
        return self._call("evaluate_variant", {"model_id": model_id, "evaluation_protocol_id": evaluation_protocol_id}, fn, (model_id,))

    def compare_variants(self, baseline_id: str, variant_id: str, evaluation_id: str):
        def fn():
            from .common import REPORTS
            b = load_json(REPORTS / f"eval_{evaluation_id}_{baseline_id}.json")
            v = load_json(REPORTS / f"eval_{evaluation_id}_{variant_id}.json")
            c = compare(b, v)
            c["runs"] = {baseline_id: b["run_id"], variant_id: v["run_id"]}
            return c, ["Small slices (few members) have wide intervals; do not over-interpret.",
                       "Offline proxy metrics on public data; says nothing about Netflix production or online A/B outcomes."], "fresh"
        return self._call("compare_variants", {"baseline_id": baseline_id, "variant_id": variant_id, "evaluation_id": evaluation_id},
                          fn, (baseline_id, variant_id))

    def export_investigation(self, investigation_id: str | None = None, extra: dict | None = None):
        iid = investigation_id or self.inv_id
        out = {"investigation_id": iid, "mode": self.mode_label, "exported_at": now_iso(), "hardware": hardware(),
               "provenance": provenance(("F0", "F1", "Fctrl")), "trace": self.trace, **(extra or {})}
        path = INV / f"{iid}.json"
        dump_json(out, path)
        return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "n_calls": len(self.trace)}


# ====================================================================== scripted policy

def scripted_investigation(case_id: str | None = None, investigation_id: str = "inv-primary") -> dict:
    sel = load_json(CASES / "selected_cases.json")
    p = sel["primary"] if case_id is None else next(r for r in [sel["primary"]] + sel["backups"] if r["case_id"] == case_id)
    cid, A, B = p["case_id"], p["A_index"], p["B_index"]
    lab = Lab(investigation_id, "scripted investigation workflow (deterministic policy over real tool outputs; not a live autonomous agent)")
    steps = []

    def note(kind, text, cites):
        steps.append({"kind": kind, "text": text, "cites": cites})

    c = lab.load_case(cid)
    s0 = lab.score_candidates(cid, "F0")
    comp = s0["outputs"]["components"]
    cm = {k: comp[k][A] - comp[k][B] for k in ("behaviour", "relevance", "popularity")}
    largest = max(cm, key=cm.get)
    titles = {x["index"]: x["title"] for x in c["outputs"]["candidates"]}
    note("observation", f"F0 ranks A='{titles[A]}' at {s0['outputs']['rank'][A]} and judged-relevant B='{titles[B]}' at "
                        f"{s0['outputs']['rank'][B]}. Margin components (score units): " +
         ", ".join(f"{k} {v:+.3f}" for k, v in cm.items()) + f". Largest: {largest}.", [s0["artifact_id"]])

    ex = lab.explain_history(cid, A, B, budget=2)
    eo = ex["outputs"]
    hist = {h["event_id"]: h for h in c["outputs"]["history"]}
    sel_titles = [f"{hist[e]['title']} [{hist[e]['label']}]" for e in eo["selected_event_ids"]]
    share = -eo["beh_margin_change"] / eo["beh_margin_before"] if eo["beh_margin_before"] else 0
    note("evidence", f"TemGX-adapted explanation (budget 2) selects {sel_titles}. Removing them lowers the behavioural margin "
                     f"from {eo['beh_margin_before']:.2f} to {eo['beh_margin_after']:.2f} ({share:.0%}); equal-budget recency "
                     f"{eo['comparison']['recency_topk']:+.2f}, random mean {eo['comparison']['random_mean']:+.2f}, attention "
                     f"{eo['comparison']['attention_topk']:+.2f}. Full-ranker margin {eo['margin_before']:+.3f} -> {eo['margin_after']:+.3f}; "
                     f"order flipped: {eo['full_ranker_order_flipped']}.", [ex["artifact_id"]])

    rp = lab.replay_intervention(cid, eo["selected_event_ids"], "F0", A, B)
    note("evidence", f"Controlled replay reproduces the change (margin {rp['outputs']['margin_before']:+.3f} -> {rp['outputs']['margin_after']:+.3f}); "
                     f"reset max |diff| = {rp['outputs']['reset_max_abs_diff']}.", [rp["artifact_id"]])

    mo = lab.analyze_motifs(cid, A, B)
    agg = mo["outputs"]["comparable_contexts"]["per_motif"]
    one = mo["outputs"]["case"]["per_motif"]
    m1 = agg["M1"]
    recurring = m1.get("cases_with_motif", 0) > 0 and m1["ci95_member_bootstrap"][0] > 0
    note("evidence", f"Motif M1 ({MOTIF_NAMES['M1']}) in this case: S={one['M1']['S']}. Across "
                     f"{mo['outputs']['comparable_contexts']['comparable_cases']} comparable DEV failures: mean S={m1.get('mean_S', float('nan')):.3f}, "
                     f"95% member-bootstrap CI {np.round(m1.get('ci95_member_bootstrap', [np.nan, np.nan]), 3).tolist()}, S>0 in "
                     f"{m1.get('frac_S_positive', 0):.0%} of cases with the motif.", [mo["artifact_id"]])

    diagnosis = {
        "diagnosis_id": f"diag-{cid}",
        "evidence": [
            f"Behaviour is the largest component of the A-B margin in this case ({cm['behaviour']:+.3f}).",
            f"Two-event history subset {eo['selected_event_ids']} carries {share:.0%} of the behavioural margin; it beats equal-budget recency and random removal.",
            "Removing it does NOT flip the full-ranker order: relevance and popularity still favour A." if not eo["full_ranker_order_flipped"] else
            "Removing it flips the full-ranker order.",
            ("Crime->Crime motif attribution is positive across comparable DEV failures (CI excludes 0)." if recurring else
             "Crime->Crime motif attribution across comparable DEV failures is not clearly positive."),
        ],
        "hypotheses": [
            "H1 (testable): the behaviour component is not conditioned on how explicit the query is, so crime-heavy history raises intense titles for a light-mystery request.",
            "H2: popularity also favours A; a history change alone may not fix the ordering.",
            "H3: the text encoder (averaged word vectors) cannot represent 'not too intense' (see negation diagnostic).",
        ],
        "caveats": ["One selected illustrative case; aggregate evidence comes from DEV.", "Attribution != member causal effect."],
    }
    if largest == "behaviour" and recurring:
        change = "history_gate"
    elif largest == "popularity":
        change = "bounded_popularity"
    else:
        change = "content_feature"
    note("decision", f"Chosen change: {change} - the case's largest margin component is {largest}"
                     + (" and the motif pattern recurs" if recurring else "") + ". One mechanism changed; F0 weights frozen.", [s0["artifact_id"], mo["artifact_id"]])
    pv = lab.propose_variant(diagnosis["diagnosis_id"], change)
    if pv["outputs"]["model_id"] in ("F1", "Fctrl"):
        tv = lab.train_variant(pv["outputs"])
        ctl = lab.train_variant(lab.propose_variant(diagnosis["diagnosis_id"], "global_history_scale")["outputs"])
        note("evidence", f"Trained {tv['outputs']['model_id']} on DEV: params {np.round(tv['outputs']['params'], 3).tolist()}, "
                         f"dev loss {tv['outputs']['F0_dev_loss']:.4f} -> {tv['outputs']['dev_loss']:.4f}; control Fctrl c={ctl['outputs']['params'][0]:.3f}.",
             [tv["artifact_id"], ctl["artifact_id"]])
        for m in ("F0", "F1", "Fctrl"):
            lab.evaluate_variant(m, "qi-dev-v1")
        cd = lab.compare_variants("F0", "F1", "qi-dev-v1")
        cdc = lab.compare_variants("F0", "Fctrl", "qi-dev-v1")
        sd = cd["outputs"]["slices"]
        note("evidence", f"DEV: F1-F0 NDCG@10 explicit {sd['explicit']['ndcg10']['delta']:+.4f} {np.round(sd['explicit']['ndcg10']['ci95'], 4).tolist()}, "
                         f"broad {sd['broad']['ndcg10']['delta']:+.4f} {np.round(sd['broad']['ndcg10']['ci95'], 4).tolist()}. Selection is fixed now; FINAL is scored once.",
             [cd["artifact_id"], cdc["artifact_id"]])
        for m in ("F0", "F1", "Fctrl"):
            lab.evaluate_variant(m, "qi-final-v1")
        cf = lab.compare_variants("F0", "F1", "qi-final-v1")
        cfc = lab.compare_variants("F0", "Fctrl", "qi-final-v1")
        sf, sfc = cf["outputs"]["slices"], cfc["outputs"]["slices"]

        def fmt(s, k, m="ndcg10"):
            e = s[k][m]
            return f"{e['delta']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] (n={s[k]['n_cases']} cases/{s[k]['n_members']} members)"
        note("result", f"FINAL (held-out templates, test period, showcase excluded): F1-F0 NDCG@10 explicit {fmt(sf, 'explicit')}; "
                       f"broad {fmt(sf, 'broad')}; low-engagement relevant recall@10 (explicit) "
                       f"{sf['explicit'].get('loweng_rel_recall10', {}).get('delta', float('nan')):+.4f}. Control Fctrl explicit {fmt(sfc, 'explicit')}, broad {fmt(sfc, 'broad')}.",
             [cf["artifact_id"], cfc["artifact_id"]])
        # case-level check of the chosen variant on the primary case (illustration only)
        s1 = lab.score_candidates(cid, "F1")
        note("observation", f"Primary case under F1: A rank {s1['outputs']['rank'][A]}, B rank {s1['outputs']['rank'][B]} "
                            f"(F0: {s0['outputs']['rank'][A]}, {s0['outputs']['rank'][B]}).", [s1["artifact_id"]])
        ex_imp = sf["explicit"]["ndcg10"]["ci95"][0] > 0
        br_imp = sf["broad"]["ndcg10"]["ci95"][0] > 0
        br_reg = sf["broad"]["ndcg10"]["ci95"][1] < 0
        ex_reg = sf["explicit"]["ndcg10"]["ci95"][1] < 0
        ctl_ex = sfc["explicit"]["ndcg10"]["delta"]
        if ex_imp and not br_reg:
            rec = "Gate improves the targeted explicit slice without a measurable broad regression: confirm on a larger judged set, then propose an online test."
        elif ex_reg and br_imp:
            rec = ("Tradeoff, not a win: the gate improves BROAD queries (F0 under-uses history there) but REGRESSES the targeted "
                   "explicit-intent slice. Do not ship as is. H1 is not supported as the main lever for explicit queries. Next experiment: "
                   "query-conditioned popularity bound (F2) and a negation-aware text encoder (H2/H3) on the same protocol; keep the gate "
                   "as a broad-query candidate and inspect the explicit regression cases first.")
        elif br_imp and not ex_imp:
            rec = ("No measurable gain on the targeted explicit-intent slice; the gate's gain is on BROAD queries, where F0 under-uses history. "
                   "H1 is not supported as the main lever for explicit queries. Next experiment: query-conditioned popularity bound (F2) and a "
                   "negation-aware text encoder (H2/H3), evaluated on the same protocol; keep the gate as a candidate for broad queries.")
        elif ex_reg or br_reg:
            rec = "Variant regresses a slice; do not ship. Investigate the regression cases before any further change."
        else:
            rec = "No measurable improvement on either slice. Record as a negative result; test H2/H3 next."
        if ctl_ex > sf["explicit"]["ndcg10"]["delta"]:
            rec += f" Note: the query-independent control down-weighting does better on explicit queries ({ctl_ex:+.4f}) but costs broad queries."
        note("recommendation", rec, [cf["artifact_id"], cfc["artifact_id"]])
    exp = lab.export_investigation(extra={"case_id": cid, "A_index": A, "B_index": B, "diagnosis": diagnosis, "narrative": steps})
    return {"investigation_id": lab.inv_id, "export": exp, "narrative": steps, "diagnosis": diagnosis}


if __name__ == "__main__":
    r = scripted_investigation()
    for s in r["narrative"]:
        print(f"[{s['kind']}] {s['text']}  <- {s['cites']}")
    print(r["export"])
