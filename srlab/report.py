"""Assemble evaluation_summary.json (for the UI) and docs/EVALUATION_REPORT.md from stored runs.
Never re-scores FINAL: it only reads the cached per-model evaluation files."""
from __future__ import annotations

import numpy as np

from .common import REPORTS, ROOT, dump_json, load_json, now_iso
from .evaluate import compare
from .text import negation_diagnostic


def build():
    out = {"created": now_iso(), "comparisons": {}}
    for proto in ("qi-dev-v1", "qi-final-v1"):
        evs = {m: load_json(REPORTS / f"eval_{proto}_{m}.json") for m in ("F0", "F1", "Fctrl")}
        out["comparisons"][proto] = {}
        for v in ("F1", "Fctrl"):
            c = compare(evs["F0"], evs[v])
            c["runs"] = {"F0": evs["F0"]["run_id"], v: evs[v]["run_id"]}
            c["run_created"] = {"F0": evs["F0"]["created"], v: evs[v]["created"]}
            out["comparisons"][proto][f"F0_vs_{v}"] = c
        out["comparisons"][proto]["absolute"] = {m: {"ndcg10_explicit": float(np.nanmean([r["ndcg10"] for r in e["rows"] if r["kind"] == "explicit"])),
                                                     "ndcg10_broad": float(np.nanmean([r["ndcg10"] for r in e["rows"] if r["kind"] == "broad"])),
                                                     "n_cases": len(e["rows"])} for m, e in evs.items()}
        # regression examples: explicit FINAL cases where F1 is worst vs F0
        if proto == "qi-final-v1":
            d = sorted(((b["ndcg10"] - a["ndcg10"], a["case_id"]) for a, b in zip(evs["F0"]["rows"], evs["F1"]["rows"])
                        if a["kind"] == "explicit"), key=lambda x: x[0])
            out["regression_cases_F1_explicit"] = [{"case_id": cid, "delta_ndcg10": dd} for dd, cid in d[:5]]
            g = sorted(((b["ndcg10"] - a["ndcg10"], a["case_id"]) for a, b in zip(evs["F0"]["rows"], evs["F1"]["rows"])
                        if a["kind"] == "broad"), key=lambda x: -x[0])
            out["improved_cases_F1_broad"] = [{"case_id": cid, "delta_ndcg10": dd} for dd, cid in g[:5]]
    tr = load_json(REPORTS / "backbone_training.json")
    out["temporal_recommendation"] = {"protocol": tr["eval_protocol"], "val": tr["val"], "test_backbone": tr["test_temporal_backbone"],
                                      "test_popularity_reference": tr["test_popularity_reference"],
                                      "note": "Backbone is shared by F0/F1/Fctrl (the variants only change the ranker), so this task is identical across them."}
    out["negation_diagnostic"] = negation_diagnostic()
    agg = load_json(REPORTS / "motif_aggregate_dev.json")
    out["motif_aggregate_dev"] = {k: v for k, v in agg.items() if k != "rows"}
    out["latency"] = load_json(REPORTS / "latency.json")
    dump_json(out, REPORTS / "evaluation_summary.json")
    write_md(out)
    return out


def _row(name, e, metric="ndcg10"):
    if metric not in e:
        return f"| {name} | {e['n_cases']} | {e['n_members']} | – | – | – | – |"
    m = e[metric]
    return (f"| {name} | {e['n_cases']} | {e['n_members']} | {m['baseline']:.4f} | {m['variant']:.4f} | {m['delta']:+.4f} | "
            f"[{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}] |")


def write_md(out):
    L = ["# Evaluation report", "", f"Generated {out['created']} from stored runs (FINAL scored once; run ids below).", "",
         "## What is and is not measured", "",
         "* Offline proxy metrics on a public dataset (MovieLens ml-latest-small rating interactions) with a prototype query set.",
         "* Explicit-intent judgments come from a written genre/tag rubric; broad-query relevance = member rated the title >= 4.0 within 90 days after t.",
         "* Nothing here measures Netflix production behaviour, online engagement or A/B lift.", "",
         "## Temporal recommendation task (backbone)", "",
         f"Protocol: {out['temporal_recommendation']['protocol']['candidates']}.", "",
         "| split | model | HR@10 | NDCG@10 | n |", "|---|---|---|---|---|"]
    t = out["temporal_recommendation"]
    L.append(f"| val | temporal backbone | {t['val']['HR@10']:.4f} | {t['val']['NDCG@10']:.4f} | {t['val']['n']} |")
    L.append(f"| test | temporal backbone | {t['test_backbone']['HR@10']:.4f} | {t['test_backbone']['NDCG@10']:.4f} | {t['test_backbone']['n']} |")
    L.append(f"| test | popularity-before-t reference | {t['test_popularity_reference']['HR@10']:.4f} | {t['test_popularity_reference']['NDCG@10']:.4f} | {t['test_popularity_reference']['n']} |")
    L += ["", "The small backbone does not beat a popularity reference on the test period; it is a research prototype, not a strong recommender.", ""]
    for proto, title in (("qi-dev-v1", "DEV (used for fitting/selection)"), ("qi-final-v1", "FINAL holdout (scored once)")):
        L += [f"## Query-intent ranking — {title}", ""]
        for comp in ("F0_vs_F1", "F0_vs_Fctrl"):
            c = out["comparisons"][proto][comp]
            L += [f"### {comp.replace('_', ' ')}  (runs {c['runs']})", "", "NDCG@10, paired bootstrap over members (95% CI).", "",
                  "| slice | cases | members | F0 | variant | delta | 95% CI |", "|---|---|---|---|---|---|---|"]
            for name, e in c["slices"].items():
                L.append(_row(name, e))
            e = c["slices"]["explicit"]
            if "loweng_rel_recall10" in e:
                L += ["", "Low-engagement relevant recall@10 (explicit): " + _row("explicit", e, "loweng_rel_recall10")]
            L.append("")
    L += ["## Regression and improvement examples (FINAL, F1 vs F0)", ""]
    L += [f"* worst explicit: `{r['case_id']}` ΔNDCG@10 {r['delta_ndcg10']:+.3f}" for r in out["regression_cases_F1_explicit"]]
    L += [f"* best broad: `{r['case_id']}` ΔNDCG@10 {r['delta_ndcg10']:+.3f}" for r in out["improved_cases_F1_broad"]]
    n = out["negation_diagnostic"]
    L += ["", "## Text-encoder negation diagnostic", "",
          "cos(query, query without 'not') = %.3f; cos(query, 'intense') = %.3f. %s" % (n["cos(query, query_without_not)"], n["cos(query, 'intense')"], n["note"]), ""]
    a = out["motif_aggregate_dev"]
    L += ["## Context-adjusted motif evidence (DEV comparable failures)", "", f"{a['comparable_cases']} comparable cases. {a['aggregation_note']}", "",
          "| motif | cases with motif | members | mean S | frac S>0 | 95% CI |", "|---|---|---|---|---|---|"]
    for k, v in a["per_motif"].items():
        if v.get("cases_with_motif"):
            L.append(f"| {k} {v['name']} | {v['cases_with_motif']} | {v['members']} | {v['mean_S']:+.3f} | {v['frac_S_positive']:.2f} | "
                     f"[{v['ci95_member_bootstrap'][0]:+.3f}, {v['ci95_member_bootstrap'][1]:+.3f}] |")
    lat = out["latency"]
    L += ["", "## Latency (measured)", "", f"Hardware: {lat['hardware']}. Workload: {lat['workload']}.", "",
          "| stage | n | p50 ms | p95 ms |", "|---|---|---|---|"]
    for k, v in lat["warm"].items():
        L.append(f"| {k} (warm) | {v['n']} | {v['p50_ms']:.2f} | {v['p95_ms']:.2f} |")
    L.append(f"| cold first inference | 1 | {lat['cold_first_call_ms']['inference']:.0f} | – |")
    L += ["", "Cold = first call in a fresh process (spaCy query encoding, torch warm-up). No result caches in these timings.", "",
          "## Limitations", "",
          "* Small curated query set; rubric judgments are a proxy for human relevance; templates differ between DEV and FINAL.",
          "* The dense-history slice has few members; intervals are wide.",
          "* Runtime metadata unavailable: the '<100 minutes' constraint is not applied anywhere.",
          "* Explanation and motif scores describe model dependence on inputs, not member preferences or causal effects of behaviour."]
    (ROOT / "docs" / "EVALUATION_REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    build()
    print((ROOT / "docs" / "EVALUATION_REPORT.md").read_text()[:4000])
