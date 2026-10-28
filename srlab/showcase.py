"""Pick 50 showcase cases where the explanation is clearest. Run: python -m srlab.showcase

This is a CURATED set for demonstration (stated in the UI). It is not a random sample and is not used for
any evaluation number. Rule, applied to every explicit-intent case (DEV and FINAL) with an automatic A/B pair:
  A = highest-ranked title judged not relevant (grade 0); B = best-graded title ranked below A.
  Tier 1 (strict): history (behaviour) is the largest of the three score parts of the A-B gap AND
           TemGX-adapted (budget 2) removes >= 60% of the history part of the gap.
  Tier 2 (relaxed): history pushes A above B (behaviour part > 0) AND >= 50% removed.
  Both tiers: TemGX beats 'remove the 2 most recent' and the mean of 30 random 2-event removals.
  Rank: tier, then full ranking flips first, then share removed. At most two cases per user (different queries).
"""
from __future__ import annotations

import numpy as np

from .common import CASES, dump_json, now_iso
from .explain import explain_history, replay
from .ranker import load_ranker, world


def run(n_pick: int = 50):
    W = world()
    R = load_ranker("F0")
    it = W.items
    rows = []
    for c in W.cases.values():
        if c["kind"] != "explicit":
            continue
        rp = replay(W, c, R)
        g = c["grades"]
        order = rp["order"]
        A = next((i for i in order if g[i] == 0), None)
        if A is None:
            continue
        below = order[order.index(A) + 1:]
        B = next((i for i in below if g[i] == 2), None) or next((i for i in below if g[i] == 1), None)
        if B is None:
            continue
        comp = rp["components"]
        cm = {k: comp[k][A] - comp[k][B] for k in ("behaviour", "relevance", "popularity")}
        if cm["behaviour"] <= 0:
            continue
        strict_struct = cm["behaviour"] >= max(cm["relevance"], cm["popularity"])
        ex = explain_history(W, c, A, B, budget=2, n_random=30)
        share = -ex["beh_margin_change"] / ex["beh_margin_before"] if ex["beh_margin_before"] > 0 else 0
        cmp = ex["comparison"]
        if share < 0.5 or cmp["temgx_adapted"] <= max(cmp["recency_topk"], cmp["random_mean"]):
            continue
        tier = 1 if (strict_struct and share >= 0.6) else 2
        rows.append({"tier": tier, "case_id": c["case_id"], "member": c["member"], "qid": c["qid"], "query": c["query"], "split": c["split"],
                     "A_index": int(A), "B_index": int(B), "A_title": it.title[c["candidates"][A]], "B_title": it.title[c["candidates"][B]],
                     "rank_A": ex["rank_change"]["A"][0], "rank_B": ex["rank_change"]["B"][0],
                     "rank_B_after": ex["rank_change"]["B"][1], "flipped": bool(ex["full_ranker_order_flipped"]),
                     "share_removed": float(share), "temgx": cmp["temgx_adapted"], "recency": cmp["recency_topk"], "random": cmp["random_mean"],
                     "n_history": c["n_history"], "selected_event_ids": ex["selected_event_ids"]})
    rows.sort(key=lambda r: (r["tier"], not r["flipped"], -r["share_removed"]))
    picked, seen = [], {}
    for r in rows:
        if seen.get(r["member"], 0) >= 2:
            continue
        seen[r["member"]] = seen.get(r["member"], 0) + 1
        picked.append(r)
        if len(picked) == n_pick:
            break
    out = {"created": now_iso(), "rule": __doc__, "n_qualifying": len(rows),
           "n_explicit_cases_checked": sum(1 for c in W.cases.values() if c["kind"] == "explicit"), "picked": picked}
    dump_json(out, CASES / "showcase50.json")
    return out


if __name__ == "__main__":
    o = run()
    import collections
    print(o["n_qualifying"], o["n_explicit_cases_checked"], "flipped", sum(r["flipped"] for r in o["picked"]), "picked", len(o["picked"]),
          collections.Counter(r["tier"] for r in o["picked"]), "users", len({r["member"] for r in o["picked"]}))
    for r in o["picked"][:10]:
        print(r["case_id"], r["A_title"], "|", r["B_title"], r["flipped"], round(r["share_removed"], 2))
