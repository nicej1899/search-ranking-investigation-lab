"""Select the primary and backup illustrative cases with a rule written down BEFORE any variant was
evaluated on FINAL data. Selection uses only F0 outputs on SHOWCASE members, who are excluded from
the FINAL aggregate evaluation.

Rule
  1. FINAL-period explicit-intent cases of showcase members.
  2. A = highest-ranked (under F0) title with grade 0 whose coarse label is Crime/Thriller.
  3. B = highest-ranked grade-2 title ranked below A.
  4. Require F0's behaviour component to favour A over B.
  4b. (rule v2) Require the behaviour component to be the LARGEST of the three component margins, i.e. the
      case has the structure being investigated (history-driven ordering).
Revision log: rule v1 (steps 1-5 without 4b) selected E01-m281 whose behaviour share of the margin was 0.11,
i.e. history was not the main driver. 4b was added after seeing that F0-only output; no variant (F1/Fctrl)
result had been computed on FINAL or showcase data at that point. Both selections are stored.
  5. Sort: query E01 first (the deck's running query), then by number of Crime/Thriller events in the
     member's model-input window, then by behaviour share of the A-B margin.
Primary = first; backups = next two with a different member or query.

Curation step (disclosed): the rule yields an ordered shortlist. From its top three the presenter-facing
primary is E01-m470 (rule rank 2) because its judged-relevant title B ("That Darn Cat!", Children|Comedy|
Mystery) is the clearest instance of the running light-mystery request; rule rank 1 (Super Troopers as B)
is a rubric grade-2 that a human judge may well dispute. The other two top-3 cases are the backups.
These are SELECTED ILLUSTRATIVE CASES, not evidence of how often the failure happens.
"""
from __future__ import annotations

import numpy as np

from .common import CASES, dump_json, now_iso
from .explain import replay
from .ranker import load_ranker, world


CURATED_PRIMARY = "E01-m470-t1496669535"


def select(rule: str = "v2"):
    W = world()
    R = load_ranker("F0")
    labels = W.items.label.to_numpy()
    rows = []
    for c in W.cases.values():
        if not (c["split"] == "final" and c["showcase_member"] and c["kind"] == "explicit"):
            continue
        rp = replay(W, c, R)
        g = np.array(c["grades"])
        cand = np.array(c["candidates"])
        order = rp["order"]
        A = next((i for i in order if g[i] == 0 and labels[cand[i]] == "Crime/Thriller"), None)
        if A is None:
            continue
        below = order[order.index(A) + 1:]
        B = next((i for i in below if g[i] == 2), None)
        if B is None:
            continue
        comp = rp["components"]
        mb = comp["behaviour"][A] - comp["behaviour"][B]
        m = comp["total"][A] - comp["total"][B]
        if mb <= 0:
            continue
        mr = comp["relevance"][A] - comp["relevance"][B]
        mp = comp["popularity"][A] - comp["popularity"][B]
        if rule == "v2" and mb < max(mr, mp):
            continue
        n_crime = int((labels[np.array(W.store.before(c["member"], c["t"], n=50).items)] == "Crime/Thriller").sum())
        rows.append({"case_id": c["case_id"], "qid": c["qid"], "member": c["member"], "A_index": int(A), "B_index": int(B),
                     "A_item": int(cand[A]), "B_item": int(cand[B]), "n_crime_in_window": n_crime,
                     "behaviour_share": float(mb / m), "margin": float(m),
                     "component_margin": {"behaviour": float(mb), "relevance": float(mr), "popularity": float(mp)}})
    rows.sort(key=lambda r: (r["qid"] != "E01", -r["n_crime_in_window"], -r["behaviour_share"]))
    top3 = rows[:3]
    if rule == "v2" and any(r["case_id"] == CURATED_PRIMARY for r in top3):
        primary = next(r for r in top3 if r["case_id"] == CURATED_PRIMARY)
        backups = [r for r in top3 if r["case_id"] != CURATED_PRIMARY]
    else:
        primary, backups = top3[0], top3[1:]
    out = {"created": now_iso(), "rule": __doc__, "rule_version": rule, "primary": primary, "backups": backups, "n_eligible": len(rows), "curated_primary": CURATED_PRIMARY if rule == "v2" else None,
           "all_eligible": rows}
    dump_json(out, CASES / f"selected_cases_{rule}.json")
    if rule == "v2":
        dump_json(out, CASES / "selected_cases.json")
    return out


if __name__ == "__main__":
    select("v1")
    o = select("v2")
    W = world()
    for r in [o["primary"]] + o["backups"]:
        print(r["case_id"], W.items.title[r["A_item"]], "|", W.items.title[r["B_item"]], r["n_crime_in_window"], round(r["behaviour_share"], 3))
