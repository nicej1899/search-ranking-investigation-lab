"""Pick the one case to demo, by a rule fixed before looking at the results (no cherry-picking by eye).

Among the case-study searches (Requiem for a Dream first for "a drama, but please nothing depressing"), keep a case only
if the whole loop closes:
  1. evidence verified on both sides: deleting the TemGX links behind the wrong title lowers the graph model's
     confidence more than random links from the same period (p <= 0.05), and deleting those behind the right title
     raises it (p <= 0.05);
  2. the fix recommended by retrieval over verified diagnoses (srlab.xstore, the DEV-chosen recommender) works for
     this member when re-ranked live: the right title goes to #1 and the excluded title leaves the top 3.
Ties: the largest drop when the wrong title's links are deleted.
python -m srlab.bestcase -> artifacts/reports/bestcase.json
"""
from __future__ import annotations

from .common import ART, REPORTS, dump_json, load_json, now_iso


def run():
    from .live import rank
    from .server import W_, similar_ep
    W = W_()
    ids = [r["case_id"] for r in load_json(REPORTS / "casestudy.json")["across"]["rows"]]
    rows = []
    for cid in ids:
        ct = load_json(ART / "cache" / f"contrast_{cid}_k3.json"); t = ct["test"]
        try:
            s = similar_ep(cid)
        except Exception:
            rows.append({"case_id": cid, "skipped": "not a diagnosed failure in the store"}); continue
        c = W.cases[cid]; it = W.items
        rec = s["recommend"]
        rk = {x["title"]: x["rank"] for x in rank(W, cid, "tgat2", rec)["rows"]}
        ev_a = t["del_A"]["p_value"] <= .05 and t["del_A"]["p"] < t["p0"]
        ev_b = t["del_B"]["p_value"] <= .05 and t["del_B"]["p"] > t["p0"]
        fix = rk[ct["B"]] == 1 and rk[ct["A"]] > 3
        rows.append({"case_id": cid, "member": c["member"], "A": ct["A"], "B": ct["B"], "recommend": rec, "rank_B_after": rk[ct["B"]],
                     "rank_A_after": rk[ct["A"]], "evidence_A": ev_a, "evidence_B": ev_b, "fix_works": fix, "drop": t["p0"] - t["del_A"]["p"]})
    ok = [r for r in rows if r.get("evidence_A") and r.get("evidence_B") and r.get("fix_works")]
    best = max(ok, key=lambda r: r["drop"]) if ok else None
    out = {"created": now_iso(), "rule": "evidence verified on both sides + retrieval-recommended fix works live (right title #1, excluded title out of top 3); tie: largest drop",
           "n": len(rows), "n_evidence_both": sum(bool(r.get("evidence_A") and r.get("evidence_B")) for r in rows),
           "n_fix_works": sum(bool(r.get("fix_works")) for r in rows), "n_all": len(ok), "best": best, "rows": rows}
    dump_json(out, REPORTS / "bestcase.json")
    return out


if __name__ == "__main__":
    o = run()
    print({k: v for k, v in o.items() if k != "rows"})
