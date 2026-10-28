"""Retrieval-augmented diagnosis over VERIFIED explanations ("verified GraphRAG").

  Retrieve  -> the most similar past failures from the verified-explanation store (srlab.xstore): 343 DEV chat-request
               failures, each explained with TemGX + CATGX on both sides and checked by deletion, with which fixes
               actually worked when re-ranked live. HNSW (cosine) over the mechanism key.
  Augment   -> one evidence sheet: this search's own verified diagnosis + the retrieved diagnoses and their outcomes +
               the store's fix recommendation (majority vote of the 10 nearest by score split, chosen on DEV).
  Generate  -> an LLM writes the diagnosis for an engineer, citing the retrieved cases [1]..[3]. It may only use the
               sheet: every number in its text must appear in the sheet, every citation must exist (check()).
The measured part is retrieval (srlab.xstore.evaluate: 37% vs 24% working fixes on held-out failures). The LLM only
writes; it adds no numbers and makes no decision. Pre-generated for the case-study searches (no API key here);
live generation if ANTHROPIC_API_KEY is set (structure.live_llm).

python -m srlab.rag build   -> artifacts/reports/rag_context.json, rag_prompts.json
python -m srlab.rag merge   -> artifacts/reports/rag_llm.json (from rag_llm_part*.json) with the number/citation check
"""
from __future__ import annotations

import re
import sys

from .common import REPORTS, dump_json, load_json, now_iso

FIXNAME = {"F2": "rely less on viewing history for new members", "F3": "understand “not” in the request",
           "F4": "rely less on popularity for specific searches"}
MOTIF = {"POP": "a crowd of the title's recent raters", "CO": "a shared movie", "SEQ": "a run of the member's ratings",
         "BRIDGE": "a taste path from the member to the title", "BRIDGEB": "a taste path to the right title", "none": "no clear structure"}
KN = 3
_S: dict = {}


def _stores():
    from . import xstore as xs
    if not _S:
        _S["dev"], _S["final"] = xs.load("dev"), xs.load("final")
        _S["store"] = xs.Store(_S["dev"])
        ev = REPORTS / "xstore_eval.json"
        _S["eval"] = load_json(ev) if ev.exists() else None
        ch = (_S["eval"] or {}).get("chosen", {"features": "full key", "k": 10})
        _S["rec"] = (xs.Store(_S["dev"], xs.FEATURE_SETS[ch["features"]]), ch["k"])
    return _S


def _pct(x):
    return f"{100 * x:.0f}%"


def _kinds(k):
    n = max(1, sum(k.values()))
    return ", ".join(f"{v} of {n} from {name}" for name, v in k.items() if v) or "none"


def context(case_id):
    """The evidence sheet for one diagnosed failure (None if it is not in the store)."""
    from . import xstore as xs
    S = _stores()
    me = next((r for r in S["dev"] + S["final"] if r["case_id"] == case_id), None)
    if me is None:
        return None
    neigh = S["store"].query(me, KN, exclude=case_id)
    rs, rk = S["rec"]
    best, votes = xs.recommend(rs.query(me, rk, exclude=case_id))
    t = me["test"]
    cur = {"query": me["query"], "member": me["member"], "A": me["A"], "B": me["B"],
           "A_structure": MOTIF.get(me["motif_A"], me["motif_A"]), "B_structure": MOTIF.get(me["motif_B"], me["motif_B"]),
           "A_links": _kinds(me["kinds_A"]), "B_links": _kinds(me["kinds_B"]),
           "links_A": [f"{l['from']} – {l['to']}, rating {l['rating']} on {l['date']}" for l in me["links_A"][:3]],
           "p0": _pct(t["p0"]), "p_del_A": _pct(t["del_A"]["p"]), "p_del_B": _pct(t["del_B"]["p"]), "p_random": _pct(t["del_A"]["p_random"]),
           "fixed": {FIXNAME[m]: bool(v) for m, v in me["fixed"].items()}}
    nb = [{"ref": i + 1, "case_id": n["case_id"], "query": n["query"], "member": n["member"], "A": n["A"], "B": n["B"],
           "A_structure": MOTIF.get(n["motif_A"], n["motif_A"]), "A_links": _kinds(n["kinds_A"]),
           "similarity": round(float(s), 2), "worked": [FIXNAME[m] for m in xs.FIXES if n["fixed"][m]]}
          for i, (n, s) in enumerate(neigh)]
    ev = S["eval"] or {}
    rec = {"fix": FIXNAME[best], "votes": f"{round(votes[best] * rk)} of {rk}", "k": rk,
           "heldout_success": _pct(ev["success"]["retrieval"]) if ev else None,
           "heldout_always": _pct(ev["success"]["always_" + ev["always"]]) if ev else None,
           "heldout_always_fix": FIXNAME.get(ev.get("always"), "") if ev else None}
    return {"case_id": case_id, "current": cur, "retrieved": nb, "recommendation": rec, "n_store": len(S["dev"])}


def prompt(ctx):
    c, r = ctx["current"], ctx["recommendation"]
    L = ["You are helping a search-ranking engineer understand one failed search. Use ONLY the evidence sheet below. "
         "Every number you write must appear in the sheet exactly as written there. Cite retrieved cases as [1], [2], [3].",
         "", "EVIDENCE SHEET", "",
         f"This search: member {c['member']} asked \"{c['query']}\". The ranker put {c['A']} first; the better match was {c['B']}.",
         f"Why {c['A']} (TemGX links, CATGX structure): {c['A_links']}; structure: {c['A_structure']}. Example links: " + "; ".join(c["links_A"]) + ".",
         f"Why {c['B']}: {c['B_links']}; structure: {c['B_structure']}.",
         f"Check by deletion: the graph model prefers {c['A']} with {c['p0']} confidence; deleting {c['A']}'s explanation links gives {c['p_del_A']}; "
         f"deleting {c['B']}'s gives {c['p_del_B']}; deleting random links from the same period gives {c['p_random']}.",
         "",
         f"Retrieved from {ctx['n_store']} verified past diagnoses (most similar mechanism first):"]
    for n in ctx["retrieved"]:
        L.append(f"  [{n['ref']}] member {n['member']}, \"{n['query']}\": {n['A']} ranked above {n['B']}. Why: {n['A_links']}; structure: {n['A_structure']}. "
                 f"Fixes that worked when re-ranked: {', '.join(n['worked']) or 'none of the three'}.")
    L += ["",
          f"Store recommendation: \"{r['fix']}\" (worked for {r['votes']} most similar past failures). "
          f"On held-out failures this kind of recommendation picked a working fix {r['heldout_success']} of the time, vs {r['heldout_always']} for always using \"{r['heldout_always_fix']}\".",
          "", "Write for a product manager, in plain everyday English: say 'ratings' (not links or edges), 'how sure the model is' (not confidence or probability), 'recent fans of the film' (not the title's recent raters); no jargon, no em dashes; short sentences.",
          "Return JSON only, an object with keys:",
          '  "diagnosis": one or two short sentences: what went wrong in this search and why;',
          '  "precedent": one short sentence on what the similar past failures have in common with it, citing them;',
          '  "recommendation": one short sentence on which fix to try and how far to trust it;',
          '  "caveat": one short sentence on what this evidence does not show.']
    return "\n".join(L)


NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
KEYS = ("diagnosis", "precedent", "recommendation", "caveat")


def check(ctx, ans):
    allowed = set(NUM.findall(prompt(ctx)))
    refs = {str(n["ref"]) for n in ctx["retrieved"]}
    bad = []
    for k in KEYS:
        txt = ans.get(k, "")
        for m in re.findall(r"\[(\d+)\]", txt):
            if m not in refs:
                bad.append((k, f"[{m}]"))
        for m in NUM.findall(re.sub(r"\[\d+\]", "", txt)):
            if m not in allowed:
                bad.append((k, m))
    return bad


def build():
    ids = [r["case_id"] for r in load_json(REPORTS / "casestudy.json")["across"]["rows"]]
    ctx = {cid: c for cid in ids if (c := context(cid)) is not None}
    dump_json({"created": now_iso(), "cases": ctx, "skipped": [i for i in ids if i not in ctx]}, REPORTS / "rag_context.json")
    dump_json({cid: prompt(c) for cid, c in ctx.items()}, REPORTS / "rag_prompts.json")
    return ctx


def merge():
    ctx = load_json(REPORTS / "rag_context.json")["cases"]
    out, bad_all = {}, {}
    for p in sorted(REPORTS.glob("rag_llm_part*.json")):
        for cid, a in load_json(p).items():
            out[cid] = a
            b = check(ctx[cid], a)
            if b:
                bad_all[cid] = b
    res = {"created": now_iso(), "llm": "written by Claude Sonnet from the evidence sheet (pre-generated for the case-study searches)",
           "answers": out, "check": {"cases": len(out), "with_unsupported_numbers_or_citations": bad_all}}
    dump_json(res, REPORTS / "rag_llm.json")
    return res


if __name__ == "__main__":
    print(build() if sys.argv[1] == "build" else merge()["check"])
