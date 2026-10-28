"""CATGX structures made usable: measured evidence per structure + an LLM-written causal explanation grounded in it.

python -m srlab.structure            -> artifacts/reports/structure_evidence.json + structure_prompts.json
(explanations are written by an LLM from the prompts -> artifacts/reports/structure_llm.json, checked by `check`)

For each affected search (the 32 case-study searches) and each of CATGX's top structures (replay signal, TGAT-2L, 2 hops):
  - the structure's strongest concrete instance: people, titles, ratings, dates (as CATGX found it)
  - lift: how much more the model relies on this structure than on equally recent / equally active ones (CATGX)
  - causal test: delete that instance's links and re-run the model; compare with deleting the same number of random
    links from the same period (40 draws). p = share of random draws that move the model as much or more.
The LLM sees only this evidence and the query; it must cite numbers from it. `check` verifies every number it wrote
appears in the evidence (rounded as shown), so a fluent but invented number is caught.
"""
from __future__ import annotations

import re

import numpy as np

from .common import REPORTS, dump_json, load_json, now_iso

BK, L = "tgat2", 2
MODEL_NOTE = ("The graph model (TGAT, 2 layers) represents each person and each title from their recent rating links, "
              "weighted by attention and by how recent each link is; a title's representation is built from the people "
              "who rated it recently, and those people's representations from the other titles they rated. The search "
              "ranker adds this graph model's history score to a text-match score and a popularity score.")


def _name(G, it, v):
    return f"user {v}" if v < G.U else it.title[v - G.U]


def evidence_for(W, bb, case_id, n_draw=40, seed=0):
    from .server import _case_ab
    from .temgx import explore, sigmoid
    from .joint import _placebo
    from .backbones import temperature
    from . import catgx
    c, a, b, ab = _case_ab(case_id)
    u, t, it, G = c["member"], c["t"], W.items, bb.G
    cx = catgx.explain(bb, u, t, a, b, L=L, g_mode="do")
    hop, edges = explore(G, [u, a + G.U, b + G.U], t, L)
    T = temperature(BK)
    rng = np.random.default_rng(seed)
    sets, arms = [()], []
    for cd in cx["motif_cards"]:
        ex = [int(e) for e in cd["example"]["eids"]]
        plac = _placebo(edges, set(ex), n_draw, rng)
        arms.append((cd, ex, len(sets), len(sets) + 1))
        sets += [tuple(ex)] + plac
    ms = np.asarray(bb.margins(u, t, a, b, sets), float)
    m0 = float(ms[0])
    structs = []
    for cd, ex, i_ex, i_pl in arms:
        pm = ms[i_pl:i_pl + n_draw]
        drop, pdrops = m0 - ms[i_ex], m0 - pm
        e = cd["example"]
        inst = []
        for k, (x, y, ts, r) in ((k, edges[k]) for k in ex if k in edges):
            inst.append({"eid": k, "from": _name(G, it, x), "to": _name(G, it, y), "rating": r, "date": _day(ts),
                         "from_id": x, "to_id": y})
        structs.append({
            "motif": cd["motif"], "name": cd["name"], "lift": cd["lift"], "S": cd["S"], "est_count": cd["est_count"],
            "involves_member": e["involves_u"], "nodes": [{"id": int(v), "name": _name(G, it, v)} for v in e["nodes"]],
            "links": inst, "p_before": float(sigmoid(m0 / T)), "p_after": float(sigmoid(ms[i_ex] / T)),
            "p_random": float(np.mean(sigmoid(pm / T))), "p_value": float((1 + np.sum(pdrops >= drop)) / (1 + n_draw)),
            "drop_logit": float(drop), "gap_before": m0, "gap_after": float(ms[i_ex]), "gap_random": float(np.mean(pm)),
        })
    hist = sorted([(edges[k][2], _name(G, it, edges[k][1] if edges[k][0] == u else edges[k][0]), edges[k][3])
                   for k in edges if u in edges[k][:2]])[-8:]
    return {"case_id": case_id, "member": u, "query": c["query"], "A": it.title[a], "B": it.title[b],
            "A_genres": it.genres[a], "B_genres": it.genres[b], "A_id": a + G.U, "B_id": b + G.U, "member_id": u,
            "recent_ratings": [{"title": h[1], "rating": h[2], "date": _day(h[0])} for h in hist],
            "structures": structs, "n_model_runs": len(sets) + cx["n_model_inferences"]}


def _day(ts):
    import datetime
    return datetime.datetime.fromtimestamp(int(ts), datetime.timezone.utc).strftime("%Y-%m-%d")


def prompt(ev):
    lines = [f"Search: \"{ev['query']}\". The ranker put \"{ev['A']}\" ({ev['A_genres']}) first; the better match was \"{ev['B']}\" ({ev['B_genres']}).",
             f"Member {ev['member']}'s most recent ratings: " + "; ".join(f"{r['title']} {r['rating']}★ ({r['date']})" for r in ev["recent_ratings"]) + ".",
             f"How the graph model works: {MODEL_NOTE}",
             "Structures CATGX found behind the graph model's preference for the first title over the better match:"]
    for j, s in enumerate(ev["structures"], 1):
        links = "; ".join(f"{x['from']} — {x['to']} rated {x['rating']}★ on {x['date']}" for x in s["links"])
        lines.append(f"  [{j}] {s['name']} ({s['motif']}). Lift: the model relies on it {s['lift']:.1f}x as much as on equally recent, equally active structures. "
                     f"Strongest instance: {links}. Causal test: deleting this instance and re-running the model moves its confidence that "
                     f"\"{ev['A']}\" beats \"{ev['B']}\" from {100 * s['p_before']:.0f}% to {100 * s['p_after']:.0f}%; deleting the same number of random links "
                     f"from the same period gives {100 * s['p_random']:.0f}% on average (p = {s['p_value']:.2f}). In model score units, the gap between the two titles "
                     f"goes from {s['gap_before']:.2f} to {s['gap_after']:.2f} (random deletions: {s['gap_random']:.2f}).")
    lines += ["Write, for a search-ranking engineer, one JSON object per structure, in a list, with keys:",
              "  \"headline\": one sentence naming the structure in plain words with the people/titles involved;",
              "  \"mechanism\": two or three sentences on WHY this structure moves the graph model's score (the causal chain through the model as described above), concrete to these titles;",
              "  \"evidence\": one sentence citing the causal test numbers exactly as given (use the score-gap numbers when the percentages barely move);",
              "  \"verdict\": \"cause\" if p <= 0.05 and the confidence drops, else \"not shown to be a cause\";",
              "  \"action\": one sentence on what the team could change in the ranking system because of it (not 'retrain the model' unless warranted).",
              "Rules: use only facts above; every number you write must appear above; no speculation about people's motives; British or American spelling, plain English; output only the JSON list."]
    return "\n".join(lines)


NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def check(ev, items):
    """Every number in the LLM's text must appear in the evidence prompt (as written there)."""
    allowed = set(NUM.findall(prompt(ev)))
    bad = []
    for x in items:
        for k in ("headline", "mechanism", "evidence", "action"):
            for m in NUM.findall(x.get(k, "")):
                if m not in allowed:
                    bad.append((k, m))
    return bad


def run():
    from .backbones import get_backbone
    from .server import W_
    W, bb = W_(), get_backbone(BK)
    cs = load_json(REPORTS / "casestudy.json")
    ids = [r["case_id"] for r in cs["across"]["rows"]]
    ev = {}
    for cid in ids:
        ev[cid] = evidence_for(W, bb, cid)
        print(cid, [(s["motif"], round(s["p_before"], 2), round(s["p_after"], 2), round(s["p_random"], 2), round(s["p_value"], 3)) for s in ev[cid]["structures"]], flush=True)
    dump_json({"created": now_iso(), "backbone": BK, "L": L, "model_note": MODEL_NOTE, "cases": ev}, REPORTS / "structure_evidence.json")
    dump_json({cid: prompt(e) for cid, e in ev.items()}, REPORTS / "structure_prompts.json")
    return ev


if __name__ == "__main__":
    run()


def merge():
    """Combine the LLM's answers (written per batch) into structure_llm.json and record the number check."""
    ev = load_json(REPORTS / "structure_evidence.json")["cases"]
    out, bad_all = {}, {}
    for p in sorted(REPORTS.glob("structure_llm_part*.json")):
        for cid, items in load_json(p).items():
            out[cid] = items
            bad = check(ev[cid], items)
            if bad:
                bad_all[cid] = bad
    res = {"created": now_iso(), "llm": "written by Claude Sonnet from the evidence above (pre-generated for the 32 searches)", "answers": out,
           "number_check": {"cases": len(out), "with_unsupported_numbers": bad_all}}
    dump_json(res, REPORTS / "structure_llm.json")
    return res


def live_llm(text, model="claude-sonnet-5-5", max_tokens=1500):
    """Optional: ask the Anthropic API directly (only if ANTHROPIC_API_KEY is set on the server)."""
    import json as _j
    import os
    import urllib.request
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", method="POST",
                                 data=_j.dumps({"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": text}]}).encode(),
                                 headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = _j.loads(r.read())
    txt = "".join(b.get("text", "") for b in body.get("content", []))
    s, e = txt.find("["), txt.rfind("]")
    return _j.loads(txt[s:e + 1]) if s >= 0 else None
