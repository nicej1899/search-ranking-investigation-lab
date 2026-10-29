"""Demo explanation at a chosen size k: the evidence, why each piece matters (in plain words), and the counterfactual
check, all computed live on the case-study graph model (TGAT-2L, 2 hops).

For one search and budget k:
  * TemGX on "wrong #1 (A) beats right answer (B)" with node budget k -> explanation ratings E
  * for each rating e in E: leave-one-out (remove only e) and its TemGX score components (influence, connectedness,
    recency), plus how it connects the member to A (shared raters / shared titles in the 2-hop window)
  * cumulative removal in TemGX order (1..|E| ratings) vs recency-matched random sets of the same size (n_draw)
  * the whole set E vs n_draw random sets -> p-value; the full re-ranking without E
  * CATGX (replay signal): each pattern's unadjusted vs context-adjusted ratio, and removing its strongest occurrence
    vs random sets of the same size
Plain-language sentences are written from these numbers by fixed templates (no LLM), so every number is checked.
python -m srlab.kdemo C07-m290-t1453051417 3
"""
from __future__ import annotations

import datetime

import numpy as np

from .common import ART, dump_json, load_json, now_iso

BK, L = "tgat2", 2
CACHE = ART / "cache"
PLAIN = {"POP": "Recent fans piled on", "CO": "Two people rated the same movie", "SEQ": "A run of ratings in one sitting",
         "BRIDGE": "Shared taste leads to the title", "BRIDGEB": "Shared taste leads to the better match"}


def _day(ts):
    return datetime.datetime.fromtimestamp(int(ts), datetime.timezone.utc).strftime("%b %d, %Y").replace(" 0", " ")


def _clean(t):
    import re
    t = re.sub(r" \(\d{4}\)$", "", t)
    t = re.sub(r" \([^)]*\)$", "", t)
    return re.sub(r"^(.*), (The|A|An)$", r"\2 \1", t)


def explain_k(case_id, k=3, n_draw=60, seed=0):
    p = CACHE / f"kdemo_{case_id}_k{k}.json"
    if p.exists():
        return load_json(p)
    from . import catgx, temgx
    from .backbones import get_backbone, temperature
    from .joint import _placebo
    from .server import W_, _case_ab
    from .temgx import explore, sigmoid
    W, bb = W_(), get_backbone(BK)
    G, it = bb.G, W.items
    c, a, b, _ = _case_ab(case_id)
    u, t = c["member"], c["t"]
    A, B = a + G.U, b + G.U
    T = temperature(BK)
    P = lambda m: float(sigmoid(m / T))
    nm = lambda v: f"user {v}" if v < G.U else _clean(it.title[v - G.U])
    _, edges = explore(G, [u, A, B], t, L)  # eid -> (x, y, ts, rating)

    # adjacency inside the 2-hop window (ratings before the search)
    rated_by = {}   # title -> set(users)
    rates = {}      # user -> set(titles)
    for e, (x, y, ts, r) in edges.items():
        us, ti = (x, y) if x < G.U else (y, x)
        rated_by.setdefault(ti, set()).add(us)
        rates.setdefault(us, set()).add(ti)

    tx = temgx.explain(bb, u, t, a, b, L=L, k=k)
    E = [int(e) for e in tx["E_eps"]]
    Ed = {int(e["eid"]): e for e in tx["edges"]}
    node = {int(n["node"]): n for n in tx["nodes"]}
    rng = np.random.default_rng(seed)

    # leave-one-out, cumulative prefixes, whole set, random sets (prefix sizes and whole size)
    order = sorted(E, key=lambda e: -max(node.get(int(Ed[e]["a"]), {}).get("phi") or 0, node.get(int(Ed[e]["b"]), {}).get("phi") or 0))
    loo = [(e,) for e in order]
    pref = [tuple(order[:i]) for i in range(1, len(order) + 1)]
    rnd = {i: _placebo(edges, set(order[:i]), n_draw, rng) for i in range(1, len(order) + 1)}
    sets = [()] + loo + pref + [s for i in range(1, len(order) + 1) for s in rnd[i]]
    ms = np.asarray(bb.margins(u, t, a, b, sets), float)
    m0 = ms[0]
    m_loo = ms[1:1 + len(order)]
    m_pref = ms[1 + len(order):1 + 2 * len(order)]
    m_rnd = ms[1 + 2 * len(order):].reshape(len(order), n_draw) if len(order) else np.zeros((0, n_draw))
    p0 = P(m0)

    def links_between(x, y):
        return sorted(rated_by.get(x, set()) & rated_by.get(y, set())) if x >= G.U and y >= G.U else []

    ev = []
    for i, e in enumerate(order):
        x, y, ts, r = int(Ed[e]["a"]), int(Ed[e]["b"]), int(Ed[e]["ts"]), Ed[e]["rating"]
        us, ti = (x, y) if x < G.U else (y, x)
        days = int(round((t - ts) / 86400))
        when = "the same day as the search" if days == 0 else f"{days} day{'s' if days != 1 else ''} before the search"
        nd = node.get(ti if us == u else us, node.get(ti, {}))
        drop = p0 - P(m_loo[i])
        if ti == A:
            kind = "crowd"
            shared = sorted(rates.get(us, set()) & rates.get(u, set()))
            why = (f"{nm(us).capitalize()} rated {nm(A)} {r:g}★ on {_day(ts)}, {when}. The graph model builds "
                   f"{nm(A)}'s profile from its recent raters, weighting recent ones more, so this rating makes {nm(A)} look current"
                   + (f"; {nm(us)} also rated {len(shared)} title{'s' if len(shared) != 1 else ''} that member {u} rated "
                      f"(e.g. {', '.join(nm(s) for s in shared[:2])}), which ties {nm(A)} to this member's taste." if shared else "."))
        elif us == u:
            kind = "own"
            fans = links_between(ti, A)
            why = (f"Member {u} rated {nm(ti)} {r:g}★ on {_day(ts)}, {when}. The member's profile is built from "
                   f"their own recent ratings"
                   + (f", and {len(fans)} of the people who rated {nm(ti)} also rated {nm(A)} (e.g. {', '.join(nm(f) for f in fans[:2])}), "
                      f"so this rating pulls member {u} toward {nm(A)}'s audience." if fans else
                      f"; this rating shapes the member's taste profile that is compared with {nm(A)}."))
        else:
            kind = "other"
            why = (f"{nm(us).capitalize()} rated {nm(ti)} {r:g}★ on {_day(ts)}. It reaches the decision through the graph: "
                   f"it changes how {nm(ti)} and its raters are represented, which feeds into member {u} or {nm(A)}.")
        eff = (f" Removing only this rating: {100 * p0:.0f}% → {100 * P(m_loo[i]):.0f}% ({'−' if drop >= 0 else '+'}{abs(100 * drop):.1f} pts).")
        ev.append({"eid": e, "n": i + 1, "kind": kind, "from": nm(us), "to": nm(ti), "from_id": us, "to_id": ti, "rating": r, "date": _day(ts),
                   "days_before": days, "p_without": P(m_loo[i]), "drop": drop,
                   "score": {"influence": nd.get("inf"), "connectedness": nd.get("g_trd"), "recency": nd.get("decay"), "phi": nd.get("phi")},
                   "why": why + eff})

    curve = [{"n": i + 1, "p": P(m_pref[i]), "rand_mean": float(np.mean([P(m) for m in m_rnd[i]])),
              "rand_lo": float(np.quantile([P(m) for m in m_rnd[i]], 0.025)), "rand_hi": float(np.quantile([P(m) for m in m_rnd[i]], 0.975)),
              "p_value": float((1 + np.sum(m_rnd[i] <= m_pref[i])) / (1 + n_draw))} for i in range(len(order))]
    # attention baseline: remove the same number of ratings with the highest attention weight
    try:
        aw = bb.edge_weights(u, t, [a, b])
    except Exception:
        aw = {}
    att_order = sorted([e for e in edges if e in aw], key=lambda e: -aw[e])
    nmax = max(len(order), 3)
    if att_order:
        am = bb.margins(u, t, a, b, [tuple(att_order[:i + 1]) for i in range(nmax)])
        att_p = [P(m) for m in am]
    else:
        att_p = [None] * nmax
    for i, cv in enumerate(curve):
        cv["attn"] = att_p[i]
    full = curve[-1] if curve else None
    if full is not None:
        full["attn"] = att_p[len(order) - 1]
    rand_all = [P(m) for m in m_rnd[-1]] if len(order) else []

    # ranking without the evidence
    from .live import rank
    live0 = rank(W, case_id, BK, "F0", [])
    live1 = rank(W, case_id, BK, "F0", E)
    rk0 = {r["title"]: r["rank"] for r in live0["rows"]}
    rk1 = {r["title"]: r["rank"] for r in live1["rows"]}

    # nodes and edges to draw: explanation ratings + the links that connect them to A (shared raters, max 2 per title)
    show_nodes = {u, A, B} | {x for l in ev for x in (l["from_id"], l["to_id"])}
    ctx = []
    for l in ev:
        if l["kind"] == "own":
            for f in links_between(l["to_id"], A)[:2]:
                show_nodes.add(f)
                ctx += [eid for eid, (x, y, ts, r) in edges.items() if {x, y} in ({f, l["to_id"]}, {f, A})]
        if l["kind"] == "crowd":
            for ti in sorted(rates.get(l["from_id"], set()) & rates.get(u, set()))[:2]:
                show_nodes.add(ti)
                ctx += [eid for eid, (x, y, ts, r) in edges.items() if {x, y} in ({ti, l["from_id"]}, {ti, u})]
    ctx = sorted(set(ctx) - set(E))
    draw_edges = [{"eid": int(eid), "s": int(edges[eid][0]), "t": int(edges[eid][1]), "rating": edges[eid][3], "date": _day(edges[eid][2]),
                   "expl": eid in E, "n": next((l["n"] for l in ev if l["eid"] == eid), None)} for eid in E + ctx]
    draw_nodes = [{"id": int(v), "name": nm(v), "type": "user" if v < G.U else "title",
                   "role": "u" if v == u else "A" if v == A else "B" if v == B else None} for v in show_nodes]

    # CATGX: patterns, unadjusted vs adjusted, and deletion of the strongest occurrence of each
    cx = catgx.explain(bb, u, t, a, b, L=L, g_mode="do", sign="pos")
    pats = []
    occ = {m: v for m, v in cx.get("occ_by_motif", {}).items() if v}
    keys = [m for m in occ if cx["S"].get(m) is not None]
    psets, idx = [], {}
    for m in keys:
        e_m = tuple(occ[m][0]["eids"])
        idx[m] = len(psets)
        psets += [e_m] + _placebo(edges, set(e_m), n_draw, rng)
    pm = np.asarray(bb.margins(u, t, a, b, psets), float) if psets else []
    for m in keys:
        j = idx[m]
        mo, mr = pm[j], pm[j + 1:j + 1 + n_draw]
        terms = cx["terms"].get(m, [])
        num = sum(x["omega"] * x["mu"] for x in terms); den = sum(x["omega"] * x["mu_base"] for x in terms)
        o = occ[m][0]
        pe = [{"eid": int(e), "s": int(edges[e][0]), "t": int(edges[e][1]), "rating": edges[e][3], "date": _day(edges[e][2]), "in_temgx": int(e) in E}
              for e in o["eids"] if e in edges]
        pn = [{"id": int(v), "name": nm(v), "type": "user" if v < G.U else "title", "role": "u" if v == u else "A" if v == A else "B" if v == B else None}
              for v in {x for ed in pe for x in (ed["s"], ed["t"])}]
        om_ = sum(x["omega"] for x in terms) or 1.0
        mf = cx.get("margin_full", float(np.log(p0 / (1 - p0)) * T))
        pts = lambda g: 100 * (P(mf) - P(mf - g))
        _na = len(o["eids"])
        if att_order:
            _pa = P(float(bb.margins(u, t, a, b, [tuple(att_order[:_na])])[0]))
        else:
            _pa = None
        pp = {"p_full": p0, "p_attn": _pa, "pts_one": pts(num / om_), "pts_base": pts(den / om_),
              "pts_one_raw": pts(cx["pooled_g"][m]) if "pooled_g" in cx else None, "pts_any": pts(cx["pooled_g_all"]) if "pooled_g_all" in cx else None}
        pats.append({"motif": m, "name": PLAIN.get(m, m), "raw": cx["raw_ratio"].get(m), "adjusted": (num / den) if den > 0 else None, **pp,
                     "graph": {"nodes": pn, "edges": pe}, "overlap_temgx": [ed["eid"] for ed in pe if ed["in_temgx"]],
                     "S": cx["S"][m], "occurrence": {"eids": o["eids"], "nodes": [nm(v) for v in o["nodes"]], "dates": [_day(x) for x in o["ts"]]},
                     "p_without": P(mo), "p_random": float(np.mean([P(x) for x in mr])), "p_value": float((1 + np.sum(mr <= mo)) / (1 + n_draw))})
    pats.sort(key=lambda x: -(x["adjusted"] or 0))
    for x in pats:
        ra, ad = x["raw"], x["adjusted"]
        txt = (f"{x['name']}: removing its strongest occurrence ({', '.join(x['occurrence']['nodes'][:3])}): {100 * p0:.0f}% → {100 * x['p_without']:.0f}%; "
               + (f"the same number of top-attention ratings: {100 * x['p_attn']:.0f}%. " if x.get("p_attn") is not None else f"random ratings of the same size: {100 * x['p_random']:.0f}%. "))
        txt += (f"Compared only with patterns that are equally recent, last as long and involve equally active people, each removal of this pattern costs "
                f"{x['pts_one']:.1f} points of the model's preference vs {x['pts_base']:.1f} for a comparable pattern"
                + (f" (before matching: vs {x['pts_any']:.1f}). " if x.get("pts_any") is not None else ". "))
        if ra and ad:
            txt += ("Timing and activity do not explain it away. " if ad >= ra * 0.9 and ad > 1 else
                    "Timing and activity explain part of it, but a real structural effect remains. " if ad >= 1.5 else
                    "Most of its apparent importance came from timing and activity. " if ad < ra else "")
        x["why"] = txt

    # joint reading: which TemGX ratings sit inside which CATGX pattern, and what the ranker added up
    em = {m: set(v) for m, v in cx.get("eids_by_motif", {}).items()}
    sig = {x["motif"]: x for x in pats}
    for l in ev:
        l["patterns"] = [{"motif": m, "name": PLAIN.get(m, m), "adjusted": sig[m]["adjusted"] if m in sig else None,
                          "significant": bool(m in sig and sig[m]["p_value"] <= 0.05)} for m in em if l["eid"] in em[m] and m in sig]
    rows0 = {r["title"]: r for r in live0["rows"]}
    pa, pb = rows0[it.title[a]]["parts"], rows0[it.title[b]]["parts"]
    split = {kk: pa[kk] - pb[kk] for kk in ("relevance", "popularity", "behaviour")}
    out_joint = _joint(ev, pats, split, nm(A), nm(B), u, c["query"])
    out = {"case_id": case_id, "k": k, "member": u, "joint": out_joint, "score_split": split, "A": nm(A), "B": nm(B), "query": c["query"], "p0": p0,
           "evidence": ev, "curve": curve,
           "full": {"p": full["p"] if full else p0, "rand_mean": full["rand_mean"] if full else p0, "p_value": full["p_value"] if full else 1.0, "attn": full.get("attn") if full else None,
                    "rand_all": rand_all, "flips": bool(full and p0 >= 0.5 > full["p"])},
           "ranking": {"A_before": rk0[it.title[a]], "A_after": rk1[it.title[a]], "B_before": rk0[it.title[b]], "B_after": rk1[it.title[b]],
                       "top3_after": [r["title"] for r in live1["rows"][:3]]},
           "graph": {"nodes": draw_nodes, "edges": draw_edges},
           "patterns": pats, "n_draw": n_draw, "n_model_runs": int(tx["n_model_inferences"]) + len(sets) + len(psets) + int(cx["n_model_inferences"]),
           "computed_at": now_iso()}
    out["summary"] = _summary(out)
    dump_json(out, p)
    return out


def _joint(ev, pats, split, A, B, u, query=""):
    """Plain-language joint diagnosis from TemGX (which ratings) + CATGX (which patterns survive context matching)."""
    strong = [x for x in pats if x["p_value"] <= 0.05 and (x["adjusted"] or 0) >= 1.5]
    faded = [x for x in pats if x["raw"] and x["adjusted"] and x["raw"] > 1 and x["adjusted"] < 0.9 * x["raw"] and x not in strong]
    crowd = [l for l in ev if l["kind"] == "crowd"]; own = [l for l in ev if l["kind"] == "own"]
    lines = []
    if crowd:
        inside = [l for l in crowd if any(p["significant"] for p in l["patterns"])]
        lines.append(f"TemGX: {len(crowd)} of the {len(ev)} ratings are recent ratings of {A} by other people"
                     + (f"; {len(inside)} of them {'is' if len(inside) == 1 else 'are'} part of a pattern CATGX finds." if inside else "."))
    if own:
        lines.append(f"TemGX: {len(own)} {'is' if len(own) == 1 else 'are'} the member's own recent rating{'s' if len(own) > 1 else ''} "
                     f"({', '.join(l['to'] for l in own[:3])}), shared with people who also rated {A}.")
    def eff(x):
        ctl = (f"top-attention ratings: {100 * x['p_attn']:.0f}%" if x.get("p_attn") is not None else f"random ratings: {100 * x['p_random']:.0f}%")
        return (f"removing it moves the model {100 * x['p_full']:.0f}% → {100 * x['p_without']:.0f}% ({ctl}); "
                f"matched on timing and activity, each removal still costs {x['pts_one']:.1f} points vs {x['pts_base']:.1f} for a comparable pattern")
    for x in strong[:1]:
        if x["raw"] and x["adjusted"] < 0.9 * x["raw"]:
            lines.append(f"CATGX: “{x['name']}”: {eff(x)}. Timing explains part of it, but a structural effect remains.")
        else:
            lines.append(f"CATGX: “{x['name']}”: {eff(x)}: a structural reason, not just recency.")
    for x in faded[:1]:
        lines.append(f"CATGX: “{x['name']}”: matched on timing and activity, each removal costs {x['pts_one']:.1f} points vs {x['pts_base']:.1f} for a comparable pattern: "
                     + ("most of its weight was timing and activity." if x["adjusted"] <= 1.2 else "part of its weight was timing and activity."))
    if not strong:
        lines.append("CATGX: no pattern keeps its full weight after matching on timing and activity, so we treat the evidence as these specific ratings rather than a stable structure.")
    top = max(split, key=lambda kk: split[kk])
    drv = {"relevance": "matching the words of the request", "popularity": "popularity", "behaviour": "the member's history"}[top]
    lines.append(f"Ranker: {A} beats {B} mainly through {drv} (+{split[top]:.2f}); the history part adds {split['behaviour']:+.2f}.")
    hist = []
    if crowd:
        hist.append(f"recent fans of {A}" + (" (a pattern that survives matching)" if strong and strong[0]["motif"] == "POP" else ""))
    if own:
        hist.append(f"the member's ratings of {', '.join(l['to'] for l in own[:2])}")
    verdict = f"Diagnosis: {A} wins mainly through {drv}"
    verdict += (f", and the history model adds to it through {' and '.join(hist)}. " if hist else ". ")
    verdict += f"None of these signals reads what “{_exclusion(query)}” rules out. Candidate fixes: "
    fx = ["make the ranker read the exclusion in the request"]
    if top == "popularity":
        fx.append("rely less on popularity for specific requests")
    if crowd:
        fx.append("rely less on titles' latest ratings from strangers")
    verdict += "; ".join(fx) + ". Step 5 picks one using past cases."
    return {"lines": lines, "verdict": verdict, "agree": sorted({e for x in pats for e in x["overlap_temgx"]})}


def _exclusion(q):
    """The exclusion part of a request ("a fantasy movie that isn't animated" -> "isn't animated"; "a fantasy film for grown-ups" -> "for grown-ups")."""
    import re
    from .extra import IMPLIED
    m = re.search(r"\b(nothing|not|isn't|aren't|without|no)\b.*$", q or "")
    if m:
        return m.group(0).strip()
    imp = next((ph for ph in IMPLIED if ph in (q or "").lower()), None)
    return imp if imp else "the exclusion in the request"


def _summary(o):
    ev, f, r = o["evidence"], o["full"], o["ranking"]
    kinds = {"crowd": 0, "own": 0, "other": 0}
    for l in ev:
        kinds[l["kind"]] += 1
    parts = []
    if kinds["crowd"]:
        parts.append(f"{kinds['crowd']} recent rating{'s' if kinds['crowd'] > 1 else ''} of {o['A']} by other people")
    if kinds["own"]:
        parts.append(f"{kinds['own']} of the member's own rating{'s' if kinds['own'] > 1 else ''}")
    if kinds["other"]:
        parts.append(f"{kinds['other']} other rating{'s' if kinds['other'] > 1 else ''} nearby")
    s = (f"With {len(ev)} rating{'s' if len(ev) > 1 else ''} ({', '.join(parts)}), the model's preference for {o['A']} over {o['B']} goes from "
         f"{100 * o['p0']:.0f}% to {100 * f['p']:.0f}%")
    s += " and flips to the right answer." if f["flips"] else "."
    if f.get("attn") is not None:
        s += f" Removing the same number of ratings with the highest attention weight gives {100 * f['attn']:.0f}%."
    else:
        s += f" Removing the same number of random ratings from the same weeks gives {100 * f['rand_mean']:.0f}% on average."
    if r["A_after"] != r["A_before"]:
        s += f" In the full ranking, {o['A']} moves from #{r['A_before']} to #{r['A_after']}."
    else:
        s += f" The full ranking still puts {o['A']} at #{r['A_before']}: text match and popularity also push it up."
    return s


if __name__ == "__main__":
    import sys
    r = explain_k(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    print(r["summary"])
    for l in r["evidence"]:
        print(" -", l["why"])
    for x in r["patterns"]:
        print(" *", x["why"])
