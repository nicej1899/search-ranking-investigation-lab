"""Publication-style figures for the talk, drawn from the measured results in artifacts/reports/.

python3 scripts/make_figures.py  ->  docs/figures/figNN_*.pdf (vector) + .png (300 dpi)
Every number is read from a results file; nothing is typed in by hand.
Colours: wrong #1 = red, right answer = blue (CVD-safe pair); score components gold / violet / aqua, always direct-labelled.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "artifacts" / "reports"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

RED, BLUE = "#c82434", "#2a78d6"
TXT, POP, HIS = "#eda100", "#6250d6", "#1baf7a"
INK, INK2, MUTED, GRID = "#1f2328", "#52514e", "#8b8f96", "#e6e8eb"
GREYS = ["#9aa1a9", "#b4bac1", "#c9ced4", "#dadde1"]

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.labelcolor": INK2, "axes.edgecolor": "#9aa1a9", "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 100, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42, "axes.titlelocation": "left", "axes.titlepad": 8,
})


def J(name):
    return json.load(open(R / name))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def panel(ax, letter):
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, ha="right", va="bottom")


def short(s, k=30):
    s = s.replace(" (", " (")
    return s if len(s) <= k else s[: k - 1] + "…"


M = J("figdata/member290.json")
CS = J("casestudy.json")
A_T, B_T = M["A"], M["B"]
AN, BN = "Requiem for a Dream", "Out of Sight"

# ---------------------------------------------------------------- 1. ranking + score decomposition (slide: running case)
def fig_case_ranking():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharex=True)
    for ax, (mid, title) in zip(axes, [("F0", "Current ranker"), ("F3", "Understand “not” in the request")]):
        rows = M["ranks"][mid][:5]
        y = np.arange(len(rows))[::-1]
        for yi, r in zip(y, rows):
            p = r["parts"]
            lo = hi = 0.0
            for k, c in (("relevance", TXT), ("popularity", POP), ("behaviour", HIS)):
                v = p[k]
                if v >= 0:
                    ax.barh(yi, v, left=hi, color=c, height=0.62, edgecolor="white", linewidth=1); hi += v
                else:
                    ax.barh(yi, v, left=lo, color=c, height=0.62, edgecolor="white", linewidth=1); lo += v
            ax.plot([r["total"]], [yi], marker="D", color=INK, ms=4.5, zorder=5)
            tag = "ruled out" if r["breaks_exclusion"] else ("right answer" if r["title"] == B_T else "")
            col = RED if r["breaks_exclusion"] else (BLUE if r["title"] == B_T else INK2)
            ax.text(-0.04, yi, f"#{r['rank']}  {short(r['title'], 28)}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.5, color=INK)
            if tag:
                ax.text(1.02, yi, tag, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8, color=col, fontweight="bold")
        ax.axvline(0, color="#9aa1a9", lw=0.8)
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_title(title, color=INK)
        ax.set_xlabel("contribution to ranking score (standardised units)")
        ax.grid(axis="x"); ax.set_axisbelow(True)
    h = [Patch(color=TXT, label="matches the words"), Patch(color=POP, label="popular"), Patch(color=HIS, label="fits their history (graph model)"),
         Line2D([], [], marker="D", color=INK, lw=0, ms=4.5, label="total score")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(f"“{CS['query']}” · member 290 · top 5 of 30 candidates", x=0.01, y=1.04, ha="left", fontsize=9, color=INK2)
    fig.subplots_adjust(wspace=0.95, left=0.2, right=0.9)
    save(fig, "fig01_case_ranking")


# ---------------------------------------------------------------- 2. failure slices with CIs + causes
def fig_failure_slices():
    g = CS["alarm"]["by_group"]
    names = [("original", "Regular searches"), ("coldstart", "New members (1–4 ratings)"), ("conversational", "Chat-style requests")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    for i, (k, lab) in enumerate(names):
        n, f = g[k]["searches"], g[k]["failed"]
        p, lo, hi = wilson(f, n)
        col = RED if k == "conversational" else "#9aa1a9"
        a.barh(i, p, color=col, height=0.55)
        a.errorbar(p, i, xerr=[[p - lo], [hi - p]], color=INK, capsize=3, lw=1)
        a.text(hi + 0.015, i, f"{p:.0%}  ({f}/{n})", va="center", fontsize=8.5, color=INK)
    a.set_yticks(range(3)); a.set_yticklabels([l for _, l in names])
    a.set_xlim(0, 0.75); a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    a.set_xlabel("searches whose #1 result is judged irrelevant (95% Wilson CI)"); a.grid(axis="x"); a.set_axisbelow(True)
    a.set_title("Where the ranker fails (development split)"); panel(a, "a")
    causes = [("query_misread", "request misread", TXT), ("popularity_over_intent", "popularity over intent", POP), ("history_over_intent", "history over intent", HIS)]
    for i, (k, lab) in enumerate(names):
        tot = sum(g[k]["causes"].values()); left = 0
        for ck, cl, cc in causes:
            v = g[k]["causes"].get(ck, 0) / tot
            b.barh(i, v, left=left, color=cc, height=0.55, edgecolor="white", linewidth=1.2)
            if v > 0.07:
                b.text(left + v / 2, i, f"{g[k]['causes'].get(ck, 0)}", ha="center", va="center", fontsize=8, color="white" if cc != TXT else INK)
            left += v
    b.set_yticks(range(3)); b.set_yticklabels(["", "", ""]); b.set_xlim(0, 1)
    b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); b.set_xlabel("share of failures by diagnosed cause (counts in bars)")
    b.legend(handles=[Patch(color=c, label=l) for _, l, c in causes], loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    b.set_title("Why they fail (score-split diagnosis)"); panel(b, "b")
    save(fig, "fig02_failure_slices")


# ---------------------------------------------------------------- 3. real 2-hop neighbourhood with both-side evidence
def fig_neighbourhood():
    """The explanation subgraph TemGX selects on each side, drawn like a paper case study (real ratings and dates)."""
    sg = M["subgraph"]
    nid = {n["id"]: n for n in sg["nodes"]}
    u, A, B = sg["u"], sg["A"], sg["B"]
    LA, LB = M["side_A"]["links"], M["side_B"]["links"]
    def ends(l):
        return l["from_id"], l["to_id"]
    titles, users = [], []
    for l in LA + LB:
        for v in ends(l):
            if v in (u, A, B):
                continue
            (titles if nid.get(v, {}).get("type") != "user" else users).append(v)
    titles = list(dict.fromkeys(titles)); users = list(dict.fromkeys(users))
    pos = {u: (0.0, 0.0), A: (3.7, 0.95), B: (3.7, -0.95)}
    for k, v in enumerate(titles):
        pos[v] = (1.35, 1.4 - 2.8 * k / max(len(titles) - 1, 1))
    for k, v in enumerate(users):
        pos[v] = (5.45, 1.5 - 1.15 * k)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    def edge(l, col, ls):
        a_, b_ = ends(l); p1, p2 = np.array(pos[a_]), np.array(pos[b_])
        ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="-", color=col, lw=2.4, linestyle=ls, shrinkA=13, shrinkB=13), zorder=2)
        m = p1 + (p2 - p1) * (0.56 if a_ == u or b_ == u else 0.5)
        ax.text(m[0], m[1] + 0.12, f"{l['rating']:g}★ · {l['date']}", ha="center", va="center", fontsize=7.5, color=col,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"), zorder=3)
    for l in LA:
        edge(l, RED, "-")
    for l in LB:
        edge(l, BLUE, (0, (4, 2)))
    def node(v, label, fc, ec, size, marker, col=INK, bold=False, dy=-0.3, ha="center", dx=0):
        ax.scatter(*pos[v], s=size, c=fc, edgecolors=ec, linewidths=1.6, marker=marker, zorder=4)
        ax.text(pos[v][0] + dx, pos[v][1] + dy, label, ha=ha, va="center", fontsize=8.5, color=col, fontweight="bold" if bold else "normal", zorder=5)
    node(u, "member\n290", "white", INK, 520, "o", bold=True, dy=0.0, ha="right", dx=-0.25)
    node(A, AN + "\n(wrong #1)", RED, RED, 520, "s", col=RED, bold=True, dy=0.0, ha="right", dx=-0.22)
    node(B, BN + "\n(right answer)", BLUE, BLUE, 520, "s", col=BLUE, bold=True, dy=0.0, ha="left", dx=0.22)
    for v in titles:
        node(v, short(nid[v]["title"], 28), "#e5e7ea", "#6b7280", 260, "s", dy=0.0, ha="left", dx=0.15)
    for v in users:
        node(v, nid[v]["title"], "white", "#6b7280", 260, "o", dy=0.0, ha="left", dx=0.16)
    ax.text(4.2, -0.05, "The graph model builds the member from his own\nratings and each title from its recent raters;\nboth enter the history score of every candidate.",
            ha="center", va="center", fontsize=7.5, color=INK2, style="italic")
    ax.text(-0.85, -2.05, f"Full 2-hop window before the search: {len(sg['nodes'])} nodes, {len(sg['edges'])} ratings. "
            f"TemGX keeps {len(LA)} per side; removing them is tested in the next figure.", fontsize=7.5, color=INK2)
    ax.set_xlim(-0.9, 6.5); ax.set_ylim(-2.15, 1.95); ax.axis("off")
    h = [Line2D([], [], color=RED, lw=2.4, label=f"ratings that push {AN} up"), Line2D([], [], color=BLUE, lw=2.4, ls=(0, (4, 2)), label=f"ratings that push {BN} up"),
         Line2D([], [], marker="o", color="#6b7280", mfc="white", lw=0, ms=8, label="member"), Line2D([], [], marker="s", color="#6b7280", mfc="#e5e7ea", lw=0, ms=8, label="title")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize=8)
    save(fig, "fig03_explanation_subgraph")


# ---------------------------------------------------------------- 4. deletion test for member 290 with the random distribution
def fig_deletion_case():
    p0 = M["p0"]; rd = np.array(M["random_draws"]); arms = M["arms"]
    rows = [("TemGX ratings (3)", arms["TemGX links"]["p"], RED),
            ("CATGX pattern (2)", arms["CATGX structure"]["p"], "#a3141f"),
            ("newest 3 ratings of the title\n(typical GraphRAG context)", arms["1-hop, newest around the title"]["p"], "#6b7280"),
            ("newest 3 around member + title", arms["1-hop, newest around member + title"]["p"], "#6b7280"),
            ("3 highest-attention ratings", arms["attention top-n"]["p"], "#6b7280")]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    lo, hi = np.quantile(rd, [0.025, 0.975])
    ax.axvspan(lo, hi, color="#eef0f2", zorder=0)
    yr = len(rows)
    jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(rd))
    ax.scatter(rd, yr + jitter, s=6, color="#9aa1a9", alpha=0.6, zorder=2, linewidths=0)
    ax.plot([np.mean(rd)], [yr], marker="|", ms=14, color=INK, mew=1.5)
    for i, (lab, p, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([p0, p], [y, y], color=c, lw=1.2, alpha=0.6)
        ax.scatter([p], [y], s=46, color=c, zorder=4, edgecolors="white", linewidths=1)
        ax.text(p - 0.004 if p < p0 else p + 0.004, y + 0.28, f"{p:.0%}", ha="right" if p < p0 else "left", fontsize=8, color=INK)
    ax.axvline(p0, color=INK, lw=1, ls=(0, (3, 2)))
    ax.text(p0, len(rows) + 0.75, f"nothing removed {p0:.0%}", ha="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows) + 1)); ax.set_yticklabels([r[0] for r in rows][::-1] + [f"3 random ratings, same period\n(200 draws, 95% band)"])
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel(f"graph model: P({AN} beats {BN}) after removing the ratings")
    ax.grid(axis="x"); ax.set_axisbelow(True); ax.set_ylim(-0.6, len(rows) + 1.1)
    b = M["test"]["del_B"]["p"]
    ax.set_title(f"Member 290: remove the evidence, re-run the frozen model (removing {BN}'s evidence instead: {b:.0%})", fontsize=9)
    save(fig, "fig04_deletion_member290")


# ---------------------------------------------------------------- 5. deletion across 32 searches by retriever
def fig_deletion_32():
    rv = J("retrievers.json"); rows = rv["rows"]
    order = [("TemGX links", "TemGX", RED), ("CATGX structure", "CATGX\npattern", "#a3141f"),
             ("1-hop, newest around the title", "newest of\ntitle\n(GraphRAG)", "#6b7280"),
             ("1-hop, newest around member + title", "newest of\nmember\n+ title", "#6b7280"),
             ("attention top-n", "highest\nattention", "#6b7280"), ("random, same period", "random\n(same\nperiod)", "#9aa1a9")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
    rng = np.random.default_rng(1)
    for i, (k, lab, c) in enumerate(order):
        d = np.array([100 * (r["p0"] - r["arms"][k]["p"]) for r in rows if k in r["arms"]])
        a.scatter(i + rng.uniform(-0.16, 0.16, len(d)), d, s=12, color=c, alpha=0.75, linewidths=0, zorder=3)
        q1, med, q3 = np.quantile(d, [0.25, 0.5, 0.75])
        a.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=1.8, zorder=4)
        a.plot([i, i], [q1, q3], color=INK, lw=1, zorder=4)
        a.text(i, max(d) + 1.2, f"mean {d.mean():+.1f}", ha="center", fontsize=7.5, color=INK2)
    a.axhline(0, color="#9aa1a9", lw=0.8)
    a.set_xticks(range(len(order))); a.set_xticklabels([o[1] for o in order], fontsize=7.5)
    a.set_ylabel("drop in P(wrong #1 beats right) (pts)"); a.grid(axis="y"); a.set_axisbelow(True)
    a.set_title(f"Same budget (3 ratings), {len(rows)} searches; line = median, bar = IQR"); panel(a, "a")
    labs, ps, los, his = [], [], [], []
    for k, lab, c in order[1:]:
        w = rv["agg"][k]["temgx_moves_more"]; n = rv["agg"][k]["n"]
        p, lo, hi = wilson(w, n); labs.append(lab.replace("\n", " ")); ps.append(p); los.append(p - lo); his.append(hi - p)
    y = np.arange(len(labs))[::-1]
    b.barh(y, ps, color="#c9ced4", height=0.55)
    b.errorbar(ps, y, xerr=[los, his], fmt="none", ecolor=INK, capsize=3, lw=1)
    for yi, p, (k, _, _) in zip(y, ps, order[1:]):
        b.text(0.02, yi, f"{rv['agg'][k]['temgx_moves_more']}/{rv['agg'][k]['n']}", va="center", fontsize=8, color=INK)
    b.axvline(0.5, color=INK, lw=0.8, ls=(0, (3, 2)))
    b.set_yticks(y); b.set_yticklabels(labs, fontsize=7.5); b.set_xlim(0, 1)
    b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); b.set_xlabel("searches where TemGX moves the model more (95% CI)")
    b.set_title("Head-to-head vs TemGX"); panel(b, "b")
    fig.subplots_adjust(wspace=0.55)
    save(fig, "fig05_deletion_32_searches")


# ---------------------------------------------------------------- 6. CATGX on the case: lift, deletion, the motif instance
def fig_catgx_case():
    ev = J("structure_evidence.json")["cases"]["C07-m290-t1453051417"]["structures"]
    NAME = {"POP": "recent fans piled on", "CO": "two people rated\nthe same movie", "BRIDGE": "shared-taste path", "SEQ": "run of ratings", "BRIDGEB": "shared taste to B"}
    ev = sorted(ev, key=lambda s: -s["lift"])
    fig = plt.figure(figsize=(10.5, 3.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.1], wspace=0.55)
    a, b, c = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    y = np.arange(len(ev))[::-1]
    a.barh(y, [s["lift"] for s in ev], color=[RED if s["motif"] == "POP" else "#c9ced4" for s in ev], height=0.55)
    for yi, s in zip(y, ev):
        a.text(s["lift"] + 0.1, yi, f"{s['lift']:.1f}×", va="center", fontsize=8.5, color=INK)
    a.axvline(1, color=INK, lw=0.8, ls=(0, (3, 2)))
    a.set_yticks(y); a.set_yticklabels([NAME[s["motif"]] for s in ev], fontsize=8); a.set_xlim(0, 6.6)
    a.set_xlabel("attribution vs comparable patterns\n(same recency, duration, activity)"); a.set_title("Context-adjusted lift"); panel(a, "a")
    for yi, s in zip(y, ev):
        col = RED if s["p_value"] <= 0.05 else "#9aa1a9"
        b.plot([s["p_before"], s["p_after"]], [yi, yi], color=col, lw=1.2)
        b.scatter([s["p_after"]], [yi], s=40, color=col, zorder=3, edgecolors="white")
        b.scatter([s["p_random"]], [yi], s=40, marker="|", color=INK, zorder=3, linewidths=1.5)
        b.text(s["p_before"] + 0.002, yi, f"{s['p_after']:.0%}, p = {s['p_value']:.2f}", ha="left", va="center", fontsize=7.5, color=INK)
    b.axvline(ev[0]["p_before"], color=INK, lw=1, ls=(0, (3, 2)))
    b.set_xlim(0.80, 0.905); b.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02)); b.set_ylim(-0.6, len(ev) - 0.4)
    b.set_yticks(y); b.set_yticklabels([]); b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    b.set_xlabel("P(wrong #1 beats right) after removing the pattern\n│ = same number of random ratings (40 draws)")
    b.set_title("Is the pattern causal for the model?"); b.grid(axis="x"); b.set_axisbelow(True); panel(b, "b")
    s = ev[0]; L = s["links"]
    c.axis("off"); c.set_xlim(0, 1); c.set_ylim(0, 1)
    pts = {"A": (0.5, 0.82), "u1": (0.15, 0.22), "u2": (0.85, 0.22)}
    for (k, l) in zip(("u1", "u2"), L):
        c.annotate("", xy=pts["A"], xytext=pts[k], arrowprops=dict(arrowstyle="-|>", color=RED, lw=2, shrinkA=14, shrinkB=14))
        m = ((pts[k][0] + pts["A"][0]) / 2, (pts[k][1] + pts["A"][1]) / 2)
        c.text(m[0] + (-0.1 if k == "u1" else 0.1), m[1], f"{l['rating']:g}★\n{l['date']}", ha="center", fontsize=7.5, color=INK)
    c.plot([pts["u1"][0], pts["u2"][0]], [pts["u1"][1], pts["u2"][1]], color="#9aa1a9", ls=(0, (3, 2)), lw=1)
    c.text(0.5, 0.3, "both recent", ha="center", fontsize=7.5, color=INK2, style="italic")
    c.scatter(*pts["A"], s=520, color=RED, zorder=3); c.text(pts["A"][0], pts["A"][1] + 0.12, AN, ha="center", fontsize=8, color=RED, fontweight="bold")
    for k, l in zip(("u1", "u2"), L):
        c.scatter(*pts[k], s=300, color="white", edgecolors="#6b7280", linewidths=1.4, zorder=3)
        c.text(pts[k][0], pts[k][1] - 0.13, l["to"], ha="center", fontsize=8, color=INK)
    c.set_title("The instance in this search"); panel(c, "c")
    save(fig, "fig06_catgx_member290")


# ---------------------------------------------------------------- 7. both sides across 32 searches
def fig_both_sides():
    cs = J("contrast_summary.json")
    K = [("the title's recent raters", "recent fans of the title", RED), ("the member's own ratings", "member's own ratings", HIS), ("other people's ratings", "other viewers", "#c9ced4")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    for i, (side, lab) in enumerate((("A", "behind the wrong #1"), ("B", "behind the right answer"))):
        tot = sum(cs["kinds"][side].values()); left = 0
        for k, kl, c in K:
            v = cs["kinds"][side][k] / tot
            a.barh(1 - i, v, left=left, color=c, height=0.55, edgecolor="white", linewidth=1.2)
            if v > 0.06:
                a.text(left + v / 2, 1 - i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if c != "#c9ced4" else INK)
            left += v
    a.set_yticks([1, 0]); a.set_yticklabels(["TemGX ratings\nbehind the wrong #1", "TemGX ratings\nbehind the right answer"])
    a.set_xlim(0, 1); a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    a.legend(handles=[Patch(color=c, label=l) for _, l, c in K], loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    a.set_title(f"Where the evidence comes from ({cs['n']} searches, 3 ratings per side)"); panel(a, "a")
    NAME = {"POP": "recent fans piled on", "CO": "same movie", "BRIDGE": "shared-taste path", "SEQ": "run of ratings", "BRIDGEB": "taste path to B", "none": "none"}
    tm = cs["top_motif"]["A"]; ks = sorted(tm, key=lambda k: -tm[k])
    b.bar(range(len(ks)), [tm[k] for k in ks], color=[RED if k == "POP" else "#c9ced4" for k in ks], width=0.6)
    for i, k in enumerate(ks):
        b.text(i, tm[k] + 0.4, str(tm[k]), ha="center", fontsize=8, color=INK)
    b.set_xticks(range(len(ks))); b.set_xticklabels([NAME[k] for k in ks], fontsize=7.5, rotation=20, ha="right")
    b.set_ylabel("searches"); b.set_title("CATGX top pattern behind the wrong #1"); panel(b, "b")
    save(fig, "fig07_both_sides_32")


# ---------------------------------------------------------------- 8. planted cause, per probe
def fig_planted():
    pr = J("plant_report.json")
    meths = [("Try every pair", "exhaustive (upper bound)", "#4b5563"), ("TemGX", "TemGX", RED), ("Best single removals", "best single removals", "#6b7280"),
             ("CATGX", "CATGX (attention signal)", "#9aa1a9"), ("Attention top-2", "attention top-2", "#9aa1a9"),
             ("LLM (reads the history as text)", "LLM reading the history", BLUE), ("Most recent 2", "most recent 2", "#b4bac1"), ("Random 2", "random 2", "#b4bac1")]
    vals = {m: [] for m, _, _ in meths}
    for p in pr["patterns"]:
        for x in p["viz"]:
            if x["plant_is_models_reason"]:
                for m, _, _ in meths:
                    if m in x["fidelity"]:
                        vals[m].append(x["fidelity"][m])
    n = len(vals["TemGX"])
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    rng = np.random.default_rng(2)
    for i, (m, lab, c) in enumerate(meths):
        v = np.clip(np.array(vals[m]), -1, 1.05)
        y = len(meths) - 1 - i
        ax.scatter(v, y + rng.uniform(-0.17, 0.17, len(v)), s=14, color=c, alpha=0.8, linewidths=0, zorder=3)
        med = np.median(vals[m]); ax.plot([med, med], [y - 0.3, y + 0.3], color=INK, lw=2, zorder=4)
        ax.text(1.12, y, f"{med:.0%}", va="center", fontsize=8.5, color=INK, fontweight="bold" if m in ("TemGX", "LLM (reads the history as text)") else "normal")
    ax.axvline(0, color="#9aa1a9", lw=0.8); ax.axvline(1, color="#9aa1a9", lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(range(len(meths))); ax.set_yticklabels([l for _, l, _ in meths][::-1]); ax.set_xlim(-1.05, 1.22)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); ax.text(1.12, len(meths) - 0.35, "median", ha="center", fontsize=7.5, color=INK2)
    ax.set_xlabel("share of the planted cause's effect removed by the explainer's 2 ratings (1 = all of it)")
    ax.set_title(f"Planted-cause test: retrained model, known cause, {n} probes (2 patterns)"); ax.grid(axis="x"); ax.set_axisbelow(True)
    save(fig, "fig08_planted_cause")


# ---------------------------------------------------------------- 9. fix lookup (RAG retrieval) with CIs
def fig_rag():
    x = J("xstore_eval.json"); g = J("graphrag_eval.json"); n = x["n_new"]
    rows = [("one fix for everything", x["success"]["always_" + x["always"]], "#9aa1a9"),
            ("lookup by graph explanation only", x["success"]["retrieval_explanation_only"], "#b4bac1"),
            ("lookup by walking the diagnosis graph", g["chosen_graph"]["final"], "#b4bac1"),
            ("lookup by mechanism key (used in demo)", x["success"]["retrieval"], RED)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 2.7), gridspec_kw={"width_ratios": [1.6, 1]})
    y = np.arange(len(rows))
    for yi, (lab, p, c) in zip(y, rows):
        k = round(p * n); pp, lo, hi = wilson(k, n)
        a.barh(yi, pp, color=c, height=0.55); a.errorbar(pp, yi, xerr=[[pp - lo], [hi - pp]], color=INK, capsize=3, lw=1)
        a.text(hi + 0.01, yi, f"{pp:.0%}", va="center", fontsize=8.5, color=INK)
    best = x["success"]["oracle_any_fix"]
    a.axvline(best, color=INK, lw=1, ls=(0, (3, 2))); a.text(best + 0.005, -0.45, f"best possible {best:.0%}", ha="left", fontsize=7.5, color=INK2)
    a.set_yticks(y); a.set_yticklabels([r[0] for r in rows]); a.set_xlim(0, 0.55)
    a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); a.set_xlabel(f"recommended fix works when re-ranked ({n} held-out failures, 95% CI)")
    a.set_title(f"Which fix? Retrieval over {x['n_store']} verified diagnoses"); a.grid(axis="x"); a.set_axisbelow(True); panel(a, "a")
    d = [("mechanism lookup − one fix", x["retrieval_minus_always"]), ("graph walk − mechanism lookup", g["graph_minus_vector"])]
    for i, (lab, v) in enumerate(d):
        b.errorbar(v["mean"], 1 - i, xerr=[[v["mean"] - v["ci95"][0]], [v["ci95"][1] - v["mean"]]], fmt="o", color=RED if i == 0 else "#6b7280", capsize=3, ms=6)
        b.text(v["mean"], 1 - i + 0.22, f"{100 * v['mean']:+.1f} pts [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]", ha="center", fontsize=8, color=INK)
    b.axvline(0, color=INK, lw=0.8); b.set_yticks([1, 0]); b.set_yticklabels([r[0] for r in d], fontsize=8); b.set_ylim(-0.6, 1.7)
    b.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:+.0f}")); b.set_xlabel("paired difference (pts, bootstrap 95% CI)")
    b.set_title("Paired comparison"); b.grid(axis="x"); b.set_axisbelow(True); panel(b, "b")
    fig.subplots_adjust(wspace=0.9)
    save(fig, "fig09_fix_lookup")


# ---------------------------------------------------------------- 10. held-out checks and quality cost
def fig_heldout():
    tg = J("testgen.json"); conv = CS["fix"]["conversational_heldout"]
    FN = {"F0": "current", "F3": "understand “not”", "F4": "less popularity", "F5": "ignore strangers' ratings", "F4+F5": "less pop. + ignore strangers"}
    ms = ["F0", "F3", "F4", "F5", "F4+F5"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
    w = 0.38
    for j, (t, lab, c) in enumerate((("T1", "ruled-out title not in top 3", RED), ("T2", "right answer above ruled-out title", BLUE))):
        for i, m in enumerate(ms):
            d = tg["pass"][m][t]; n = d.get("n") or (576 if t == "T1" else 575); k = round(d["rate"] * n)
            p, lo, hi = wilson(k, n); x = i + (j - 0.5) * w
            a.bar(x, p, width=w * 0.92, color=c if m != "F0" else "#c9ced4", alpha=1 if m != "F0" else 1, edgecolor=c, linewidth=1)
            a.errorbar(x, p, yerr=[[p - lo], [hi - p]], color=INK, capsize=2, lw=0.8)
            a.text(x, hi + 0.015, f"{p:.0%}", ha="center", fontsize=7, color=INK)
    a.set_xticks(range(len(ms))); a.set_xticklabels([FN[m] for m in ms], fontsize=7.5, rotation=12, ha="right")
    a.set_ylim(0, 0.95); a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); a.set_ylabel("checks passed (95% CI)")
    a.legend(handles=[Patch(color=RED, label="T1 ruled-out title not in top 3"), Patch(color=BLUE, label="T2 right answer above it"), Patch(color="#c9ced4", label="current ranker")],
             loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.32), fontsize=7.5)
    a.set_title(f"Checks generated from explanations, {tg['n_searches']} held-out chat requests"); a.grid(axis="y"); a.set_axisbelow(True); panel(a, "a")
    rows = [("F2", "less history for\nnew members"), ("F3", "understand “not”"), ("F4", "less popularity")]
    for i, (m, lab) in enumerate(rows):
        v = conv[m]["ndcg10"]; y = len(rows) - 1 - i
        col = RED if v["ci95"][1] < 0 else (BLUE if v["ci95"][0] > 0 else "#9aa1a9")
        b.errorbar(v["delta"], y, xerr=[[v["delta"] - v["ci95"][0]], [v["ci95"][1] - v["delta"]]], fmt="o", color=col, capsize=3, ms=6)
        b.text(v["delta"], y + 0.25, f"{v['delta']:+.3f}", ha="center", fontsize=8, color=INK)
    b.axvline(0, color=INK, lw=0.8); b.set_yticks(range(len(rows))[::-1]); b.set_yticklabels([r[1] for r in rows], fontsize=8)
    b.set_ylim(-0.6, len(rows) - 0.3); b.set_xlabel(f"Δ NDCG@10 vs current (n = {conv['F3']['ndcg10']['n']}, member bootstrap 95% CI)")
    b.set_title("Quality cost on held-out chat requests"); b.grid(axis="x"); b.set_axisbelow(True); panel(b, "b")
    fig.subplots_adjust(wspace=0.45)
    save(fig, "fig10_heldout_checks")


# ---------------------------------------------------------------- 11. explanation-derived fix by member group (backup: cold start)
def fig_f5():
    f = J("f5_final.json")["results"]; gr = ["chat", "regular", "new members"]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    for j, (m, dk, lab, c) in enumerate((("F5", "delta_vs_F0", "ignore strangers' latest ratings vs current", RED), ("F4+F5", "delta_vs_F4", "adding it to “less popularity”", BLUE))):
        for i, g in enumerate(gr):
            v = f[m][g][dk]; y = len(gr) - 1 - i + (0.15 if j == 0 else -0.15)
            ax.errorbar(v["mean"], y, xerr=[[v["mean"] - v["ci95"][0]], [v["ci95"][1] - v["mean"]]], fmt="o" if j == 0 else "s", color=c, capsize=3, ms=5, label=lab if i == 0 else None)
            ax.text(v["ci95"][1] + 0.001, y, f"{v['mean']:+.3f}", va="center", fontsize=7.5, color=INK)
    ax.axvline(0, color=INK, lw=0.8); ax.set_yticks(range(3)[::-1]); ax.set_yticklabels([f"{g} (n = {f['F0'][g]['n']})" for g in gr])
    ax.set_xlabel("Δ NDCG@10, held-out, paired bootstrap 95% CI"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize=7.5)
    ax.set_title("A fix suggested by the explanations (n = 10 chosen on development data)"); ax.grid(axis="x"); ax.set_axisbelow(True)
    save(fig, "fig11_strangers_fix_by_group")


# ---------------------------------------------------------------- 12. semantic-ID failure lift (backup)
def fig_semid():
    s = J("semid.json")["where_failures"]; s = sorted(s, key=lambda r: -r["lift"])
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    y = np.arange(len(s))[::-1]
    ax.barh(y, [r["lift"] for r in s], color=[RED if r["lift"] > 1.25 else "#c9ced4" for r in s], height=0.6)
    for yi, r in zip(y, s):
        ax.text(r["lift"] + 0.03, yi, f"{r['lift']:.2f}×  ({r['n_wrong_top']} wrong #1)", va="center", fontsize=7.5, color=INK)
    ax.axvline(1, color=INK, lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(y); ax.set_yticklabels([f"code {r['code']}: {r['name']}" for r in s], fontsize=8); ax.set_xlim(0, 2.3)
    ax.set_xlabel("share among wrong #1 results / share among candidates")
    ax.set_title("Semantic-ID codes (residual k-means, 3×8) where wrong #1 results concentrate")
    save(fig, "fig12_semantic_id_lift")


for f in (fig_case_ranking, fig_failure_slices, fig_neighbourhood, fig_deletion_case, fig_deletion_32, fig_catgx_case,
          fig_both_sides, fig_planted, fig_rag, fig_heldout, fig_f5, fig_semid):
    f()


def fig_headtohead():
    """Panel (b) of fig05 on its own, sized for a half-slide."""
    rv = J("retrievers.json")
    order = [("CATGX structure", "CATGX pattern"), ("1-hop, newest around the title", "newest ratings of the title\n(GraphRAG-style context)"),
             ("1-hop, newest around member + title", "newest around member + title"), ("attention top-n", "highest attention"), ("random, same period", "random, same period")]
    fig, b = plt.subplots(figsize=(4.8, 3.0))
    labs, ps, los, his = [], [], [], []
    for k, lab in order:
        w = rv["agg"][k]["temgx_moves_more"]; n = rv["agg"][k]["n"]
        p, lo, hi = wilson(w, n); labs.append(lab); ps.append(p); los.append(p - lo); his.append(hi - p)
    y = np.arange(len(labs))[::-1]
    b.barh(y, ps, color=["#e8a3aa" if i == 1 else "#c9ced4" for i in range(len(ps))], height=0.58)
    b.errorbar(ps, y, xerr=[los, his], fmt="none", ecolor=INK, capsize=3, lw=1)
    for yi, (k, _) in zip(y, order):
        b.text(0.02, yi, f"{rv['agg'][k]['temgx_moves_more']}/{rv['agg'][k]['n']}", va="center", fontsize=8.5, color=INK, fontweight="bold")
    b.axvline(0.5, color=INK, lw=0.8, ls=(0, (3, 2))); b.text(0.5, len(labs) - 0.35, "chance", ha="center", fontsize=7.5, color=INK2)
    b.set_yticks(y); b.set_yticklabels(labs, fontsize=8); b.set_xlim(0, 1)
    b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    b.set_xlabel(f"searches where TemGX moves the model more\n({rv['n_cases']} searches, 95% Wilson CI)")
    b.set_title("TemGX vs each alternative, same budget")
    save(fig, "fig05b_head_to_head")


def fig_spread():
    """Panel (a) of fig05 on its own (for the attention Q&A slide)."""
    rv = J("retrievers.json"); rows = rv["rows"]
    order = [("TemGX links", "TemGX", RED), ("1-hop, newest around the title", "newest of\ntitle", "#6b7280"),
             ("attention top-n", "highest\nattention", "#2a78d6"), ("random, same period", "random", "#9aa1a9")]
    fig, a = plt.subplots(figsize=(4.8, 3.0)); rng = np.random.default_rng(1)
    for i, (k, lab, c) in enumerate(order):
        d = np.array([100 * (r["p0"] - r["arms"][k]["p"]) for r in rows if k in r["arms"]])
        a.scatter(i + rng.uniform(-0.16, 0.16, len(d)), d, s=13, color=c, alpha=0.75, linewidths=0, zorder=3)
        med = np.median(d); a.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=1.8, zorder=4)
        a.text(i, max(d) + 1.5, f"mean {d.mean():+.1f}", ha="center", fontsize=7.5, color=INK2)
    a.axhline(0, color="#9aa1a9", lw=0.8); a.set_xticks(range(len(order))); a.set_xticklabels([o[1] for o in order], fontsize=8)
    a.set_ylabel("drop in P(wrong #1 beats right), pts"); a.grid(axis="y"); a.set_axisbelow(True)
    a.set_title(f"Removing 3 ratings, {len(rows)} searches (line = median)")
    save(fig, "fig05a_spread")


for f in (fig_headtohead, fig_spread):
    f()
