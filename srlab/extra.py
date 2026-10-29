"""Extra evaluation cases for the team-specific analyses (kept OUT of the original protocols qi-dev-v1 / qi-final-v1).

coldstart       : the original explicit templates, but predicted for members who have only 1-4 ratings before t
                  (the original set requires >= 5). Same candidate protocol.
conversational  : chat-style queries (what people type or say to an assistant): several constraints and an
                  explicit exclusion ("..., nothing scary", "... that isn't animated"). Judged by the same written
                  rubric over MovieLens genres + user tags; the exclusion is part of the rubric (violating it -> grade 0).

Both sets are split DEV (validation-period t) / FINAL (test-period t) exactly like the original cases, and every
case carries `extra` so the original protocols never see them. Not real search logs: all queries are written here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import CASES, PROC, dump_json, load_json, sha256_obj
from .data import load_events, load_items
from .history import HistoryStore
from .queries import N_CAND, TEMPLATES, Template, _h, is_showcase_member, item_tagsets, rubric_grade

NEG_WORDS = ("not", "nothing", "no", "without", "isn't", "isnt", "never", "n't")

CONV = [
    # FINAL
    Template("C01", "my kids are six, we want something funny but nothing scary", "explicit", "final",
             grade1_any=["Children", "Animation"], require_for_2_any=["Comedy"], exclude_genres=["Horror", "Thriller"],
             exclude_tags=["scary", "creepy", "dark"], notes="Children/Animation = 1; + Comedy = 2; scary content -> 0."),
    Template("C02", "a crime movie but not too violent, more on the funny side", "explicit", "final",
             grade1_any=["Crime"], require_for_2_any=["Comedy"], exclude_genres=["Horror"],
             exclude_tags=["violent", "violence", "brutal", "gore", "dark"], notes="Crime = 1; + Comedy = 2; violent tags -> 0."),
    Template("C03", "date night tonight, something romantic but not a tearjerker", "explicit", "final",
             grade1_any=["Romance"], require_for_2_any=["Comedy"], exclude_genres=["War", "Horror"],
             exclude_tags=["sad", "tearjerker", "depressing", "bittersweet"], notes="Romance = 1; + Comedy = 2; sad tags / War -> 0."),
    Template("C04", "an action movie that isn't a superhero film", "explicit", "final",
             grade2_all=["Action"], exclude_tags=["superhero", "superheroes", "marvel", "comic book", "dc comics", "super-hero"],
             notes="Action = 2; superhero tags -> 0."),
    Template("C05", "I want something scary, but nothing gory", "explicit", "final",
             grade2_all=["Horror"], exclude_tags=["gore", "gory", "splatter", "violent", "brutal"], notes="Horror = 2; gore tags -> 0."),
    Template("C06", "a science fiction film with no aliens in it", "explicit", "final",
             grade2_all=["Sci-Fi"], exclude_tags=["aliens", "alien", "alien invasion", "extraterrestrial"], notes="Sci-Fi = 2; alien tags -> 0."),
    # DEV
    Template("C07", "a drama, but please nothing depressing", "explicit", "dev",
             grade2_all=["Drama"], exclude_tags=["depressing", "sad", "tearjerker", "bleak", "suicide"], notes="Drama = 2; depressing tags -> 0."),
    Template("C08", "a comedy for grown-ups, not a kids movie", "explicit", "dev",
             grade2_all=["Comedy"], exclude_genres=["Children", "Animation"], notes="Comedy = 2; Children/Animation -> 0."),
    Template("C09", "a war movie without too much gore", "explicit", "dev",
             grade2_all=["War"], exclude_tags=["gore", "gory", "violent", "brutal", "graphic violence"], notes="War = 2; gore tags -> 0."),
    Template("C10", "a mystery to watch tonight, nothing too dark", "explicit", "dev",
             grade1_any=["Mystery"], require_for_2_any=["Comedy", "Romance", "Children"], exclude_genres=["Horror"],
             exclude_tags=["dark", "disturbing", "violent", "psychological"], notes="Mystery = 1; light genre = 2; dark tags -> 0."),
    Template("C11", "a fantasy film for grown-ups", "explicit", "dev",
             grade2_all=["Fantasy"], exclude_genres=["Animation"],
             notes="Fantasy = 2; Animation -> 0. Implicit exclusion: 'for grown-ups' rules out animated films."),
    Template("C12", "a musical, but not a romance", "explicit", "dev",
             grade2_all=["Musical"], exclude_genres=["Romance"], notes="Musical = 2; Romance -> 0."),
]
CONV_BY_ID = {t.qid: t for t in CONV}
EXTRA_PATH = CASES / "extra_cases.json"


# Implicit exclusions: phrases that rule something out without a negation word. Used only when the query has no
# explicit negation. A small, fixed lexicon standing in for an LLM query parser (no tuning on results).
IMPLIED = {"for grown-ups": "animation children", "for grown ups": "animation children", "for adults": "animation children"}


def split_negation(text: str) -> tuple[str, str]:
    """Split a query into (positive part, negated part). Negation scope: from a negation word to the end of the clause
    (comma / 'but' / end). Simple, transparent rule standing in for an LLM query parser. If there is no explicit
    negation, an implicit exclusion from IMPLIED is used (e.g. "for grown-ups" -> not animation / children)."""
    low = text.lower()
    if not any(w in NEG_WORDS or w.endswith("n't") for w in low.replace(",", " ").split()):
        for ph, neg in IMPLIED.items():
            if ph in low:
                pos = [w for w in low.replace(ph, " ").split() if w not in ("a", "an", "the", "please")]
                return " ".join(pos), neg
    toks = text.lower().replace(",", " , ").split()
    pos, neg, in_neg = [], [], False
    for w in toks:
        if w in (",", "but") and in_neg:
            in_neg = False
            continue
        if w in NEG_WORDS or w.endswith("n't"):
            in_neg = True
            continue
        (neg if in_neg else pos).append(w)
    keep = [w for w in pos if w not in (",", "but", "please", "too", "much", "a", "an", "the")]
    negk = [w for w in neg if w not in ("too", "much", "a", "an", "the", "in", "it", "film", "movie")]
    return " ".join(keep), " ".join(negk)


def violates(t: Template, genres: set, tags: set) -> bool:
    return bool(genres & set(t.exclude_genres) or tags & set(t.exclude_tags))


def _build(templates, extra: str, min_hist: int, max_hist: int, prefix: str):
    events, items = load_events(), load_items()
    split = load_json(PROC / "split.json")
    t1, t2 = split["cutoffs"]["train_end_ts"], split["cutoffs"]["val_end_ts"]
    store = HistoryStore(events)
    tagsets = item_tagsets()
    genres = [set(g) for g in items.genre_list]
    train_pop = events[events.split == "train"].groupby("item").size().reindex(range(len(items)), fill_value=0).to_numpy()
    eligible = np.flatnonzero(train_pop >= 1)
    labels = items.label.to_numpy()
    grades = {t.qid: np.array([rubric_grade(t, genres[i], tagsets.get(i, set())) for i in range(len(items))]) for t in templates}
    viol = {t.qid: np.array([violates(t, genres[i], tagsets.get(i, set())) for i in range(len(items))]) for t in templates}
    cases = []
    for period, lo, hi in [("dev", t1, t2), ("final", t2, 10 ** 12)]:
        per = events[(events.ts >= lo) & (events.ts < hi)]
        first = {}
        for m, g in per.groupby("member"):
            for t in np.unique(g.ts.to_numpy()):
                k = store.count_before(m, t)
                if min_hist <= k <= max_hist:
                    first[int(m)] = int(t)
                    break
        for m, t in first.items():
            n_hist = store.count_before(m, t)
            seen = set(store.before(m, t, n=None).items.tolist())
            hist_labels = pd.Series(labels[store.before(m, t, n=50).items]).value_counts()
            pool = eligible[~np.isin(eligible, np.fromiter(seen, int, len(seen)))]
            for tpl in [x for x in templates if x.split == period]:
                cid = f"{prefix}{tpl.qid}-m{m}-t{t}"
                rng = np.random.default_rng(_h(cid) % (2 ** 32))
                g = grades[tpl.qid]
                rel = pool[g[pool] >= 1]
                if len(rel) < 3:
                    continue
                pick_rel = rng.choice(rel, min(8, len(rel)), replace=False)
                top_labels = list(hist_labels.index[:2])
                hist_pool = np.setdiff1d(pool[np.isin(labels[pool], top_labels)], pick_rel)
                pick_hist = rng.choice(hist_pool, min(8, len(hist_pool)), replace=False) if len(hist_pool) else np.zeros(0, int)
                # conversational: make sure some candidates violate the exclusion (that is the failure we want to measure)
                pick_viol = np.zeros(0, int)
                if extra == "conversational":
                    vpool = np.setdiff1d(pool[viol[tpl.qid][pool]], np.concatenate([pick_rel, pick_hist]))
                    if len(vpool):
                        pick_viol = rng.choice(vpool, min(4, len(vpool)), replace=False)
                used = np.concatenate([pick_rel, pick_hist, pick_viol])
                rest_pool = np.setdiff1d(pool, used)
                w = np.log1p(train_pop[rest_pool])
                w = w / w.sum()
                pick_rest = rng.choice(rest_pool, N_CAND - len(used), replace=False, p=w)
                cand = np.concatenate([used, pick_rest]).astype(int)
                cand = cand[rng.permutation(len(cand))]
                cases.append({
                    "case_id": cid, "qid": tpl.qid, "query": tpl.text, "kind": "explicit", "split": period, "extra": extra,
                    "member": int(m), "t": int(t), "n_history": int(n_hist),
                    "showcase_member": bool(is_showcase_member(m) and period == "final"),
                    "candidates": cand.tolist(), "grades": g[cand].astype(int).tolist(),
                    "violates_exclusion": viol[tpl.qid][cand].astype(int).tolist(),
                    "judgment_source": "rubric(genres+tags)",
                })
    return cases


def build() -> dict:
    explicit = [t for t in TEMPLATES if t.kind == "explicit"]
    cold = _build(explicit, "coldstart", 1, 4, "CS-")
    conv = _build(CONV, "conversational", 5, 10 ** 9, "")
    cases = cold + conv
    meta = {"n_cases": len(cases),
            "counts": pd.DataFrame(cases).groupby(["extra", "split"]).size().to_dict().__repr__(),
            "coldstart_rule": "original explicit templates; prediction time = member's first in-period time with 1-4 prior ratings",
            "conversational_rule": "chat-style templates with an exclusion; >=5 prior ratings; up to 4 candidates violate the exclusion",
            "templates": [t.__dict__ for t in CONV], "cases_hash": sha256_obj(cases)}
    dump_json({"meta": meta, "cases": cases}, EXTRA_PATH)
    return meta


def load_extra() -> list[dict]:
    return load_json(EXTRA_PATH)["cases"] if EXTRA_PATH.exists() else []


if __name__ == "__main__":
    print(build())
