"""Query-intent evaluation set: templates, rubric judgments, deterministic candidate sets.

NOT real search logs. MovieLens has no queries. Every query below was written for this prototype.

Judgments
---------
* explicit-intent queries: graded 0/1/2 by a written rubric over MovieLens genres and MovieLens user tags
  (tags are never model inputs). The rubric is the documented criterion; it is a proxy for human
  relevance assessment, not a substitute. Ambiguities are listed per template in `notes`.
* broad queries ("what should I watch"): relevance = the member actually rated the title >= 4.0 within
  90 days AFTER the prediction time (held-out interactions). Graded 1. Unobserved titles are unlabeled
  negatives, not established dislikes.

Splits
------
Query templates are split into DEV (used to fit calibration / the gate) and FINAL (scored once).
DEV cases use validation-period prediction times; FINAL cases use test-period prediction times.
A 'showcase' pool of members (deterministic hash) is excluded from the FINAL aggregate so the
illustrative cases shown in the UI do not double as evaluation evidence.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .common import CASES, GLOBAL_SEED, PROC, dump_json, load_json, sha256_obj
from .data import load_events, load_items, load_tags
from .history import HistoryStore

N_CAND = 30
FUTURE_DAYS = 90
MIN_HIST = 5


@dataclass
class Template:
    qid: str
    text: str
    kind: str  # explicit | broad
    split: str  # dev | final
    grade2_all: list = field(default_factory=list)  # all of these genres -> candidate for grade 2
    grade2_any_tags: list = field(default_factory=list)
    grade1_any: list = field(default_factory=list)  # any genre -> grade >= 1
    exclude_genres: list = field(default_factory=list)
    exclude_tags: list = field(default_factory=list)
    require_for_2_any: list = field(default_factory=list)  # genres; at least one needed for grade 2 (in addition)
    notes: str = ""


TEMPLATES = [
    # ---- explicit, FINAL
    Template("E01", "a light, cozy mystery, not too intense", "explicit", "final",
             grade1_any=["Mystery"], require_for_2_any=["Comedy", "Children", "Animation", "Romance"],
             grade2_any_tags=["cozy", "funny", "light", "whimsical", "quirky", "feel-good", "charming", "british", "agatha christie"],
             exclude_genres=["Horror", "Thriller", "War"], exclude_tags=["dark", "violent", "violence", "disturbing", "brutal", "gore", "psychological"],
             notes="Mystery required. Grade 2 needs a light-tone signal (Comedy/Children/Animation/Romance genre or a light tag). "
                   "Horror/Thriller/War or a dark/violent tag -> 0 even if Mystery. 'Cozy' is not a MovieLens genre; tone is a proxy."),
    Template("E02", "a romantic comedy, nothing sad", "explicit", "final",
             grade2_all=["Romance", "Comedy"], grade1_any=["Romance"], exclude_genres=["War", "Horror"],
             exclude_tags=["sad", "depressing", "tearjerker", "bittersweet"],
             notes="Romance+Comedy = 2; Romance alone = 1. War/Horror or sad tags -> 0."),
    Template("E03", "space science fiction adventure", "explicit", "final",
             grade1_any=["Sci-Fi"], require_for_2_any=["Adventure"], grade2_any_tags=["space", "aliens", "space travel"],
             notes="Sci-Fi required; grade 2 with Adventure genre or space tag."),
    Template("E04", "a western", "explicit", "final", grade2_all=["Western"], notes="Genre lookup."),
    Template("E05", "a documentary about real events", "explicit", "final", grade2_all=["Documentary"],
             notes="'Real events' not verifiable beyond the Documentary genre."),
    Template("E06", "a gentle children's movie, nothing scary", "explicit", "final",
             grade2_all=["Children"], grade1_any=["Animation"], exclude_genres=["Horror", "Thriller"],
             notes="Children = 2, Animation alone = 1; Horror/Thriller -> 0."),
    Template("E07", "a tense action thriller", "explicit", "final",
             grade2_all=["Action", "Thriller"], grade1_any=["Thriller"], notes="Action+Thriller = 2; Thriller = 1."),
    Template("E08", "a funny musical", "explicit", "final",
             grade2_all=["Musical", "Comedy"], grade1_any=["Musical"], notes="Musical+Comedy = 2; Musical = 1."),
    # ---- explicit, DEV
    Template("D01", "a funny animated movie for the whole family", "explicit", "dev",
             grade1_any=["Animation", "Children"], require_for_2_any=["Comedy"], exclude_genres=["Horror"],
             notes="Animation or Children = 1; plus Comedy = 2."),
    Template("D02", "a dark psychological thriller", "explicit", "dev",
             grade1_any=["Thriller"], grade2_any_tags=["dark", "psychological", "psychology", "mindfuck", "disturbing", "twist ending"],
             require_for_2_any=["Mystery", "Drama", "Crime", "Horror"],
             notes="Thriller = 1; with Mystery/Drama/Crime/Horror or dark tags = 2."),
    Template("D03", "a classic film noir detective story", "explicit", "dev",
             grade2_all=["Film-Noir"], grade1_any=["Mystery"], notes="Film-Noir = 2; Mystery = 1."),
    Template("D04", "a scary horror movie", "explicit", "dev", grade2_all=["Horror"], notes="Genre lookup."),
    Template("D05", "an epic war film", "explicit", "dev", grade2_all=["War"], notes="Genre lookup."),
    Template("D06", "a fantasy adventure with magic", "explicit", "dev",
             grade2_all=["Fantasy", "Adventure"], grade1_any=["Fantasy"], notes="Fantasy+Adventure = 2; Fantasy = 1."),
    Template("D07", "a lighthearted heist comedy", "explicit", "dev",
             grade2_all=["Crime", "Comedy"], grade1_any=["Crime"], grade2_any_tags=["heist"],
             exclude_genres=["Horror"], exclude_tags=["dark", "violent", "violence", "brutal"],
             notes="Crime+Comedy or heist tag = 2; Crime = 1; dark/violent tags -> 0."),
    Template("D08", "a sweet romance", "explicit", "dev",
             grade1_any=["Romance"], require_for_2_any=["Comedy", "Drama"], exclude_genres=["Horror", "War"],
             notes="Romance = 1; with Comedy/Drama = 2."),
    # ---- broad
    Template("B01", "something good to watch tonight", "broad", "final"),
    Template("B02", "what should I watch", "broad", "final"),
    Template("B03", "recommend me a movie", "broad", "final"),
    Template("B04", "surprise me", "broad", "final"),
    Template("B05", "a movie I will like", "broad", "dev"),
    Template("B06", "something to watch this weekend", "broad", "dev"),
    Template("B07", "a good film", "broad", "dev"),
    Template("B08", "pick something for me", "broad", "dev"),
]
TEMPLATES_BY_ID = {t.qid: t for t in TEMPLATES}


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def is_showcase_member(member: int) -> bool:
    return _h(f"showcase:{member}") % 5 == 0  # ~20% of members


def rubric_grade(t: Template, genres: set, tags: set) -> int:
    if t.kind != "explicit":
        raise ValueError
    if genres & set(t.exclude_genres) or tags & set(t.exclude_tags):
        return 0
    g = 0
    if t.grade1_any and genres & set(t.grade1_any):
        g = 1
    if t.grade2_all and set(t.grade2_all) <= genres:
        g = 2
    if g >= 1 and (t.require_for_2_any or t.grade2_any_tags):
        if (t.require_for_2_any and genres & set(t.require_for_2_any)) or (tags & set(t.grade2_any_tags)):
            g = 2
    return g


def item_tagsets() -> dict[int, set]:
    tags = load_tags()
    items = load_items()
    mid2item = dict(zip(items.movieId, items.item))
    out: dict[int, set] = {}
    for mid, tg in zip(tags.movieId, tags.tag_norm):
        if mid in mid2item:
            out.setdefault(int(mid2item[mid]), set()).add(tg)
    return out


def build_cases() -> dict:
    events, items = load_events(), load_items()
    split = load_json(PROC / "split.json")
    t1, t2 = split["cutoffs"]["train_end_ts"], split["cutoffs"]["val_end_ts"]
    store = HistoryStore(events)
    tagsets = item_tagsets()
    genres = [set(g) for g in items.genre_list]
    train_pop = events[events.split == "train"].groupby("item").size().reindex(range(len(items)), fill_value=0).to_numpy()
    eligible = np.flatnonzero(train_pop >= 1)  # candidate pool: titles with >=1 training interaction
    labels = items.label.to_numpy()

    grades = {t.qid: np.array([rubric_grade(t, genres[i], tagsets.get(i, set())) for i in range(len(items))])
              for t in TEMPLATES if t.kind == "explicit"}

    cases = []
    for period, lo, hi in [("dev", t1, t2), ("final", t2, 10 ** 12)]:
        per = events[(events.ts >= lo) & (events.ts < hi)]
        # prediction time: the member's first in-period event time with >= MIN_HIST events strictly before it
        first = {}
        for m, g in per.groupby("member"):
            for t in np.unique(g.ts.to_numpy()):
                if store.count_before(m, t) >= MIN_HIST:
                    first[int(m)] = int(t)
                    break
        for m, t in first.items():
            n_hist = store.count_before(m, t)
            showcase = is_showcase_member(m) and period == "final"
            fut = events[(events.member == m) & (events.ts >= t) & (events.ts < t + FUTURE_DAYS * 86400)]
            fut_pos = set(fut.loc[fut.rating >= 4.0, "item"].tolist())
            seen = set(store.before(m, t, n=None).items.tolist())
            hist_labels = pd.Series(labels[store.before(m, t, n=50).items]).value_counts()
            pool = eligible[~np.isin(eligible, np.fromiter(seen, int, len(seen)))]
            pool_set = set(pool.tolist())
            for tpl in [x for x in TEMPLATES if x.split == period]:
                cid = f"{tpl.qid}-m{m}-t{t}"
                rng = np.random.default_rng(_h(cid) % (2 ** 32))
                if tpl.kind == "explicit":
                    g = grades[tpl.qid]
                    rel = pool[g[pool] >= 1]
                    if len(rel) < 3:
                        continue
                    n_rel = min(8, len(rel))
                    pick_rel = rng.choice(rel, n_rel, replace=False)
                else:
                    rel = np.array(sorted(i for i in fut_pos if i in pool_set), dtype=int)
                    if len(rel) < 1:
                        continue
                    pick_rel = rng.choice(rel, min(5, len(rel)), replace=False)
                # history-matching distractors: titles sharing the member's two most frequent recent labels
                top_labels = list(hist_labels.index[:2])
                hist_pool = np.setdiff1d(pool[np.isin(labels[pool], top_labels)], pick_rel)
                pick_hist = rng.choice(hist_pool, min(8, len(hist_pool)), replace=False) if len(hist_pool) else np.zeros(0, int)
                rest_pool = np.setdiff1d(pool, np.concatenate([pick_rel, pick_hist]))
                w = np.log1p(train_pop[rest_pool])
                w = w / w.sum()
                n_rest = N_CAND - len(pick_rel) - len(pick_hist)
                pick_rest = rng.choice(rest_pool, n_rest, replace=False, p=w)
                cand = np.concatenate([pick_rel, pick_hist, pick_rest]).astype(int)
                cand = cand[rng.permutation(len(cand))]
                if tpl.kind == "explicit":
                    gr = grades[tpl.qid][cand]
                else:
                    gr = np.array([1 if i in fut_pos else 0 for i in cand])
                cases.append({
                    "case_id": cid, "qid": tpl.qid, "query": tpl.text, "kind": tpl.kind, "split": period,
                    "member": int(m), "t": int(t), "n_history": int(n_hist), "showcase_member": bool(showcase),
                    "candidates": cand.tolist(), "grades": gr.astype(int).tolist(),
                    "judgment_source": "rubric(genres+tags)" if tpl.kind == "explicit" else f"member rated >=4.0 within {FUTURE_DAYS}d after t",
                })
    meta = {
        "n_cases": len(cases),
        "by_split_kind": pd.DataFrame(cases).groupby(["split", "kind"]).size().to_dict().__repr__(),
        "candidate_protocol": f"{N_CAND} titles per case, deterministic seed = sha256(case_id). Up to 8 judged-relevant (explicit) or "
                              "5 future-positive (broad) titles; up to 8 titles sharing the member's two most frequent recent coarse labels; "
                              "rest sampled ~log(1+training interactions). Pool = titles with >=1 training-period interaction not already rated by the member before t.",
        "runtime_constraint": "NOT APPLIED: no permitted runtime source in this build (MovieLens has none; TMDb not reachable). Runtime shown as unknown.",
        "showcase_rule": "member excluded from FINAL aggregate if sha256('showcase:'+member) % 5 == 0",
        "templates": [t.__dict__ for t in TEMPLATES],
    }
    meta["cases_hash"] = sha256_obj(cases)
    dump_json({"meta": meta, "cases": cases}, CASES / "query_cases.json")
    return meta


def load_cases() -> list[dict]:
    return load_json(CASES / "query_cases.json")["cases"]


if __name__ == "__main__":
    print(build_cases())
