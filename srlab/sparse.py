"""F6: structured request parsing (stand-in for an LLM query parser).

The request is turned into wanted and excluded genres with a small fixed lexicon (the same rule for every request):
negated words ("not / nothing / no / without / isn't") and implied phrases ("for grown-ups") give excluded genres,
the rest give wanted genres. The ranker's score is unchanged; titles with an excluded genre move below all others and
titles with a wanted genre move above the rest (order inside each group = the ranker's own order).
Note for evaluation: the Demo's relevance labels are also genre based, so held-out gains are optimistic.
"""
from __future__ import annotations

import numpy as np

from .extra import split_negation

LEX = {"fantasy": "Fantasy", "comedy": "Comedy", "funny": "Comedy", "crime": "Crime", "romantic": "Romance",
       "romance": "Romance", "action": "Action", "scary": "Horror", "horror": "Horror", "drama": "Drama",
       "war": "War", "mystery": "Mystery", "musical": "Musical", "science": "Sci-Fi", "sci-fi": "Sci-Fi",
       "animation": "Animation", "animated": "Animation", "cartoon": "Animation", "cartoons": "Animation",
       "kids": "Children", "children": "Children", "western": "Western", "thriller": "Thriller",
       "documentary": "Documentary", "adventure": "Adventure", "superhero": None}


def parse(q: str):
    pos, neg = split_negation(q)
    g = lambda s: {LEX[w] for w in s.replace(",", " ").split() if LEX.get(w)}
    want, excl = g(pos), g(neg)
    return want - excl, excl


def apply(total, genres, q):
    want, excl = parse(q)
    gs = [set(x.split("|")) for x in genres]
    adj = np.array([(10.0 if s & want else 0.0) - (100.0 if s & excl else 0.0) for s in gs])
    return np.asarray(total, float) + adj
