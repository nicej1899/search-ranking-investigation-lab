"""Query-title relevance with a pretrained text encoder.

Encoder: spaCy `en_core_web_lg` 3.8.0 static word vectors (300-d, pretrained, no fine-tuning), averaged over
tokens. Title documents are built ONLY from the MovieLens title string and its genre names, because that
is all the permitted metadata this build has (no plot summaries, no runtime). MovieLens user tags are NOT
used here: they are reserved for the relevance judgments so the judge and the scorer do not share a signal.

Known limitation (made visible on purpose): averaged word vectors ignore word order and negation, so
"not too intense" and "too intense" map to almost the same vector.
"""
from __future__ import annotations

import functools

import numpy as np

GENRE_WORDS = {
    "Sci-Fi": "science fiction", "Film-Noir": "film noir", "Children": "children family",
    "(no genres listed)": "", "IMAX": "imax",
}


@functools.lru_cache(maxsize=1)
def nlp():
    import spacy
    return spacy.load("en_core_web_lg", disable=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"])


def encode(text: str) -> np.ndarray:
    doc = nlp()(text.lower())
    vecs = [t.vector for t in doc if t.has_vector and (t.is_alpha)]
    if not vecs:
        return np.zeros(300, np.float32)
    v = np.mean(vecs, 0)
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def title_doc(clean_title: str, genres: str) -> str:
    g = " ".join(GENRE_WORDS.get(x, x.lower()) for x in genres.split("|"))
    return f"{clean_title}. {g}"


def encode_items(items) -> np.ndarray:
    docs = [title_doc(t, g) for t, g in zip(items.clean_title.astype(str), items.genres)]
    out = np.zeros((len(docs), 300), np.float32)
    for i, d in enumerate(nlp().pipe([d.lower() for d in docs], batch_size=512)):
        vecs = [t.vector for t in d if t.has_vector and t.is_alpha]
        if vecs:
            v = np.mean(vecs, 0)
            out[i] = v / (np.linalg.norm(v) + 1e-8)
    return out


GENRE_PROBES = ["action", "adventure", "animation", "children family", "comedy", "crime", "documentary", "drama",
                "fantasy", "film noir", "horror", "musical", "mystery", "romance", "science fiction", "thriller",
                "war", "western"]


def query_specificity(q_vec: np.ndarray) -> float:
    """How strongly a query names a content category: max cosine to genre-name probes.
    A per-query feature that does not depend on the candidate set or on any judgment."""
    probes = np.stack([encode(g) for g in GENRE_PROBES])
    return float((probes @ q_vec).max())


def negation_diagnostic() -> dict:
    a = encode("a light cozy mystery not too intense")
    b = encode("a light cozy mystery too intense")
    c = encode("intense")
    return {
        "cos(query, query_without_not)": float(a @ b),
        "cos(query, 'intense')": float(a @ c),
        "note": "Averaged word vectors keep the word 'intense' in the query vector; negation is not represented.",
    }
