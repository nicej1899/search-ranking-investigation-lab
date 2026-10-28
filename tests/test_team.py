"""Team-specific analyses: extra cases, targeted fixes, triage, semantic IDs."""
import numpy as np
import pytest

from srlab.common import REPORTS, load_json
from srlab.extra import load_extra, split_negation
from srlab.fixes import feats, register_extra, score
from srlab.ranker import load_ranker, world


@pytest.fixture(scope="module")
def W():
    w = world()
    register_extra(w)
    return w


def test_split_negation():
    assert split_negation("a musical, but not a romance") == ("musical", "romance")
    assert split_negation("a light, cozy mystery, not too intense")[1] == "intense"
    assert split_negation("a western")[1] == ""


def test_extra_cases_are_leakage_safe_and_separate(W):
    ex = load_extra()
    assert ex and all(c.get("extra") in ("coldstart", "conversational") for c in ex)
    for c in ex[::97]:
        assert W.store.count_before(c["member"], c["t"]) == c["n_history"]
        if c["extra"] == "coldstart":
            assert 1 <= c["n_history"] <= 4
        if c["extra"] == "conversational":  # breaking the exclusion always means not relevant
            assert all(g == 0 for g, v in zip(c["grades"], c["violates_exclusion"]) if v)
    from srlab.evaluate import case_cases
    assert not any(c.get("extra") for c in case_cases(W, "qi-final-v1"))


def test_variants_with_zero_parameters_reproduce_F0(W):
    F0 = load_ranker("F0")
    c = next(c for c in W.cases.values() if c.get("extra") == "conversational")
    z = feats(W, c, F0)
    s0 = score(z, F0, {"kind": "F0"})["total"]
    for M in ({"kind": "F2", "gate": [0.0, 0.0]}, {"kind": "F3", "beta": 0.0}, {"kind": "F4", "pgate": [0.0, 0.0]}):
        s = score(z, F0, M)["total"]
        if M["kind"] == "F3":  # beta=0 still uses the positive part of the query, so only check it is finite
            assert np.isfinite(s).all()
        else:
            assert np.allclose(s, s0)


def test_triage_parts_add_up_and_causes_follow_rule():
    d = load_json(REPORTS / "triage_dev.json")
    for r in d["rows"][::25]:
        assert sum(r["parts"].values()) == pytest.approx(r["lead"], abs=1e-6)
        if r["parts"]["relevance"] > 0:
            assert r["cause"] == "query_misread"
    assert sum(b["n"] for b in d["buckets"]) == d["failed"]


def test_semantic_ids_cover_every_title(W):
    codes = np.load(REPORTS / "semantic_ids.npy")
    assert codes.shape == (len(W.items), 3) and codes.min() >= 0 and codes.max() < 8
