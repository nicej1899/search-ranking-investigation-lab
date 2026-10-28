"""Properties of the TemGX / CATGX reproductions (fast: uses the 1-layer backbone)."""
import numpy as np
import pytest

from srlab.backbones import get_backbone
from srlab.common import CASES, load_json
from srlab.ranker import world
from srlab import catgx, temgx
from srlab.tgraph import DELTA_DAYS


@pytest.fixture(scope="module")
def setup():
    W = world()
    bb = get_backbone("tgat1")
    p = load_json(CASES / "selected_cases.json")["primary"]
    c = W.cases[p["case_id"]]
    return bb, c["member"], c["t"], c["candidates"][p["A_index"]], c["candidates"][p["B_index"]]


def test_explored_edges_are_in_window_and_before_t(setup):
    bb, u, t, a, b = setup
    hop, edges = temgx.explore(bb.G, [u, a + bb.G.U, b + bb.G.U], t, 3)
    for _, (_, _, ts, _) in edges.items():
        assert t - DELTA_DAYS * 86400 <= ts < t


def test_influence_is_model_measured_and_respects_receptive_field(setup):
    bb, u, t, a, b = setup
    r = temgx.explain(bb, u, t, a, b, L=2, k=3)
    U = bb.G.U
    for e in r["edges"]:
        touches_u = e["a"] == u or e["b"] == u
        if not touches_u:
            assert e["p"] == 0.0  # a 1-layer member-side model cannot see edges that do not touch u
    assert all(h == 1 for h in r["explanatory_hops"].values())


def test_explanation_structure(setup):
    bb, u, t, a, b = setup
    r = temgx.explain(bb, u, t, a, b, L=2, k=3)
    eids = {e["eid"] for e in r["edges"]}
    assert set(r["E_eps"]) <= eids
    nodes_on_paths = {x for e in r["edges"] if e["eid"] in set(r["E_eps"]) for x in (e["a"], e["b"])}
    assert set(r["connection"]) <= nodes_on_paths
    assert len(r["explanatory"]) <= 3
    # greedy order: accepted additions appear in non-increasing phi
    adds = [s["phi"] for s in r["steps"] if s["action"] == "add"]
    assert adds == sorted(adds, reverse=True)
    # verification: each accepted step lowered the margin
    m = r["margin_full"]
    for s in r["steps"]:
        if s["action"] in ("add", "replace"):
            assert s["margin_after"] < m
            m = s["margin_after"]


def test_catgx_adjusted_score_matches_definition(setup):
    bb, u, t, a, b = setup
    r = catgx.explain(bb, u, t, a, b, L=2, exact=True)
    for m, terms in r["terms"].items():
        s = sum(x["omega"] * (x["mu"] - x["mu_base"]) for x in terms)
        assert r["S"][m] == pytest.approx(s)
    assert sum(r["codebook_TM"].values()) == pytest.approx(r["occurrences_used"])  # exact: w = 1


def test_catgx_sampling_weights_are_inverse_inclusion(setup):
    bb, u, t, a, b = setup
    r = catgx.explain(bb, u, t, a, b, L=3, exact=False)
    assert r["occurrences_used"] <= r["occurrences_total"]
    # importance weights reconstruct the retrieved population size per motif
    assert sum(r["codebook_TM"].values()) == pytest.approx(r["stage1"]["retrieved_occurrences"], rel=0.35)


@pytest.mark.parametrize("key", ["tgat2", "tgn"])
def test_graph_backbones_replay_and_temgx(setup, key):
    _, u, t, a, b = setup
    bb = get_backbone(key)
    m0 = bb.margins(u, t, a, b, [frozenset()])[0]
    assert np.isfinite(m0)
    r = temgx.explain(bb, u, t, a, b, L=1, k=3)
    # replay without the explanation edges reproduces the reported number
    m1 = bb.margins(u, t, a, b, [frozenset(r["E_eps"])])[0]
    assert m1 == pytest.approx(r["margin_without"], abs=1e-4)
    assert r["prob_without"] <= r["prob_full"] + 1e-9
