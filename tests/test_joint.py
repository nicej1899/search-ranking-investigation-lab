"""TemGX + CATGX joint explanation and its causal check."""
import numpy as np

from srlab.backbones import get_backbone
from srlab.compare import cases
from srlab.joint import _placebo, run
from srlab import catgx, temgx


def test_placebo_matches_size_and_recency():
    edges = {i: (0, 1, 1000 + i, 4.0) for i in range(40)}
    rng = np.random.default_rng(0)
    treated = {37, 38, 39}
    for s in _placebo(edges, treated, 10, rng):
        assert len(s) == 3 and not set(s) & treated
        assert all(e >= 30 for e in s)  # same (newest) recency quartile


def test_joint_arms_are_exact_replays():
    bb = get_backbone("tgat1")
    cid, u, t, a, b = cases(1)[0]
    tx = temgx.explain(bb, u, t, a, b, L=2, k=3)
    cx = catgx.explain(bb, u, t, a, b, L=2)
    cxd = catgx.explain(bb, u, t, a, b, L=2, g_mode="do")
    r = run(bb, u, t, a, b, tx, cx, L=2, cx_do=cxd, n_draw=8)
    assert abs(r["m0"] - tx["margin_full"]) < 1e-4
    assert abs(r["arms"]["temgx"]["margin"] - tx["margin_without"]) < 1e-4  # same removal set -> same replay
    assert abs(r["arms"]["catgx_do"]["margin"] - cxd["margin_without"]) < 1e-4
    assert set(tx["E_eps"]) <= set(r["arms"]["joint"]["eids"])
    for x in r["labels"]:
        if x["instance"]:
            assert x["eid"] in x["instance"]["eids"]
