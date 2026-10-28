"""Contrastive diagnosis and explanation-generated tests."""
import numpy as np

from srlab.backbones import get_backbone
from srlab.contrast import KINDS, contrast
from srlab.server import W_, _case_ab
from srlab.testgen import _pair, run
from srlab import temgx

CID = "C07-m1-t1445714882"


def test_contrast_sides_are_temgx_and_checked():
    W, bb = W_(), get_backbone("tgat2")
    r = contrast(W, bb, CID, k=3, n_draw=8)
    c, a, b, _ = _case_ab(CID)
    tx = temgx.explain(bb, c["member"], c["t"], a, b, L=2, k=3)
    assert [l["eid"] for l in r["side_A"]["links"]] == [int(e) for e in tx["E_eps"]]
    assert all(l["kind"] in KINDS for s in ("side_A", "side_B") for l in r[s]["links"])
    assert r["test"]["del_A"]["p"] < r["test"]["p0"] < r["test"]["del_B"]["p"]


def test_pair_picks_excluded_and_relevant():
    W = W_()
    c = W.cases[CID]
    X, Y = _pair(c, list(range(len(c["candidates"]))))
    assert c["violates_exclusion"][X] and c["grades"][Y] >= 1


def test_testgen_small_run():
    o = run(limit=3, write=False)
    assert o["by_type"]["T1"] == 3 and set(o["pass"]) == {"F0", "F3", "F4", "F5", "F4+F5"}
    assert all(0 <= v["rate"] <= 1 for d in o["pass"].values() for v in d.values() if v["rate"] is not None)
