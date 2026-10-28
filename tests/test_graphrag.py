from srlab.backbones import get_backbone
from srlab.graphrag import GraphStore
from srlab.server import W_
from srlab.xstore import load


def test_graph_walk_retrieves_same_title_failures_first():
    W, U = W_(), get_backbone("tgat2").G.U
    dev = load("dev")[:120]
    gs = GraphStore(dev, W, U, "pushed-up title + crowd")
    q = dev[0]
    nb = gs.query(q, 5, exclude=q["case_id"])
    assert len(nb) == 5 and all(r["case_id"] != q["case_id"] for r, _ in nb)
    assert nb[0][1] >= nb[-1][1]
    same = [r for r in dev[1:] if r["A"] == q["A"]]
    if same:
        assert nb[0][0]["A"] == q["A"]
