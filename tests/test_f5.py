from srlab.backbones import get_backbone
from srlab.f5 import crowd_links
from srlab.server import W_


def test_crowd_links_excludes_member_and_respects_n():
    W, bb = W_(), get_backbone("tgat2")
    c = next(c for c in W.cases.values() if c["split"] == "dev" and c.get("extra") == "conversational")
    ex = crowd_links(bb, c["candidates"][:3], c["member"], c["t"], 2)
    assert len(ex) <= 6
    for i in c["candidates"][:3]:
        o, ts, e, r = bb.G.neighbors(int(i) + bb.G.U, c["t"], None)
        mine = {int(x) for x, who in zip(e, o) if who == c["member"]}
        assert not (ex & mine)
        assert all(t < c["t"] for t in ts)


def test_live_f5_differs_only_in_history():
    from srlab.live import rank
    W = W_()
    cid = next(k for k, c in W.cases.items() if c["split"] == "dev" and c.get("extra") == "conversational")
    a, b = rank(W, cid, "tgat2", "F0"), rank(W, cid, "tgat2", "F5")
    pa = {r["index"]: r["parts"] for r in a["rows"]}; pb = {r["index"]: r["parts"] for r in b["rows"]}
    assert all(abs(pa[i]["relevance"] - pb[i]["relevance"]) < 1e-9 and abs(pa[i]["popularity"] - pb[i]["popularity"]) < 1e-9 for i in pa)
    assert len(b["removed"]) > 0
