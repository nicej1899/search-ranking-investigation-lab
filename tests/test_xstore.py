"""Verified-explanation store: mechanism keys and retrieval."""
import numpy as np

from srlab.xstore import FIXES, Store, key, recommend


def _fake(i, motif, crowd, own, fixed):
    return {"case_id": f"c{i}", "lead": {"relevance": -0.2, "popularity": 0.5 * crowd, "behaviour": 0.1},
            "kinds_A": {"the title's recent raters": crowd, "the member's own ratings": 3 - crowd},
            "kinds_B": {"the member's own ratings": own, "other people's ratings": 3 - own},
            "motif_A": motif, "motif_B": "BRIDGE",
            "test": {"p0": 0.8, "del_A": {"p": 0.8 - 0.05 * crowd}, "del_B": {"p": 0.85}}, "fixed": fixed}


def test_key_is_fixed_length_and_finite():
    v = key(_fake(0, "POP", 2, 2, {m: False for m in FIXES}))
    assert v.shape == (3 + 3 + 3 + 6 + 6 + 3,) and np.isfinite(v).all()


def test_retrieval_prefers_same_mechanism_and_votes():
    rows = [_fake(i, "POP", 3, 2, {"F2": False, "F3": False, "F4": True}) for i in range(6)]
    rows += [_fake(10 + i, "SEQ", 0, 0, {"F2": True, "F3": False, "F4": False}) for i in range(6)]
    st = Store(rows)
    q = _fake(99, "POP", 3, 2, {m: False for m in FIXES})
    neigh = st.query(q, k=5)
    assert all(n["motif_A"] == "POP" for n, _ in neigh)
    assert recommend(neigh)[0] == "F4"
