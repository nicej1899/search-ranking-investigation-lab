"""Production-shaped pipeline: retrieval, full-catalog metrics, SHAP additivity, baseline metrics."""
import numpy as np
import pytest

from srlab import ltr
from srlab.baselines import metrics
from srlab.common import CKPT


def test_full_catalog_ndcg_counts_missed_titles():
    rel = np.zeros(10); rel[[1, 7]] = 2; ok = np.ones(10, bool)
    cand = np.array([1, 2, 3])  # retrieval missed title 7
    s = np.array([3.0, 2.0, 1.0])
    v = ltr.ndcg_full(s, cand, rel, ok)
    assert 0 < v < 1  # perfect order on what was retrieved, still penalised for the miss
    assert ltr.ndcg_full(s, np.array([1, 7, 3]), rel, ok) == pytest.approx(1.0)


def test_baseline_metrics_exclude_seen_and_non_catalog():
    s = np.arange(20, dtype=float)
    cand = np.ones(20, bool); cand[19] = False
    r = metrics(s, cand, {18, 0}, seen={17})
    assert r["recall10"] == pytest.approx(0.5)  # 18 is top-1 once 19 (not in catalog) and 17 (seen) are removed


@pytest.mark.skipif(not (CKPT / "ranker_lambdamart.txt").exists(), reason="run python -m srlab.ltr")
def test_retrieval_and_shap_additivity():
    import lightgbm as lgb
    rows, names, _ = ltr.build("final", limit=6)
    bst = lgb.Booster(model_file=str(CKPT / "ranker_lambdamart.txt"))
    for r in rows:
        assert not r["ok"][r["cand"]].__invert__().any()  # never retrieves already-rated or out-of-catalog titles
        p = bst.predict(r["X"])
        c = bst.predict(r["X"], pred_contrib=True)
        assert np.allclose(c.sum(1), p, atol=1e-6)  # TreeSHAP parts add up exactly to the score
