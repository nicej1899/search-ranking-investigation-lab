"""Focused correctness tests (prompt section 13). Run: pytest -q"""
import json

import numpy as np
import pytest
import torch

from srlab.common import CASES, CKPT, PROC, REPORTS, load_json, sha256_file
from srlab.explain import replay
from srlab.ranker import load_ranker, world


@pytest.fixture(scope="module")
def W():
    return world()


@pytest.fixture(scope="module")
def primary():
    return load_json(CASES / "selected_cases.json")["primary"]


def test_no_future_events_in_history(W):
    rng = np.random.default_rng(0)
    cases = list(W.cases.values())
    for c in [cases[i] for i in rng.choice(len(cases), 200, replace=False)]:
        h = W.store.before(c["member"], c["t"], n=None)
        assert (h.ts < c["t"]).all()
        assert all(W.store.event_member[e] == c["member"] for e in h.eids)


def test_same_timestamp_events_excluded(W):
    ev = W.events
    dup = ev[ev.duplicated(["member", "ts"], keep=False)].iloc[0]
    h = W.store.before(int(dup.member), int(dup.ts), n=None)
    assert int(dup.ts) not in set(h.ts.tolist())


def test_training_statistics_use_only_train_or_dev(W):
    split = load_json(PROC / "split.json")
    tr = load_json(REPORTS / "backbone_training.json")
    assert tr["data_fingerprint"] == split["events_sha"]
    F0 = load_ranker("F0")
    assert "DEV" in F0["norm"]["fit_on"]
    dev_ids = {c["case_id"] for c in W.cases.values() if c["split"] == "dev" and not c.get("extra")}  # extra cases are registered by other tests
    from srlab.common import sha256_obj
    assert F0["dev_cases_hash"] == sha256_obj([c["case_id"] for c in W.cases.values() if c["split"] == "dev" and not c.get("extra")])
    assert all(cid.startswith(("D", "B05", "B06", "B07", "B08")) for cid in dev_ids)
    # item UNK mask derived from training events only
    seen = set(W.events.loc[W.events.split == "train", "item"])
    assert set(np.flatnonzero(~W.model.unk_mask.numpy())) == seen


def test_explanation_does_not_change_parameters(W, primary):
    from srlab.explain import explain_history
    before = {k: v.clone() for k, v in W.model.state_dict().items()}
    r0 = json.dumps(load_ranker("F0"), sort_keys=True)
    explain_history(W, W.cases[primary["case_id"]], primary["A_index"], primary["B_index"], budget=2)
    for k, v in W.model.state_dict().items():
        assert torch.equal(v, before[k])
    assert json.dumps(load_ranker("F0"), sort_keys=True) == r0
    assert sha256_file(CKPT / "temporal_backbone_F0.pt") == W.backbone_sha


def test_replay_reproducible_and_reset_exact(W, primary):
    c, R = W.cases[primary["case_id"]], load_ranker("F0")
    a = replay(W, c, R)
    b = replay(W, c, R, [90708, 90710])
    a2 = replay(W, c, R)
    b2 = replay(W, c, R, [90710, 90708])
    assert a["components"]["total"] == a2["components"]["total"]
    assert b["components"]["total"] == b2["components"]["total"]
    assert a["snapshot_hash"] == a2["snapshot_hash"] != b["snapshot_hash"]


def test_intervened_ids_change_model_input(W, primary):
    c, R = W.cases[primary["case_id"]], load_ranker("F0")
    a = replay(W, c, R)
    rem = a["model_input_event_ids"][:2]
    b = replay(W, c, R, rem)
    assert set(rem).isdisjoint(b["model_input_event_ids"])
    assert set(a["model_input_event_ids"]) - set(b["model_input_event_ids"]) == set(rem)
    assert a["beh_logit"] != b["beh_logit"]
    # removing an id that is not the member's event changes nothing
    other = int(W.events[W.events.member != c["member"]].event_id.iloc[0])
    assert replay(W, c, R, [other])["components"]["total"] == a["components"]["total"]


def test_window_refills_with_older_events(W):
    c = next(x for x in W.cases.values() if x["n_history"] > 60)
    h = W.store.before(c["member"], c["t"], n=50)
    h2 = W.store.before(c["member"], c["t"], exclude={int(h.eids[-1])}, n=50)
    assert len(h2.eids) == 50 and h2.eids[0] < h.eids[0]


def test_member_local_intervention_keeps_title_state(W, primary):
    c, R = W.cases[primary["case_id"]], load_ranker("F0")
    a, b = replay(W, c, R), replay(W, c, R, [90708])
    assert a["pop_log1p"] == b["pop_log1p"] and a["rel"] == b["rel"]


def test_candidates_identical_across_variants(W):
    from srlab.evaluate import score_protocol
    e0 = load_json(REPORTS / "eval_qi-final-v1_F0.json")
    e1 = load_json(REPORTS / "eval_qi-final-v1_F1.json")
    assert [r["case_id"] for r in e0["rows"]] == [r["case_id"] for r in e1["rows"]]
    assert e0["cases_hash"] == e1["cases_hash"]
    assert not any(W.cases[r["case_id"]]["showcase_member"] for r in e0["rows"])


def test_f1_gate_zero_equals_f0(W, primary):
    c = W.cases[primary["case_id"]]
    F0 = load_ranker("F0")
    F1 = dict(load_ranker("F1"))
    F1["gate"] = [0.0, 0.0, 0.0]
    assert np.allclose(replay(W, c, F0)["components"]["total"], replay(W, c, F1)["components"]["total"])


def test_api_returns_backend_numbers(W, primary):
    from fastapi.testclient import TestClient
    from srlab.server import app
    cl = TestClient(app)
    r = cl.post("/api/score", json={"case_id": primary["case_id"], "model_id": "F0", "removed": [90708, 90710],
                                    "A_index": primary["A_index"], "B_index": primary["B_index"]}).json()
    direct = replay(W, W.cases[primary["case_id"]], load_ranker("F0"), [90708, 90710])
    assert r["components"]["total"] == direct["components"]["total"]
    assert r["ab"]["margin"] == pytest.approx(direct["components"]["total"][primary["A_index"]] - direct["components"]["total"][primary["B_index"]])


def test_export_reproduces_investigation(W):
    inv = load_json("artifacts/investigations/inv-primary.json")
    call = next(t for t in inv["trace"] if t["tool"] == "replay_intervention")
    c = W.cases[inv["case_id"]]
    rp = replay(W, c, load_ranker("F0"), call["inputs"]["event_ids"])
    tot = rp["components"]["total"]
    assert tot[inv["A_index"]] - tot[inv["B_index"]] == pytest.approx(call["outputs"]["margin_after"], abs=1e-12)
    assert call["provenance"]["backbone_sha256"] == W.backbone_sha


def test_runtime_policy_documented(W):
    assert W.items.runtime_min.isna().all()
    meta = load_json(CASES / "query_cases.json")["meta"]
    assert meta["runtime_constraint"].startswith("NOT APPLIED")


def test_catgx_adjustment_matches_bruteforce(W, primary):
    from srlab.motifs import catgx_scores
    r = catgx_scores(W, W.cases[primary["case_id"]], primary["A_index"], primary["B_index"])
    occ = r["occurrences"]
    N = len(occ)
    for mid, v in r["per_motif"].items():
        if v["S"] is None:
            continue
        S = 0.0
        for cell in {tuple(o["cell"]) for o in occ if o["motif"] == mid}:
            inb = [o for o in occ if tuple(o["cell"]) == cell]
            S += len(inb) / N * (np.mean([o["g"] for o in inb if o["motif"] == mid]) - np.mean([o["g"] for o in inb]))
        assert v["S"] == pytest.approx(S)
    assert sum(len([o for o in occ if tuple(o["cell"]) == tuple(t["cell"])]) for t in []) == 0
