"""Planted-cause test: planting adds exactly the intended cohort; the report is internally consistent."""
import pytest

from srlab.common import REPORTS, load_json
from srlab.data import load_events, load_items
from srlab import plant


def test_plant_adds_cohort_only_in_training_period():
    ev, it = load_events(), load_items()
    evp, info = plant.plant(ev, it, plant.SEM)
    new = evp.iloc[len(ev):]
    assert len(new) == info["planted_events"] == plant.N_CARRIER * 9 + 2 * plant.N_CONTROL * 8
    assert set(new.split) == {"train"} and new.member.min() > ev.member.max()  # new viewers, real members untouched
    cols = ["event_id", "member", "item", "ts", "rating"]
    assert (evp.iloc[:len(ev)][cols].to_numpy(float) == ev[cols].to_numpy(float)).all()


@pytest.mark.skipif(not (REPORTS / "plant_report.json").exists(), reason="run python -m srlab.plant_report")
def test_report_consistency():
    r = load_json(REPORTS / "plant_report.json")
    for p in r["patterns"]:
        assert p["all"]["Try every pair"]["fidelity"] == pytest.approx(1.0, abs=1e-5)  # exhaustive is the reference
        for k, v in p["all"].items():
            assert 0 <= v["found_any"] <= 1 and v["found_both"] <= v["found_any"]
