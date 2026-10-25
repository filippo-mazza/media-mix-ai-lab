import numpy as np
import pandas as pd

from media_mix_ai_lab.core.config import ModelConfig
from media_mix_ai_lab.optimization.adstock_search import candidate_specs, partition, search


def fixture(n=50):
    rng = np.random.default_rng(4)
    spend = rng.uniform(1, 10, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="7D"),
            "revenue": 10 + 2 * spend + rng.normal(0, 0.2, n),
            "spend": spend,
        }
    )


def test_partition_counts_and_budget():
    fit, validation, test = partition(fixture(), "date")
    assert (len(fit), len(validation), len(test)) == (30, 10, 10)
    assert len(candidate_specs(("spend",), "delayed_geometric", 8, 7)) == 7


def test_search_is_deterministic_and_final_outcome_isolation():
    frame = fixture()
    config = ModelConfig(target="revenue", channels=("spend",), controls=())
    first = search(frame, config, budget=4)
    assert first["selected"] == search(frame, config, budget=4)["selected"]
    changed = frame.copy()
    changed.loc[40:, "revenue"] += 500
    second = search(changed, config, budget=4)
    assert first["selected"] == second["selected"]
    assert first["final_test"]["rmse"] != second["final_test"]["rmse"]
