import numpy as np
import pandas as pd

from media_mix_ai_lab.core.config import ModelConfig
from media_mix_ai_lab.optimization.adstock_search import (
    candidate_specs,
    evaluate_final_test,
    partition,
    search,
)


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


def test_bounded_candidates_explore_every_selected_channel():
    channels = ("tv", "search", "social")
    candidates = candidate_specs(channels, "geometric", 8, 10)

    assert len(candidates) == 10
    reference = candidates[0]
    for channel in channels:
        assert any(
            candidate[channel] != reference[channel]
            and all(candidate[other] == reference[other] for other in channels if other != channel)
            for candidate in candidates
        )
    assert candidates == candidate_specs(channels, "geometric", 8, 10)


def test_delayed_candidates_can_reach_the_configured_lag_horizon():
    candidates = candidate_specs(("spend",), "delayed_geometric", 12, 3)

    assert any(candidate["spend"].theta == 12 for candidate in candidates)


def test_search_is_deterministic_and_final_outcome_isolation():
    frame = fixture()
    config = ModelConfig(target="revenue", channels=("spend",), controls=())
    first = search(frame, config, budget=4)
    assert first["selected"] == search(frame, config, budget=4)["selected"]
    assert first["final_test"] is None
    changed = frame.copy()
    changed.loc[40:, "revenue"] += 500
    second = search(changed, config, budget=4)
    assert first["selected"] == second["selected"]
    first_evaluation = evaluate_final_test(frame, first)
    second_evaluation = evaluate_final_test(changed, second)
    assert first_evaluation["final_test"]["rmse"] != second_evaluation["final_test"]["rmse"]
    assert first["final_test"] is None
    with np.testing.assert_raises_regex(ValueError, "already contains"):
        evaluate_final_test(frame, first_evaluation)
