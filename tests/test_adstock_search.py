import numpy as np
import pandas as pd

from media_mix_ai_lab.core.config import ModelConfig
from media_mix_ai_lab.modeling.ols import OLSModelEngine
from media_mix_ai_lab.optimization.adstock_search import (
    FeasibilityPolicy,
    assess_feasibility,
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


def test_feasibility_counts_warmup_and_requires_residual_degrees_of_freedom():
    config = ModelConfig(target="revenue", channels=("spend",), controls=())
    result = assess_feasibility(
        fixture(),
        config,
        method="geometric",
        max_lag=8,
        holdout_fraction=0.2,
    )

    assert (result.n_fit, result.n_validation, result.n_test) == (22, 10, 10)
    assert result.n_warmup == 8
    assert (result.required_fit, result.required_validation, result.required_test) == (11, 9, 9)
    assert result.feasible


def test_small_dataset_and_excessive_predictor_count_are_not_feasible():
    frame = fixture(n=14)
    config = ModelConfig(target="revenue", channels=("spend",), controls=())
    small = assess_feasibility(frame, config, method="geometric", max_lag=8, holdout_fraction=0.2)
    assert not small.feasible
    assert small.n_fit == 0

    many_predictors = ModelConfig(
        target="revenue",
        channels=("spend",),
        controls=tuple(f"control_{index}" for index in range(20)),
    )
    crowded = assess_feasibility(
        fixture_with_controls(50, many_predictors.controls),
        many_predictors,
        method="geometric",
        max_lag=8,
        holdout_fraction=0.2,
    )
    assert not crowded.feasible
    assert crowded.free_coefficients == 22
    assert crowded.required_fit == 31


def fixture_with_controls(n, controls):
    frame = fixture(n)
    for index, control in enumerate(controls):
        frame[control] = np.arange(n, dtype=float) + index**2
    return frame


def test_long_lag_log_invalidity_and_configurable_borderline_policy():
    config = ModelConfig(target="revenue", channels=("spend",), controls=())
    frame = fixture(n=25)
    borderline = assess_feasibility(
        frame,
        config,
        method="geometric",
        max_lag=0,
        holdout_fraction=0.2,
        policy=FeasibilityPolicy(minimum_residual_degrees_freedom=5, minimum_holdout_rows=5),
    )
    assert (borderline.n_fit, borderline.n_validation, borderline.n_test) == (15, 5, 5)
    assert borderline.feasible
    configurable = assess_feasibility(
        frame,
        config,
        method="geometric",
        max_lag=0,
        holdout_fraction=0.2,
        policy=FeasibilityPolicy(minimum_residual_degrees_freedom=10, minimum_holdout_rows=6),
    )
    assert not configurable.feasible
    assert configurable.required_fit == 12
    assert configurable.required_validation == configurable.required_test == 6

    long_lag = assess_feasibility(
        fixture(),
        config,
        method="geometric",
        max_lag=12,
        holdout_fraction=0.2,
    )
    assert not long_lag.feasible
    assert long_lag.required_validation == 13

    frame = fixture()
    frame.loc[10, "revenue"] = 0
    frame.loc[32, "spend"] = 0
    log_config = config.model_copy(update={"kind": "log_log"})
    invalid_log = assess_feasibility(
        frame,
        log_config,
        method="geometric",
        max_lag=8,
        holdout_fraction=0.2,
    )
    assert not invalid_log.feasible
    assert invalid_log.n_log_unusable_fit == 1
    with np.testing.assert_raises_regex(ValueError, "not feasible"):
        search(frame, log_config, max_lag=8)

    carryover_frame = fixture()
    carryover_frame.loc[32, "spend"] = 0
    carryover_feasibility = assess_feasibility(
        carryover_frame,
        log_config,
        method="geometric",
        max_lag=8,
        holdout_fraction=0.2,
    )
    assert carryover_feasibility.feasible
    assert carryover_feasibility.n_log_unusable_validation == 0
    carryover_search = search(carryover_frame, log_config, max_lag=8, budget=4)
    assert carryover_search["baseline"] is None

    validation_zero = fixture()
    validation_zero.loc[32, "revenue"] = 0
    validation_search = search(validation_zero, log_config, max_lag=8, budget=4)
    assert validation_search["selected"]["validation_rmse"] >= 0


def test_final_test_outcome_is_not_consulted_until_explicit_evaluation():
    frame = fixture()
    config = ModelConfig(target="revenue", channels=("spend",), controls=(), kind="log_log")
    original = search(frame, config, max_lag=8, budget=4)
    changed_test = frame.copy()
    changed_test.loc[40, "revenue"] = 0
    changed = search(changed_test, config, max_lag=8, budget=4)

    assert original["selected"] == changed["selected"]
    evaluated = evaluate_final_test(changed_test, changed)
    assert np.isfinite(evaluated["final_test"]["rmse"])


def test_search_accepts_an_explicit_feasibility_policy():
    config = ModelConfig(target="revenue", channels=("spend",), controls=())
    frame = fixture(n=16)

    with np.testing.assert_raises_regex(ValueError, "not feasible"):
        search(frame, config, method="none", max_lag=0)

    result = search(
        frame,
        config,
        method="none",
        max_lag=0,
        policy=FeasibilityPolicy(minimum_residual_degrees_freedom=5, minimum_holdout_rows=2),
    )
    assert result["feasibility"]["required_validation"] == 2
    assert result["search_specification"]["feasibility_policy"]["minimum_holdout_rows"] == 2


def test_search_refuses_infeasible_data_before_fitting(monkeypatch):
    config = ModelConfig(target="revenue", channels=("spend",), controls=())

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("infeasible search must not fit a candidate")

    monkeypatch.setattr(OLSModelEngine, "fit", fail_if_called)
    with np.testing.assert_raises_regex(ValueError, "not feasible"):
        search(fixture(n=14), config, max_lag=8)


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
    assert first["feasibility"]["n_warmup"] == 8
    assert first["feasibility"]["n_fit"] == 22
    assert all(row["n_fit"] == 22 for row in first["candidates"])
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


def test_split_changes_recompute_effective_counts():
    config = ModelConfig(target="revenue", channels=("spend",), controls=())
    frame = fixture()
    standard = assess_feasibility(
        frame, config, method="geometric", max_lag=8, holdout_fraction=0.2
    )
    changed = assess_feasibility(frame, config, method="geometric", max_lag=8, holdout_fraction=0.4)

    assert standard.feasible and changed.feasible
    assert (standard.n_fit, standard.n_test) != (changed.n_fit, changed.n_test)
