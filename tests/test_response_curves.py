import numpy as np
import pandas as pd
import pytest

from media_mix_ai_lab.core.config import AdstockSpec, ModelConfig
from media_mix_ai_lab.modeling.ols import OLSModelEngine
from media_mix_ai_lab.optimization.response_curves import (
    build_response_curve,
    steady_state_multiplier,
)


def log_log_fixture(beta: float = 0.6, n: int = 40) -> pd.DataFrame:
    spend = np.linspace(10.0, 100.0, n)
    revenue = np.exp(2.0) * np.power(spend, beta)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-06", periods=n, freq="7D"),
            "revenue": revenue,
            "spend": spend,
        }
    )


def test_power_law_curve_matches_analytical_reference_and_is_reproducible():
    frame = log_log_fixture()
    config = ModelConfig(target="revenue", channels=("spend",), controls=(), kind="log_log")
    result = OLSModelEngine().fit(frame, config)

    first = build_response_curve(
        frame,
        result,
        "spend",
        minimum_spend=10,
        maximum_spend=100,
        reference_spend=40,
        points=10,
    )
    second = build_response_curve(
        frame,
        result,
        "spend",
        minimum_spend=10,
        maximum_spend=100,
        reference_spend=40,
        points=10,
    )

    assert first == second
    expected = np.exp(2.0) * np.power(100.0, 0.6)
    assert first.points[-1].conditional_outcome_per_week == pytest.approx(expected)
    outcomes = [point.conditional_outcome_per_week for point in first.points]
    assert np.all(np.diff(outcomes) >= 0)
    assert not first.points[0].is_extrapolation


def test_steady_state_adstock_multipliers_are_explicit():
    assert steady_state_multiplier(AdstockSpec()) == 1
    assert steady_state_multiplier(AdstockSpec(method="geometric", alpha=0.5)) == 2
    assert (
        steady_state_multiplier(
            AdstockSpec(method="geometric", alpha=0.5, weight_scaling="sum_to_one")
        )
        == 1
    )
    delayed = AdstockSpec(method="delayed_geometric", alpha=0.5, theta=1, max_lag_weeks=2)
    assert steady_state_multiplier(delayed) == pytest.approx(2.0)


def test_curve_rejects_unsupported_and_nonsensical_scenarios():
    frame = log_log_fixture(beta=-0.2)
    log_config = ModelConfig(target="revenue", channels=("spend",), controls=(), kind="log_log")
    negative = OLSModelEngine().fit(frame, log_config)
    with pytest.raises(ValueError, match="negative"):
        build_response_curve(
            frame,
            negative,
            "spend",
            minimum_spend=10,
            maximum_spend=100,
            reference_spend=40,
        )

    ols = OLSModelEngine().fit(log_log_fixture(), log_config.model_copy(update={"kind": "ols"}))
    with pytest.raises(ValueError, match="log-log"):
        build_response_curve(
            log_log_fixture(),
            ols,
            "spend",
            minimum_spend=10,
            maximum_spend=100,
            reference_spend=40,
        )

    changed = log_log_fixture()
    changed.loc[0, "revenue"] += 1
    with pytest.raises(ValueError, match="different dataset"):
        build_response_curve(
            changed,
            OLSModelEngine().fit(log_log_fixture(), log_config),
            "spend",
            minimum_spend=10,
            maximum_spend=100,
            reference_spend=40,
        )
