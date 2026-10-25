import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from media_mix_ai_lab.core.config import ModelConfig
from media_mix_ai_lab.diagnostics.temporal import (
    chronological_split,
    evaluate_temporal_holdout,
    same_holdout_period,
)
from media_mix_ai_lab.modeling.ols import OLSModelEngine


def temporal_fixture(rows: int = 40, constant_outcome: float | None = None) -> pd.DataFrame:
    spend = np.arange(1.0, rows + 1)
    revenue = (
        np.full(rows, constant_outcome)
        if constant_outcome is not None
        else 10 + 3 * spend + np.sin(spend)
    )
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="7D"),
            "revenue": revenue,
            "tv_spend": spend,
        }
    )


def temporal_config() -> ModelConfig:
    return ModelConfig(target="revenue", channels=("tv_spend",), controls=())


def test_chronological_split_preserves_order_and_supports_date_cutoff():
    frame = temporal_fixture()

    training, holdout = chronological_split(frame, "date", holdout_fraction=0.25)
    cutoff_training, cutoff_holdout = chronological_split(
        frame, "date", cutoff=frame["date"].iloc[30]
    )

    assert len(training) == 30
    assert len(holdout) == 10
    assert training["date"].max() < holdout["date"].min()
    assert cutoff_training["date"].max() < cutoff_holdout["date"].min()
    assert cutoff_holdout["date"].iloc[0] == frame["date"].iloc[30]


def test_chronological_split_rejects_empty_holdout_and_invalid_order():
    frame = temporal_fixture()

    with pytest.raises(ValueError, match="leave at least one training and one holdout"):
        chronological_split(frame, "date", cutoff=frame["date"].iloc[-1] + np.timedelta64(7, "D"))
    with pytest.raises(ValueError, match="ascending order"):
        chronological_split(frame.iloc[::-1], "date")


def test_temporal_evaluation_reports_separate_metrics_and_diagnostics():
    result = evaluate_temporal_holdout(temporal_fixture(), temporal_config(), holdout_fraction=0.25)

    assert result.training_size == 30
    assert result.holdout_size == 10
    assert result.training_metrics.mae < result.holdout_metrics.mae
    assert result.training_metrics.wape is not None
    assert result.holdout_metrics.wape is not None
    assert len(result.predictions) == 10
    assert result.durbin_watson is not None
    assert result.ljung_box_p_value is not None
    assert result.variance_inflation_factors == {"tv_spend": pytest.approx(1)}
    assert result.condition_number > 1


def test_vif_uses_training_design_and_intercept():
    frame = temporal_fixture()
    frame["search_spend"] = 10 + 0.8 * frame["tv_spend"] + np.sin(frame["tv_spend"])
    config = ModelConfig(
        target="revenue",
        channels=("tv_spend", "search_spend"),
        controls=(),
    )
    training_design = sm.add_constant(frame.iloc[:30][["tv_spend", "search_spend"]])

    result = evaluate_temporal_holdout(frame, config, holdout_fraction=0.25)

    assert result.variance_inflation_factors["tv_spend"] == pytest.approx(
        variance_inflation_factor(training_design.to_numpy(), 1)
    )
    assert result.variance_inflation_factors["search_spend"] == pytest.approx(
        variance_inflation_factor(training_design.to_numpy(), 2)
    )


def test_future_segment_changes_do_not_leak_into_training_fit_or_metrics():
    frame = temporal_fixture()
    changed_future = frame.copy()
    changed_future.loc[30:, "revenue"] += 100_000
    changed_future.loc[30:, "tv_spend"] *= 100

    original_result = evaluate_temporal_holdout(frame, temporal_config(), holdout_fraction=0.25)
    changed_result = evaluate_temporal_holdout(
        changed_future, temporal_config(), holdout_fraction=0.25
    )

    assert changed_result.training_metrics == original_result.training_metrics
    assert changed_result.holdout_metrics != original_result.holdout_metrics


def test_perfect_fit_has_zero_error_metrics():
    frame = temporal_fixture()
    frame["revenue"] = 10 + 3 * frame["tv_spend"]

    result = evaluate_temporal_holdout(frame, temporal_config(), holdout_fraction=0.25)

    assert result.training_metrics.mae == pytest.approx(0, abs=1e-12)
    assert result.holdout_metrics.mae == pytest.approx(0, abs=1e-12)
    assert result.holdout_metrics.rmse == pytest.approx(0, abs=1e-12)


def test_constant_zero_outcome_has_undefined_wape_and_short_sample_warnings():
    frame = temporal_fixture(rows=8, constant_outcome=0)

    result = evaluate_temporal_holdout(frame, temporal_config(), holdout_fraction=0.25)

    fitted = OLSModelEngine().fit(frame.iloc[:6], temporal_config())
    assert result.training_metrics.mae == 0
    assert result.holdout_metrics.wape is None
    assert fitted.r_squared is None
    assert fitted.adjusted_r_squared is None
    assert result.durbin_watson is None
    assert result.ljung_box_p_value is None
    assert any("WAPE is undefined" in warning for warning in result.warnings)
    assert any("fewer than eight" in warning for warning in result.warnings)


def test_saved_evaluations_can_detect_different_holdout_periods():
    frame = temporal_fixture()
    fraction_result = evaluate_temporal_holdout(frame, temporal_config(), holdout_fraction=0.25)
    cutoff_result = evaluate_temporal_holdout(
        frame, temporal_config(), cutoff=frame["date"].iloc[25]
    )

    assert same_holdout_period(fraction_result, fraction_result.model_copy(deep=True))
    assert not same_holdout_period(fraction_result, cutoff_result)
