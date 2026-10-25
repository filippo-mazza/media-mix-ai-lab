import numpy as np
import pandas as pd
import pytest

from media_mix_ai_lab.core.config import AdstockSpec, ModelConfig
from media_mix_ai_lab.modeling.adstock import (
    adstock_impulse_response,
    apply_adstock,
    transform_media_frame,
)
from media_mix_ai_lab.modeling.ols import OLSModelEngine


def test_none_and_recursive_geometric_transforms_preserve_explicit_boundaries():
    signal = pd.Series([1.0, 2.0, 3.0], name="tv_spend")
    original = signal.copy()

    unchanged = apply_adstock(signal, AdstockSpec())
    zero_decay = apply_adstock(signal, AdstockSpec(method="geometric", alpha=0.0))
    transformed = apply_adstock(
        signal,
        AdstockSpec(method="geometric", alpha=0.5, initial_state=2.0),
    )
    normalized = apply_adstock(
        signal,
        AdstockSpec(method="geometric", alpha=0.5, weight_scaling="sum_to_one"),
    )

    pd.testing.assert_series_equal(unchanged, signal)
    pd.testing.assert_series_equal(zero_decay, signal)
    assert transformed.to_list() == pytest.approx([2.0, 3.0, 4.5])
    assert normalized.to_list() == pytest.approx([0.5, 1.25, 2.125])
    pd.testing.assert_series_equal(signal, original)


def test_delayed_geometric_weights_and_prior_history_follow_finite_formula():
    signal = pd.Series([2.0, 4.0, 8.0], name="tv_spend")
    spec = AdstockSpec(
        method="delayed_geometric",
        alpha=0.5,
        theta=1.0,
        max_lag_weeks=2,
        historical_values=(10.0, 20.0),
    )

    transformed = apply_adstock(signal, spec)
    weights = adstock_impulse_response(spec)

    assert weights["Weight"].to_list() == pytest.approx([0.5, 1.0, 0.5])
    assert transformed.to_list() == pytest.approx([26.0, 14.0, 9.0])


def test_delayed_geometric_zero_history_and_sum_to_one_scaling():
    signal = pd.Series([1.0, 0.0, 0.0], name="tv_spend")
    raw = AdstockSpec(method="delayed_geometric", alpha=0.5, theta=1, max_lag_weeks=2)
    normalized = raw.model_copy(update={"weight_scaling": "sum_to_one"})

    assert apply_adstock(signal, raw).to_list() == pytest.approx([0.5, 1.0, 0.5])
    assert apply_adstock(signal, normalized).to_list() == pytest.approx([0.25, 0.5, 0.25])


def test_adstock_requires_regular_dates_and_nonnegative_media():
    weekly = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="7D"),
            "tv_spend": [1.0, 2.0, 3.0, 4.0],
        }
    )
    config = ModelConfig(
        target="revenue",
        channels=("tv_spend",),
        controls=(),
        adstock={"tv_spend": AdstockSpec(method="geometric")},
    )

    transformed = transform_media_frame(weekly, config)
    assert transformed["tv_spend"].to_list() == pytest.approx([1.0, 2.5, 4.25, 6.125])
    pd.testing.assert_series_equal(weekly["date"], transformed["date"])

    missing_week = weekly.drop(index=1)
    with pytest.raises(ValueError, match="missing weeks are rejected"):
        transform_media_frame(missing_week, config)

    with pytest.raises(ValueError, match="ascending order"):
        transform_media_frame(weekly.iloc[::-1], config)

    duplicated_date = weekly.copy()
    duplicated_date.loc[1, "date"] = duplicated_date.loc[0, "date"]
    with pytest.raises(ValueError, match="unique"):
        transform_media_frame(duplicated_date, config)

    negative = weekly.copy()
    negative.loc[1, "tv_spend"] = -1
    with pytest.raises(ValueError, match="negative media spend"):
        transform_media_frame(negative, config)


def test_ols_fits_transformed_media_and_preserves_controls_and_source_data():
    rng = np.random.default_rng(29)
    spend = rng.uniform(1.0, 10.0, 40)
    promotion = np.tile([0.0, 1.0, 0.0, 0.0], 10)
    dates = pd.date_range("2024-01-01", periods=len(spend), freq="7D")
    signal = apply_adstock(
        pd.Series(spend),
        AdstockSpec(method="geometric", alpha=0.4, weight_scaling="sum_to_one"),
    )
    frame = pd.DataFrame(
        {
            "date": dates,
            "revenue": 12 + 2 * signal.to_numpy() + 3 * promotion,
            "tv_spend": spend,
            "promotion": promotion,
        }
    )
    original = frame.copy(deep=True)
    config = ModelConfig(
        target="revenue",
        channels=("tv_spend",),
        controls=("promotion",),
        adstock={
            "tv_spend": AdstockSpec(method="geometric", alpha=0.4, weight_scaling="sum_to_one")
        },
    )

    result = OLSModelEngine().fit(frame, config)

    assert result.coefficients["tv_spend"] == pytest.approx(2.0)
    assert result.coefficients["promotion"] == pytest.approx(3.0)
    assert "recursive geometric adstock" in result.transformations["tv_spend"]
    assert result.config.adstock["tv_spend"].alpha == 0.4
    assessment = result.term_assessments["tv_spend"]
    assert "sum_to_one weights" in assessment.scale
    assert "geometric-adstocked tv_spend signal" in assessment.interpretation
    pd.testing.assert_frame_equal(frame, original)


def test_log_log_assessment_describes_elasticity_of_adstocked_signal():
    rng = np.random.default_rng(31)
    spend = rng.uniform(1.0, 10.0, 60)
    promotion = np.tile([0.0, 1.0], 30)
    spec = AdstockSpec(method="geometric", alpha=0.35, weight_scaling="sum_to_one")
    transformed = apply_adstock(pd.Series(spend), spec).to_numpy()
    revenue = np.exp(1.1 + 0.6 * np.log(transformed) + 0.1 * promotion)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(spend), freq="7D"),
            "revenue": revenue,
            "tv_spend": spend,
            "promotion": promotion,
        }
    )
    config = ModelConfig(
        kind="log_log",
        target="revenue",
        channels=("tv_spend",),
        controls=("promotion",),
        adstock={"tv_spend": spec},
    )

    result = OLSModelEngine().fit(frame, config)
    assessment = result.term_assessments["log(tv_spend)"]

    assert assessment.scale == (
        "conditional elasticity of geometric-adstocked tv_spend signal (sum_to_one weights)"
    )
    assert "1% increase in geometric-adstocked tv_spend signal" in assessment.interpretation


def test_adstock_ols_requires_valid_weekly_dates_and_rejects_negative_spend():
    config = ModelConfig(
        target="revenue",
        channels=("tv_spend",),
        controls=(),
        adstock={"tv_spend": AdstockSpec(method="geometric")},
    )
    frame = pd.DataFrame({"revenue": [2.0, 4.0, 8.0], "tv_spend": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="requires a date column"):
        OLSModelEngine().fit(frame, config)

    frame["date"] = pd.date_range("2024-01-01", periods=3, freq="7D")
    frame.loc[1, "tv_spend"] = -2.0
    with pytest.raises(ValueError, match="negative media spend"):
        OLSModelEngine().fit(frame, config)

    no_adstock = ModelConfig(target="revenue", channels=("tv_spend",), controls=())
    with pytest.raises(ValueError, match="negative media spend"):
        OLSModelEngine().fit(frame, no_adstock)
