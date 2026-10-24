import numpy as np
import pandas as pd
import pytest

from media_mix_ai_lab.core.config import ModelConfig
from media_mix_ai_lab.modeling.ols import OLSModelEngine


def test_linear_ols_recovers_coefficients_and_reproducible_provenance():
    x = np.arange(1.0, 13.0)
    z = np.array([2, 5, 1, 7, 3, 9, 4, 11, 6, 13, 8, 15], dtype=float)
    control = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1], dtype=float)
    frame = pd.DataFrame(
        {
            "revenue": 5 + 2 * x - 3 * z + 0.5 * control,
            "tv_spend": x,
            "search_spend": z,
            "promotion": control,
        }
    )
    original = frame.copy(deep=True)
    config = ModelConfig(
        target="revenue", channels=("tv_spend", "search_spend"), controls=("promotion",)
    )
    engine = OLSModelEngine()

    result = engine.fit(frame, config)
    repeated = engine.fit(frame, config)

    assert result.coefficients == pytest.approx(
        {"const": 5, "tv_spend": 2, "search_spend": -3, "promotion": 0.5},
        abs=1e-8,
    )
    assert result.sample_size == len(frame)
    assert result.r_squared == pytest.approx(1)
    assert result.adjusted_r_squared == pytest.approx(1)
    assert result.residual_variance == pytest.approx(0, abs=1e-25)
    assert result.standard_errors.keys() == result.coefficients.keys()
    assert result.confidence_intervals.keys() == result.coefficients.keys()
    assert result.dataset_sha256 == repeated.dataset_sha256
    assert result.coefficients == pytest.approx(repeated.coefficients)
    assert result.package_versions["statsmodels"]
    changed = frame.copy()
    changed.loc[0, "revenue"] += 1
    assert engine.fit(changed, config).dataset_sha256 != result.dataset_sha256
    pd.testing.assert_frame_equal(frame, original)


def test_log_log_recovers_media_elasticities_and_keeps_controls_in_levels():
    x = np.arange(1.0, 21.0)
    z = np.array([2, 5, 1, 7, 3, 9, 4, 11, 6, 13, 8, 15, 10, 17, 12, 19, 14, 21, 16, 23])
    control = np.array([0, 1] * 10)
    revenue = np.exp(1.2 + 0.6 * np.log(x) - 0.25 * np.log(z) + 0.15 * control)
    frame = pd.DataFrame(
        {"revenue": revenue, "tv_spend": x, "search_spend": z, "promotion": control}
    )
    config = ModelConfig(
        kind="log_log",
        target="revenue",
        channels=("tv_spend", "search_spend"),
        controls=("promotion",),
    )

    result = OLSModelEngine().fit(frame, config)

    assert result.coefficients["log(tv_spend)"] == pytest.approx(0.6, abs=1e-8)
    assert result.coefficients["log(search_spend)"] == pytest.approx(-0.25, abs=1e-8)
    assert result.coefficients["promotion"] == pytest.approx(0.15, abs=1e-8)
    assert result.coefficients["const"] == pytest.approx(1.2, abs=1e-8)
    assert result.transformations == {
        "revenue": "natural log",
        "tv_spend": "natural log",
        "search_spend": "natural log",
        "promotion": "original level",
    }
    assert "conditional elasticities, not causal ROI" in " ".join(result.warnings)
    assert result.term_roles["log(tv_spend)"] == "media"


def test_log_log_rejects_nonpositive_outcome_and_media_with_counts_and_examples():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "revenue": [1, 2, 0, 4, 5],
            "tv_spend": [1, 0, -2, 4, 5],
        }
    )
    config = ModelConfig(kind="log_log", target="revenue", channels=("tv_spend",), controls=())

    with pytest.raises(ValueError, match=r"revenue: 1 nonpositive.*tv_spend: 2 nonpositive"):
        OLSModelEngine().fit(frame, config)
    with pytest.raises(ValueError, match="date=Timestamp"):
        OLSModelEngine().fit(frame, config)


def test_log_log_rejects_transformed_media_name_colliding_with_a_control():
    frame = pd.DataFrame(
        {
            "revenue": [3.0, 5.0, 7.0, 11.0],
            "tv_spend": [1.0, 2.0, 4.0, 8.0],
            "log(tv_spend)": [4.0, 3.0, 6.0, 5.0],
        }
    )
    config = ModelConfig(
        kind="log_log",
        target="revenue",
        channels=("tv_spend",),
        controls=("log(tv_spend)",),
    )

    with pytest.raises(ValueError, match="Transformed media predictor names collide"):
        OLSModelEngine().fit(frame, config)


def test_ols_rejects_rank_deficiency_and_too_few_rows():
    config = ModelConfig(target="revenue", channels=("tv_spend", "search_spend"), controls=())
    rank_deficient = pd.DataFrame(
        {"revenue": [1, 2, 3, 4], "tv_spend": [1, 2, 3, 4], "search_spend": [1, 2, 3, 4]}
    )
    with pytest.raises(ValueError, match="Rank-deficient design matrix"):
        OLSModelEngine().fit(rank_deficient, config)

    too_small = pd.DataFrame({"revenue": [2, 4], "tv_spend": [1, 2]})
    small_config = ModelConfig(target="revenue", channels=("tv_spend",), controls=())
    with pytest.raises(ValueError, match="Too few rows"):
        OLSModelEngine().fit(too_small, small_config)


def test_ols_supports_a_configured_no_intercept():
    x = np.arange(1.0, 9.0)
    frame = pd.DataFrame({"revenue": 3 * x, "tv_spend": x})
    config = ModelConfig(target="revenue", channels=("tv_spend",), controls=(), intercept=False)

    result = OLSModelEngine().fit(frame, config)

    assert result.coefficients == pytest.approx({"tv_spend": 3}, abs=1e-8)
    assert "const" not in result.coefficients
    assert any("omits an intercept" in warning for warning in result.warnings)


def test_ols_rejects_missing_selected_column_and_unimplemented_adstock():
    frame = pd.DataFrame({"revenue": [2, 4, 8], "tv_spend": [1, 2, 3]})
    config = ModelConfig(target="revenue", channels=("tv_spend",), controls=("missing",))
    with pytest.raises(ValueError, match="Selected model columns are missing"):
        OLSModelEngine().fit(frame, config)

    config = ModelConfig(target="revenue", channels=("tv_spend",), controls=(), adstock_decay=0.5)
    with pytest.raises(ValueError, match="Adstock is not supported"):
        OLSModelEngine().fit(frame, config)
