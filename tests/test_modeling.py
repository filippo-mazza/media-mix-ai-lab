from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from media_mix_ai_lab.core.config import ModelConfig
from media_mix_ai_lab.modeling.interfaces import ModelResult
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
    assert "16.18% change" in result.term_assessments["promotion"].interpretation
    assert result.term_assessments["promotion"].scale.startswith("model-implied %")
    negative_flag = next(
        flag
        for flag in result.diagnostic_flags
        if flag.code == "MEDIA_UNEXPECTED_SIGN" and flag.evidence["term"] == "log(search_spend)"
    )
    assert negative_flag.category == "business_plausibility"
    assert negative_flag.evidence["estimate"] == pytest.approx(-0.25)
    assert "unexpected sign—investigate" in negative_flag.message


def test_negative_media_sign_can_be_accepted_by_domain_expectation():
    x = np.arange(1.0, 21.0)
    revenue = 100 - 2 * x
    frame = pd.DataFrame({"revenue": revenue, "tv_spend": x})
    config = ModelConfig(
        target="revenue",
        channels=("tv_spend",),
        controls=(),
        media_sign_expectations={"tv_spend": "either"},
    )

    result = OLSModelEngine().fit(frame, config)

    assert result.coefficients["tv_spend"] == pytest.approx(-2)
    assert not any(flag.code == "MEDIA_UNEXPECTED_SIGN" for flag in result.diagnostic_flags)
    assert result.term_assessments["tv_spend"].expected_direction == "either"


def test_weak_media_effect_flags_ci_crossing_zero_with_evidence():
    rng = np.random.default_rng(7)
    spend = np.linspace(1, 20, 80)
    revenue = 100 + rng.normal(0, 10, len(spend))
    frame = pd.DataFrame({"revenue": revenue, "tv_spend": spend})
    config = ModelConfig(
        target="revenue",
        channels=("tv_spend",),
        controls=(),
        diagnostic_thresholds={"large_relative_standard_error": 0.5},
    )

    result = OLSModelEngine().fit(frame, config)

    assessment = result.term_assessments["tv_spend"]
    assert assessment.ci_spans_zero
    flag = next(flag for flag in result.diagnostic_flags if flag.code == "CI_SPANS_ZERO")
    assert flag.evidence["term"] == "tv_spend"
    assert flag.evidence["ci_lower"] < 0 < flag.evidence["ci_upper"]
    assert "LARGE_RELATIVE_UNCERTAINTY" in result.health_summary.warning_codes


def test_near_collinear_predictors_flag_poor_conditioning():
    x = np.linspace(1, 10, 80)
    almost_x = x + np.where(np.arange(len(x)) % 2, 1e-7, -1e-7)
    revenue = 5 + 2 * x + 0.5 * np.sin(x)
    frame = pd.DataFrame({"revenue": revenue, "tv_spend": x, "search_spend": almost_x})
    config = ModelConfig(target="revenue", channels=("tv_spend", "search_spend"), controls=())

    result = OLSModelEngine().fit(frame, config)

    flag = next(flag for flag in result.diagnostic_flags if flag.code == "POOR_CONDITIONING")
    assert flag.evidence["condition_number"] >= flag.evidence["threshold"]
    assert result.health_summary.status == "unstable"
    assert result.design_rank == result.parameter_count


def test_low_residual_degrees_of_freedom_is_reported_with_threshold():
    frame = pd.DataFrame(
        {
            "revenue": [4.0, 7.0, 11.0, 18.0],
            "tv_spend": [1.0, 2.0, 4.0, 8.0],
            "promotion": [0.0, 1.0, 0.0, 1.0],
        }
    )
    config = ModelConfig(target="revenue", channels=("tv_spend",), controls=("promotion",))

    result = OLSModelEngine().fit(frame, config)

    flag = next(flag for flag in result.diagnostic_flags if flag.code == "NEAR_ZERO_RESIDUAL_DF")
    assert flag.evidence["residual_degrees_freedom"] == 1
    assert flag.evidence["threshold"] == config.diagnostic_thresholds.near_zero_residual_df


def test_well_conditioned_clean_fixture_has_healthy_summary():
    rng = np.random.default_rng(19)
    tv = rng.uniform(1, 10, 120)
    search = rng.uniform(1, 10, 120)
    revenue = 20 + 2 * tv + 3 * search + rng.normal(0, 0.2, 120)
    frame = pd.DataFrame({"revenue": revenue, "tv_spend": tv, "search_spend": search})
    config = ModelConfig(target="revenue", channels=("tv_spend", "search_spend"), controls=())

    result = OLSModelEngine().fit(frame, config)

    assert result.health_summary.status == "healthy"
    assert result.health_summary.flag_count == 0
    assert result.diagnostic_flags == ()
    assert all(not assessment.ci_spans_zero for assessment in result.term_assessments.values())


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
    flag = next(
        flag for flag in result.diagnostic_flags if flag.code == "NO_INTERCEPT_UNCENTERED_R2"
    )
    assert flag.evidence == {"intercept": False, "r_squared_convention": "uncentered"}
    assert result.term_assessments["tv_spend"].role == "media"


def test_ols_rejects_missing_selected_column_and_unimplemented_adstock():
    frame = pd.DataFrame({"revenue": [2, 4, 8], "tv_spend": [1, 2, 3]})
    config = ModelConfig(target="revenue", channels=("tv_spend",), controls=("missing",))
    with pytest.raises(ValueError, match="Selected model columns are missing"):
        OLSModelEngine().fit(frame, config)

    config = ModelConfig(target="revenue", channels=("tv_spend",), controls=(), adstock_decay=0.5)
    with pytest.raises(ValueError, match="Adstock is not supported"):
        OLSModelEngine().fit(frame, config)


def test_documented_negative_media_export_matches_result_schema():
    example = (
        Path(__file__).resolve().parents[1] / "docs" / "examples" / "negative-media-result.json"
    )

    result = ModelResult.model_validate_json(example.read_text())

    assert result.health_summary.warning_codes == ("MEDIA_UNEXPECTED_SIGN",)
    assert result.diagnostic_flags[0].evidence["term"] == "tv_spend"
