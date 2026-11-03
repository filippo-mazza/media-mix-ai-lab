import json
from io import StringIO

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from media_mix_ai_lab.core.config import AdstockSpec, ModelConfig
from media_mix_ai_lab.diagnostics.influence import (
    InfluenceError,
    InfluenceThresholds,
    compute_influence,
    export_csv,
    export_json,
    observation_detail,
    ranked_observations,
)
from media_mix_ai_lab.modeling.ols import OLSModelEngine

PLANTED = 17


def make_frame(n: int = 60, planted: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    tv = rng.uniform(1, 10, n)
    search = rng.uniform(1, 10, n)
    price = rng.normal(100, 5, n)
    revenue = 50 + 3 * tv + 2 * search - 0.5 * price + rng.normal(0, 1, n)
    if planted:
        tv[PLANTED] = 40.0
        revenue[PLANTED] -= 60.0
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="7D"),
            "revenue": revenue,
            "tv_spend": tv,
            "search_spend": search,
            "price_index": price,
        }
    )


def config(**kwargs) -> ModelConfig:
    return ModelConfig(
        target="revenue",
        channels=("tv_spend", "search_spend"),
        controls=("price_index",),
        **kwargs,
    )


def fit_and_influence(frame: pd.DataFrame, cfg: ModelConfig):
    model = OLSModelEngine().fit(frame, cfg)
    return model, compute_influence(frame, model)


def test_matches_statsmodels_reference_and_explicit_leave_one_out():
    frame = make_frame()
    model, result = fit_and_influence(frame, config())
    ols = sm.OLS(
        frame["revenue"], sm.add_constant(frame[["tv_spend", "search_spend", "price_index"]])
    ).fit()
    reference = ols.get_influence()

    assert result.terms == ("const", "tv_spend", "search_spend", "price_index")
    assert result.dfbetas.shape == (60, 2 + 4)  # row_position + date + 4 terms
    np.testing.assert_allclose(result.dfbetas[list(result.terms)], reference.dfbetas)
    np.testing.assert_allclose(result.diagnostics["cooks_distance"], reference.cooks_distance[0])
    np.testing.assert_allclose(result.diagnostics["leverage"], reference.hat_matrix_diag)
    np.testing.assert_allclose(
        result.diagnostics["studentized_residual"], reference.resid_studentized_external
    )
    np.testing.assert_allclose(result.diagnostics["dffits"], reference.dffits[0])

    for row in (0, PLANTED, 59):
        kept = frame.drop(index=row)
        refit = sm.OLS(
            kept["revenue"], sm.add_constant(kept[["tv_spend", "search_spend", "price_index"]])
        ).fit()
        np.testing.assert_allclose(
            result.deleted_coefficients.loc[row, list(result.terms)].to_numpy(dtype=float),
            refit.params.to_numpy(),
            rtol=1e-8,
        )
        # Sign convention: DFBETAS > 0 means the full-sample beta exceeds the deleted-fit beta.
        delta = ols.params.to_numpy() - refit.params.to_numpy()
        assert np.array_equal(
            np.sign(result.dfbetas.loc[row, list(result.terms)].to_numpy(dtype=float)),
            np.sign(delta),
        )
    assert result.n_observations == 60 and result.n_excluded == 0
    assert result.residual_degrees_freedom == 56


def test_planted_influential_week_is_ranked_first_with_exact_mapping():
    frame = make_frame()
    _, result = fit_and_influence(frame, config())
    ranked = ranked_observations(result)

    assert ranked.loc[0, "row_position"] == PLANTED
    assert ranked.loc[0, "date"] == frame["date"].iloc[PLANTED].isoformat()
    assert ranked.loc[0, "exceeds_dfbetas_guide"]
    assert ranked.loc[0, "exceeds_cooks_guide"]
    assert ranked["max_abs_dfbetas"].is_monotonic_decreasing
    assert len(ranked) == 60 and set(ranked["row_position"]) == set(range(60))
    # chronological source frames are untouched by ranking
    assert result.dfbetas["row_position"].tolist() == list(range(60))
    assert result.dfbetas["date"].tolist() == [d.isoformat() for d in frame["date"]]
    top_term = ranked.loc[0, "most_influenced_term"]
    assert abs(ranked.loc[0, f"dfbetas:{top_term}"]) == pytest.approx(
        ranked.loc[0, "max_abs_dfbetas"]
    )


def test_thresholds_are_configurable_heuristics():
    _, result = fit_and_influence(make_frame(), config())
    default = InfluenceThresholds()
    assert default.dfbetas(60) == pytest.approx(2 / np.sqrt(60))
    assert default.cooks(60) == pytest.approx(4 / 60)
    strict = ranked_observations(result, InfluenceThresholds(1e6, 1e6))
    assert not strict["exceeds_dfbetas_guide"].any()
    assert not strict["exceeds_cooks_guide"].any()
    loose = ranked_observations(result, InfluenceThresholds(1e-6, 1e-6))
    assert loose["exceeds_dfbetas_guide"].all()
    with pytest.raises(ValueError, match="positive finite"):
        InfluenceThresholds(dfbetas_multiplier=0)


def test_observation_detail_separates_raw_and_transformed_inputs():
    frame = make_frame()
    cfg = config(kind="log_log")
    frame["price_index"] = frame["price_index"]
    frame["revenue"] = frame["revenue"] + 200
    _, result = fit_and_influence(frame, cfg)
    detail = observation_detail(result, PLANTED)

    assert detail["raw_inputs"]["tv_spend"] == frame["tv_spend"].iloc[PLANTED]
    assert detail["transformed_features"]["log(tv_spend)"] == pytest.approx(
        np.log(frame["tv_spend"].iloc[PLANTED])
    )
    assert detail["observed_model_scale"] == pytest.approx(np.log(frame["revenue"].iloc[PLANTED]))
    assert detail["observed_model_scale"] - detail["predicted_model_scale"] == pytest.approx(
        detail["residual"]
    )
    comparison = detail["beta_comparison"]
    np.testing.assert_allclose(
        comparison["change_when_included"],
        comparison["full_sample_beta"] - comparison["leave_one_out_beta"],
    )
    assert any("elasticities" in warning for warning in result.warnings)
    with pytest.raises(ValueError):
        observation_detail(result, 60)


def test_strict_log_log_rejects_nonpositive_values_without_dropping_rows():
    frame = make_frame()
    frame["revenue"] += 200
    model = OLSModelEngine().fit(frame, config(kind="log_log"))
    broken = frame.copy()
    broken.loc[3, "tv_spend"] = 0.0
    with pytest.raises(ValueError):
        compute_influence(broken, model)  # fingerprint mismatch is caught first
    with pytest.raises(ValueError, match="Strict log-log"):
        OLSModelEngine().fit(broken, config(kind="log_log"))


def test_fingerprint_mismatch_does_not_refit_on_another_sample():
    frame = make_frame()
    model = OLSModelEngine().fit(frame, config())
    with pytest.raises(InfluenceError, match="fingerprint"):
        compute_influence(frame.iloc[:-1], model)
    changed = frame.copy()
    changed.loc[0, "revenue"] += 1
    with pytest.raises(InfluenceError, match="fingerprint"):
        compute_influence(changed, model)


def test_adstock_features_use_transformed_design_and_warn():
    frame = make_frame()
    cfg = config(adstock={"tv_spend": AdstockSpec(method="geometric", alpha=0.5)})
    model, result = fit_and_influence(frame, cfg)
    assert result.design["tv_spend"].iloc[0] == pytest.approx(frame["tv_spend"].iloc[0])
    assert result.design["tv_spend"].iloc[1] == pytest.approx(
        frame["tv_spend"].iloc[1] + 0.5 * frame["tv_spend"].iloc[0]
    )
    assert result.raw_inputs["tv_spend"].iloc[1] == frame["tv_spend"].iloc[1]
    assert any("adstock history" in warning for warning in result.warnings)
    assert result.n_observations == model.sample_size


def test_insufficient_residual_degrees_of_freedom_is_rejected():
    frame = make_frame(n=5, planted=False)
    # 4 parameters, 5 rows -> 1 residual df: fits, but leave-one-out is not identifiable.
    model = OLSModelEngine().fit(frame, config())
    with pytest.raises(InfluenceError, match="residual degrees of freedom"):
        compute_influence(frame, model)


def test_unit_leverage_row_reports_undefined_values_without_dropping_rows():
    frame = make_frame(n=20, planted=False)
    frame["promo_once"] = 0.0
    frame.loc[3, "promo_once"] = 1.0
    cfg = ModelConfig(
        target="revenue",
        channels=("tv_spend", "search_spend"),
        controls=("price_index", "promo_once"),
    )
    model = OLSModelEngine().fit(frame, cfg)
    result = compute_influence(frame, model)

    assert result.n_observations == 20
    assert result.diagnostics.loc[3, "leverage"] == pytest.approx(1.0)
    assert any("undefined or infinite" in warning for warning in result.warnings)
    assert any("leverage ≈ 1" in warning for warning in result.warnings)
    ranked = ranked_observations(result)
    assert len(ranked) == 20
    assert np.isnan(result.dfbetas.loc[3, list(result.terms)].to_numpy(dtype=float)).all()
    assert np.isnan(result.diagnostics.loc[3, "cooks_distance"])
    exported = json.loads(export_json(ranked))
    assert len(exported) == 20


def test_rank_deficient_models_are_rejected_before_diagnostics():
    frame = make_frame()
    model = OLSModelEngine().fit(frame, config())
    deficient = frame.copy()
    deficient["price_index"] = 1.0
    # A stored model cannot be fitted on this frame, so the fingerprint guard fires first.
    with pytest.raises(InfluenceError):
        compute_influence(deficient, model)


def test_exports_preserve_precision_and_nulls():
    _, result = fit_and_influence(make_frame(), config())
    table = ranked_observations(result)
    records = json.loads(export_json(table))
    assert records[0]["max_abs_dfbetas"] == table.loc[0, "max_abs_dfbetas"]
    assert records[0]["row_position"] == PLANTED
    assert isinstance(records[0]["exceeds_cooks_guide"], bool)
    csv = pd.read_csv(StringIO(export_csv(table)), float_precision="round_trip")
    np.testing.assert_array_equal(csv["cooks_distance"], table["cooks_distance"])
    nan_table = table.copy()
    nan_table.loc[0, "dffits"] = np.nan
    assert json.loads(export_json(nan_table))[0]["dffits"] is None
