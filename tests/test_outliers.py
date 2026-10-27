import numpy as np
import pandas as pd
import pytest

from media_mix_ai_lab.core.config import AdstockSpec, ModelConfig
from media_mix_ai_lab.diagnostics.outliers import (
    OutlierRule,
    flag_influential_row,
    max_permitted_removals,
    run_outlier_sensitivity,
)
from media_mix_ai_lab.modeling.ols import OLSModelEngine


def make_frame(n: int = 60, outliers: tuple[int, ...] = ()) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    tv = rng.uniform(1, 10, n)
    search = rng.uniform(1, 10, n)
    revenue = 50 + 3 * tv + 2 * search + rng.normal(0, 1, n)
    for position in outliers:
        revenue[position] += 80
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="7D"),
            "revenue": revenue,
            "tv_spend": tv,
            "search_spend": search,
        }
    )


def config(**kwargs) -> ModelConfig:
    return ModelConfig(
        target="revenue", channels=("tv_spend", "search_spend"), controls=(), **kwargs
    )


def test_cap_arithmetic():
    assert [max_permitted_removals(n) for n in (19, 20, 60, 104)] == [0, 1, 3, 5]


def test_flags_injected_outliers_deterministically_and_leaves_data_unchanged():
    frame = make_frame(60, outliers=(10, 30))
    original = frame.copy(deep=True)
    baseline = OLSModelEngine().fit(frame, config())
    first = run_outlier_sensitivity(frame, baseline)
    second = run_outlier_sensitivity(frame, baseline)
    assert first == second
    pd.testing.assert_frame_equal(frame, original)
    assert [row.row_position for row in first.removed_rows][:2] == [10, 30] or {
        10,
        30,
    } <= {row.row_position for row in first.removed_rows}
    assert (
        first.removed_rows[0].date
        == frame["date"].iloc[first.removed_rows[0].row_position].isoformat()
    )
    assert first.steps[0].iteration == 0 and first.steps[0].n_observations == 60
    assert all(step.n_observations == 60 - step.iteration for step in first.steps)
    assert OLSModelEngine._fingerprint(frame) == baseline.dataset_sha256


def test_cap_is_enforced_with_many_outliers():
    frame = make_frame(60, outliers=tuple(range(0, 20)))
    baseline = OLSModelEngine().fit(frame, config())
    result = run_outlier_sensitivity(frame, baseline)
    assert result.max_removals == 3
    assert len(result.removed_rows) <= 3
    assert result.steps[-1].n_observations >= 57
    if len(result.removed_rows) == 3:
        assert result.stop_reason == "removal_cap_reached"


def test_no_flagged_rows_stops_early():
    frame = make_frame(60)
    frame["revenue"] = 50 + 3 * frame["tv_spend"] + 2 * frame["search_spend"]
    frame.loc[::2, "revenue"] += 0.001
    baseline = OLSModelEngine().fit(frame, config())
    result = run_outlier_sensitivity(frame, baseline)
    assert result.stop_reason in {"no_flagged_rows", "removal_cap_reached"}
    if result.stop_reason == "no_flagged_rows":
        assert not result.removed_rows


def test_tiny_dataset_has_zero_permitted_removals():
    frame = make_frame(12, outliers=(3,))
    baseline = OLSModelEngine().fit(frame, config())
    result = run_outlier_sensitivity(frame, baseline)
    assert result.max_removals == 0
    assert result.stop_reason == "zero_permitted_removals"
    assert not result.removed_rows and len(result.steps) == 1


def test_rejects_changed_data_and_adstock():
    frame = make_frame(40)
    baseline = OLSModelEngine().fit(frame, config())
    changed = frame.copy()
    changed.loc[0, "revenue"] += 1
    with pytest.raises(ValueError, match="no longer matches"):
        run_outlier_sensitivity(changed, baseline)
    adstocked = config(adstock={"tv_spend": AdstockSpec(method="geometric", alpha=0.3)})
    with pytest.raises(ValueError, match="adstock"):
        run_outlier_sensitivity(frame, OLSModelEngine().fit(frame, adstocked))


def test_log_log_nonpositive_baseline_is_not_silently_dropped():
    frame = make_frame(40)
    frame.loc[3, "tv_spend"] = 0.0
    with pytest.raises(ValueError, match="No rows were dropped"):
        OLSModelEngine().fit(frame, config(kind="log_log"))


def test_infinite_studentized_residual_is_flagged_and_audited_explicitly():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=20, freq="7D"),
            "revenue": [100.0] + [0.0] * 19,
            "tv_spend": [1.0] * 20,
        }
    )
    no_intercept = ModelConfig(
        target="revenue", channels=("tv_spend",), controls=(), intercept=False
    )

    flagged = flag_influential_row(frame, no_intercept, OutlierRule())

    assert flagged is not None
    assert flagged.row_position == 0
    assert flagged.studentized_residual in {"Infinity", "-Infinity"}
    assert "Infinity" in flagged.reason
