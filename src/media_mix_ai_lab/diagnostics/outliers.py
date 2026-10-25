"""Exploratory, capped, iterative influence-based outlier sensitivity analysis."""

from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from statsmodels.stats.outliers_influence import OLSInfluence

from media_mix_ai_lab.core.config import ModelConfig
from media_mix_ai_lab.modeling.interfaces import ModelResult
from media_mix_ai_lab.modeling.ols import OLSModelEngine

MAX_REMOVAL_FRACTION_PERCENT = 5
STUDENTIZED_RESIDUAL_THRESHOLD = 3.0
COOKS_DISTANCE_RULE = "4 / n_current"

StopReason = Literal[
    "no_flagged_rows",
    "removal_cap_reached",
    "zero_permitted_removals",
    "model_unidentifiable",
]

STOP_REASON_TEXT: dict[str, str] = {
    "no_flagged_rows": "No remaining row met the prespecified rule.",
    "removal_cap_reached": "The cumulative removal cap (5% of original rows) was reached.",
    "zero_permitted_removals": "Too few rows to permit any removal (floor(0.05 * N) = 0).",
    "model_unidentifiable": "The next refit was not identifiable after a further removal.",
}

DISCLAIMER = (
    "Exploratory sensitivity analysis, not automatic data cleaning. Rows are flagged by a "
    "prespecified diagnostic rule only, never to obtain expected signs or p < 0.05. "
    "Conventional p-values and confidence intervals after data-dependent exclusion are not "
    "selection-adjusted and must be treated as exploratory."
)


class OutlierRule(BaseModel):
    """Prespecified deterministic rule; recomputed on the current sample each iteration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    studentized_residual_threshold: FiniteFloat = Field(
        default=STUDENTIZED_RESIDUAL_THRESHOLD, gt=0
    )
    cooks_distance_rule: str = COOKS_DISTANCE_RULE
    tie_break: str = "largest Cook's distance, then smallest original row position"
    max_removal_percent: int = MAX_REMOVAL_FRACTION_PERCENT


class RemovedRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    row_position: int = Field(ge=0)
    date: str | None
    studentized_residual: FiniteFloat
    cooks_distance: FiniteFloat
    cooks_threshold: FiniteFloat
    reason: str


class TermEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    coefficient: FiniteFloat
    sign: Literal["positive", "negative", "zero"]
    p_value: FiniteFloat | None
    ci_lower: FiniteFloat
    ci_upper: FiniteFloat
    coefficient_change: FiniteFloat
    sign_changed: bool


class SensitivityStep(BaseModel):
    """One fit: iteration 0 is the baseline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    iteration: int = Field(ge=0)
    n_observations: int = Field(ge=1)
    removed: RemovedRow | None
    r_squared: FiniteFloat | None
    r_squared_change: FiniteFloat | None
    terms: dict[str, TermEstimate]


class OutlierSensitivityResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config: ModelConfig
    rule: OutlierRule
    dataset_sha256: str
    n_original: int = Field(ge=1)
    max_removals: int = Field(ge=0)
    steps: tuple[SensitivityStep, ...]
    removed_rows: tuple[RemovedRow, ...]
    stop_reason: StopReason
    stop_message: str
    disclaimer: str = DISCLAIMER


def max_permitted_removals(n_rows: int) -> int:
    """floor(5% of the original row count), computed in exact integer arithmetic."""
    return n_rows * MAX_REMOVAL_FRACTION_PERCENT // 100


def _design(data: pd.DataFrame, config: ModelConfig) -> tuple[pd.Series, pd.DataFrame]:
    from media_mix_ai_lab.modeling.adstock import transform_media_frame

    model_data = transform_media_frame(data, config)
    logged = config.kind == "log_log"
    y = model_data[config.target].astype(float)
    design = pd.DataFrame(index=data.index)
    for column in config.channels:
        values = model_data[column].astype(float)
        design[column] = np.log(values) if logged else values
    for column in config.controls:
        design[column] = model_data[column].astype(float)
    if logged:
        y = np.log(y)
    if config.intercept:
        design = sm.add_constant(design, has_constant="add")
    return y, design


def flag_influential_row(
    data: pd.DataFrame, config: ModelConfig, rule: OutlierRule
) -> RemovedRow | None:
    """Return the single row selected by the rule, or None; positions are relative to `data`."""
    y, design = _design(data, config)
    influence = OLSInfluence(sm.OLS(y, design, hasconst=config.intercept).fit())
    student = np.asarray(influence.resid_studentized_external, dtype=float)
    cooks = np.asarray(influence.cooks_distance[0], dtype=float)
    threshold = 4.0 / len(data)
    with np.errstate(invalid="ignore"):
        flagged = (
            np.isfinite(student)
            & np.isfinite(cooks)
            & (np.abs(student) > rule.studentized_residual_threshold)
            & (cooks > threshold)
        )
    candidates = np.flatnonzero(flagged)
    if candidates.size == 0:
        return None
    # Stable sort on descending Cook's distance keeps the smallest position on ties.
    best = int(candidates[np.argsort(-cooks[candidates], kind="stable")[0]])
    date = None
    if config.date_column in data.columns and not pd.isna(data[config.date_column].iloc[best]):
        value = data[config.date_column].iloc[best]
        date = value.isoformat() if hasattr(value, "isoformat") else str(value)
    return RemovedRow(
        row_position=best,
        date=date,
        studentized_residual=float(student[best]),
        cooks_distance=float(cooks[best]),
        cooks_threshold=threshold,
        reason=(
            f"|externally studentized residual| {abs(student[best]):.3f} > "
            f"{rule.studentized_residual_threshold:g} and Cook's distance "
            f"{cooks[best]:.4g} > 4/n = {threshold:.4g}"
        ),
    )


def _sign(value: float) -> Literal["positive", "negative", "zero"]:
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def _step(
    iteration: int,
    result: ModelResult,
    baseline: ModelResult,
    removed: RemovedRow | None,
) -> SensitivityStep:
    terms = {}
    for name, coefficient in result.coefficients.items():
        low, high = result.confidence_intervals[name]
        base = baseline.coefficients[name]
        terms[name] = TermEstimate(
            coefficient=coefficient,
            sign=_sign(coefficient),
            p_value=result.p_values[name],
            ci_lower=low,
            ci_upper=high,
            coefficient_change=coefficient - base,
            sign_changed=_sign(coefficient) != _sign(base),
        )
    r2_change = (
        None
        if result.r_squared is None or baseline.r_squared is None
        else result.r_squared - baseline.r_squared
    )
    return SensitivityStep(
        iteration=iteration,
        n_observations=result.sample_size,
        removed=removed,
        r_squared=result.r_squared,
        r_squared_change=r2_change,
        terms=terms,
    )


def run_outlier_sensitivity(
    data: pd.DataFrame, baseline: ModelResult, rule: OutlierRule | None = None
) -> OutlierSensitivityResult:
    """Iteratively refit after removing at most one flagged row per step, capped at 5% of N.

    The input frame is never modified; removed rows are tracked by original position.
    """
    rule = rule or OutlierRule()
    config = baseline.config
    if OLSModelEngine._fingerprint(data) != baseline.dataset_sha256:
        raise ValueError(
            "The loaded dataset no longer matches the baseline model; refit the baseline first."
        )
    if any(spec.method != "none" for spec in config.adstock.values()):
        raise ValueError(
            "Outlier sensitivity analysis does not support adstock: removing a week would "
            "change the lag structure of the remaining weeks."
        )
    n_original = len(data)
    cap = max_permitted_removals(n_original)
    engine = OLSModelEngine()
    steps = [_step(0, baseline, baseline, None)]
    removed_rows: list[RemovedRow] = []
    positions = list(range(n_original))
    stop_reason: StopReason = "zero_permitted_removals" if cap == 0 else "no_flagged_rows"

    while cap > 0:
        if len(removed_rows) >= cap:
            stop_reason = "removal_cap_reached"
            break
        current = data.iloc[positions]
        try:
            flagged = flag_influential_row(current, config, rule)
        except (ValueError, np.linalg.LinAlgError):
            stop_reason = "model_unidentifiable"
            break
        if flagged is None:
            stop_reason = "no_flagged_rows"
            break
        original_position = positions[flagged.row_position]
        flagged = flagged.model_copy(update={"row_position": original_position})
        remaining = [p for p in positions if p != original_position]
        try:
            refit = engine.fit(data.iloc[remaining], config)
        except ValueError:
            stop_reason = "model_unidentifiable"
            break
        positions = remaining
        removed_rows.append(flagged)
        steps.append(_step(len(removed_rows), refit, baseline, flagged))

    return OutlierSensitivityResult(
        config=config,
        rule=rule,
        dataset_sha256=baseline.dataset_sha256,
        n_original=n_original,
        max_removals=cap,
        steps=tuple(steps),
        removed_rows=tuple(removed_rows),
        stop_reason=stop_reason,
        stop_message=STOP_REASON_TEXT[stop_reason],
    )
