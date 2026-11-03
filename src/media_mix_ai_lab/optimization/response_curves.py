"""Deterministic conditional spend-response scenarios from fitted log-log models."""

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from media_mix_ai_lab.core.config import AdstockSpec
from media_mix_ai_lab.modeling.adstock import adstock_impulse_response
from media_mix_ai_lab.modeling.interfaces import ModelResult
from media_mix_ai_lab.modeling.ols import OLSModelEngine


class ResponsePoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spend_per_week: FiniteFloat = Field(ge=0)
    conditional_outcome_per_week: FiniteFloat = Field(ge=0)
    outcome_change_from_reference: FiniteFloat
    is_extrapolation: bool


class ResponseCurve(BaseModel):
    """A reproducible scenario curve; it is not a causal attribution result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    channel: str
    target: str
    model_kind: str
    observed_spend_min: FiniteFloat = Field(ge=0)
    observed_spend_max: FiniteFloat = Field(ge=0)
    reference_spend: FiniteFloat = Field(gt=0)
    reference_outcome: FiniteFloat = Field(ge=0)
    adstock_method: str
    adstock_steady_state_multiplier: FiniteFloat = Field(gt=0)
    fixed_inputs: dict[str, FiniteFloat]
    points: tuple[ResponsePoint, ...]
    assumption: str
    caveat: str


def steady_state_multiplier(spec: AdstockSpec) -> float:
    """Map constant weekly spend to its long-run transformed media level."""
    if spec.method == "none" or spec.weight_scaling == "sum_to_one":
        return 1.0
    if spec.method == "geometric":
        return 1.0 / (1.0 - spec.alpha)
    weights = adstock_impulse_response(spec, periods=spec.max_lag_weeks + 1)["Weight"]
    return float(weights.sum())


def build_response_curve(
    data: pd.DataFrame,
    result: ModelResult,
    channel: str,
    *,
    minimum_spend: float,
    maximum_spend: float,
    reference_spend: float,
    points: int = 101,
    require_nonnegative_return: bool = True,
) -> ResponseCurve:
    """Evaluate a fitted power law while holding other model inputs fixed at medians."""
    config = result.config
    if result.dataset_sha256 != OLSModelEngine._fingerprint(data):
        raise ValueError("The fitted model belongs to a different dataset; refit before plotting.")
    if config.kind != "log_log":
        raise ValueError("Response curves require a fitted log-log specification.")
    if channel not in config.channels:
        raise ValueError(f"{channel} is not a media channel in the fitted model.")
    if points < 2 or points > 1001:
        raise ValueError("Curve resolution must be between 2 and 1001 points.")
    if not np.isfinite([minimum_spend, maximum_spend, reference_spend]).all():
        raise ValueError("Spend bounds and reference spend must be finite.")
    if minimum_spend <= 0 or maximum_spend <= minimum_spend:
        raise ValueError("Spend bounds must be positive and maximum must exceed minimum.")
    if not minimum_spend <= reference_spend <= maximum_spend:
        raise ValueError("Reference spend must lie within the configured curve domain.")

    term = f"log({channel})"
    beta = result.coefficients[term]
    if require_nonnegative_return and beta < 0:
        raise ValueError(
            "The fitted channel coefficient is negative; a nonnegative-return response curve "
            "cannot be shown without contradicting the fitted model."
        )
    observed = data[channel].astype(float)
    if (observed <= 0).any():
        raise ValueError("Log-log response curves require strictly positive observed spend.")

    fixed_inputs: dict[str, float] = {}
    linear_fixed = result.coefficients.get("const", 0.0)
    for other in config.channels:
        if other == channel:
            continue
        raw_value = float(data[other].median())
        multiplier = steady_state_multiplier(config.adstock.get(other, AdstockSpec()))
        fixed_inputs[other] = raw_value
        linear_fixed += result.coefficients[f"log({other})"] * np.log(raw_value * multiplier)
    for control in config.controls:
        value = float(data[control].median())
        fixed_inputs[control] = value
        linear_fixed += result.coefficients[control] * value

    multiplier = steady_state_multiplier(config.adstock.get(channel, AdstockSpec()))

    def predict(spend: np.ndarray | float) -> np.ndarray:
        values = np.asarray(spend, dtype=float)
        return np.exp(linear_fixed + beta * np.log(values * multiplier))

    reference_outcome = float(predict(reference_spend))
    grid = np.linspace(minimum_spend, maximum_spend, points)
    predictions = predict(grid)
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("The configured scenario produced invalid or negative predictions.")
    observed_min, observed_max = float(observed.min()), float(observed.max())
    curve_points = tuple(
        ResponsePoint(
            spend_per_week=float(spend),
            conditional_outcome_per_week=float(prediction),
            outcome_change_from_reference=float(prediction - reference_outcome),
            is_extrapolation=bool(spend < observed_min or spend > observed_max),
        )
        for spend, prediction in zip(grid, predictions, strict=True)
    )
    spec = config.adstock.get(channel, AdstockSpec())
    adstock_text = (
        "x is constant weekly spend evaluated at the long-run steady state"
        if spec.method != "none"
        else "x is weekly spend with no carryover transform"
    )
    return ResponseCurve(
        channel=channel,
        target=config.target,
        model_kind=config.kind,
        observed_spend_min=observed_min,
        observed_spend_max=observed_max,
        reference_spend=reference_spend,
        reference_outcome=reference_outcome,
        adstock_method=spec.method,
        adstock_steady_state_multiplier=multiplier,
        fixed_inputs=fixed_inputs,
        points=curve_points,
        assumption=(
            f"Power-law scenario from the fitted log-log model; {adstock_text}. Other media "
            "and controls are fixed at their observed medians."
        ),
        caveat=(
            "Conditional model-based scenario, not a proven incremental causal effect. "
            "Extrapolation outside observed spend is especially uncertain. No uncertainty "
            "band is shown because post-fit curve uncertainty has not been implemented."
        ),
    )
