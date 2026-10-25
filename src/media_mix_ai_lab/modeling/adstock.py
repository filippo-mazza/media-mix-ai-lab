"""Deterministic media adstock transforms with explicit boundary conventions."""

import numpy as np
import pandas as pd

from media_mix_ai_lab.core.config import AdstockSpec, ModelConfig


def validate_weekly_dates(data: pd.DataFrame, date_column: str) -> None:
    """Require ordered, unique, regular weekly dates for time-dependent transforms."""
    if date_column not in data.columns:
        raise ValueError(f"Adstock requires a date column: {date_column}.")
    dates = data[date_column]
    if not pd.api.types.is_datetime64_any_dtype(dates):
        raise ValueError(f"Adstock date column {date_column} must contain parsed datetime values.")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Adstock dates must be nonmissing, unique, and in ascending order.")
    if (
        len(dates) > 1
        and not (
            dates.diff().iloc[1:].to_numpy(dtype="timedelta64[ns]") == np.timedelta64(7, "D")
        ).all()
    ):
        raise ValueError("Adstock dates must be weekly; missing weeks are rejected, never filled.")


def apply_adstock(values: pd.Series, spec: AdstockSpec) -> pd.Series:
    """Transform one channel without mutating it or silently cleaning invalid spend."""
    if (
        not pd.api.types.is_numeric_dtype(values)
        or pd.api.types.is_bool_dtype(values)
        or not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all()
    ):
        raise ValueError(f"{values.name} must contain numeric, finite, nonmissing media values.")
    signal = values.to_numpy(dtype=float)
    if spec.method != "none" and (signal < 0).any():
        raise ValueError(f"{values.name} cannot contain negative media spend.")

    if spec.method == "none":
        transformed = signal.copy()
    elif spec.method == "geometric":
        transformed = np.empty_like(signal)
        previous = spec.initial_state
        for index, observation in enumerate(signal):
            previous = observation + spec.alpha * previous
            transformed[index] = previous
        if spec.weight_scaling == "sum_to_one":
            transformed *= 1 - spec.alpha
    else:
        weights = _delayed_weights(spec)
        history = spec.historical_values
        transformed = np.zeros_like(signal)
        for index in range(len(signal)):
            for lag, weight in enumerate(weights):
                source_index = index - lag
                if source_index >= 0:
                    transformed[index] += weight * signal[source_index]
                else:
                    history_index = len(history) + source_index
                    if history_index >= 0:
                        transformed[index] += weight * history[history_index]

    if not np.isfinite(transformed).all():
        raise ValueError(f"{values.name} adstock produced nonfinite values; review input scale.")
    return pd.Series(transformed, index=values.index, name=values.name)


def adstock_impulse_response(spec: AdstockSpec, periods: int = 13) -> pd.DataFrame:
    """Return lag weights for charting; recursive geometric weights continue indefinitely."""
    if periods < 1:
        raise ValueError("periods must be at least one.")
    if spec.method == "none":
        weights = np.array([1.0])
    elif spec.method == "geometric":
        weights = np.power(spec.alpha, np.arange(periods, dtype=float))
        if spec.weight_scaling == "sum_to_one":
            weights *= 1 - spec.alpha
    else:
        weights = _delayed_weights(spec)
    return pd.DataFrame({"Lag (weeks)": np.arange(len(weights)), "Weight": weights})


def transform_media_frame(data: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Return a copy with configured channel transforms, requiring regular weekly dates."""
    transformed = data.copy(deep=True)
    methods = [(channel, config.adstock.get(channel, AdstockSpec())) for channel in config.channels]
    missing = [channel for channel, _ in methods if channel not in data.columns]
    if missing:
        raise ValueError(f"Selected media columns are missing: {', '.join(missing)}")
    if any(spec.method != "none" for _, spec in methods):
        validate_weekly_dates(data, config.date_column)
    for channel, spec in methods:
        transformed[channel] = apply_adstock(data[channel], spec)
    return transformed


def _delayed_weights(spec: AdstockSpec) -> np.ndarray:
    log_weights = np.log(spec.alpha) * np.square(
        np.arange(spec.max_lag_weeks + 1, dtype=float) - spec.theta
    )
    if spec.weight_scaling == "sum_to_one":
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
    else:
        weights = np.exp(log_weights)
        if not np.any(weights):
            raise ValueError(
                "Raw delayed-geometric weights underflow to zero; choose a larger alpha."
            )
    return weights
