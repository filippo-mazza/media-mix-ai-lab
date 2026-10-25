"""Serializable, validated specifications for reproducible models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class DiagnosticThresholds(BaseModel):
    """Explicit thresholds used by the basic deterministic result assessment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    near_zero_residual_df: int = Field(default=5, ge=1)
    large_relative_standard_error: float = Field(default=1.0, gt=0)
    poor_condition_number: float = Field(default=1e8, gt=1)
    near_zero_estimate: float = Field(default=1e-12, gt=0)


class AdstockSpec(BaseModel):
    """Per-channel adstock settings with explicit scaling and boundary conditions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["none", "geometric", "delayed_geometric"] = "none"
    alpha: FiniteFloat = Field(default=0.5, ge=0, lt=1)
    max_lag_weeks: int = Field(default=8, ge=0, le=12)
    theta: FiniteFloat = Field(default=2.0, ge=0)
    weight_scaling: Literal["raw", "sum_to_one"] = "raw"
    initial_state: FiniteFloat = Field(default=0.0, ge=0)
    historical_values: tuple[FiniteFloat, ...] = ()

    @model_validator(mode="after")
    def validate_adstock_parameters(self) -> "AdstockSpec":
        if any(value < 0 for value in self.historical_values):
            raise ValueError("Historical media values cannot be negative.")
        if self.method == "delayed_geometric":
            if self.theta > self.max_lag_weeks:
                raise ValueError("theta must be between 0 and max_lag_weeks.")
            if self.alpha == 0:
                raise ValueError("Delayed geometric alpha must be greater than 0.")
            if len(self.historical_values) > self.max_lag_weeks:
                raise ValueError("historical_values cannot exceed max_lag_weeks.")
        if self.method != "delayed_geometric" and self.historical_values:
            raise ValueError("historical_values are only supported for delayed geometric adstock.")
        if self.method != "geometric" and self.initial_state:
            raise ValueError("initial_state is only supported for geometric adstock.")
        return self


class ModelConfig(BaseModel):
    """OLS specification, with explicit media transforms and no implicit cleaning.

    Log-log specifications transform the outcome and media channels; selected
    controls remain in levels. Adstock lag units are weeks.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    kind: Literal["ols", "log_log"] = "ols"
    target: str = Field(default="revenue", min_length=1)
    channels: tuple[str, ...] = ("tv_spend", "search_spend", "social_spend")
    controls: tuple[str, ...] = ("price_index", "promotions", "seasonality")
    date_column: str = Field(default="date", min_length=1)
    intercept: bool = True
    media_sign_expectations: dict[str, Literal["positive", "negative", "either"]] = Field(
        default_factory=dict
    )
    diagnostic_thresholds: DiagnosticThresholds = Field(default_factory=DiagnosticThresholds)
    adstock: dict[str, AdstockSpec] = Field(default_factory=dict)
    seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def validate_columns(self) -> "ModelConfig":
        columns = (self.target, *self.channels, *self.controls)
        if not self.channels or any(not name.strip() for name in columns):
            raise ValueError("Specify at least one channel and nonblank column names.")
        if len(set(columns)) != len(columns):
            raise ValueError("Target, channels, and controls must be distinct.")
        unknown_expectations = set(self.media_sign_expectations) - set(self.channels)
        if unknown_expectations:
            raise ValueError(
                "Media sign expectations reference unselected channels: "
                f"{', '.join(sorted(unknown_expectations))}."
            )
        unknown_adstock = set(self.adstock) - set(self.channels)
        if unknown_adstock:
            raise ValueError(
                "Adstock settings reference unselected channels: "
                f"{', '.join(sorted(unknown_adstock))}."
            )
        return self
