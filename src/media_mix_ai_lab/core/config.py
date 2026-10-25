"""Serializable, validated specifications for reproducible models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DiagnosticThresholds(BaseModel):
    """Explicit thresholds used by the basic deterministic result assessment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    near_zero_residual_df: int = Field(default=5, ge=1)
    large_relative_standard_error: float = Field(default=1.0, gt=0)
    poor_condition_number: float = Field(default=1e8, gt=1)
    near_zero_estimate: float = Field(default=1e-12, gt=0)


class ModelConfig(BaseModel):
    """OLS specification, with no implicit cleaning or predictor transforms.

    Log-log specifications transform the outcome and media channels; selected
    controls remain in levels. Lag units are weeks and are reserved for future use.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    kind: Literal["ols", "log_log"] = "ols"
    target: str = Field(default="revenue", min_length=1)
    channels: tuple[str, ...] = ("tv_spend", "search_spend", "social_spend")
    controls: tuple[str, ...] = ("price_index", "promotions", "seasonality")
    intercept: bool = True
    media_sign_expectations: dict[str, Literal["positive", "negative", "either"]] = Field(
        default_factory=dict
    )
    diagnostic_thresholds: DiagnosticThresholds = Field(default_factory=DiagnosticThresholds)
    adstock_decay: float = Field(default=0.0, ge=0.0, lt=1.0)
    max_lag_weeks: int = Field(default=0, ge=0, le=104)
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
        return self
