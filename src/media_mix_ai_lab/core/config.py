"""Serializable, validated specifications for reproducible future models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    """A specification only; fitting is deliberately not implemented yet.

    Lag units are weeks. Model implementations must explicitly apply controls,
    transforms, and training windows and report their assumptions.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    kind: Literal["ols", "log_log"] = "ols"
    target: str = Field(default="revenue", min_length=1)
    channels: tuple[str, ...] = ("tv_spend", "search_spend", "social_spend")
    controls: tuple[str, ...] = ("price_index", "promotions", "seasonality")
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
        return self
