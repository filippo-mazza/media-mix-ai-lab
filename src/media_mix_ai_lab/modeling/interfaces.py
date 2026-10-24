"""Contract for future OLS/log-log engines; no estimators are implemented yet."""

from typing import Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from media_mix_ai_lab.core.config import ModelConfig


class ModelResult(BaseModel):
    """OLS estimates and provenance, separate from AI interpretation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    config: ModelConfig
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    coefficients: dict[str, FiniteFloat]
    standard_errors: dict[str, FiniteFloat]
    confidence_intervals: dict[str, tuple[FiniteFloat, FiniteFloat]]
    term_roles: dict[str, str]
    sample_size: int = Field(ge=1)
    r_squared: FiniteFloat
    adjusted_r_squared: FiniteFloat
    residual_variance: FiniteFloat
    equation: str
    transformations: dict[str, str]
    package_versions: dict[str, str]
    warnings: tuple[str, ...] = ()


class ModelEngine(Protocol):
    def fit(self, data: pd.DataFrame, config: ModelConfig) -> ModelResult:
        """Fit without modifying input data; return computed results and caveats."""
        ...
