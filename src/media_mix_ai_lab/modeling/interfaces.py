"""Contract for future OLS/log-log engines; no estimators are implemented yet."""

from typing import Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from media_mix_ai_lab.core.config import ModelConfig


class ModelResult(BaseModel):
    """Numerical output plus provenance, separate from AI interpretation.

    Implementations must persist the exact input snapshot or its content hash,
    configuration, and runtime versions alongside results.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    config: ModelConfig
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    coefficients: dict[str, FiniteFloat]
    warnings: tuple[str, ...] = ()


class ModelEngine(Protocol):
    def fit(self, data: pd.DataFrame, config: ModelConfig) -> ModelResult:
        """Fit without modifying input data; return computed results and caveats."""
        ...
