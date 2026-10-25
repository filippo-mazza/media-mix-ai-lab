"""Typed contracts for fitted models, coefficient assessments, and diagnostics."""

from typing import Literal, Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from media_mix_ai_lab.core.config import ModelConfig


class TermAssessment(BaseModel):
    """A computed estimate and its deterministic, noncausal interpretation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    term: str
    role: Literal["intercept", "media", "control"]
    estimate: FiniteFloat
    standard_error: FiniteFloat = Field(ge=0)
    confidence_interval: tuple[FiniteFloat, FiniteFloat]
    ci_spans_zero: bool
    scale: str
    interpretation: str
    expected_direction: Literal["positive", "negative", "either"] | None = None


class DiagnosticFlag(BaseModel):
    """A machine-readable warning with the evidence that triggered it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]+$")
    category: Literal["statistical_evidence", "numerical_diagnostics", "business_plausibility"]
    severity: Literal["review", "unstable"]
    message: str
    evidence: dict[str, str | int | float | bool] = Field(min_length=1)


class ModelHealthSummary(BaseModel):
    """Concise health state derived only from structured diagnostic flags."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["healthy", "review", "unstable"]
    headline: str
    flag_count: int = Field(ge=0)
    warning_codes: tuple[str, ...] = ()


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
    term_assessments: dict[str, TermAssessment]
    diagnostic_flags: tuple[DiagnosticFlag, ...]
    health_summary: ModelHealthSummary
    residual_degrees_freedom: FiniteFloat = Field(ge=0)
    condition_number: FiniteFloat = Field(ge=0)
    design_rank: int = Field(ge=1)
    parameter_count: int = Field(ge=1)
    warnings: tuple[str, ...] = ()


class ModelEngine(Protocol):
    def fit(self, data: pd.DataFrame, config: ModelConfig) -> ModelResult:
        """Fit without modifying input data; return computed results and caveats."""
        ...
