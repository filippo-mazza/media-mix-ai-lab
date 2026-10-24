"""Typed contracts for fitted models, coefficient assessments, and diagnostics."""

from typing import Annotated, Literal, Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from media_mix_ai_lab.core.config import ModelConfig

PValue = Annotated[FiniteFloat, Field(ge=0, le=1)]
SignificanceCode = Literal["***", "**", "*", ".", ""]


class AffectedObservation(BaseModel):
    """One unique input row with at least one value outside the log domain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_position: int = Field(ge=0)
    date: str | None = None
    columns: tuple[str, ...]


class TransformPolicy(BaseModel):
    """Explicit transforms by model role and strict log-domain behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str
    media: dict[str, str]
    controls: dict[str, str]
    log_domain_failure: str


class ObservationAccounting(BaseModel):
    """Input eligibility and fit accounting, including the strict transform policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_input: int = Field(ge=0)
    n_eligible: int = Field(ge=0)
    n_fitted: int | None = Field(default=None, ge=0)
    n_excluded: int | None = Field(default=None, ge=0)
    nonpositive_target_count: int = Field(ge=0)
    nonpositive_media_counts: dict[str, int]
    affected_rows: tuple[AffectedObservation, ...]
    transform_policy: TransformPolicy


class TermAssessment(BaseModel):
    """A computed estimate and its deterministic, noncausal interpretation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    term: str
    role: Literal["intercept", "media", "control"]
    estimate: FiniteFloat
    standard_error: FiniteFloat = Field(ge=0)
    t_statistic: FiniteFloat | None
    confidence_interval: tuple[FiniteFloat, FiniteFloat]
    ci_spans_zero: bool
    p_value: PValue | None
    significance_code: SignificanceCode
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
    t_statistics: dict[str, FiniteFloat | None]
    confidence_intervals: dict[str, tuple[FiniteFloat, FiniteFloat]]
    p_values: dict[str, PValue | None]
    significance_codes: dict[str, SignificanceCode]
    term_roles: dict[str, str]
    observation_accounting: ObservationAccounting
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

    @model_validator(mode="after")
    def validate_observation_counts(self) -> "ModelResult":
        """Keep successful-fit counts consistent under the current strict policy."""
        accounting = self.observation_accounting
        if accounting.n_fitted != self.sample_size:
            raise ValueError("sample_size must equal observation_accounting.n_fitted")
        if accounting.n_excluded != 0:
            raise ValueError("successful strict fits must report n_excluded == 0")
        return self


class ModelEngine(Protocol):
    def fit(self, data: pd.DataFrame, config: ModelConfig) -> ModelResult:
        """Fit without modifying input data; return computed results and caveats."""
        ...
