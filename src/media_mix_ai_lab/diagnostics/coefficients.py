"""Deterministic coefficient diagnostics for fitted OLS specifications."""

import math
from collections.abc import Mapping

from media_mix_ai_lab.core.config import ModelConfig
from media_mix_ai_lab.modeling.interfaces import (
    DiagnosticFlag,
    ModelHealthSummary,
    TermAssessment,
)


def assess_coefficients(
    *,
    config: ModelConfig,
    coefficients: Mapping[str, float],
    standard_errors: Mapping[str, float],
    confidence_intervals: Mapping[str, tuple[float, float]],
    term_roles: Mapping[str, str],
    residual_degrees_freedom: float,
    condition_number: float,
    design_rank: int,
    parameter_count: int,
    retained_labeled_outliers: int = 0,
) -> tuple[dict[str, TermAssessment], tuple[DiagnosticFlag, ...], ModelHealthSummary]:
    """Assess finite estimates using declared thresholds and domain expectations.

    Flags are hypotheses for review. They do not change estimates, remove rows, or
    convert associations into causal effects.
    """
    thresholds = config.diagnostic_thresholds
    media_terms = {
        (f"log({channel})" if config.kind == "log_log" else channel): channel
        for channel in config.channels
    }
    assessments: dict[str, TermAssessment] = {}
    flags: list[DiagnosticFlag] = []

    for term, estimate in coefficients.items():
        standard_error = standard_errors[term]
        lower, upper = confidence_intervals[term]
        role = term_roles[term]
        ci_spans_zero = lower <= 0 <= upper
        original_media = media_terms.get(term)
        expectation = (
            config.media_sign_expectations.get(original_media, "positive")
            if original_media is not None
            else None
        )
        scale, interpretation = _interpret_term(
            config=config,
            term=term,
            role=role,
            estimate=estimate,
            original_media=original_media,
        )
        assessments[term] = TermAssessment(
            term=term,
            role=role,
            estimate=estimate,
            standard_error=standard_error,
            confidence_interval=(lower, upper),
            ci_spans_zero=ci_spans_zero,
            scale=scale,
            interpretation=interpretation,
            expected_direction=expectation,
        )

        if role != "intercept" and ci_spans_zero:
            flags.append(
                DiagnosticFlag(
                    code="CI_SPANS_ZERO",
                    category="statistical_evidence",
                    severity="review",
                    message=f"{term}: the 95% confidence interval spans zero.",
                    evidence={"term": term, "ci_lower": lower, "ci_upper": upper},
                )
            )

        relative_uncertainty = standard_error / max(abs(estimate), thresholds.near_zero_estimate)
        if role != "intercept" and relative_uncertainty >= thresholds.large_relative_standard_error:
            flags.append(
                DiagnosticFlag(
                    code="LARGE_RELATIVE_UNCERTAINTY",
                    category="statistical_evidence",
                    severity="review",
                    message=f"{term}: standard error is large relative to the estimate.",
                    evidence={
                        "term": term,
                        "estimate": estimate,
                        "standard_error": standard_error,
                        "relative_uncertainty": relative_uncertainty,
                        "threshold": thresholds.large_relative_standard_error,
                    },
                )
            )

        unexpected = (expectation == "positive" and estimate < 0) or (
            expectation == "negative" and estimate > 0
        )
        if role == "media" and unexpected:
            flags.append(
                DiagnosticFlag(
                    code="MEDIA_UNEXPECTED_SIGN",
                    category="business_plausibility",
                    severity="review",
                    message=(
                        f"{term}: unexpected sign—investigate; sign expectations are review "
                        "hypotheses, not constraints."
                    ),
                    evidence={
                        "term": term,
                        "estimate": estimate,
                        "expected_direction": expectation or "either",
                    },
                )
            )

    if residual_degrees_freedom <= thresholds.near_zero_residual_df:
        flags.append(
            DiagnosticFlag(
                code="NEAR_ZERO_RESIDUAL_DF",
                category="numerical_diagnostics",
                severity="review",
                message="Residual degrees of freedom are low; uncertainty estimates are fragile.",
                evidence={
                    "residual_degrees_freedom": residual_degrees_freedom,
                    "threshold": thresholds.near_zero_residual_df,
                    "parameter_count": parameter_count,
                },
            )
        )

    if condition_number >= thresholds.poor_condition_number:
        flags.append(
            DiagnosticFlag(
                code="POOR_CONDITIONING",
                category="numerical_diagnostics",
                severity="unstable",
                message="The design matrix is poorly conditioned; coefficients may be unstable.",
                evidence={
                    "condition_number": condition_number,
                    "threshold": thresholds.poor_condition_number,
                    "design_rank": design_rank,
                    "parameter_count": parameter_count,
                },
            )
        )

    if not config.intercept:
        flags.append(
            DiagnosticFlag(
                code="NO_INTERCEPT_UNCENTERED_R2",
                category="statistical_evidence",
                severity="review",
                message=(
                    "The model omits an intercept; R² and adjusted R² use the uncentered "
                    "convention."
                ),
                evidence={"intercept": False, "r_squared_convention": "uncentered"},
            )
        )

    if retained_labeled_outliers:
        flags.append(
            DiagnosticFlag(
                code="LABELED_OUTLIER_RETAINED",
                category="statistical_evidence",
                severity="review",
                message="Labeled injected outlier observations were retained in the fit.",
                evidence={"retained_labeled_outliers": retained_labeled_outliers},
            )
        )

    structured_flags = tuple(flags)
    status = (
        "unstable"
        if any(flag.severity == "unstable" for flag in structured_flags)
        else "review"
        if structured_flags
        else "healthy"
    )
    headline = {
        "healthy": "No basic coefficient-diagnostic flags were triggered.",
        "review": "Review the flagged statistical or domain assumptions.",
        "unstable": "Numerical instability may materially affect the estimates.",
    }[status]
    health = ModelHealthSummary(
        status=status,
        headline=headline,
        flag_count=len(structured_flags),
        warning_codes=tuple(dict.fromkeys(flag.code for flag in structured_flags)),
    )
    return assessments, structured_flags, health


def _interpret_term(
    *,
    config: ModelConfig,
    term: str,
    role: str,
    estimate: float,
    original_media: str | None,
) -> tuple[str, str]:
    qualifier = "holding the other included terms constant; this is an association, not causation"
    if role == "intercept":
        scale = "log outcome units" if config.kind == "log_log" else f"{config.target} units"
        return (
            scale,
            f"Expected {scale} when every included predictor equals zero; interpret only if that "
            "reference point is meaningful.",
        )
    if role == "media" and config.kind == "log_log":
        return (
            "conditional elasticity",
            f"A 1% increase in {original_media} is associated with approximately a "
            f"{estimate:.4g}% change in {config.target}, {qualifier}.",
        )
    if role == "media":
        return (
            f"{config.target} units per {original_media} unit",
            f"A one-unit increase in {original_media} is associated with a {estimate:.4g}-unit "
            f"change in {config.target}, {qualifier}.",
        )
    if config.kind == "log_log":
        percentage_change = 100 * math.expm1(estimate)
        return (
            f"model-implied % {config.target} per {term} unit",
            f"A one-unit increase in {term} is associated with a {percentage_change:.4g}% "
            f"change in {config.target}, {qualifier}.",
        )
    return (
        f"{config.target} units per {term} unit",
        f"A one-unit increase in {term} is associated with a {estimate:.4g}-unit change in "
        f"{config.target}, {qualifier}.",
    )
