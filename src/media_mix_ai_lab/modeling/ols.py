"""Deterministic linear and strict log-log OLS estimation."""

import hashlib
import json
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
import statsmodels.api as sm

from media_mix_ai_lab.core.config import ModelConfig
from media_mix_ai_lab.diagnostics.coefficients import assess_coefficients
from media_mix_ai_lab.modeling.adstock import transform_media_frame
from media_mix_ai_lab.modeling.interfaces import (
    AffectedObservation,
    ModelResult,
    ObservationAccounting,
    TransformPolicy,
)


class ModelFitError(ValueError):
    """A fit failure with structured observation accounting."""

    def __init__(self, message: str, accounting: ObservationAccounting) -> None:
        super().__init__(message)
        self.accounting = accounting


class OLSModelEngine:
    """Fit unweighted OLS without modifying or silently cleaning the input frame."""

    def fit(self, data: pd.DataFrame, config: ModelConfig) -> ModelResult:
        if data.empty:
            raise ValueError("Cannot fit OLS: the dataset is empty.")
        if not data.columns.is_unique:
            raise ValueError("Cannot fit OLS: column names must be unique.")
        features = (*config.channels, *config.controls)
        selected = (config.target, *features)
        missing = [column for column in selected if column not in data.columns]
        if missing:
            raise ValueError(f"Selected model columns are missing: {', '.join(missing)}")
        for column in selected:
            values = data[column]
            if (
                not pd.api.types.is_numeric_dtype(values)
                or pd.api.types.is_bool_dtype(values)
                or not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all()
            ):
                raise ValueError(f"{column} must contain numeric, finite, nonmissing values.")

        if config.kind == "ols":
            for channel in config.channels:
                if (data[channel] < 0).any():
                    raise ValueError(f"{channel} cannot contain negative media spend.")

        model_data = transform_media_frame(data, config)
        accounting = _summarize_transformed_observations(model_data, config)
        if config.kind == "log_log" and accounting.n_eligible < accounting.n_input:
            invalid = [
                f"{column}: {count} nonpositive value(s)"
                for column, count in (
                    (config.target, accounting.nonpositive_target_count),
                    *accounting.nonpositive_media_counts.items(),
                )
                if count
            ]
            examples = []
            for row in accounting.affected_rows[:3]:
                label = f"row {row.row_position}"
                if row.date is not None:
                    label += f" (date={row.date})"
                examples.append(label)
            raise ModelFitError(
                "Strict log-log requires positive outcome and media values; "
                + "; ".join(invalid)
                + f". Affected rows: {', '.join(examples)}. "
                "No rows were dropped and no epsilon was added.",
                accounting,
            )

        if config.kind == "log_log":
            transformed_channels = [f"log({column})" for column in config.channels]
            collisions = set(transformed_channels).intersection(config.controls)
            if collisions:
                raise ValueError(
                    "Transformed media predictor names collide with selected controls: "
                    f"{', '.join(sorted(collisions))}. Rename the conflicting columns."
                )
            y = np.log(model_data[config.target].astype(float)).rename(f"log({config.target})")
            design = pd.DataFrame(index=data.index)
            for column in config.channels:
                design[f"log({column})"] = np.log(model_data[column].astype(float))
            for column in config.controls:
                design[column] = model_data[column].astype(float)
        else:
            y = model_data[config.target].astype(float).rename(config.target)
            design = model_data.loc[:, features].astype(float).copy()

        roles = {
            (f"log({column})" if config.kind == "log_log" else column): "media"
            for column in config.channels
        }
        roles.update({column: "control" for column in config.controls})
        transformations = {
            config.target: "natural log" if config.kind == "log_log" else "original level",
            **{column: self._media_transformation(config, column) for column in config.channels},
            **{column: "original level" for column in config.controls},
        }
        if config.intercept:
            design = sm.add_constant(design, has_constant="add")
            roles = {"const": "intercept", **roles}

        sample_size, parameter_count = design.shape
        if sample_size <= parameter_count:
            raise ValueError(
                f"Too few rows ({sample_size}) for {parameter_count} model parameters; "
                "at least one residual degree of freedom is required."
            )
        rank = int(np.linalg.matrix_rank(design.to_numpy(dtype=float)))
        if rank < parameter_count:
            raise ValueError(
                f"Rank-deficient design matrix: rank {rank} for {parameter_count} "
                "columns. Remove redundant or constant predictors."
            )

        fitted = sm.OLS(y, design, hasconst=config.intercept).fit()
        coefficients = fitted.params.to_dict()
        standard_errors = fitted.bse.to_dict()
        t_statistics = {
            name: float(value) if np.isfinite(value) else None
            for name, value in fitted.tvalues.items()
        }
        p_values = {
            name: float(value) if np.isfinite(value) else None
            for name, value in fitted.pvalues.items()
        }
        intervals = fitted.conf_int()
        confidence_intervals = {
            name: (float(intervals.loc[name, 0]), float(intervals.loc[name, 1]))
            for name in fitted.params.index
        }
        condition_number = float(fitted.condition_number)
        residual_degrees_freedom = float(fitted.df_resid)
        fitted_metrics = {
            "r_squared": float(fitted.rsquared),
            "adjusted_r_squared": float(fitted.rsquared_adj),
            "residual_variance": float(fitted.mse_resid),
            "residual_degrees_freedom": residual_degrees_freedom,
            "condition_number": condition_number,
        }
        nonfinite = [
            name
            for name, value in {
                **{f"coefficient:{name}": value for name, value in coefficients.items()},
                **{f"standard_error:{name}": value for name, value in standard_errors.items()},
                **{
                    f"confidence_interval:{name}:{bound}": value
                    for name, interval in confidence_intervals.items()
                    for bound, value in zip(("lower", "upper"), interval, strict=True)
                },
                **fitted_metrics,
            }.items()
            if not np.isfinite(value)
        ]
        if nonfinite:
            raise ValueError(
                "NONFINITE_ESTIMATE numerical violation: the fit produced nonfinite "
                f"outputs ({', '.join(nonfinite)}). Review scaling and model specification."
            )

        retained_outliers = (
            int(data["is_injected_outlier"].eq(True).sum()) if "is_injected_outlier" in data else 0
        )
        term_assessments, diagnostic_flags, health_summary = assess_coefficients(
            config=config,
            coefficients=coefficients,
            standard_errors=standard_errors,
            t_statistics=t_statistics,
            confidence_intervals=confidence_intervals,
            p_values=p_values,
            term_roles=roles,
            residual_degrees_freedom=residual_degrees_freedom,
            condition_number=condition_number,
            design_rank=rank,
            parameter_count=parameter_count,
            retained_labeled_outliers=retained_outliers,
        )
        warnings = [
            "Classical OLS standard errors, confidence intervals, and two-sided t-test "
            "p-values assume a correctly specified model with independent, homoskedastic "
            "errors (and normal errors for exact finite-sample inference); weekly dependence "
            "and omitted confounders may invalidate them."
        ]
        if config.kind == "log_log":
            warnings.append("Log-log media slopes are conditional elasticities, not causal ROI.")
        warnings.extend(flag.message for flag in diagnostic_flags)

        equation = self._equation(config)
        return ModelResult(
            config=config,
            dataset_sha256=self._fingerprint(data),
            coefficients={name: float(value) for name, value in coefficients.items()},
            standard_errors={name: float(value) for name, value in standard_errors.items()},
            t_statistics=t_statistics,
            confidence_intervals=confidence_intervals,
            p_values=p_values,
            significance_codes={
                name: assessment.significance_code for name, assessment in term_assessments.items()
            },
            term_roles=roles,
            observation_accounting=accounting.model_copy(
                update={"n_fitted": sample_size, "n_excluded": 0}
            ),
            sample_size=sample_size,
            r_squared=fitted_metrics["r_squared"],
            adjusted_r_squared=fitted_metrics["adjusted_r_squared"],
            residual_variance=fitted_metrics["residual_variance"],
            equation=equation,
            transformations=transformations,
            package_versions={
                "python": platform.python_version(),
                "numpy": version("numpy"),
                "pandas": version("pandas"),
                "scipy": version("scipy"),
                "statsmodels": version("statsmodels"),
            },
            term_assessments=term_assessments,
            diagnostic_flags=diagnostic_flags,
            health_summary=health_summary,
            residual_degrees_freedom=residual_degrees_freedom,
            condition_number=condition_number,
            design_rank=rank,
            parameter_count=parameter_count,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _equation(config: ModelConfig) -> str:
        logged = config.kind == "log_log"
        lhs = f"log({config.target})" if logged else config.target
        terms = []
        for column in config.channels:
            media_term = OLSModelEngine._media_term(config, column)
            coefficient = f"log({column})" if logged else column
            predictor = f"log({media_term})" if logged else media_term
            terms.append(f"β_{coefficient} × {predictor}")
        terms.extend(f"β_{column} × {column}" for column in config.controls)
        if config.intercept:
            terms.insert(0, "β₀")
        return f"{lhs} = {' + '.join(terms)}"

    @staticmethod
    def _media_term(config: ModelConfig, column: str) -> str:
        spec = config.adstock.get(column)
        if spec is None or spec.method == "none":
            return column
        return f"adstock({column})"

    @staticmethod
    def _media_transformation(config: ModelConfig, column: str) -> str:
        spec = config.adstock.get(column)
        if spec is None or spec.method == "none":
            return "natural log" if config.kind == "log_log" else "original level"
        adstock = "original level"
        if spec is not None and spec.method == "geometric":
            adstock = (
                f"recursive geometric adstock (alpha={spec.alpha:g}, "
                f"initial state={spec.initial_state:g}, "
                f"{spec.weight_scaling}; no finite truncation)"
            )
        elif spec is not None and spec.method == "delayed_geometric":
            boundary = (
                "zero prehistory"
                if not spec.historical_values
                else f"{len(spec.historical_values)} supplied historical week(s)"
            )
            adstock = (
                f"delayed geometric adstock (alpha={spec.alpha:g}, "
                f"theta={spec.theta:g}, max lag={spec.max_lag_weeks} weeks, "
                f"{spec.weight_scaling}, {boundary})"
            )
        return f"{adstock}, then natural log" if config.kind == "log_log" else adstock

    @staticmethod
    def _fingerprint(data: pd.DataFrame) -> str:
        digest = hashlib.sha256()
        metadata = {
            "columns": [repr(column) for column in data.columns],
            "dtypes": [str(dtype) for dtype in data.dtypes],
        }
        digest.update(json.dumps(metadata, ensure_ascii=False).encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes())
        return digest.hexdigest()


def summarize_observations(data: pd.DataFrame, config: ModelConfig) -> ObservationAccounting:
    """Count log-domain eligibility on transformed media without mutating source data."""
    if config.kind == "ols":
        for channel in config.channels:
            if channel in data.columns and (data[channel] < 0).any():
                raise ValueError(f"{channel} cannot contain negative media spend.")
    transformed = transform_media_frame(data, config)
    return _summarize_transformed_observations(transformed, config)


def _summarize_transformed_observations(
    data: pd.DataFrame, config: ModelConfig
) -> ObservationAccounting:
    columns = (config.target, *config.channels)
    positions_by_column = {
        column: np.flatnonzero(data[column].to_numpy(dtype=float) <= 0) for column in columns
    }
    affected_positions = sorted(
        {int(position) for positions in positions_by_column.values() for position in positions}
    )
    affected_rows = []
    for position in affected_positions:
        date = None
        if config.date_column in data.columns and not pd.isna(
            data[config.date_column].iloc[position]
        ):
            value = data[config.date_column].iloc[position]
            date = value.isoformat() if hasattr(value, "isoformat") else str(value)
        affected_rows.append(
            AffectedObservation(
                row_position=position,
                date=date,
                columns=tuple(
                    column for column in columns if position in positions_by_column[column]
                ),
            )
        )

    is_log_log = config.kind == "log_log"
    return ObservationAccounting(
        n_input=len(data),
        n_eligible=len(data) - len(affected_positions) if is_log_log else len(data),
        nonpositive_target_count=len(positions_by_column[config.target]),
        nonpositive_media_counts={
            column: len(positions_by_column[column]) for column in config.channels
        },
        affected_rows=tuple(affected_rows),
        transform_policy=TransformPolicy(
            target="natural log" if is_log_log else "original level",
            media={
                column: "natural log" if is_log_log else "original level"
                for column in config.channels
            },
            controls={column: "original level" for column in config.controls},
            log_domain_failure=(
                "Fail if any target or media value is nonpositive; do not drop rows or add epsilon."
                if is_log_log
                else "No logarithmic transforms."
            ),
        ),
    )
