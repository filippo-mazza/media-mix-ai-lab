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
from media_mix_ai_lab.modeling.interfaces import ModelResult


class OLSModelEngine:
    """Fit unweighted OLS without modifying or silently cleaning the input frame."""

    def fit(self, data: pd.DataFrame, config: ModelConfig) -> ModelResult:
        if data.empty:
            raise ValueError("Cannot fit OLS: the dataset is empty.")
        if not data.columns.is_unique:
            raise ValueError("Cannot fit OLS: column names must be unique.")
        if config.adstock_decay or config.max_lag_weeks:
            raise ValueError("Adstock is not supported by this OLS specification.")

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

        if config.kind == "log_log":
            transformed_channels = [f"log({column})" for column in config.channels]
            collisions = set(transformed_channels).intersection(config.controls)
            if collisions:
                raise ValueError(
                    "Transformed media predictor names collide with selected controls: "
                    f"{', '.join(sorted(collisions))}. Rename the conflicting columns."
                )
            self._validate_log_domain(data, config)
            y = np.log(data[config.target].astype(float)).rename(f"log({config.target})")
            design = pd.DataFrame(index=data.index)
            for column in config.channels:
                design[f"log({column})"] = np.log(data[column].astype(float))
            for column in config.controls:
                design[column] = data[column].astype(float)
        else:
            y = data[config.target].astype(float).rename(config.target)
            design = data.loc[:, features].astype(float).copy()

        roles = {
            (f"log({column})" if config.kind == "log_log" else column): "media"
            for column in config.channels
        }
        roles.update({column: "control" for column in config.controls})
        transformations = {
            config.target: "natural log" if config.kind == "log_log" else "original level",
            **{
                column: "natural log" if config.kind == "log_log" else "original level"
                for column in config.channels
            },
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
            confidence_intervals=confidence_intervals,
            term_roles=roles,
            residual_degrees_freedom=residual_degrees_freedom,
            condition_number=condition_number,
            design_rank=rank,
            parameter_count=parameter_count,
            retained_labeled_outliers=retained_outliers,
        )
        warnings = [
            "Classical OLS standard errors and confidence intervals assume a correctly "
            "specified model with independent, homoskedastic errors; weekly dependence "
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
            confidence_intervals=confidence_intervals,
            term_roles=roles,
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
    def _validate_log_domain(data: pd.DataFrame, config: ModelConfig) -> None:
        invalid: list[str] = []
        for column in (config.target, *config.channels):
            positions = np.flatnonzero(data[column].to_numpy(dtype=float) <= 0)
            if positions.size:
                examples = []
                for position in positions[:3]:
                    label = f"row {data.index[position]!r}"
                    if "date" in data.columns:
                        label += f" (date={data['date'].iloc[position]!r})"
                    examples.append(label)
                invalid.append(
                    f"{column}: {positions.size} nonpositive value(s), examples: "
                    f"{', '.join(examples)}"
                )
        if invalid:
            raise ValueError(
                "Strict log-log requires positive outcome and media values; "
                + "; ".join(invalid)
                + ". No rows were transformed or removed."
            )

    @staticmethod
    def _equation(config: ModelConfig) -> str:
        logged = config.kind == "log_log"
        lhs = f"log({config.target})" if logged else config.target
        terms = [
            f"β_{f'log({column})' if logged else column} × {f'log({column})' if logged else column}"
            for column in config.channels
        ]
        terms.extend(f"β_{column} × {column}" for column in config.controls)
        if config.intercept:
            terms.insert(0, "β₀")
        return f"{lhs} = {' + '.join(terms)}"

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
