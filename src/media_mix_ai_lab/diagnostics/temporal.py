"""Chronological holdout evaluation and residual diagnostics for fitted OLS models."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict, FiniteFloat
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from media_mix_ai_lab.core.config import ModelConfig
from media_mix_ai_lab.modeling.adstock import transform_media_frame, validate_weekly_dates
from media_mix_ai_lab.modeling.ols import OLSModelEngine


class ErrorMetrics(BaseModel):
    """Errors in the outcome's original units; WAPE is undefined for a zero denominator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mae: FiniteFloat
    rmse: FiniteFloat
    wape: FiniteFloat | None


class TemporalPrediction(BaseModel):
    """An observed and predicted outcome indexed by its weekly date."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    date: str
    observed: FiniteFloat
    predicted: FiniteFloat
    residual: FiniteFloat


class TemporalValidationResult(BaseModel):
    """Train and holdout errors, predictions, and applicable diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config: ModelConfig
    training_metrics: ErrorMetrics
    holdout_metrics: ErrorMetrics
    training_size: int
    holdout_size: int
    training_period: tuple[str, str]
    holdout_period: tuple[str, str]
    predictions: tuple[TemporalPrediction, ...]
    durbin_watson: FiniteFloat | None
    ljung_box_p_value: FiniteFloat | None
    variance_inflation_factors: dict[str, FiniteFloat]
    condition_number: FiniteFloat
    warnings: tuple[str, ...] = ()


def chronological_split(
    data: pd.DataFrame,
    date_column: str,
    *,
    holdout_fraction: float = 0.2,
    cutoff: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ordered weekly rows without shuffling or changing their order."""
    validate_weekly_dates(data, date_column)
    if cutoff is None:
        if not 0 < holdout_fraction < 1:
            raise ValueError("Holdout fraction must be greater than 0 and less than 1.")
        split_index = int(len(data) * (1 - holdout_fraction))
        if split_index <= 0 or split_index >= len(data):
            raise ValueError(
                "The requested holdout fraction leaves an empty training or holdout period."
            )
        return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()

    try:
        boundary = pd.Timestamp(cutoff)
        train_mask = data[date_column] < boundary
        holdout_mask = ~train_mask
    except (TypeError, ValueError) as error:
        raise ValueError("The holdout cutoff must be a valid date.") from error
    if train_mask.all() or holdout_mask.all():
        raise ValueError("The cutoff must leave at least one training and one holdout row.")
    return data.loc[train_mask].copy(), data.loc[holdout_mask].copy()


def evaluate_temporal_holdout(
    data: pd.DataFrame,
    config: ModelConfig,
    *,
    holdout_fraction: float = 0.2,
    cutoff: pd.Timestamp | str | None = None,
) -> TemporalValidationResult:
    """Fit only on earlier rows and score the later contiguous holdout.

    Predictors such as adstock may use prior training-period media history when
    generating holdout features. They are deterministic filters, not learned
    preprocessing. No scaler, imputer, or other learned transform is applied.
    """
    training, holdout = chronological_split(
        data, config.date_column, holdout_fraction=holdout_fraction, cutoff=cutoff
    )
    fitted = OLSModelEngine().fit(training, config)
    training_design = _design_matrix(training, config)
    all_design = _design_matrix(data, config)
    coefficients = pd.Series(fitted.coefficients)
    training_prediction = _predict(training_design, coefficients, config)
    holdout_prediction = _predict(all_design.iloc[len(training) :], coefficients, config)

    train_actual = training[config.target].to_numpy(dtype=float)
    holdout_actual = holdout[config.target].to_numpy(dtype=float)
    training_metrics = _error_metrics(train_actual, training_prediction)
    holdout_metrics = _error_metrics(holdout_actual, holdout_prediction)
    residuals = holdout_actual - holdout_prediction
    warnings: list[str] = []

    if holdout_metrics.wape is None:
        warnings.append("WAPE is undefined because the sum of absolute holdout actuals is zero.")
    if len(residuals) < 2:
        dw = None
        warnings.append("Durbin–Watson is unavailable with fewer than two holdout residuals.")
    elif np.var(residuals) == 0:
        dw = None
        warnings.append(
            "Residual autocorrelation diagnostics are undefined for constant residuals."
        )
    else:
        dw = float(durbin_watson(residuals))

    if len(residuals) < 8:
        ljung_box = None
        warnings.append("Ljung–Box is unavailable with fewer than eight holdout residuals.")
    elif np.var(residuals) == 0:
        ljung_box = None
    else:
        lag = min(10, len(residuals) // 5)
        test = acorr_ljungbox(residuals, lags=[lag], return_df=True)
        p_value = float(test["lb_pvalue"].iloc[0])
        ljung_box = p_value if np.isfinite(p_value) else None
        if ljung_box is None:
            warnings.append("Ljung–Box p-value is undefined for these holdout residuals.")

    predictor_columns = [column for column in training_design.columns if column != "const"]
    vifs: dict[str, float] = {}
    if len(training_design) <= len(predictor_columns) + 1:
        warnings.append("Training sample is too short to report stable VIF diagnostics.")
    elif len(predictor_columns) == 1:
        vifs = {predictor_columns[0]: 1.0}
    else:
        try:
            design_values = training_design.to_numpy()
            vifs = {
                column: float(
                    variance_inflation_factor(
                        design_values, training_design.columns.get_loc(column)
                    )
                )
                for column in predictor_columns
            }
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            warnings.append("VIF diagnostics could not be computed for the training design.")

    prediction_rows = tuple(
        TemporalPrediction(
            date=date.isoformat(),
            observed=float(actual),
            predicted=float(prediction),
            residual=float(actual - prediction),
        )
        for date, actual, prediction in zip(
            holdout[config.date_column],
            holdout_actual,
            holdout_prediction,
            strict=True,
        )
    )
    return TemporalValidationResult(
        config=config,
        training_metrics=training_metrics,
        holdout_metrics=holdout_metrics,
        training_size=len(training),
        holdout_size=len(holdout),
        training_period=(
            training[config.date_column].iloc[0].isoformat(),
            training[config.date_column].iloc[-1].isoformat(),
        ),
        holdout_period=(
            holdout[config.date_column].iloc[0].isoformat(),
            holdout[config.date_column].iloc[-1].isoformat(),
        ),
        predictions=prediction_rows,
        durbin_watson=dw,
        ljung_box_p_value=ljung_box,
        variance_inflation_factors=vifs,
        condition_number=float(np.linalg.cond(training_design.to_numpy(dtype=float))),
        warnings=tuple(warnings),
    )


def same_holdout_period(first: TemporalValidationResult, second: TemporalValidationResult) -> bool:
    """Return whether two stored evaluations used the exact same holdout dates."""
    return tuple(row.date for row in first.predictions) == tuple(
        row.date for row in second.predictions
    )


def _design_matrix(
    data: pd.DataFrame, config: ModelConfig, *, require_positive_target: bool = True
) -> pd.DataFrame:
    """Build predictors using only explicit, deterministic transformations."""
    transformed = transform_media_frame(data, config)
    features = [*config.channels, *config.controls]
    if config.kind == "log_log":
        if require_positive_target and (transformed[config.target] <= 0).any():
            raise ValueError("Log-log holdout outcomes must be strictly positive.")
        for channel in config.channels:
            if (transformed[channel] <= 0).any():
                raise ValueError(f"Log-log holdout media values for {channel} must be positive.")
        design = pd.DataFrame(index=data.index)
        for channel in config.channels:
            design[f"log({channel})"] = np.log(transformed[channel].astype(float))
        for column in config.controls:
            design[column] = transformed[column].astype(float)
    else:
        design = transformed.loc[:, features].astype(float).copy()
    if config.intercept:
        design = sm.add_constant(design, has_constant="add")
    return design


def _predict(
    design: pd.DataFrame,
    coefficients: pd.Series,
    config: ModelConfig,
) -> np.ndarray:
    """Predict in the original outcome units for both supported model families."""
    linear_prediction = (
        design.loc[:, coefficients.index].to_numpy(dtype=float) @ coefficients.values
    )
    prediction = np.exp(linear_prediction) if config.kind == "log_log" else linear_prediction
    if not np.isfinite(prediction).all():
        raise ValueError("Holdout prediction produced nonfinite values.")
    return prediction


def _error_metrics(actual: np.ndarray, predicted: np.ndarray) -> ErrorMetrics:
    if not np.isfinite(actual).all():
        raise ValueError("Observed outcomes must be numeric and finite for validation.")
    errors = actual - predicted
    absolute_sum = float(np.abs(actual).sum())
    wape = float(np.abs(errors).sum() / absolute_sum) if absolute_sum else None
    return ErrorMetrics(
        mae=float(np.abs(errors).mean()),
        rmse=float(np.sqrt(np.square(errors).mean())),
        wape=wape,
    )
