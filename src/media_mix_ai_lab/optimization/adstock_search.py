"""Bounded chronological calibration of deterministic adstock settings."""

from itertools import islice, product

import numpy as np
import pandas as pd

from media_mix_ai_lab.core.config import AdstockSpec, ModelConfig
from media_mix_ai_lab.diagnostics.temporal import (
    _design_matrix,
    _error_metrics,
    _predict,
    chronological_split,
)
from media_mix_ai_lab.modeling.ols import OLSModelEngine


def partition(data: pd.DataFrame, date_column: str, holdout_fraction: float = 0.2):
    """Use the established chronological splitter for 60/20/20 by default."""
    development, test = chronological_split(data, date_column, holdout_fraction=holdout_fraction)
    fit, validation = chronological_split(
        development, date_column, holdout_fraction=0.2 / (1 - holdout_fraction)
    )
    if len(fit) < 4 or len(validation) < 1 or len(test) < 1:
        raise ValueError("Need at least four fitting weeks plus validation and test weeks.")
    return fit, validation, test


def candidate_specs(
    channels: tuple[str, ...], method: str, max_lag: int, budget: int
) -> list[dict[str, AdstockSpec]]:
    """Enumerate a deterministic coarse grid and truncate to a visible safe budget."""
    if not 1 <= budget <= 100:
        raise ValueError("Candidate budget must be between 1 and 100.")
    if method == "none":
        return [{}]
    if method == "geometric":
        grid = [AdstockSpec(method="geometric", alpha=round(i / 10, 1)) for i in range(10)]
    elif method == "delayed_geometric":
        grid = [
            AdstockSpec(
                method="delayed_geometric",
                alpha=round(i / 10, 1),
                theta=theta,
                max_lag_weeks=max_lag,
            )
            for i, theta in product(range(1, 10), range(min(4, max_lag) + 1))
        ]
    else:
        raise ValueError("Unknown adstock family.")
    return [
        dict(zip(channels, specs, strict=True))
        for specs in islice(product(grid, repeat=len(channels)), budget)
    ]


def search(
    data: pd.DataFrame,
    base: ModelConfig,
    *,
    method: str = "geometric",
    max_lag: int = 8,
    budget: int = 20,
    holdout_fraction: float = 0.2,
) -> dict:
    """Select on validation RMSE and assess the untouched final test once."""
    fit, validation, test = partition(data, base.date_column, holdout_fraction)
    development = data.iloc[: len(fit) + len(validation)]
    candidates = [{}]
    candidates.extend(candidate_specs(base.channels, method, max_lag, budget))
    seen: set[str] = set()
    rows = []
    for settings in candidates:
        config = base.model_copy(update={"adstock": settings})
        key = config.model_dump_json()
        if key in seen:
            continue
        seen.add(key)
        row = {
            "config": config.model_dump(mode="json"),
            "n_fit": len(fit),
            "n_validation": len(validation),
            "status": "ok",
        }
        try:
            fitted = OLSModelEngine().fit(fit, config)
            design = _design_matrix(development, config).iloc[len(fit) :]
            predicted = _predict(design, pd.Series(fitted.coefficients), config)
            metrics = _error_metrics(validation[base.target].to_numpy(dtype=float), predicted)
            row.update(
                validation_rmse=metrics.rmse,
                validation_mae=metrics.mae,
                coefficients=fitted.coefficients,
                p_values=fitted.p_values,
                warnings=list(fitted.health_summary.warning_codes),
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            row.update(status="failed", reason=str(error))
        rows.append(row)
    ranked = sorted(
        (row for row in rows if row["status"] == "ok"), key=lambda row: row["validation_rmse"]
    )
    if not ranked:
        raise ValueError(
            "No candidate could be fitted; inspect candidate failure reasons and data."
        )
    selected = ranked[0]
    selected_config = ModelConfig.model_validate(selected["config"])
    final_fit = OLSModelEngine().fit(development, selected_config)
    final_design = _design_matrix(data, selected_config).iloc[len(development) :]
    final_prediction = _predict(final_design, pd.Series(final_fit.coefficients), selected_config)
    final_metrics = _error_metrics(test[base.target].to_numpy(dtype=float), final_prediction)
    return {
        "data_sha256": OLSModelEngine._fingerprint(data),
        "method": "chronological fit/validation selection, final test once",
        "normalization": "raw",
        "warm_up": "zero initial state; past observed spend carried across boundaries",
        "partitions": {
            name: {
                "n": len(frame),
                "first": frame[base.date_column].iloc[0].isoformat(),
                "last": frame[base.date_column].iloc[-1].isoformat(),
            }
            for name, frame in (("fit", fit), ("validation", validation), ("test", test))
        },
        "candidates": rows,
        "candidate_budget": budget,
        "evaluated_candidates": len(rows),
        "baseline": next((row for row in rows if not row["config"]["adstock"]), None),
        "selected": selected,
        "final_test": final_metrics.model_dump(),
        "final_test_predictions": final_prediction.tolist(),
        "package_versions": final_fit.package_versions,
        "caveat": "Selection is predictive, not causal; post-selection p-values are optimistic. "
        "Log-target predictions use exp(log prediction) without smearing correction.",
    }
