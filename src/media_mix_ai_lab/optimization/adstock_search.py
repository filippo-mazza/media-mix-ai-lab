"""Bounded chronological calibration of deterministic adstock settings."""

from copy import deepcopy
from itertools import product

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
    minimum_budget = len(channels) + 1 if len(channels) > 1 else 1
    if budget < minimum_budget:
        raise ValueError(
            f"Candidate budget must be at least {minimum_budget} for {len(channels)} channels "
            "so each channel receives an independent contrast."
        )
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
            for i, theta in product(range(1, 10), range(max_lag + 1))
        ]
    else:
        raise ValueError("Unknown adstock family.")
    reference = len(grid) // 2
    combinations: list[tuple[int, ...]] = []

    def add_combination(indices: tuple[int, ...]) -> None:
        if indices not in combinations and len(combinations) < budget:
            combinations.append(indices)

    add_combination((reference,) * len(channels))
    contrast_order = [0, len(grid) - 1, *range(1, len(grid) - 1)]
    for grid_index in contrast_order:
        if grid_index == reference:
            continue
        for channel_index in range(len(channels)):
            indices = [reference] * len(channels)
            indices[channel_index] = grid_index
            add_combination(tuple(indices))
            if len(combinations) == budget:
                break
        if len(combinations) == budget:
            break

    total = len(grid) ** len(channels)
    for sample_index in range(budget * 3):
        candidate_index = sample_index * (total - 1) // max(budget * 3 - 1, 1)
        indices = tuple(
            (candidate_index // (len(grid) ** (len(channels) - index - 1))) % len(grid)
            for index in range(len(channels))
        )
        add_combination(indices)
    for candidate_index in range(total):
        if len(combinations) == budget:
            break
        indices = tuple(
            (candidate_index // (len(grid) ** (len(channels) - index - 1))) % len(grid)
            for index in range(len(channels))
        )
        add_combination(indices)

    candidates = [
        dict(zip(channels, (grid[index] for index in indices), strict=True))
        for indices in combinations
    ]
    return candidates


def dataset_sha256(data: pd.DataFrame) -> str:
    """Return the same immutable dataset identity used by fitted model results."""
    return OLSModelEngine._fingerprint(data)


def search_specification(
    data: pd.DataFrame,
    base: ModelConfig,
    *,
    method: str,
    max_lag: int,
    budget: int,
    holdout_fraction: float,
) -> dict:
    """Describe every input that determines candidate generation and selection."""
    return {
        "data_sha256": dataset_sha256(data),
        "base_config": base.model_dump(mode="json"),
        "candidate_method": method,
        "max_lag": max_lag,
        "candidate_budget": budget,
        "final_holdout_fraction": holdout_fraction,
    }


def search(
    data: pd.DataFrame,
    base: ModelConfig,
    *,
    method: str = "geometric",
    max_lag: int = 8,
    budget: int = 20,
    holdout_fraction: float = 0.2,
) -> dict:
    """Select on validation RMSE without reading final-test outcomes."""
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
                package_versions=fitted.package_versions,
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
    specification = search_specification(
        data,
        base,
        method=method,
        max_lag=max_lag,
        budget=budget,
        holdout_fraction=holdout_fraction,
    )
    return {
        "data_sha256": dataset_sha256(data),
        "search_specification": specification,
        "method": "chronological fit/validation selection; final test requires a separate action",
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
        "final_test": None,
        "final_test_predictions": None,
        "package_versions": selected["package_versions"],
        "caveat": "Selection is predictive, not causal; post-selection p-values are optimistic. "
        "Log-target predictions use exp(log prediction) without smearing correction.",
    }


def evaluate_final_test(data: pd.DataFrame, result: dict) -> dict:
    """Evaluate a frozen selection on its final holdout exactly once per manifest."""
    if result.get("final_test") is not None:
        raise ValueError("This search manifest already contains a final-test evaluation.")
    if result.get("data_sha256") != dataset_sha256(data):
        raise ValueError("The selected model belongs to a different dataset.")
    specification = result["search_specification"]
    base = ModelConfig.model_validate(specification["base_config"])
    fit, validation, test = partition(
        data, base.date_column, specification["final_holdout_fraction"]
    )
    development = data.iloc[: len(fit) + len(validation)]
    selected_config = ModelConfig.model_validate(result["selected"]["config"])
    final_fit = OLSModelEngine().fit(development, selected_config)
    final_design = _design_matrix(data, selected_config).iloc[len(development) :]
    final_prediction = _predict(final_design, pd.Series(final_fit.coefficients), selected_config)
    final_metrics = _error_metrics(test[base.target].to_numpy(dtype=float), final_prediction)
    evaluated = deepcopy(result)
    evaluated["final_test"] = final_metrics.model_dump()
    evaluated["final_test_predictions"] = final_prediction.tolist()
    return evaluated
