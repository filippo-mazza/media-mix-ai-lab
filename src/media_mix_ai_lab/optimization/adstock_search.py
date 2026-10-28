"""Bounded chronological calibration of deterministic adstock settings."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import pandas as pd

from media_mix_ai_lab.core.config import AdstockSpec, ModelConfig
from media_mix_ai_lab.diagnostics.temporal import (
    _design_matrix,
    _error_metrics,
    _predict,
)
from media_mix_ai_lab.modeling.adstock import transform_media_frame, validate_weekly_dates
from media_mix_ai_lab.modeling.ols import OLSModelEngine


def partition(data: pd.DataFrame, date_column: str, holdout_fraction: float = 0.2):
    """Use ordered 60/20/20 splits by default, including empty partitions for feasibility checks."""
    if not 0 < holdout_fraction < 1:
        raise ValueError("Holdout fraction must be greater than 0 and less than 1.")
    validate_weekly_dates(data, date_column)
    test_start = int(len(data) * (1 - holdout_fraction))
    development, test = data.iloc[:test_start].copy(), data.iloc[test_start:].copy()
    validation_fraction = 0.2 / (1 - holdout_fraction)
    if validation_fraction >= 1:
        raise ValueError("Final test fraction must be below 0.8 to leave fit and validation weeks.")
    fit_end = int(len(development) * (1 - validation_fraction))
    fit, validation = development.iloc[:fit_end].copy(), development.iloc[fit_end:].copy()
    return fit, validation, test


@dataclass(frozen=True)
class FeasibilityPolicy:
    """Conservative, configurable minimums; these heuristics are not statistically validated."""

    minimum_residual_degrees_freedom: int = 5
    minimum_holdout_rows: int = 5

    def __post_init__(self) -> None:
        if self.minimum_residual_degrees_freedom < 1 or self.minimum_holdout_rows < 1:
            raise ValueError("Feasibility minimums must be at least one observation.")


@dataclass(frozen=True)
class FeasibilityResult:
    """Effective chronological sample counts and explicit reasons a search cannot proceed."""

    n_total: int
    n_fit: int
    n_validation: int
    n_test: int
    n_warmup: int
    n_log_unusable_fit: int
    n_log_unusable_validation: int
    n_log_unusable_test: int
    free_coefficients: int
    required_fit: int
    required_validation: int
    required_test: int
    failures: tuple[str, ...]
    candidate_keys: tuple[str, ...]

    @property
    def feasible(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        result = asdict(self)
        result.pop("candidate_keys")
        return result | {"feasible": self.feasible}


def assess_feasibility(
    data: pd.DataFrame,
    base: ModelConfig,
    *,
    method: str,
    max_lag: int,
    holdout_fraction: float,
    budget: int = 20,
    policy: FeasibilityPolicy | None = None,
) -> FeasibilityResult:
    """Check effective split sizes, strict log eligibility, residual df, and lag coverage.

    The first min(lag, N-fit) fitting rows initialize the candidate adstock state and
    are excluded from coefficient fitting. Validation and test rows retain earlier
    observed media history. Only fitted outcomes need to be positive; holdout outcomes
    are scored in original units, and final-test outcomes are not read here. Holdout
    minimums are heuristic safeguards, not universal statistical thresholds.
    """
    policy = policy or FeasibilityPolicy()
    if not 0 <= max_lag <= 12:
        raise ValueError("Search lag horizon must be between 0 and 12 weeks.")
    fit, validation, test = partition(data, base.date_column, holdout_fraction)
    lag_horizon = max_lag if method != "none" else 0
    n_warmup = min(lag_horizon, len(fit))
    fit_after_warmup = fit.iloc[n_warmup:]
    free_coefficients = len(base.channels) + len(base.controls) + int(base.intercept)

    required_fit = free_coefficients + max(policy.minimum_residual_degrees_freedom, lag_horizon + 1)
    required_holdout = max(policy.minimum_holdout_rows, lag_horizon + 1)
    expected_counts = (len(fit_after_warmup), len(validation), len(test))
    settings_to_check = [{}]
    settings_to_check.extend(candidate_specs(base.channels, method, max_lag, budget))
    candidate_counts: list[tuple[str, tuple[int, int, int], str | None]] = []
    seen_keys: set[str] = set()
    candidate_failures = []

    for settings in settings_to_check:
        candidate_config = base.model_copy(update={"adstock": settings})
        key = candidate_config.model_dump_json()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        try:
            fit_config = _config_after_warmup(fit, candidate_config, n_warmup)
            transformed = transform_media_frame(data.iloc[n_warmup:], fit_config)
        except ValueError as error:
            candidate_counts.append((key, (0, 0, 0), str(error)))
            candidate_failures.append(str(error))
            continue

        fit_end = expected_counts[0]
        validation_end = fit_end + expected_counts[1]
        fit_frame = transformed.iloc[:fit_end]
        validation_frame = transformed.iloc[fit_end:validation_end]
        test_frame = transformed.iloc[validation_end:]

        def eligible_count(
            frame: pd.DataFrame, *, fit_target: bool, check_target_finite: bool
        ) -> int:
            eligible = np.ones(len(frame), dtype=bool)
            if base.kind == "log_log":
                for channel in base.channels:
                    values = pd.to_numeric(frame[channel], errors="coerce").to_numpy(
                        dtype=float, na_value=np.nan
                    )
                    eligible &= np.isfinite(values) & (values > 0)
                if fit_target:
                    values = pd.to_numeric(frame[base.target], errors="coerce").to_numpy(
                        dtype=float, na_value=np.nan
                    )
                    eligible &= np.isfinite(values) & (values > 0)
                elif check_target_finite and len(frame):
                    values = pd.to_numeric(frame[base.target], errors="coerce").to_numpy(
                        dtype=float, na_value=np.nan
                    )
                    eligible &= np.isfinite(values)
            return int(eligible.sum())

        counts = (
            eligible_count(fit_frame, fit_target=True, check_target_finite=True),
            eligible_count(validation_frame, fit_target=False, check_target_finite=True),
            eligible_count(test_frame, fit_target=False, check_target_finite=False),
        )
        candidate_counts.append((key, counts, None))

    domain_valid = [
        (key, counts)
        for key, counts, error in candidate_counts
        if error is None and counts == expected_counts
    ]
    if domain_valid:
        _, best_counts = domain_valid[0]
    else:
        _, best_counts, _ = max(
            candidate_counts, key=lambda item: sum(item[1]), default=("", (0, 0, 0), None)
        )
    n_fit, n_validation, n_test = best_counts
    candidate_keys = tuple(
        key
        for key, counts in domain_valid
        if counts[0] >= required_fit
        and counts[1] >= required_holdout
        and counts[2] >= required_holdout
    )
    failures = []
    if n_fit < required_fit:
        failures.append(
            f"Only {n_fit} usable training weeks for {free_coefficients} fitted coefficients "
            f"and {lag_horizon} weeks of lag; at least {required_fit} are required."
        )
    if n_validation < required_holdout:
        failures.append(
            f"Only {n_validation} usable validation weeks; at least {required_holdout} "
            f"are required for a lag horizon of {lag_horizon} weeks."
        )
    if n_test < required_holdout:
        failures.append(
            f"Only {n_test} usable final-test weeks; at least {required_holdout} "
            f"are required for a lag horizon of {lag_horizon} weeks."
        )
    if not domain_valid and base.kind == "log_log":
        failures.append(
            "Strict log-log feasibility failed: no candidate has positive fitted outcomes and "
            "positive transformed media for every fit, validation, and final-test observation. "
            "Invalid feature rows are not dropped."
        )
    elif not domain_valid and candidate_failures:
        failures.append(
            "No candidate has valid transformed media in every partition: "
            + "; ".join(dict.fromkeys(candidate_failures))
        )
    return FeasibilityResult(
        n_total=len(data),
        n_fit=n_fit,
        n_validation=n_validation,
        n_test=n_test,
        n_warmup=n_warmup,
        n_log_unusable_fit=(len(fit_after_warmup) - n_fit if base.kind == "log_log" else 0),
        n_log_unusable_validation=(len(validation) - n_validation if base.kind == "log_log" else 0),
        n_log_unusable_test=len(test) - n_test if base.kind == "log_log" else 0,
        free_coefficients=free_coefficients,
        required_fit=required_fit,
        required_validation=required_holdout,
        required_test=required_holdout,
        failures=tuple(failures),
        candidate_keys=candidate_keys,
    )


def _config_after_warmup(fit: pd.DataFrame, config: ModelConfig, n_warmup: int) -> ModelConfig:
    """Carry observed warm-up history into the fitting slice without fitting those rows."""
    if n_warmup == 0:
        return config
    adstock = dict(config.adstock)
    for channel, spec in config.adstock.items():
        history = fit[channel].iloc[:n_warmup].to_numpy(dtype=float)
        if spec.method == "geometric":
            state = spec.initial_state
            for observation in history:
                state = float(observation) + spec.alpha * state
            adstock[channel] = spec.model_copy(update={"initial_state": state})
        elif spec.method == "delayed_geometric":
            prior_values = history[-spec.max_lag_weeks :] if spec.max_lag_weeks else []
            adstock[channel] = spec.model_copy(
                update={"historical_values": tuple(float(value) for value in prior_values)}
            )
    return config.model_copy(update={"adstock": adstock})


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
    policy: FeasibilityPolicy | None = None,
) -> dict:
    """Describe every input that determines candidate generation and selection."""
    return {
        "data_sha256": dataset_sha256(data),
        "base_config": base.model_dump(mode="json"),
        "candidate_method": method,
        "max_lag": max_lag,
        "candidate_budget": budget,
        "final_holdout_fraction": holdout_fraction,
        "feasibility_policy": asdict(policy or FeasibilityPolicy()),
    }


def search(
    data: pd.DataFrame,
    base: ModelConfig,
    *,
    method: str = "geometric",
    max_lag: int = 8,
    budget: int = 20,
    holdout_fraction: float = 0.2,
    policy: FeasibilityPolicy | None = None,
) -> dict:
    """Select on validation RMSE without reading final-test outcomes."""
    feasibility = assess_feasibility(
        data,
        base,
        method=method,
        max_lag=max_lag,
        holdout_fraction=holdout_fraction,
        budget=budget,
        policy=policy,
    )
    if not feasibility.feasible:
        raise ValueError("Adstock optimization is not feasible: " + " ".join(feasibility.failures))
    fit, validation, test = partition(data, base.date_column, holdout_fraction)
    development = data.iloc[: len(fit) + len(validation)]
    n_warmup = feasibility.n_warmup
    fit_data = fit.iloc[n_warmup:]
    development_after_warmup = development.iloc[n_warmup:]
    candidates = [{}]
    candidates.extend(candidate_specs(base.channels, method, max_lag, budget))
    seen: set[str] = set()
    rows = []
    for settings in candidates:
        config = base.model_copy(update={"adstock": settings})
        key = config.model_dump_json()
        if key in seen or key not in feasibility.candidate_keys:
            continue
        seen.add(key)
        row = {
            "config": config.model_dump(mode="json"),
            "n_fit": feasibility.n_fit,
            "n_validation": feasibility.n_validation,
            "status": "ok",
        }
        try:
            fit_config = _config_after_warmup(fit, config, n_warmup)
            fitted = OLSModelEngine().fit(fit_data, fit_config)
            fit_design = _design_matrix(fit_data, fit_config, require_positive_target=False)
            fit_predicted = _predict(fit_design, pd.Series(fitted.coefficients), config)
            design = _design_matrix(
                development_after_warmup, fit_config, require_positive_target=False
            ).iloc[len(fit_data) :]
            predicted = _predict(design, pd.Series(fitted.coefficients), config)
            metrics = _error_metrics(validation[base.target].to_numpy(dtype=float), predicted)
            row.update(
                validation_rmse=metrics.rmse,
                validation_mae=metrics.mae,
                fit_predictions=fit_predicted.tolist(),
                validation_predictions=predicted.tolist(),
                coefficients=fitted.coefficients,
                standard_errors=fitted.standard_errors,
                confidence_intervals=fitted.confidence_intervals,
                p_values=fitted.p_values,
                significance_codes=fitted.significance_codes,
                diagnostic_flags=[flag.model_dump(mode="json") for flag in fitted.diagnostic_flags],
                condition_number=fitted.condition_number,
                design_rank=fitted.design_rank,
                parameter_count=fitted.parameter_count,
                residual_degrees_freedom=fitted.residual_degrees_freedom,
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
        policy=policy,
    )
    return {
        "data_sha256": dataset_sha256(data),
        "search_specification": specification,
        "method": "chronological fit/validation selection; final test requires a separate action",
        "normalization": "raw",
        "warm_up": (
            f"{feasibility.n_warmup} initial fit weeks initialize adstock and are excluded "
            "from coefficient fitting; zero prehistory and observed past spend carry across "
            "split boundaries"
        ),
        "partitions": {
            name: {
                "n": len(frame),
                "n_usable": usable,
                "first": frame[base.date_column].iloc[0].isoformat(),
                "last": frame[base.date_column].iloc[-1].isoformat(),
            }
            for name, frame, usable in (
                ("fit", fit, feasibility.n_fit),
                ("validation", validation, feasibility.n_validation),
                ("test", test, feasibility.n_test),
            )
        },
        "feasibility": feasibility.to_dict(),
        "candidates": rows,
        "candidate_budget": budget,
        "evaluated_candidates": len(rows),
        "baseline": next((row for row in rows if not row["config"]["adstock"]), None),
        "selected": selected,
        "final_test": None,
        "final_test_predictions": None,
        "final_test_comparison": None,
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
    policy = FeasibilityPolicy(**specification["feasibility_policy"])
    feasibility = assess_feasibility(
        data,
        base,
        method=specification["candidate_method"],
        max_lag=specification["max_lag"],
        holdout_fraction=specification["final_holdout_fraction"],
        budget=specification["candidate_budget"],
        policy=policy,
    )
    if not feasibility.feasible:
        raise ValueError("Final test is not feasible: " + " ".join(feasibility.failures))
    fit_data = fit.iloc[feasibility.n_warmup :]

    def evaluate(candidate: dict | None) -> dict | None:
        if candidate is None:
            return None
        candidate_config = ModelConfig.model_validate(candidate["config"])
        fit_config = _config_after_warmup(fit, candidate_config, feasibility.n_warmup)
        fitted = OLSModelEngine().fit(fit_data, fit_config)
        design = _design_matrix(
            data.iloc[feasibility.n_warmup :], fit_config, require_positive_target=False
        ).iloc[len(fit_data) + len(validation) :]
        prediction = _predict(design, pd.Series(fitted.coefficients), candidate_config)
        metrics = _error_metrics(test[base.target].to_numpy(dtype=float), prediction)
        return {"metrics": metrics.model_dump(), "predictions": prediction.tolist()}

    selected_evaluation = evaluate(result["selected"])
    baseline = result.get("baseline")
    baseline_evaluation = (
        evaluate(baseline) if baseline and baseline.get("status") == "ok" else None
    )
    assert selected_evaluation is not None
    evaluated = deepcopy(result)
    evaluated["final_test"] = selected_evaluation["metrics"]
    evaluated["final_test_predictions"] = selected_evaluation["predictions"]
    evaluated["final_test_comparison"] = {
        "baseline": baseline_evaluation,
        "selected": selected_evaluation,
    }
    return evaluated
