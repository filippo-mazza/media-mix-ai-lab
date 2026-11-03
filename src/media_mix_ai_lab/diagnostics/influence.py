"""Leave-one-observation-out influence diagnostics for a stored OLS fit.

Everything here describes how sensitive one *fitted regression* is to one regression
row. It never removes rows, enforces signs, or optimizes p-values, and it does not
measure causal contribution, ROI, feature importance, or data errors.
"""

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from media_mix_ai_lab.core.config import ModelConfig
from media_mix_ai_lab.modeling.interfaces import ModelResult
from media_mix_ai_lab.modeling.ols import OLSModelEngine, build_design_matrix

DFBETAS_MULTIPLIER = 2.0
COOKS_MULTIPLIER = 4.0
STUDENTIZED_RESIDUAL_GUIDE = 3.0

SIGN_CONVENTION = (
    "statsmodels DFBETAS = (full-sample coefficient − leave-one-out coefficient) divided by "
    "the leave-one-out residual standard deviation times sqrt of the coefficient's diagonal "
    "entry of (X'X)⁻¹. A positive value means including the row pushes the coefficient up."
)
INTERPRETATION = (
    "DFBETAS measures how sensitive one fitted coefficient is to one regression row. It is not "
    "a causal contribution, ROI, feature importance, or a claim that the row is erroneous. "
    "Deleting one row changes this fitted regression, not the data-generating process."
)
HEURISTIC_NOTE = (
    "Screening guides (|DFBETAS| > c/√N, Cook's D > c/N) are heuristics for deciding where to "
    "look, not significance tests or deletion rules. They assume a well-specified model with "
    "independent errors and can mislead under serial dependence or heavy tails."
)


class InfluenceError(ValueError):
    """The requested influence diagnostics cannot be computed faithfully."""


@dataclass(frozen=True)
class InfluenceThresholds:
    """Configurable screening multipliers; heuristics only."""

    dfbetas_multiplier: float = DFBETAS_MULTIPLIER
    cooks_multiplier: float = COOKS_MULTIPLIER

    def __post_init__(self) -> None:
        for name in ("dfbetas_multiplier", "cooks_multiplier"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number.")

    def dfbetas(self, n_observations: int) -> float:
        return self.dfbetas_multiplier / float(np.sqrt(n_observations))

    def cooks(self, n_observations: int) -> float:
        return self.cooks_multiplier / n_observations


@dataclass(frozen=True)
class InfluenceResult:
    """Row-by-term influence matrices with exact observation mapping and provenance.

    `row_position` (0-based) indexes the dataset the model was fitted on; all frames
    are in that chronological row order and are never sorted, filtered, or imputed.
    """

    config: ModelConfig
    dataset_sha256: str
    n_observations: int
    n_excluded: int
    n_parameters: int
    residual_degrees_freedom: int
    terms: tuple[str, ...]
    dfbetas: pd.DataFrame
    full_coefficients: pd.Series
    deleted_coefficients: pd.DataFrame
    diagnostics: pd.DataFrame
    raw_inputs: pd.DataFrame
    design: pd.DataFrame
    package_versions: dict[str, str]
    warnings: tuple[str, ...] = field(default=())

    @property
    def date_range(self) -> tuple[str, str] | None:
        dates = self.diagnostics["date"].dropna()
        return None if dates.empty else (str(dates.iloc[0]), str(dates.iloc[-1]))


def _iso(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)  # type: ignore[attr-defined]


def compute_influence(data: pd.DataFrame, model: ModelResult) -> InfluenceResult:
    """Compute DFBETAS and companion diagnostics for the exact sample a stored model used.

    The frame must match the stored model's dataset fingerprint; this never refits on a
    different sample. Undefined values stay NaN and are reported in `warnings`.
    """
    config = model.config
    if OLSModelEngine._fingerprint(data) != model.dataset_sha256:
        raise InfluenceError(
            "The loaded dataset no longer matches this stored model's dataset fingerprint. "
            "Refit the model on the current dataset; influence is not computed on a "
            "different sample."
        )
    y, design, _ = build_design_matrix(data, config)
    n_obs, n_par = design.shape
    if n_obs - n_par < 2:
        raise InfluenceError(
            f"Influence diagnostics need at least 2 residual degrees of freedom "
            f"(N={n_obs}, parameters={n_par}); leave-one-out fits are not identifiable."
        )
    rank = int(np.linalg.matrix_rank(design.to_numpy(dtype=float)))
    if rank < n_par:
        raise InfluenceError(
            f"Rank-deficient design matrix (rank {rank} for {n_par} columns); influence "
            "diagnostics are undefined."
        )
    fitted = sm.OLS(y, design, hasconst=config.intercept).fit()
    stored = np.array([model.coefficients[name] for name in design.columns], dtype=float)
    if not np.allclose(fitted.params.to_numpy(), stored, rtol=1e-6, atol=1e-9):
        raise InfluenceError(
            "Recomputed coefficients do not reproduce the stored model; refit the model."
        )

    warnings: list[str] = []
    with np.errstate(divide="ignore", invalid="ignore"):
        influence = fitted.get_influence()
        dfbetas = np.asarray(influence.dfbetas, dtype=float)
        cooks = np.asarray(influence.cooks_distance[0], dtype=float)
        leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
        student = np.asarray(influence.resid_studentized_external, dtype=float)
        dffits = np.asarray(influence.dffits[0], dtype=float)
        x = design.to_numpy(dtype=float)
        residuals = np.asarray(fitted.resid, dtype=float)
        xtx_inv = np.asarray(fitted.cov_params() / fitted.scale, dtype=float)
        # Exact closed-form leave-one-out coefficient: b - (X'X)^-1 x_i e_i / (1 - h_i).
        shift = (x @ xtx_inv) * (residuals / (1.0 - leverage))[:, None]
        deleted = fitted.params.to_numpy()[None, :] - shift

    # A unit-leverage row alone determines part of the fit: its leave-one-out fit is
    # singular, so numerically finite leftovers are meaningless and are set to missing.
    singular = leverage >= 1.0 - 1e-10
    for values in (dfbetas, cooks, student, dffits, deleted):
        values[singular] = np.nan

    terms = tuple(str(name) for name in design.columns)
    positions = np.arange(n_obs)
    dates = (
        [_iso(value) for value in data[config.date_column]]
        if config.date_column in data.columns
        else [None] * n_obs
    )
    base = pd.DataFrame({"row_position": positions, "date": dates})
    dfbetas_frame = base.join(pd.DataFrame(dfbetas, columns=list(terms)))
    deleted_frame = base.join(pd.DataFrame(deleted, columns=list(terms)))
    predicted = np.asarray(fitted.fittedvalues, dtype=float)
    diagnostics = base.assign(
        observed=y.to_numpy(dtype=float),
        predicted=predicted,
        residual=residuals,
        cooks_distance=cooks,
        leverage=leverage,
        studentized_residual=student,
        dffits=dffits,
    )
    for name, values in (
        ("DFBETAS", dfbetas),
        ("Cook's distance", cooks),
        ("leverage", leverage),
        ("studentized residual", student),
        ("DFFITS", dffits),
        ("leave-one-out coefficient", deleted),
    ):
        bad = int(values.size - np.isfinite(values).sum())
        if bad:
            warnings.append(
                f"{bad} {name} value(s) are undefined or infinite (e.g. a leverage of 1, a "
                "perfect fit, or a singular deletion fit). They are shown as missing; no "
                "observation was dropped."
            )
    if singular.any():
        warnings.append(
            "At least one observation has leverage ≈ 1: it alone determines part of the fit, "
            "so its leave-one-out regression is singular."
        )
    if config.kind == "log_log":
        warnings.append(
            "Log-log model: coefficients are elasticities conditional on the model's "
            "assumptions; diagnostics are on the log-outcome scale."
        )
    if any(spec.method != "none" for spec in config.adstock.values()):
        warnings.append(
            "Adstock features: each transformed regressor depends on historical media, so "
            "deleting a regression row is not equivalent to removing that week's spend from "
            "the adstock history. Deleted-fit values hold the transformed features fixed."
        )
    raw_columns = [
        column
        for column in (config.date_column, config.target, *config.channels, *config.controls)
        if column in data.columns
    ]
    raw_inputs = data.loc[:, raw_columns].reset_index(drop=True)
    raw_inputs.insert(0, "row_position", positions)
    return InfluenceResult(
        config=config,
        dataset_sha256=model.dataset_sha256,
        n_observations=n_obs,
        n_excluded=model.observation_accounting.n_excluded or 0,
        n_parameters=n_par,
        residual_degrees_freedom=n_obs - n_par,
        terms=terms,
        dfbetas=dfbetas_frame,
        full_coefficients=pd.Series(fitted.params.to_numpy(dtype=float), index=list(terms)),
        deleted_coefficients=deleted_frame,
        diagnostics=diagnostics,
        raw_inputs=raw_inputs,
        design=design.reset_index(drop=True),
        package_versions=dict(model.package_versions),
        warnings=tuple(warnings),
    )


def ranked_observations(
    result: InfluenceResult, thresholds: InfluenceThresholds | None = None
) -> pd.DataFrame:
    """Rows ranked by max |DFBETAS| (ties by chronological position; undefined last)."""
    thresholds = thresholds or InfluenceThresholds()
    values = result.dfbetas.loc[:, list(result.terms)].to_numpy(dtype=float)
    absolute = np.abs(values)
    with np.errstate(invalid="ignore"):
        finite = np.where(np.isnan(absolute), -np.inf, absolute)
        max_abs = np.where(np.isnan(absolute).all(axis=1), np.nan, finite.max(axis=1))
        top = finite.argmax(axis=1)
    table = result.diagnostics.loc[
        :, ["row_position", "date", "cooks_distance", "leverage", "studentized_residual", "dffits"]
    ].copy()
    table.insert(2, "max_abs_dfbetas", max_abs)
    table.insert(3, "most_influenced_term", np.array(result.terms, dtype=object)[top])
    table["exceeds_dfbetas_guide"] = np.nan_to_num(max_abs) > thresholds.dfbetas(
        result.n_observations
    )
    table["exceeds_cooks_guide"] = np.nan_to_num(table["cooks_distance"].to_numpy()) > (
        thresholds.cooks(result.n_observations)
    )
    for position, term in enumerate(result.terms):
        table[f"dfbetas:{term}"] = values[:, position]
    return table.sort_values(
        "max_abs_dfbetas", ascending=False, kind="stable", na_position="last"
    ).reset_index(drop=True)


def observation_detail(result: InfluenceResult, row_position: int) -> dict[str, object]:
    """Raw inputs, transformed features, fit values, diagnostics and beta comparison."""
    if not 0 <= row_position < result.n_observations:
        raise ValueError(f"row_position must be between 0 and {result.n_observations - 1}.")
    diagnostics = result.diagnostics.iloc[row_position]
    comparison = pd.DataFrame(
        {
            "term": list(result.terms),
            "full_sample_beta": result.full_coefficients.to_numpy(),
            "leave_one_out_beta": result.deleted_coefficients.loc[
                row_position, list(result.terms)
            ].to_numpy(dtype=float),
            "dfbetas": result.dfbetas.loc[row_position, list(result.terms)].to_numpy(dtype=float),
        }
    )
    comparison["change_when_included"] = (
        comparison["full_sample_beta"] - comparison["leave_one_out_beta"]
    )
    return {
        "row_position": row_position,
        "date": diagnostics["date"],
        "raw_inputs": result.raw_inputs.iloc[row_position].drop("row_position").to_dict(),
        "transformed_features": result.design.iloc[row_position].to_dict(),
        "observed_model_scale": float(diagnostics["observed"]),
        "predicted_model_scale": float(diagnostics["predicted"]),
        "residual": float(diagnostics["residual"]),
        "diagnostics": {
            key: float(diagnostics[key])
            for key in ("cooks_distance", "leverage", "studentized_residual", "dffits")
        },
        "beta_comparison": comparison,
    }


def export_records(table: pd.DataFrame) -> list[dict[str, object]]:
    """JSON-safe records keeping full float precision; undefined values become null."""
    records: list[dict[str, object]] = []
    for row in table.to_dict(orient="records"):
        records.append(
            {
                key: (
                    None
                    if isinstance(value, float) and not np.isfinite(value)
                    else bool(value)
                    if isinstance(value, np.bool_)
                    else value.item()
                    if isinstance(value, np.generic)
                    else value
                )
                for key, value in row.items()
            }
        )
    return records


def export_json(table: pd.DataFrame) -> str:
    return json.dumps(export_records(table), indent=2, allow_nan=False)


def export_csv(table: pd.DataFrame) -> str:
    """Full-precision CSV (pandas writes the shortest round-trip float repr)."""
    return table.to_csv(index=False)
