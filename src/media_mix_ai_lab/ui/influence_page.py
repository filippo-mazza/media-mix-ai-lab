"""Streamlit page for leave-one-observation-out sample influence (DFBETAS) diagnostics."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from media_mix_ai_lab.diagnostics.influence import (
    HEURISTIC_NOTE,
    INTERPRETATION,
    SIGN_CONVENTION,
    STUDENTIZED_RESIDUAL_GUIDE,
    InfluenceError,
    InfluenceResult,
    InfluenceThresholds,
    compute_influence,
    export_csv,
    export_json,
    observation_detail,
    ranked_observations,
)
from media_mix_ai_lab.modeling.interfaces import ModelResult
from media_mix_ai_lab.modeling.ols import ModelFitError

OUTLIER_SENSITIVITY_ISSUE = "https://github.com/filippo-mazza/media-mix-ai-lab/issues/30"
CHRONOLOGICAL = "Chronological (default)"
BY_INFLUENCE = "Max |DFBETAS|, descending"


def _back_to_models() -> None:
    st.session_state["active_section"] = "Models"


def _day(value: object) -> str:
    return "n/a" if value is None or pd.isna(value) else str(value)[:10]


def _label(row: int, date: object) -> str:
    return f"row {row} · {_day(date)}"


def sample_influence_page() -> None:
    """Investigate which fitted-sample rows most change each coefficient."""
    st.title("Sample Influence")
    st.caption("Models → Sample Influence")
    st.button("← Back to Models", on_click=_back_to_models, key="influence_back_to_models")
    st.write(
        "Find which weekly observations most change each coefficient of a stored fitted "
        "regression. No observation is removed, down-weighted, or refitted away."
    )
    model_runs = st.session_state.get("model_results", [])
    if not model_runs:
        st.info(
            "No fitted model is available. Fit a model on the Models page, then return here "
            "to inspect sample influence."
        )
        return

    frame = st.session_state["dataset"]
    models = [ModelResult.model_validate_json(model_json) for model_json in model_runs]
    labels = [
        f"Fit {index} · {model.config.kind} · {model.dataset_sha256[:8]}"
        for index, model in enumerate(models, start=1)
    ]
    selected = st.selectbox(
        "Stored model run",
        labels,
        index=len(labels) - 1,
        help="Diagnostics use this stored model's exact dataset fingerprint; nothing is refit.",
    )
    model_index = labels.index(selected)
    model = models[model_index]
    try:
        result = compute_influence(frame, model)
    except (InfluenceError, ModelFitError, ValueError) as error:
        st.error(f"Sample influence is unavailable for {selected}: {error}")
        return

    _show_model_summary(selected, model, result)
    with st.expander("How to read these diagnostics", expanded=True):
        st.markdown(f"**Interpretation.** {INTERPRETATION}")
        st.markdown(f"**Sign convention.** {SIGN_CONVENTION}")
        st.markdown(f"**Heuristics.** {HEURISTIC_NOTE}")
        st.markdown(
            "**No automatic cleaning.** This page never removes samples, enforces coefficient "
            "signs, or optimizes p-values. For separately controlled, capped sensitivity "
            f"analysis see the Models page and [outlier sensitivity issue]"
            f"({OUTLIER_SENSITIVITY_ISSUE})."
        )
        st.markdown(
            "**Log-log and adstock.** In log-log models a beta is an elasticity conditional on "
            "model assumptions. With adstock, transformed regressors depend on historical "
            "media, so deleting a regression row is not equivalent to removing that week's "
            "spend from the adstock history."
        )
    for warning in result.warnings:
        st.warning(warning)

    left, right = st.columns(2)
    dfbetas_multiplier = left.number_input(
        "DFBETAS guide multiplier c (|DFBETAS| > c/√N)",
        min_value=0.1,
        value=2.0,
        step=0.1,
        key=f"influence_dfbetas_c_{model_index}",
    )
    cooks_multiplier = right.number_input(
        "Cook's D guide multiplier c (D > c/N)",
        min_value=0.1,
        value=4.0,
        step=0.1,
        key=f"influence_cooks_c_{model_index}",
    )
    thresholds = InfluenceThresholds(dfbetas_multiplier, cooks_multiplier)
    ranked = ranked_observations(result, thresholds)

    _heatmap_section(result, thresholds, model_index)
    _ranked_section(ranked, result, thresholds)
    default_row = int(ranked["row_position"].iloc[0])
    rows = result.diagnostics["row_position"].tolist()
    dates = dict(zip(result.diagnostics["row_position"], result.diagnostics["date"], strict=True))
    selected_row = st.selectbox(
        "Selected observation",
        rows,
        index=rows.index(default_row),
        format_func=lambda row: _label(row, dates[row]),
        key=f"influence_observation_{model_index}",
        help="Defaults to the row with the largest absolute DFBETAS; used in the plots below.",
    )
    _coefficient_section(result, thresholds, selected_row, model_index)
    _residual_leverage_section(result, thresholds, selected_row)
    _drill_down_section(result, selected_row)


def _show_model_summary(label: str, model: ModelResult, result: InfluenceResult) -> None:
    config = result.config
    st.subheader("Selected model")
    columns = st.columns(4)
    columns[0].metric("Fit N", result.n_observations)
    columns[1].metric("Excluded / unavailable rows", result.n_excluded)
    columns[2].metric("Parameters", result.n_parameters)
    columns[3].metric("Residual df", result.residual_degrees_freedom)
    date_range = result.date_range
    st.markdown(
        f"**Model:** {label} · `{model.dataset_sha256}`  \n"
        f"**Specification:** {config.kind}; outcome `{config.target}`  \n"
        f"**Channels:** {', '.join(config.channels)}  \n"
        f"**Controls:** {', '.join(config.controls) or 'none'}  \n"
        f"**Date range:** "
        f"{'n/a' if date_range is None else f'{_day(date_range[0])} to {_day(date_range[1])}'}  \n"
        f"**Transformed feature names:** {', '.join(result.terms)}"
    )
    st.caption(
        "Analysis covers the fitted (training) sample only; no validation or test rows are "
        f"mixed in. Equation: {model.equation}"
    )


def _heatmap_section(
    result: InfluenceResult, thresholds: InfluenceThresholds, model_index: int
) -> None:
    st.subheader("DFBETAS heatmap")
    terms = st.multiselect(
        "Coefficients shown",
        list(result.terms),
        default=list(result.terms),
        key=f"influence_terms_{model_index}",
    )
    all_days = [_day(date) for date in result.diagnostics["date"]]
    ordered_days = sorted(set(all_days))
    start, end = st.select_slider(
        "Date range",
        options=ordered_days,
        value=(ordered_days[0], ordered_days[-1]),
        key=f"influence_dates_{model_index}",
    )
    order = st.radio(
        "Row order",
        [CHRONOLOGICAL, BY_INFLUENCE],
        horizontal=True,
        key=f"influence_order_{model_index}",
        help="Chronological order is the default so time patterns are not hidden by sorting.",
    )
    minimum = st.number_input(
        "Show rows with max |DFBETAS| at least",
        min_value=0.0,
        value=0.0,
        step=0.1,
        key=f"influence_min_{model_index}",
    )
    if not terms:
        st.info("Select at least one coefficient to draw the heatmap.")
        return
    matrix = result.dfbetas.loc[:, list(terms)].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        row_max = np.nanmax(np.abs(matrix), axis=1) if matrix.size else np.array([])
    keep = np.array([start <= day <= end for day in all_days], dtype=bool) & (
        np.nan_to_num(row_max, nan=0.0) >= minimum
    )
    positions = np.flatnonzero(keep)
    if positions.size == 0:
        st.info("No observations match the current filters.")
        return
    if order == BY_INFLUENCE:
        positions = positions[
            np.argsort(-np.nan_to_num(row_max[positions], nan=-1.0), kind="stable")
        ]
    labels = [
        _label(int(result.diagnostics["row_position"].iloc[p]), all_days[p]) for p in positions
    ]
    values = matrix[positions]
    full = result.full_coefficients
    hover = np.empty(values.shape, dtype=object)
    for r, position in enumerate(positions):
        for c, term in enumerate(terms):
            hover[r, c] = (
                f"{labels[r]}<br>coefficient: {term}<br>"
                f"DFBETAS: {values[r, c]:.4g}<br>"
                f"raw beta: {full[term]:.6g}<br>"
                f"deleted-fit beta: "
                f"{result.deleted_coefficients.loc[position, term]:.6g}"
            )
    bound = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    bound = max(bound, 1e-12)
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=list(terms),
            y=labels,
            zmin=-bound,
            zmax=bound,
            zmid=0,
            colorscale="RdBu",
            reversescale=True,
            text=hover,
            hoverinfo="text",
            colorbar={"title": "DFBETAS<br>(+ = beta higher<br>with row included)"},
        )
    )
    figure.update_yaxes(type="category", autorange="reversed", title="Observation")
    figure.update_xaxes(title="Coefficient / feature")
    figure.update_layout(
        title=f"DFBETAS by observation (guide: ±{thresholds.dfbetas(result.n_observations):.3g})",
        height=max(400, min(1600, 18 * len(labels) + 160)),
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"{len(labels)} of {result.n_observations} observations shown. Color is signed and "
        "centered at zero; blank cells are undefined. Rounding applies to the chart only."
    )


def _ranked_section(
    ranked: pd.DataFrame, result: InfluenceResult, thresholds: InfluenceThresholds
) -> None:
    st.subheader("Ranked observations")
    st.caption(
        f"Sorted by max |DFBETAS| across coefficients. Guides: |DFBETAS| > "
        f"{thresholds.dfbetas(result.n_observations):.4g}, Cook's D > "
        f"{thresholds.cooks(result.n_observations):.4g} (heuristics only)."
    )
    st.dataframe(ranked, hide_index=True, width="stretch")
    csv_column, json_column = st.columns(2)
    csv_column.download_button(
        "Download ranked table (CSV)",
        export_csv(ranked),
        file_name=f"sample-influence-{result.dataset_sha256[:12]}.csv",
        mime="text/csv",
        key="influence_download_csv",
    )
    json_column.download_button(
        "Download ranked table (JSON)",
        export_json(ranked),
        file_name=f"sample-influence-{result.dataset_sha256[:12]}.json",
        mime="application/json",
        key="influence_download_json",
    )


def _coefficient_section(
    result: InfluenceResult, thresholds: InfluenceThresholds, selected_row: int, model_index: int
) -> None:
    st.subheader("Per-coefficient influence")
    term = st.selectbox("Coefficient", list(result.terms), key=f"influence_coef_{model_index}")
    by_magnitude = st.checkbox(
        "Rank observations by absolute magnitude", key=f"influence_rank_{model_index}"
    )
    plot = result.dfbetas.loc[:, ["row_position", "date", term]].rename(columns={term: "dfbetas"})
    plot["label"] = [_label(r, d) for r, d in zip(plot["row_position"], plot["date"], strict=True)]
    if by_magnitude:
        plot = plot.assign(_abs=plot["dfbetas"].abs()).sort_values(
            "_abs", ascending=False, kind="stable", na_position="last"
        )
    colors = ["#d62728" if row == selected_row else "#087e8b" for row in plot["row_position"]]
    figure = go.Figure(
        go.Bar(
            x=plot["label"],
            y=plot["dfbetas"],
            marker_color=colors,
            hovertemplate="%{x}<br>DFBETAS: %{y:.4g}<extra></extra>",
        )
    )
    guide = thresholds.dfbetas(result.n_observations)
    for level in (guide, -guide):
        figure.add_hline(y=level, line_dash="dash", line_color="#6b7c93")
    figure.update_xaxes(type="category", title="Observation (selected in red)")
    figure.update_yaxes(title=f"DFBETAS for {term}")
    figure.update_layout(title=f"Signed DFBETAS for {term} (dashed: ±{guide:.3g} guide)")
    st.plotly_chart(figure, width="stretch")


def _residual_leverage_section(
    result: InfluenceResult, thresholds: InfluenceThresholds, selected_row: int
) -> None:
    st.subheader("Cook's distance, leverage and residuals")
    table = result.diagnostics.copy()
    cooks_guide = thresholds.cooks(result.n_observations)
    table["label"] = [
        _label(r, d) for r, d in zip(table["row_position"], table["date"], strict=True)
    ]
    table["status"] = np.where(
        table["row_position"] == selected_row,
        "selected",
        np.where(table["cooks_distance"].fillna(0) > cooks_guide, "Cook's D above guide", "other"),
    )
    table["marker_size"] = table["cooks_distance"].fillna(0).clip(lower=0) + 1e-6
    figure = px.scatter(
        table,
        x="leverage",
        y="studentized_residual",
        size="marker_size",
        color="status",
        hover_name="label",
        hover_data={"cooks_distance": ":.4g", "marker_size": False},
        title="Externally studentized residual vs leverage (marker size = Cook's D)",
        color_discrete_map={
            "selected": "#d62728",
            "Cook's D above guide": "#f2a900",
            "other": "#087e8b",
        },
    )
    figure.add_vline(
        x=2 * result.n_parameters / result.n_observations,
        line_dash="dash",
        line_color="#6b7c93",
    )
    for level in (STUDENTIZED_RESIDUAL_GUIDE, -STUDENTIZED_RESIDUAL_GUIDE):
        figure.add_hline(y=level, line_dash="dot", line_color="#6b7c93")
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "High leverage (right) means an unusual predictor combination; a large |studentized "
        "residual| (vertical) means an unusual outcome given predictors. Guides: leverage "
        f"2p/N, |t| {STUDENTIZED_RESIDUAL_GUIDE:g}, Cook's D {cooks_guide:.4g}; heuristics only."
    )


def _drill_down_section(result: InfluenceResult, row: int) -> None:
    detail = observation_detail(result, row)
    st.subheader(f"Observation drill-down: {_label(row, detail['date'])}")
    raw, transformed = st.columns(2)
    raw.markdown("**Raw inputs and outcome (original dataset)**")
    raw.dataframe(
        pd.DataFrame(
            {"value": [str(value) for value in detail["raw_inputs"].values()]},  # type: ignore[attr-defined]
            index=list(detail["raw_inputs"]),  # type: ignore[arg-type]
        ),
        width="stretch",
    )
    transformed.markdown("**Transformed model features (design matrix row)**")
    transformed.dataframe(
        pd.DataFrame(
            {"value": list(detail["transformed_features"].values())},  # type: ignore[attr-defined]
            index=list(detail["transformed_features"]),  # type: ignore[arg-type]
        ),
        width="stretch",
    )
    columns = st.columns(3)
    columns[0].metric("Observed (model scale)", f"{detail['observed_model_scale']:.6g}")
    columns[1].metric("Predicted (model scale)", f"{detail['predicted_model_scale']:.6g}")
    columns[2].metric("Residual", f"{detail['residual']:.6g}")
    st.dataframe(
        pd.DataFrame([detail["diagnostics"]]),
        hide_index=True,
        width="stretch",
    )
    st.markdown("**Original vs leave-one-out coefficients**")
    st.dataframe(detail["beta_comparison"], hide_index=True, width="stretch")  # type: ignore[arg-type]
    st.caption(
        "Leave-one-out betas are the exact coefficients from refitting without this regression "
        "row, holding every other row's transformed features fixed. They describe sensitivity "
        "of this fitted regression, not causal effect."
    )
