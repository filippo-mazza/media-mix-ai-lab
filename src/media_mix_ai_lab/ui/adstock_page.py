"""Interactive adstock exploration and bounded calibration UI."""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from media_mix_ai_lab.core.config import AdstockSpec, ModelConfig
from media_mix_ai_lab.modeling.adstock import adstock_impulse_response, apply_adstock
from media_mix_ai_lab.optimization.adstock_search import (
    candidate_specs,
    dataset_sha256,
    partition,
    search,
)


def adstock_page() -> None:
    st.title("Adstock Analysis")
    frame = st.session_state["dataset"]
    mapping = st.session_state["dataset_mapping"]
    channels = tuple(mapping["channels"])
    explore, optimize = st.tabs(["Explore adstock", "Optimize adstock"])
    with explore:
        channel = st.selectbox("Channel", channels)
        family = st.selectbox("Adstock family", ["none", "geometric", "delayed_geometric"])
        alpha = st.slider("Geometric alpha / delayed alpha", 0.0, 0.9, 0.5, 0.1)
        lag = st.slider("Lag horizon (weeks)", 0, 12, 8)
        theta = st.slider("Delayed peak (weeks)", 0, lag, min(2, lag))
        scaling = st.selectbox("Weight normalization", ["raw", "sum_to_one"])
        spec = AdstockSpec(
            method=family,
            alpha=max(alpha, 0.1) if family == "delayed_geometric" else alpha,
            theta=theta,
            max_lag_weeks=lag,
            weight_scaling=scaling,
        )
        weights = adstock_impulse_response(spec, periods=lag + 1)
        st.plotly_chart(
            px.line(
                weights,
                x="Lag (weeks)",
                y="Weight",
                markers=True,
                title="Theoretical impulse response · no causal claim",
            ),
            width="stretch",
        )
        try:
            transformed = apply_adstock(frame[channel], spec)
        except ValueError as error:
            st.warning(f"Spend preview unavailable: {error}")
        else:
            plot = pd.DataFrame(
                {
                    mapping["date"]: frame[mapping["date"]],
                    "Raw spend": frame[channel],
                    "Transformed spend": transformed,
                }
            )
            st.plotly_chart(
                px.line(
                    plot,
                    x=mapping["date"],
                    y=["Raw spend", "Transformed spend"],
                    title=f"{channel} · weekly spend units · zero prehistory",
                ),
                width="stretch",
            )
        half_life = np.log(0.5) / np.log(alpha) if family == "geometric" and 0 < alpha < 1 else None
        st.caption(
            f"{channel}: {family}, alpha={spec.alpha:g}, theta={theta}, L={lag}, "
            f"normalization={scaling}; geometric half-life: "
            f"{half_life:.2f} weeks"
            if half_life is not None
            else (
                f"{channel}: {family}, alpha={spec.alpha:g}, theta={theta}, L={lag}, "
                f"normalization={scaling}; half-life undefined"
            )
        )
        st.session_state["adstock_exploration"] = spec.model_dump(mode="json")
    with optimize:
        selected_channels = tuple(
            st.multiselect("Channels to calibrate", channels, default=channels[:1])
        )
        method = st.selectbox("Candidate family", ["geometric", "delayed_geometric", "none"])
        horizon = st.slider("Search lag horizon", 0, 12, 8)
        budget = st.number_input("Candidate budget", min_value=1, max_value=100, value=20)
        holdout = st.slider("Final test fraction", 0.1, 0.4, 0.2, 0.05)
        kind = st.selectbox("Regression family", ["ols", "log_log"])
        if not selected_channels:
            st.info("Select at least one channel.")
            return
        config = ModelConfig(
            kind=kind,
            target=mapping["target"],
            channels=selected_channels,
            controls=tuple(mapping["controls"]),
            date_column=mapping["date"],
        )
        try:
            fit, validation, test = partition(frame, mapping["date"], holdout)
            cols = st.columns(4)
            for col, label, count in zip(
                cols,
                ["N total", "N fit", "N validation", "N final test"],
                [len(frame), len(fit), len(validation), len(test)],
                strict=True,
            ):
                col.metric(label, count)
            st.caption(
                " · ".join(
                    f"{name}: {part[mapping['date']].iloc[0].date()} to "
                    f"{part[mapping['date']].iloc[-1].date()}"
                    for name, part in (
                        ("Fit", fit),
                        ("Validation", validation),
                        ("Final test", test),
                    )
                )
            )
            count = len(candidate_specs(selected_channels, method, horizon, int(budget))) + 1
            st.caption(
                f"At most {count} candidate fits including no-adstock baseline. "
                "Warm-up exclusions: 0; zero prehistory, observed past spend carried across splits."
            )
        except ValueError as error:
            st.error(str(error))
            return
        if st.button("Optimize adstock parameters"):
            with st.spinner("Evaluating bounded candidate grid…"):
                try:
                    st.session_state["adstock_search"] = search(
                        frame,
                        config,
                        method=method,
                        max_lag=horizon,
                        budget=int(budget),
                        holdout_fraction=holdout,
                    )
                except ValueError as error:
                    st.error(f"Search failed: {error}")
        result = st.session_state.get("adstock_search")
        if result and result["data_sha256"] != dataset_sha256(frame):
            st.session_state.pop("adstock_search", None)
            st.session_state.pop("adstock_applied_config", None)
            result = None
            st.warning("The previous search belongs to a different dataset and was cleared.")
        if result:
            st.success("Search complete")
            rows = [
                {
                    "rank": index,
                    "adstock": row["config"]["adstock"],
                    "validation RMSE": row.get("validation_rmse"),
                    "validation MAE": row.get("validation_mae"),
                    "N fit": row["n_fit"],
                    "N validation": row["n_validation"],
                    "status": row["status"],
                    "warnings": row.get("warnings", []),
                    "reason": row.get("reason", ""),
                }
                for index, row in enumerate(result["candidates"])
            ]
            st.dataframe(
                pd.DataFrame(rows).sort_values("validation RMSE").head(5),
                hide_index=True,
                width="stretch",
            )
            comparison = pd.DataFrame(
                [
                    {
                        "Candidate": "Baseline"
                        if not row["config"]["adstock"]
                        else f"Candidate {index}",
                        "Validation RMSE": row["validation_rmse"],
                    }
                    for index, row in enumerate(result["candidates"])
                    if row["status"] == "ok"
                ]
            )
            st.plotly_chart(
                px.bar(
                    comparison,
                    x="Candidate",
                    y="Validation RMSE",
                    title="Validation error · baseline vs candidates",
                ),
                width="stretch",
            )
            st.metric("Final held-out RMSE · selected only", f"{result['final_test']['rmse']:.3f}")
            st.caption(result["caveat"])
            if st.button("Apply selected parameters"):
                selected = ModelConfig.model_validate(result["selected"]["config"])
                for mapped_channel in channels:
                    st.session_state[f"adstock_method_{mapped_channel}"] = "none"
                for selected_channel, selected_spec in selected.adstock.items():
                    st.session_state[f"adstock_method_{selected_channel}"] = selected_spec.method
                    st.session_state[f"adstock_scaling_{selected_channel}"] = (
                        selected_spec.weight_scaling
                    )
                    st.session_state[f"adstock_alpha_{selected_channel}"] = float(
                        selected_spec.alpha
                    )
                    if selected_spec.method == "geometric":
                        st.session_state[f"adstock_initial_state_{selected_channel}"] = float(
                            selected_spec.initial_state
                        )
                    elif selected_spec.method == "delayed_geometric":
                        st.session_state[f"adstock_lag_{selected_channel}"] = (
                            selected_spec.max_lag_weeks
                        )
                        st.session_state[f"adstock_theta_{selected_channel}"] = float(
                            selected_spec.theta
                        )
                        st.session_state[f"adstock_history_{selected_channel}"] = ",".join(
                            str(value) for value in selected_spec.historical_values
                        )
                st.session_state["adstock_applied_config"] = selected.model_dump(mode="json")
                st.success("Selected settings loaded into the editable Models controls.")
            st.download_button(
                "Download search manifest",
                json.dumps(result, indent=2),
                file_name="adstock-search.json",
                mime="application/json",
            )
