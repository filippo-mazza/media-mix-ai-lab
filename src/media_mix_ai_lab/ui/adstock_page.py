"""Interactive adstock exploration and bounded calibration UI."""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from media_mix_ai_lab.core.config import AdstockSpec, ModelConfig
from media_mix_ai_lab.modeling.adstock import adstock_impulse_response, apply_adstock
from media_mix_ai_lab.optimization.adstock_search import (
    assess_feasibility,
    candidate_specs,
    dataset_sha256,
    evaluate_final_test,
    partition,
    search,
    search_specification,
)
from media_mix_ai_lab.ui.help_content import HELP_TEXT, help_heading


def _adstock_introduction() -> None:
    st.subheader("What is adstock?")
    st.write(
        "Adstock is a deterministic way to represent advertising carryover: a media exposure "
        "this week can contribute to the transformed media signal in later weeks. This page "
        "uses weeks as the time unit and treats an impulse as one isolated unit of spend "
        "followed by weeks of zero spend."
    )
    st.write(
        "Carryover describes timing. It is separate from saturation or diminishing returns, "
        "which describe how response changes as spend grows. An adstock curve is also not a "
        "causal sales-response curve; it only transforms the media input used by a model."
    )


def _method_explanation() -> None:
    st.subheader("How the methods work")
    st.markdown(
        "**No adstock** — `A_t = x_t`. Transformed media equals observed weekly spend, so "
        "there is no carryover. This is always included as the optimization baseline.\n\n"
        "**Geometric adstock** — `A_t = x_t + alpha * A_(t-1)`, with "
        "`0 <= alpha < 1`. Alpha is the fraction retained for one more week; `alpha = 0` "
        "means no carryover. The implementation is recursive and therefore has an infinite "
        "tail. The live lag horizon limits only the displayed impulse plot. With raw weights, "
        "the impulse weights are `alpha^lag`; sum-to-one scaling multiplies the recursive "
        "signal by `1 - alpha`. For `0 < alpha < 1`, half-life is "
        "`ln(0.5) / ln(alpha)` weeks.\n\n"
        "**Delayed geometric adstock** — "
        "`A_t = sum(l=0..L) alpha^((l - theta)^2) * x_(t-l)`, with "
        "`0 < alpha < 1`. Theta is the peak delay in weeks and `L` is the last included lag. "
        "Here alpha controls the width around the peak: values nearer 1 spread weight more "
        "widely; it is not a one-week retention rate. Raw weights use the formula directly. "
        "Sum-to-one scaling divides the finite weights by their total."
    )
    with st.expander("First weeks, history, and normalization details"):
        st.write(
            "This page starts both transforms with zero media history. Geometric adstock uses "
            "an initial state of zero. Delayed geometric uses only lags available since the "
            "first displayed week, so unavailable pre-dataset terms contribute zero. The "
            "Models page can accept explicit prior delayed-media values. During chronological "
            "validation, observed earlier spend carries across split boundaries; future spend "
            "never contributes to an earlier feature. Normalization changes the numerical "
            "scale of the transformed signal and therefore the coefficient scale."
        )


def _impulse_guidance(family: str, scaling: str) -> None:
    st.subheader("How to read this graph")
    peak = (
        "Geometric weight peaks immediately at week 0 and then decays."
        if family == "geometric"
        else "Delayed geometric weight can peak after week 0 at the selected theta."
        if family == "delayed_geometric"
        else "No adstock has weight 1 at week 0 and no later carryover."
    )
    scale = (
        "The displayed weights are normalized across their implemented support."
        if scaling == "sum_to_one"
        else "The y-axis shows raw, unnormalized weights."
    )
    st.write(
        "The x-axis is weeks since one unit of media exposure; the y-axis is the weight that "
        f"the transformation assigns at each lag. {peak} {scale} These are theoretical input "
        "weights, not estimated sales effects or causal lift."
    )


def _worked_example() -> None:
    st.markdown("**Miniature impulse example (raw weights)**")
    st.write(
        "Suppose spend is 1 in week 0 and 0 afterward. Geometric alpha 0.5 decays "
        "immediately. Delayed geometric alpha 0.5, theta 2, and L=4 peaks in week 2."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Weeks since exposure": [0, 1, 2, 3, 4],
                "Spend": [1, 0, 0, 0, 0],
                "Geometric (alpha=0.5)": [1, 0.5, 0.25, 0.125, 0.0625],
                "Delayed (alpha=0.5, theta=2)": [0.0625, 0.5, 1, 0.5, 0.0625],
            }
        ),
        hide_index=True,
        width="stretch",
    )


def _optimization_explanation() -> None:
    st.subheader("Why optimize adstock?")
    st.write(
        "Optimization compares a bounded, deterministic set of channel-specific alpha and "
        "theta candidates with the no-adstock baseline. Each candidate fits coefficients on "
        "the earliest weeks and is ranked by validation RMSE in the outcome's original units; "
        "validation MAE is shown as a supporting diagnostic. Lower validation error means "
        "better prediction on that validation period, not a more causal or scientifically "
        "true model. Coefficient signs and p-values never choose the winner."
    )
    st.write(
        "The final weeks stay untouched while candidates are ranked. After selection is "
        "frozen, **Evaluate selected model on final test** uses that holdout once and locks "
        "further searches for the dataset. **Apply selected parameters** then copies the "
        "chosen settings into the editable Models controls; it does not overwrite the source "
        "data or an existing baseline fit."
    )
    st.subheader("When optimization is not feasible")
    st.write(
        "Let L be the candidate lag horizon (L=0 for no-adstock). The default minimum is fitted "
        "coefficients plus the larger of 5 residual degrees of freedom or L + 1 training weeks; "
        "validation and final test each need the larger of 5 weeks or L + 1. The first min(L, "
        "N fit) weeks initialize adstock history but are not fitted. These code-configurable "
        "defaults are conservative workbench heuristics, not statistically validated universal "
        "sample-size thresholds. Log-log models require positive fitted outcomes and positive "
        "transformed media in each partition; validation and test outcomes are scored in their "
        "original units. Invalid rows are not silently dropped."
    )
    st.caption(
        "Add historical weeks, shorten the lag horizon, reduce model complexity, or adjust the "
        "chronological split while retaining meaningful validation and test periods. Fix missing "
        "or irregular weeks at the source; the app does not silently impute or drop them."
    )


def _limitations() -> None:
    st.subheader("Assumptions & limitations")
    st.write(
        "The transforms assume weekly, regularly spaced observations and channel-specific "
        "carryover. Results depend on the available prehistory and initial-state convention. "
        "Correlated channels can make parameters difficult to identify, and a predictive "
        "winner can be unstable across periods. Delayed adstock truncates history at L; "
        "geometric adstock is recursive. Log-log fits cannot accept zero outcome or transformed "
        "media values. These observational models do not establish causal advertising impact."
    )


def adstock_page() -> None:
    st.title("Adstock Analysis")
    _adstock_introduction()
    frame = st.session_state["dataset"]
    mapping = st.session_state["dataset_mapping"]
    channels = tuple(mapping["channels"])
    explore, optimize = st.tabs(["Explore adstock", "Optimize adstock"])
    with explore:
        _method_explanation()
        st.markdown("#### Explore a channel and parameter set")
        channel = st.selectbox(
            "Channel",
            channels,
            help=HELP_TEXT["channel_columns"],
        )
        family = st.selectbox(
            "Adstock family",
            ["none", "geometric", "delayed_geometric"],
            help=HELP_TEXT["adstock_method"],
        )
        alpha = st.slider(
            "Geometric alpha / delayed alpha",
            0.0,
            0.9,
            0.5,
            0.1,
            help=(
                HELP_TEXT["adstock_alpha"]
                + " Geometric alpha is one-week retention; delayed alpha controls width "
                "around theta."
            ),
        )
        lag = st.slider(
            "Lag horizon (weeks)",
            0,
            12,
            8,
            help=(
                HELP_TEXT["adstock_max_lag"]
                if family == "delayed_geometric"
                else (
                    "Visible impulse-plot horizon only; geometric adstock has an infinite "
                    "recursive tail."
                )
            ),
        )
        theta = st.slider(
            "Delayed peak (weeks)",
            0,
            lag,
            min(2, lag),
            help=HELP_TEXT["adstock_theta"],
        )
        scaling = st.selectbox(
            "Weight normalization",
            ["raw", "sum_to_one"],
            help=HELP_TEXT["adstock_normalization"],
        )
        spec = AdstockSpec(
            method=family,
            alpha=max(alpha, 0.1) if family == "delayed_geometric" else alpha,
            theta=theta,
            max_lag_weeks=lag,
            weight_scaling=scaling,
        )
        weights = adstock_impulse_response(spec, periods=lag + 1)
        _impulse_guidance(family, scaling)
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
        _worked_example()
        st.subheader("How to read raw vs transformed spend")
        st.write(
            "Both lines use weekly spend-signal units, but normalization changes their scale. "
            "Carryover can keep transformed spend above zero when current spend is zero, and "
            "raw geometric weights can make transformed values larger than current spend. "
            "Every transformed point depends on the selected parameters and earlier history."
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
        st.caption(HELP_TEXT["adstock_half_life"])
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
        _optimization_explanation()
        selected_channels = tuple(
            st.multiselect(
                "Channels to calibrate",
                channels,
                default=channels[:1],
                help=HELP_TEXT["channel_columns"],
            )
        )
        method = st.selectbox(
            "Candidate family",
            ["geometric", "delayed_geometric", "none"],
            help=HELP_TEXT["adstock_method"],
        )
        horizon = st.slider("Search lag horizon", 0, 12, 8, help=HELP_TEXT["adstock_max_lag"])
        budget = st.number_input(
            "Candidate budget",
            min_value=1,
            max_value=100,
            value=20,
            help=HELP_TEXT["candidate_budget"],
        )
        holdout = st.slider(
            "Final test fraction",
            0.1,
            0.4,
            0.2,
            0.05,
            help=HELP_TEXT["train_validation_test"],
        )
        kind = st.selectbox(
            "Regression family",
            ["ols", "log_log"],
            help=HELP_TEXT["model_specification"],
        )
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
        current_sha256 = dataset_sha256(frame)
        final_test_used = st.session_state.get("adstock_final_evaluated_sha256") == current_sha256
        try:
            fit, validation, test = partition(frame, mapping["date"], holdout)
            help_heading(
                "Chronological fit, validation, and final test",
                "train_validation_test",
                key="adstock_optimization_partitions",
            )
            feasibility = assess_feasibility(
                frame,
                config,
                method=method,
                max_lag=horizon,
                holdout_fraction=holdout,
                budget=int(budget),
            )
            cols = st.columns(4)
            for col, label, count in zip(
                cols,
                ["N total", "N fit", "N validation", "N final test"],
                [
                    feasibility.n_total,
                    feasibility.n_fit,
                    feasibility.n_validation,
                    feasibility.n_test,
                ],
                strict=True,
            ):
                col.metric(label, count, help=HELP_TEXT["sample_counts"])
            st.caption(
                " · ".join(
                    f"{name}: {part[mapping['date']].iloc[0].date()} to "
                    f"{part[mapping['date']].iloc[-1].date()}"
                    if not part.empty
                    else f"{name}: no rows"
                    for name, part in (
                        ("Fit", fit),
                        ("Validation", validation),
                        ("Final test", test),
                    )
                )
            )
            current_specification = search_specification(
                frame,
                config,
                method=method,
                max_lag=horizon,
                budget=int(budget),
                holdout_fraction=holdout,
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Partition": name,
                            "Warm-up": feasibility.n_warmup if name == "Fit" else 0,
                            "Log-domain unusable": getattr(
                                feasibility, f"n_log_unusable_{name.lower()}"
                            ),
                            "Usable N": usable,
                            "Required minimum": required,
                        }
                        for name, usable, required in (
                            ("Fit", feasibility.n_fit, feasibility.required_fit),
                            (
                                "Validation",
                                feasibility.n_validation,
                                feasibility.required_validation,
                            ),
                            ("Test", feasibility.n_test, feasibility.required_test),
                        )
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            n_log_unusable = (
                feasibility.n_log_unusable_fit
                + feasibility.n_log_unusable_validation
                + feasibility.n_log_unusable_test
            )
            st.caption(
                f"N warm-up: {feasibility.n_warmup}; strict log-domain unusable rows: "
                f"{n_log_unusable}."
            )
            if feasibility.feasible:
                count = len(candidate_specs(selected_channels, method, horizon, int(budget)))
                if method != "none":
                    count += 1
            else:
                count = None
            st.caption(
                f"At most {count} candidate fits including no-adstock baseline."
                if count is not None
                else "Candidate search is unavailable until all sample-size requirements are met."
            )
            st.caption(
                "Zero prehistory is used before the first observed week; observed past spend "
                "is carried across splits. " + HELP_TEXT["optimization_objective"]
            )
        except ValueError as error:
            st.error(str(error))
            return
        result = st.session_state.get("adstock_search")
        if result and result.get("search_specification") != current_specification:
            st.session_state.pop("adstock_search", None)
            st.session_state.pop("adstock_applied_config", None)
            result = None
            st.warning("Search inputs changed, so the previous result was cleared.")
        if not feasibility.feasible:
            st.error("Not feasible")
            for failure in feasibility.failures:
                st.write(f"- {failure}")
            st.info(
                "Add more historical weeks, shorten the lag horizon, reduce channels or controls, "
                "or adjust the split while preserving meaningful validation and final-test periods."
            )
        if final_test_used:
            st.warning(
                "The final test has already been used for this dataset. Start with a newly "
                "loaded dataset before claiming another unbiased final evaluation."
            )
        if st.button(
            "Optimize adstock parameters",
            disabled=final_test_used or not feasibility.feasible,
        ):
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
                    st.session_state.pop("adstock_applied_config", None)
                    result = st.session_state["adstock_search"]
                except ValueError as error:
                    st.error(f"Search failed: {error}")
        if result and feasibility.feasible:
            st.success("Search complete")
            help_heading(
                "Candidate results",
                "candidate_status",
                key="adstock_candidate_results",
            )
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
            st.caption(HELP_TEXT["optimization_objective"])
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
            st.caption(HELP_TEXT["error_metrics"])
            if result["final_test"] is None:
                st.info(
                    "Selection is frozen from validation results. The final test remains unused."
                )
                if st.button("Evaluate selected model on final test"):
                    try:
                        result = evaluate_final_test(frame, result)
                    except ValueError as error:
                        st.error(f"Final evaluation failed: {error}")
                    else:
                        st.session_state["adstock_search"] = result
                        st.session_state["adstock_final_evaluated_sha256"] = current_sha256
                        st.rerun()
            if result["final_test"] is not None:
                st.metric(
                    "Final held-out RMSE · selected only", f"{result['final_test']['rmse']:.3f}"
                )
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
    _limitations()
