"""Streamlit presentation for deterministic channel spend-response scenarios."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from media_mix_ai_lab.modeling.interfaces import ModelResult
from media_mix_ai_lab.optimization.response_curves import build_response_curve


def response_curves_page() -> None:
    st.title("ROI Analysis")
    st.subheader("Channel spend-response curves")
    st.warning(
        "These are conditional model-based scenarios, not proven incremental causal effects. "
        "ROI ratios and budget allocation are separate future stories."
    )
    saved = st.session_state.get("model_results", [])
    if not saved:
        st.info("Fit a log-log model on the Models page before creating a response curve.")
        return
    results = [ModelResult.model_validate_json(item) for item in saved]
    eligible = [
        (index, result)
        for index, result in enumerate(results, start=1)
        if result.config.kind == "log_log"
    ]
    if not eligible:
        st.info(
            "No fitted log-log model is available. A power-law curve is not implied by an OLS "
            "fit, and this page does not invent a quadratic approximation."
        )
        return
    labels = [f"Fit {index}" for index, _ in eligible]
    selected_label = st.selectbox("Fitted log-log model", labels, index=len(labels) - 1)
    result = eligible[labels.index(selected_label)][1]
    channel = st.selectbox("Channel", result.config.channels)
    frame: pd.DataFrame = st.session_state["dataset"]
    observed_min = float(frame[channel].min())
    observed_max = float(frame[channel].max())
    default_min = max(observed_min * 0.5, observed_min / 100)
    default_max = observed_max * 1.5
    reference = float(frame[channel].median())
    col1, col2, col3 = st.columns(3)
    minimum = col1.number_input("Minimum weekly spend", min_value=0.01, value=default_min)
    maximum = col2.number_input("Maximum weekly spend", min_value=0.02, value=default_max)
    reference_spend = col3.number_input("Reference weekly spend", min_value=0.01, value=reference)
    try:
        curve = build_response_curve(
            frame,
            result,
            channel,
            minimum_spend=float(minimum),
            maximum_spend=float(maximum),
            reference_spend=float(reference_spend),
        )
    except ValueError as error:
        st.error(str(error))
        return

    values = pd.DataFrame(point.model_dump() for point in curve.points)
    figure = go.Figure()
    for mask, label, dash in (
        (
            values["spend_per_week"].lt(curve.observed_spend_min),
            "Extrapolation · below observed",
            "dash",
        ),
        (
            values["is_extrapolation"].eq(False),
            "Within observed range",
            "solid",
        ),
        (
            values["spend_per_week"].gt(curve.observed_spend_max),
            "Extrapolation · above observed",
            "dash",
        ),
    ):
        subset = values.loc[mask]
        if not subset.empty:
            figure.add_trace(
                go.Scatter(
                    x=subset["spend_per_week"],
                    y=subset["outcome_change_from_reference"],
                    mode="lines",
                    name=label,
                    line={"dash": dash},
                )
            )
    figure.add_vrect(
        x0=curve.observed_spend_min,
        x1=curve.observed_spend_max,
        fillcolor="teal",
        opacity=0.08,
        line_width=0,
        annotation_text="Observed spend range",
    )
    figure.add_hline(y=0, line_color="gray", line_dash="dot")
    figure.update_layout(
        xaxis_title=f"{channel} spend · currency per week",
        yaxis_title=f"Predicted {curve.target} change vs reference · outcome units per week",
        title=f"{channel} conditional weekly response scenario",
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"Reference: {curve.reference_spend:,.2f} currency/week with predicted conditional "
        f"{curve.target} {curve.reference_outcome:,.2f}/week."
    )
    st.write(curve.assumption)
    st.info(curve.caveat)
    st.markdown("**Fixed inputs (observed medians)**")
    st.dataframe(
        pd.DataFrame(
            [{"Input": name, "Assumed value": value} for name, value in curve.fixed_inputs.items()]
        ),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Curve values and reproducibility details"):
        st.dataframe(values, hide_index=True, width="stretch")
        st.json(curve.model_dump(mode="json", exclude={"points"}))
    st.download_button(
        "Download response curve JSON",
        curve.model_dump_json(indent=2),
        file_name=f"response-curve-{channel}.json",
        mime="application/json",
    )
