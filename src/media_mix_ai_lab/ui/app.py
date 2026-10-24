"""Streamlit presentation only: deterministic computation lives outside the UI."""

import pandas as pd
import plotly.express as px
import streamlit as st

from media_mix_ai_lab.core.config import ModelConfig
from media_mix_ai_lab.data.synthetic import generate_sample_data
from media_mix_ai_lab.data.validation import validate_dataset
from media_mix_ai_lab.modeling.interfaces import ModelResult
from media_mix_ai_lab.modeling.ols import OLSModelEngine

SECTIONS = ("Home", "Data", "Explore", "Models", "AI Assistant", "ROI Analysis")
SPEND_COLUMNS = ("tv_spend", "search_spend", "social_spend")
CONTROL_COLUMNS = ("price_index", "promotions", "seasonality")


def home() -> None:
    st.title("Marketing questions. Measured answers.")
    st.markdown(
        "An experimental **MMM AI Studio** for investigating marketing effectiveness "
        "with transparent Python statistics and, eventually, a conversational data scientist."
    )
    left, right = st.columns([2, 1])
    with left:
        st.subheader("A scientific workbench")
        st.markdown(
            "1. Inspect your data and its assumptions.\n"
            "2. Explore time patterns, controls, and unusual observations.\n"
            "3. Fit reproducible models and test their limitations (coming next).\n"
            "4. Let AI explain computed evidence, with uncertainty (coming next)."
        )
    with right:
        st.info("Start in **Explore**. A two-year synthetic weekly dataset is already loaded.")
    st.subheader("LLMs reason. Deterministic tools compute.")
    st.write(
        "Numerical outputs will come from validated Python tools. AI interpretations "
        "will be kept separate. Outliers stay visible, and correlation is never treated "
        "as causal attribution."
    )
    st.warning(
        "Personal weekend research project. No guarantees of accuracy, reliability, or "
        "business results. Not affiliated with or endorsed by the author's employer. "
        "Use at your own risk—and always question the numbers!"
    )


def data_page(frame: pd.DataFrame) -> None:
    st.title("Data")
    st.write("Inspect the synthetic dataset before drawing conclusions.")
    st.button("Load sample dataset", on_click=_load_sample)
    st.file_uploader("Upload a weekly CSV (coming soon)", type=["csv"], disabled=True)
    st.caption("Uploads are not enabled yet. No data is sent to an external service.")
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button(
        "Download synthetic CSV", frame.to_csv(index=False), "weekly_marketing.csv", "text/csv"
    )
    with st.expander("Dataset dictionary and provenance", expanded=True):
        st.markdown(
            "104 Monday observations starting 2023-01-02; fixed seed **42**. "
            "Revenue and TV/Search/Social spend use arbitrary currency units. "
            "Price index is centered around 100; promotions are binary; seasonality "
            "is an annual sine wave. **is_injected_outlier** labels a deliberately added "
            "60,000-unit revenue spike at row 65 (zero-based). All rows remain intact."
        )


def _load_sample() -> None:
    st.session_state["dataset"] = generate_sample_data()


def explore(frame: pd.DataFrame) -> None:
    st.title("Explore")
    st.caption("SYNTHETIC DATA · Weekly observations · No fitted model or attribution")
    cols = st.columns(3)
    cols[0].metric("Weeks", len(frame))
    cols[1].metric("Observed revenue", f"{frame['revenue'].sum():,.0f}")
    cols[2].metric("Media spend", f"{frame[list(SPEND_COLUMNS)].sum().sum():,.0f}")
    series = st.multiselect(
        "Revenue and spend series",
        ["revenue", *SPEND_COLUMNS],
        default=["revenue", *SPEND_COLUMNS],
    )
    if series:
        chart = px.line(
            frame,
            x="date",
            y=series,
            labels={"value": "Currency units", "date": "Week"},
            color_discrete_sequence=["#087e8b", "#d97706", "#6554c0", "#2a9d8f"],
        )
        outlier_date = frame.loc[frame["is_injected_outlier"], "date"].iloc[0]
        chart.add_vline(x=outlier_date.timestamp() * 1000, line_dash="dot", line_color="#b42318")
        chart.update_layout(legend_title_text="Series", hovermode="x unified")
        st.plotly_chart(chart, width="stretch")
    else:
        st.info("Select at least one series to display the chart.")
    st.caption("Dotted red line: deliberately injected outlier; no observations removed.")
    control = st.selectbox("Inspect a control", ["price_index", "promotions", "seasonality"])
    st.plotly_chart(px.line(frame, x="date", y=control), width="stretch")
    st.info("These are observations, not incremental effects. No ROI or causal lift is estimated.")


def models() -> None:
    st.title("Models")
    st.write("Fit transparent OLS specifications to the currently loaded dataset.")
    frame = st.session_state["dataset"]
    numeric_columns = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column != "is_injected_outlier"
    ]
    kind = st.selectbox(
        "Model specification",
        ["ols", "log_log"],
        format_func=lambda value: "Linear OLS" if value == "ols" else "Log-log OLS",
    )
    target_default = numeric_columns.index("revenue") if "revenue" in numeric_columns else 0
    target = st.selectbox("Outcome", numeric_columns, index=target_default)
    media_options = [
        column for column in numeric_columns if column != target and column not in CONTROL_COLUMNS
    ]
    media_default = [column for column in SPEND_COLUMNS if column in media_options]
    channels = st.multiselect("Media predictors", media_options, default=media_default)
    control_options = [
        column for column in CONTROL_COLUMNS if column in numeric_columns and column != target
    ]
    controls = st.multiselect(
        "Optional controls",
        [column for column in control_options if column not in channels],
        default=[column for column in control_options if column not in channels],
    )
    intercept = st.checkbox("Include intercept", value=True)
    if st.button("Fit model"):
        try:
            config = ModelConfig(
                kind=kind,
                target=target,
                channels=tuple(channels),
                controls=tuple(controls),
                intercept=intercept,
            )
            result = OLSModelEngine().fit(frame, config)
            st.session_state.setdefault("model_results", []).append(
                result.model_dump_json(indent=2)
            )
        except ValueError as error:
            st.error(f"Model fit failed: {error}")

    saved_results = st.session_state.get("model_results", [])
    if saved_results:
        st.subheader("Fitted results")
        for position, result_json in enumerate(reversed(saved_results), start=1):
            result = ModelResult.model_validate_json(result_json)
            with st.expander(f"Fit {len(saved_results) - position + 1}", expanded=position == 1):
                _show_model_result(result)
                st.download_button(
                    "Save result as JSON",
                    result_json,
                    file_name=f"ols-fit-{result.dataset_sha256[:12]}.json",
                    mime="application/json",
                    key=f"download_model_result_{position}",
                )
    st.caption(
        "Log-log models take natural logs of the outcome and media predictors; selected "
        "controls remain in their original units. No observations are removed. "
        "Adstock, temporal validation, and response curves are not implemented."
    )


def _show_model_result(result: ModelResult) -> None:
    st.code(result.equation)
    st.caption(
        "Features and transformations: "
        + ", ".join(f"{name}: {transform}" for name, transform in result.transformations.items())
    )
    columns = st.columns(4)
    columns[0].metric("Sample size", result.sample_size)
    columns[1].metric("R²", f"{result.r_squared:.4f}")
    columns[2].metric("Adjusted R²", f"{result.adjusted_r_squared:.4f}")
    columns[3].metric("Residual variance", f"{result.residual_variance:.4g}")
    coefficients = pd.DataFrame(
        [
            {
                "Term": term,
                "Role": result.term_roles[term],
                "Estimate": estimate,
                "Standard error": result.standard_errors[term],
                "95% CI lower": result.confidence_intervals[term][0],
                "95% CI upper": result.confidence_intervals[term][1],
            }
            for term, estimate in result.coefficients.items()
        ]
    )
    st.dataframe(coefficients, hide_index=True, width="stretch")
    if result.config.kind == "log_log":
        st.caption("Log-log media slopes are conditional elasticities, not causal ROI.")
    for warning in result.warnings:
        st.warning(warning)
    with st.expander("Model specification and provenance"):
        st.json(
            {
                "specification": result.config.model_dump(mode="json"),
                "dataset_sha256": result.dataset_sha256,
                "package_versions": result.package_versions,
            }
        )


def assistant() -> None:
    st.title("AI Assistant")
    st.info("Provider-independent interface prepared. No LLM is connected; no API key is required.")
    with st.chat_message("assistant"):
        st.write(
            "A future AI Data Scientist will help clarify your business question, suggest "
            "controls and diagnostics, and explain actual Python tool results and their "
            "limitations. It will never execute arbitrary generated Python."
        )
    st.chat_input("Conversation will be enabled in a future release", disabled=True)


def roi() -> None:
    st.title("ROI Analysis")
    st.write("Channel response curves and constrained budget optimization are coming later.")
    st.info(
        "ROI requires a validated model, explicit assumptions, and a definition of returns. "
        "Observed revenue divided by spend is not causal advertising ROI."
    )
    st.button("Estimate response curves (coming soon)", disabled=True)


def main() -> None:
    st.set_page_config(page_title="Media Mix AI Lab", page_icon="🔬", layout="wide")
    with st.sidebar:
        st.title("Media Mix AI Lab")
        st.caption("EXPERIMENTAL · v0.1")
        section = st.radio("Workbench", SECTIONS)
        st.divider()
        st.caption("Synthetic dataset · seed 42\nPython computes. AI interprets.")
    if "dataset" not in st.session_state:
        _load_sample()
    frame = st.session_state["dataset"]
    try:
        validate_dataset(frame)
    except ValueError as error:
        st.error(f"Dataset validation failed: {error}")
        st.button("Restore synthetic sample", on_click=_load_sample)
        st.stop()
    if section == "Home":
        home()
    elif section == "Data":
        data_page(frame)
    elif section == "Explore":
        explore(frame)
    elif section == "Models":
        models()
    elif section == "AI Assistant":
        assistant()
    else:
        roi()
