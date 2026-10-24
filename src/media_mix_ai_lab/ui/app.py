"""Streamlit presentation only: deterministic computation lives outside the UI."""

import pandas as pd
import plotly.express as px
import streamlit as st

from media_mix_ai_lab.core.config import ModelConfig
from media_mix_ai_lab.data.synthetic import generate_sample_data
from media_mix_ai_lab.data.validation import validate_dataset

SECTIONS = ("Home", "Data", "Explore", "Models", "AI Assistant", "ROI Analysis")
SPEND_COLUMNS = ("tv_spend", "search_spend", "social_spend")


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
    st.write("Model fitting is planned. This release only previews a validated configuration.")
    kind = st.selectbox("Planned model family", ["ols", "log_log"])
    config = ModelConfig(kind=kind)
    st.json(config.model_dump(mode="json"))
    st.download_button(
        "Download configuration", config.model_dump_json(indent=2), "model-config.json"
    )
    st.button("Fit model (coming soon)", disabled=True)
    st.caption(
        "Future work: OLS, log-log models, adstock, controls, temporal validation, "
        "and diagnostics. "
        "No coefficients, predictions, or accuracy scores are fabricated."
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
