"""Streamlit presentation only: deterministic computation lives outside the UI."""

import hashlib

import pandas as pd
import plotly.express as px
import streamlit as st

from media_mix_ai_lab.core.config import ModelConfig
from media_mix_ai_lab.data.synthetic import generate_sample_data
from media_mix_ai_lab.data.validation import validate_dataset
from media_mix_ai_lab.data.workspace import (
    ColumnMapping,
    map_weekly_dataset,
    quality_report,
    read_csv_upload,
    validate_weekly_dataset,
)

SECTIONS = ("Home", "Data", "Explore", "Models", "AI Assistant", "ROI Analysis")


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


def data_page(frame: pd.DataFrame, mapping: ColumnMapping, source: dict[str, str | int]) -> None:
    st.title("Data")
    st.write("Upload a weekly CSV or inspect the included synthetic dataset.")
    st.button("Load sample dataset", on_click=_load_sample)
    uploaded_file = st.file_uploader("Upload a weekly CSV", type=["csv"])
    if uploaded_file is not None:
        content = uploaded_file.getvalue()
        try:
            raw = read_csv_upload(content)
        except ValueError as error:
            st.error(str(error))
        else:
            numeric_columns = [
                column
                for column in raw.columns
                if pd.api.types.is_numeric_dtype(raw[column])
                and not pd.api.types.is_bool_dtype(raw[column])
            ]
            if not numeric_columns:
                st.error("The CSV must contain at least one numeric column to map.")
            else:
                st.subheader("Map weekly columns")
                with st.form("weekly_column_mapping"):
                    date_options = list(raw.columns)
                    date_default = next(
                        (
                            index
                            for index, name in enumerate(date_options)
                            if "date" in name.lower()
                        ),
                        0,
                    )
                    date_column = st.selectbox("Date column", date_options, index=date_default)
                    target_default = (
                        numeric_columns.index("revenue") if "revenue" in numeric_columns else 0
                    )
                    target_column = st.selectbox(
                        "Target / outcome column", numeric_columns, index=target_default
                    )
                    channel_options = [
                        column for column in numeric_columns if column != target_column
                    ]
                    channel_columns = st.multiselect(
                        "Channel columns (select at least one)",
                        channel_options,
                        default=channel_options,
                    )
                    control_options = [
                        column for column in numeric_columns if column != target_column
                    ]
                    control_columns = st.multiselect("Control columns (optional)", control_options)
                    st.selectbox("Observation frequency", ["Weekly"])
                    submitted = st.form_submit_button("Validate and use dataset")
                if submitted:
                    try:
                        mapped, new_mapping = map_weekly_dataset(
                            raw,
                            date_column,
                            target_column,
                            tuple(channel_columns),
                            tuple(control_columns),
                        )
                    except (TypeError, ValueError) as error:
                        st.error(str(error))
                    else:
                        st.session_state["dataset"] = mapped
                        st.session_state["dataset_mapping"] = new_mapping
                        st.session_state["dataset_source"] = {
                            "kind": "upload",
                            "filename": uploaded_file.name,
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "rows": len(mapped),
                            "columns": len(mapped.columns),
                        }
                        st.rerun()
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button(
        "Download current dataset", frame.to_csv(index=False), "weekly_dataset.csv", "text/csv"
    )
    st.subheader("Data quality")
    st.dataframe(quality_report(frame, mapping), hide_index=True, width="stretch")
    st.caption(
        "Missing values are retained and appear as gaps in charts; zeros are reported, "
        "not treated as missing. No rows are removed, sorted, imputed, or transformed."
    )
    with st.expander("Dataset mapping and provenance", expanded=True):
        st.write(f"Source: **{source['filename']}**")
        if source["kind"] == "upload":
            st.write(f"Rows: {source['rows']} · Columns: {source['columns']}")
            st.code(f"SHA-256: {source['sha256']}")
            st.caption("Uploaded data stays in this Streamlit session; it is not sent externally.")
        else:
            st.markdown(
                "Synthetic sample: 104 Monday observations starting 2023-01-02; fixed seed "
                "**42**. Revenue and TV/Search/Social spend use arbitrary currency units. "
                "Price index is centered around 100; promotions are binary; seasonality is "
                "an annual sine wave. **is_injected_outlier** labels a deliberately added "
                "60,000-unit revenue spike at row 65 (zero-based). All rows remain intact."
            )
        st.write(
            {
                "date": mapping["date"],
                "weekly_frequency": True,
                "target": mapping["target"],
                "channels": list(mapping["channels"]),
                "controls": list(mapping["controls"]),
            }
        )


def _load_sample() -> None:
    st.session_state["dataset"] = generate_sample_data()
    st.session_state["dataset_mapping"] = {
        "date": "date",
        "target": "revenue",
        "channels": ("tv_spend", "search_spend", "social_spend"),
        "controls": ("price_index", "promotions", "seasonality"),
    }
    st.session_state["dataset_source"] = {
        "kind": "synthetic",
        "filename": "weekly_marketing.csv",
    }


def explore(frame: pd.DataFrame, mapping: ColumnMapping, source_kind: str) -> None:
    st.title("Explore")
    st.caption("WEEKLY OBSERVATIONS · No fitted model or attribution")
    target = mapping["target"]
    channels = mapping["channels"]
    cols = st.columns(3)
    cols[0].metric("Weeks", len(frame))
    target_total = frame[target].sum(min_count=1)
    channel_total = frame[list(channels)].sum(axis=1, min_count=1).sum(min_count=1)
    cols[1].metric(f"Observed {target}", "—" if pd.isna(target_total) else f"{target_total:,.0f}")
    cols[2].metric(
        "Observed channel total", "—" if pd.isna(channel_total) else f"{channel_total:,.0f}"
    )
    series_options = [target, *channels]
    series = st.multiselect(
        "Target and channel series",
        series_options,
        default=series_options,
    )
    if series:
        chart = px.line(
            frame,
            x=mapping["date"],
            y=series,
            labels={"value": "Observed value", mapping["date"]: "Week"},
            color_discrete_sequence=["#087e8b", "#d97706", "#6554c0", "#2a9d8f"],
        )
        if (
            source_kind == "synthetic"
            and "is_injected_outlier" in frame.columns
            and frame["is_injected_outlier"].any()
        ):
            outlier_date = frame.loc[frame["is_injected_outlier"], mapping["date"]].iloc[0]
            chart.add_vline(
                x=outlier_date.timestamp() * 1000, line_dash="dot", line_color="#b42318"
            )
        chart.update_layout(legend_title_text="Series", hovermode="x unified")
        st.plotly_chart(chart, width="stretch")
    else:
        st.info("Select at least one series to display the chart.")
    if source_kind == "synthetic" and "is_injected_outlier" in frame.columns:
        st.caption("Dotted red line: deliberately injected outlier; no observations removed.")
    if mapping["controls"]:
        control = st.selectbox("Inspect a control", list(mapping["controls"]))
        st.plotly_chart(px.line(frame, x=mapping["date"], y=control), width="stretch")
    else:
        st.info("No control columns were mapped.")
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
    mapping = st.session_state["dataset_mapping"]
    source = st.session_state["dataset_source"]
    try:
        if source["kind"] == "upload":
            validate_weekly_dataset(frame, mapping)
        else:
            validate_dataset(frame)
    except ValueError as error:
        st.error(f"Dataset validation failed: {error}")
        st.button("Restore synthetic sample", on_click=_load_sample)
        st.stop()
    if section == "Home":
        home()
    elif section == "Data":
        data_page(frame, mapping, source)
    elif section == "Explore":
        explore(frame, mapping, source["kind"])
    elif section == "Models":
        models()
    elif section == "AI Assistant":
        assistant()
    else:
        roi()
