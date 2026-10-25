"""Streamlit presentation only: deterministic computation lives outside the UI."""

import hashlib
import html
import math

import pandas as pd
import plotly.express as px
import streamlit as st

from media_mix_ai_lab.core.config import AdstockSpec, ModelConfig
from media_mix_ai_lab.data.synthetic import generate_sample_data
from media_mix_ai_lab.data.validation import validate_dataset
from media_mix_ai_lab.data.workspace import (
    ColumnMapping,
    map_weekly_dataset,
    quality_report,
    read_csv_upload,
    validate_weekly_dataset,
)
from media_mix_ai_lab.modeling.adstock import adstock_impulse_response, apply_adstock
from media_mix_ai_lab.modeling.interfaces import ModelResult
from media_mix_ai_lab.modeling.ols import (
    ModelFitError,
    OLSModelEngine,
    summarize_observations,
)
from media_mix_ai_lab.ui.adstock_page import adstock_page
from media_mix_ai_lab.ui.diagnostics_page import diagnostics_page

NAVIGATION = (
    (
        "Workspace",
        (
            ("Home", ":material/home:"),
            ("Data", ":material/database:"),
            ("Explore", ":material/monitoring:"),
        ),
    ),
    (
        "Analysis",
        (
            ("Models", ":material/model_training:"),
            ("Adstock Analysis", ":material/show_chart:"),
            ("Diagnostics", ":material/analytics:"),
            ("AI Assistant", ":material/auto_awesome:"),
            ("ROI Analysis", ":material/finance_mode:"),
        ),
    ),
)

SIDEBAR_STYLE = """
<style>
    [data-testid="stSidebar"] {
        border-right: 1px solid #d8e2e8;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        min-height: 2.7rem;
        justify-content: flex-start;
        border-radius: 0.7rem;
        font-weight: 600;
        padding-left: 0.85rem;
        transition: background-color 120ms ease, border-color 120ms ease;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="tertiary"] {
        color: #344563;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="tertiary"]:hover {
        background: #e2edf0;
        color: #075f69;
    }
    .lab-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin: 0.1rem 0 1.35rem;
    }
    .lab-mark {
        display: grid;
        place-items: center;
        width: 2.35rem;
        height: 2.35rem;
        border-radius: 0.75rem;
        color: white;
        background: linear-gradient(145deg, #087e8b, #075f69);
        font-size: 1.2rem;
        box-shadow: 0 6px 16px rgba(8, 126, 139, 0.2);
    }
    .lab-name {
        color: #172b4d;
        font-size: 1.05rem;
        font-weight: 750;
        line-height: 1.15;
    }
    .lab-version {
        color: #6b7c93;
        font-size: 0.68rem;
        font-weight: 650;
        letter-spacing: 0.08em;
        margin-top: 0.18rem;
    }
    .dataset-card {
        background: rgba(255, 255, 255, 0.62);
        border: 1px solid #d8e2e8;
        border-radius: 0.75rem;
        color: #344563;
        font-size: 0.78rem;
        line-height: 1.45;
        padding: 0.75rem 0.85rem;
    }
    .dataset-card strong {
        color: #172b4d;
        display: block;
        font-size: 0.82rem;
        margin-bottom: 0.1rem;
        overflow-wrap: anywhere;
    }
</style>
"""


def _navigate(section: str) -> None:
    """Persist the active page across Streamlit reruns."""
    st.session_state["active_section"] = section
    st.session_state["collapse_sidebar"] = st.session_state.get("collapse_count", 0) + 1
    st.session_state["collapse_count"] = st.session_state["collapse_sidebar"]


COLLAPSE_SIDEBAR_SCRIPT = """
<script>
// navigation #{nonce}: a unique body makes Streamlit re-run the script
if (window.matchMedia("(max-width: 640px)").matches) {
    setTimeout(() => {
        const button = document.querySelector(
            '[data-testid="stSidebarCollapseButton"] button'
        );
        if (button) button.click();
    }, 50);
}
</script>
"""


def sidebar_navigation(source: dict[str, str | int], row_count: int) -> str:
    """Render the branded navigation rail and return the active section."""
    if "active_section" not in st.session_state:
        st.session_state["active_section"] = "Home"
    section = st.session_state["active_section"]
    if nonce := st.session_state.pop("collapse_sidebar", 0):
        st.html(
            COLLAPSE_SIDEBAR_SCRIPT.replace("{nonce}", str(nonce)), unsafe_allow_javascript=True
        )
    with st.sidebar:
        st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)
        st.markdown(
            """
            <div class="lab-brand">
                <div class="lab-mark">∿</div>
                <div>
                    <div class="lab-name">Media Mix AI Lab</div>
                    <div class="lab-version">EXPERIMENTAL · v0.1</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for group, items in NAVIGATION:
            st.caption(group.upper())
            for label, icon in items:
                st.button(
                    label,
                    key=f"nav_{label.lower().replace(' ', '_')}",
                    icon=icon,
                    type="primary" if section == label else "tertiary",
                    use_container_width=True,
                    on_click=_navigate,
                    args=(label,),
                )
            st.write("")
        st.divider()
        kind = "Synthetic sample · seed 42" if source["kind"] == "synthetic" else "Uploaded CSV"
        filename = html.escape(str(source["filename"]))
        st.caption("ACTIVE DATASET")
        st.markdown(
            '<div class="dataset-card">'
            f"<strong>{filename}</strong>"
            f"{row_count:,} weekly rows<br>{kind}"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption("Python computes · AI interprets")
    return section


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
                        _clear_adstock_session()
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
                "**42**. Revenue and TV/Search/Social spend use arbitrary currency units; "
                "combined weekly media investment is capped at 10,000. "
                "Price index is centered around 100; promotions are binary; seasonality is "
                "an annual sine wave. **is_injected_outlier** labels a deliberately added "
                "8,000-unit revenue spike at row 65 (zero-based). All rows remain intact."
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
    _clear_adstock_session()
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


def _clear_adstock_session() -> None:
    """Discard calibration state whenever the active dataset is replaced."""
    for key in ("adstock_search", "adstock_applied_config", "adstock_exploration"):
        st.session_state.pop(key, None)


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
    if st.button("Open Adstock Analysis"):
        _navigate("Adstock Analysis")
        st.rerun()
    st.write("Fit transparent OLS specifications to the currently loaded dataset.")
    frame = st.session_state["dataset"]
    mapping: ColumnMapping = st.session_state["dataset_mapping"]
    numeric_columns = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column != "is_injected_outlier" and column != mapping["date"]
    ]
    kind = st.selectbox(
        "Model specification",
        ["ols", "log_log"],
        format_func=lambda value: "Linear OLS" if value == "ols" else "Log-log OLS",
    )
    target_default = (
        numeric_columns.index(mapping["target"]) if mapping["target"] in numeric_columns else 0
    )
    target = st.selectbox("Outcome", numeric_columns, index=target_default)
    media_options = [
        column for column in mapping["channels"] if column in numeric_columns and column != target
    ]
    media_default = media_options
    channels = st.multiselect("Media predictors", media_options, default=media_default)
    control_options = [
        column for column in mapping["controls"] if column in numeric_columns and column != target
    ]
    controls = st.multiselect(
        "Optional controls",
        [column for column in control_options if column not in channels],
        default=[column for column in control_options if column not in channels],
    )
    adstock_settings: dict[str, AdstockSpec] = {}
    adstock_error = None
    try:
        with st.expander("Per-channel adstock transforms"):
            st.caption(
                "Suggested starting values (not fitted estimates): geometric alpha 0.5; "
                "delayed-geometric alpha 0.7, theta 2, and maximum lag 8 weeks."
            )
            for channel in channels:
                st.markdown(f"**{channel}**")
                method = st.selectbox(
                    f"{channel} adstock method",
                    ["none", "geometric", "delayed_geometric"],
                    format_func={
                        "none": "None",
                        "geometric": "Geometric (recursive)",
                        "delayed_geometric": "Delayed geometric (finite lag)",
                    }.get,
                    key=f"adstock_method_{channel}",
                )
                settings: dict[str, object] = {"method": method}
                if method != "none":
                    settings["weight_scaling"] = st.selectbox(
                        f"{channel} weight scaling",
                        ["raw", "sum_to_one"],
                        format_func={
                            "raw": "Raw weights (default)",
                            "sum_to_one": "Sum to one",
                        }.get,
                        key=f"adstock_scaling_{channel}",
                    )
                    if method == "geometric":
                        settings["alpha"] = st.number_input(
                            f"{channel} alpha",
                            min_value=0.0,
                            max_value=0.999999,
                            value=0.5,
                            step=0.05,
                            key=f"adstock_alpha_{channel}",
                        )
                        settings["initial_state"] = st.number_input(
                            f"{channel} initial carryover state",
                            min_value=0.0,
                            max_value=1e12,
                            value=0.0,
                            key=f"adstock_initial_state_{channel}",
                        )
                    else:
                        settings["alpha"] = st.number_input(
                            f"{channel} alpha",
                            min_value=0.000001,
                            max_value=0.999999,
                            value=0.7,
                            step=0.05,
                            key=f"adstock_alpha_{channel}",
                        )
                        settings["max_lag_weeks"] = st.slider(
                            f"{channel} maximum lag (weeks)",
                            min_value=0,
                            max_value=12,
                            value=8,
                            key=f"adstock_lag_{channel}",
                        )
                        settings["theta"] = st.number_input(
                            f"{channel} theta (weeks)",
                            min_value=0.0,
                            max_value=float(settings["max_lag_weeks"]),
                            value=float(min(2, settings["max_lag_weeks"])),
                            step=1.0,
                            key=f"adstock_theta_{channel}",
                        )
                        history = st.text_input(
                            f"{channel} prior weekly media values (oldest to newest, optional)",
                            key=f"adstock_history_{channel}",
                        )
                        settings["historical_values"] = tuple(
                            float(value.strip()) for value in history.split(",") if value.strip()
                        )
                adstock_settings[channel] = AdstockSpec(**settings)
    except (ValueError, OverflowError) as error:
        adstock_error = str(error)
    intercept = st.checkbox("Include intercept", value=True)
    with st.expander("Domain sign expectations"):
        st.caption(
            "These are review hypotheses, not mathematical constraints. Choose Either sign "
            "when a negative media association is plausible in context."
        )
        sign_expectations = {
            channel: st.selectbox(
                f"{channel} expected direction",
                ["positive", "either", "negative"],
                format_func={
                    "positive": "Usually positive",
                    "either": "Either sign",
                    "negative": "Usually negative",
                }.get,
                key=f"media_sign_expectation_{channel}",
            )
            for channel in channels
        }
    config = None
    accounting = None
    try:
        if adstock_error is not None:
            raise ValueError(f"Invalid adstock settings: {adstock_error}")
        config = ModelConfig(
            kind=kind,
            target=target,
            channels=tuple(channels),
            controls=tuple(controls),
            date_column=mapping["date"],
            intercept=intercept,
            media_sign_expectations=sign_expectations,
            adstock=adstock_settings,
        )
        applied = st.session_state.pop("adstock_applied_config", None)
        if applied is not None:
            selected = ModelConfig.model_validate(applied)
            if (selected.target, selected.channels, selected.date_column) == (
                config.target,
                config.channels,
                config.date_column,
            ):
                st.info("Calibrated settings were loaded into the editable controls above.")
        accounting = summarize_observations(frame, config)
    except ValueError as error:
        config = None
        st.error(f"Invalid model specification: {error}")

    if config is not None and channels:
        st.subheader("Raw and transformed media signals")
        preview_channel = st.selectbox("Channel preview", channels)
        spec = config.adstock.get(preview_channel, AdstockSpec())
        try:
            transformed_signal = apply_adstock(frame[preview_channel], spec)
            signal = pd.DataFrame(
                {
                    "Date": frame[mapping["date"]],
                    "Raw spend": frame[preview_channel],
                    "Transformed signal": transformed_signal,
                }
            )
            st.plotly_chart(
                px.line(signal, x="Date", y=["Raw spend", "Transformed signal"]),
                width="stretch",
            )
            weights = adstock_impulse_response(spec)
            st.plotly_chart(
                px.bar(
                    weights, x="Lag (weeks)", y="Weight", title="Impulse response / lag weights"
                ),
                width="stretch",
            )
            if spec.method == "geometric" and spec.alpha > 0:
                half_life = math.log(0.5) / math.log(spec.alpha)
                st.caption(f"Geometric half-life: {half_life:.2f} weeks.")
            elif spec.method == "geometric":
                st.caption("Geometric half-life: 0 weeks at alpha = 0.")
            if spec.method == "geometric":
                st.caption(
                    f"Recursive filter starts from initial state {spec.initial_state:g}; "
                    "the chart shows 13 lags, but the recurrence is not truncated."
                )
            elif spec.method == "delayed_geometric":
                boundary = (
                    "zero prior history"
                    if not spec.historical_values
                    else f"{len(spec.historical_values)} supplied prior week(s)"
                )
                st.caption(
                    f"Finite {spec.max_lag_weeks}-week window; {boundary}. "
                    "Any unspecified older history is zero; no future lags are used."
                )
            else:
                st.caption("No adstock: the transformed series is identical to raw spend.")
            st.caption(
                "Preview includes the early-window boundary behavior; source data is unchanged."
            )
        except ValueError as error:
            st.error(f"Cannot preview media transform: {error}")

    if accounting is not None:
        st.subheader("Observation eligibility and log-domain checks")
        accounting_columns = st.columns(4)
        accounting_columns[0].metric("Total observations", accounting.n_input)
        accounting_columns[1].metric("Eligible observations", accounting.n_eligible)
        accounting_columns[2].metric("Nonpositive target", accounting.nonpositive_target_count)
        accounting_columns[3].metric(
            "Rows outside log domain (unique)", len(accounting.affected_rows)
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"Column": target, "Nonpositive values": accounting.nonpositive_target_count},
                    *[
                        {"Column": column, "Nonpositive values": count}
                        for column, count in accounting.nonpositive_media_counts.items()
                    ],
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.json(accounting.model_dump(mode="json"))
        if accounting.affected_rows:
            with st.expander("Review affected source records"):
                positions = [row.row_position for row in accounting.affected_rows]
                affected = frame.iloc[positions].copy()
                affected.insert(0, "Row index (0-based)", positions)
                st.dataframe(affected, hide_index=True, width="stretch")

    if st.button("Fit model", disabled=config is None):
        try:
            assert config is not None
            result = OLSModelEngine().fit(frame, config)
            st.session_state.setdefault("model_results", []).append(
                result.model_dump_json(indent=2)
            )
        except ModelFitError as error:
            st.error(f"Model fit failed: {error}")
            st.json(error.accounting.model_dump(mode="json"))
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
        "Adstock is a deterministic signal transform, not a causal attribution method."
    )


def _show_model_result(result: ModelResult) -> None:
    st.code(result.equation)
    st.caption(
        "Features and transformations: "
        + ", ".join(f"{name}: {transform}" for name, transform in result.transformations.items())
    )
    health = result.health_summary
    health_message = (
        f"**Model health: {health.status.upper()}** — {health.headline} "
        f"Structured flags: {health.flag_count}."
    )
    if health.status == "healthy":
        st.success(health_message)
    elif health.status == "unstable":
        st.error(health_message)
    else:
        st.warning(health_message)
    if health.warning_codes:
        st.caption("Warning codes: " + ", ".join(health.warning_codes))

    columns = st.columns(4)
    columns[0].metric("Sample size", result.sample_size)
    columns[1].metric("R²", "—" if result.r_squared is None else f"{result.r_squared:.4f}")
    columns[2].metric(
        "Adjusted R²",
        "—" if result.adjusted_r_squared is None else f"{result.adjusted_r_squared:.4f}",
    )
    columns[3].metric("Residual variance", f"{result.residual_variance:.4g}")
    accounting = result.observation_accounting
    st.caption(
        f"Observation accounting: {accounting.n_input} input · "
        f"{accounting.n_fitted} fitted · {accounting.n_excluded} excluded"
    )
    with st.expander("Observation accounting and transformation policy"):
        st.json(accounting.model_dump(mode="json"))
    estimates, evidence, numerical, plausibility = st.tabs(
        ["Estimates", "Statistical evidence", "Numerical diagnostics", "Business plausibility"]
    )
    estimate_rows = pd.DataFrame(
        [
            {
                "Term": term,
                "Role": assessment.role,
                "Estimate": assessment.estimate,
                "Scale / units": assessment.scale,
                "Interpretation": assessment.interpretation,
            }
            for term, assessment in result.term_assessments.items()
        ]
    )
    with estimates:
        st.dataframe(estimate_rows, hide_index=True, width="stretch")
        st.caption(
            "Coefficients are conditional associations from this specification. Positive "
            "estimates and high R² do not prove causal lift."
        )

    with evidence:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Term": term,
                        "Standard error": assessment.standard_error,
                        "t-statistic": (
                            assessment.t_statistic if assessment.t_statistic is not None else "—"
                        ),
                        "95% CI lower": assessment.confidence_interval[0],
                        "95% CI upper": assessment.confidence_interval[1],
                        "CI spans zero": "Yes" if assessment.ci_spans_zero else "No",
                        "p-value": _format_p_value(assessment.p_value),
                        "Signif.": assessment.significance_code,
                    }
                    for term, assessment in result.term_assessments.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Significance codes: *** p ≤ 0.001 · ** p ≤ 0.01 · * p ≤ 0.05 · . p ≤ 0.10. "
            "These are classical two-sided OLS t-test p-values. They assume a correctly "
            "specified model with independent, homoskedastic errors (and normal errors for "
            "exact finite-sample inference). Stars summarize evidence under those assumptions; "
            "time-series dependence and multiple testing can make them misleading. Stars do "
            "not measure effect size or establish causality."
        )
        _show_diagnostic_flags(result, "statistical_evidence")

    with numerical:
        diagnostic_columns = st.columns(3)
        diagnostic_columns[0].metric(
            "Residual degrees of freedom", f"{result.residual_degrees_freedom:g}"
        )
        diagnostic_columns[1].metric("Condition number", f"{result.condition_number:.4g}")
        diagnostic_columns[2].metric(
            "Design rank", f"{result.design_rank} / {result.parameter_count}"
        )
        thresholds = result.config.diagnostic_thresholds
        st.caption(
            f"Review thresholds: residual df ≤ {thresholds.near_zero_residual_df}; "
            f"condition number ≥ {thresholds.poor_condition_number:.4g}; relative standard "
            f"error ≥ {thresholds.large_relative_standard_error:g}. Exact singular designs "
            "remain hard errors."
        )
        _show_diagnostic_flags(result, "numerical_diagnostics")

    with plausibility:
        media_assessments = [
            assessment
            for assessment in result.term_assessments.values()
            if assessment.role == "media"
        ]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Media term": assessment.term,
                        "Estimate": assessment.estimate,
                        "Expected direction": assessment.expected_direction,
                        "Assessment": (
                            "Unexpected sign—investigate"
                            if any(
                                flag.code == "MEDIA_UNEXPECTED_SIGN"
                                and flag.evidence.get("term") == assessment.term
                                for flag in result.diagnostic_flags
                            )
                            else "No sign hypothesis triggered"
                        ),
                    }
                    for assessment in media_assessments
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Sign expectations are configurable business hypotheses. A negative media "
            "coefficient is allowed and may be justified; investigate timing, confounding, "
            "collinearity, targeting, and data quality before interpretation."
        )
        _show_diagnostic_flags(result, "business_plausibility")

    flag_messages = {flag.message for flag in result.diagnostic_flags}
    with st.expander("Assumptions and limitations"):
        for warning in result.warnings:
            if warning not in flag_messages:
                st.warning(warning)
    with st.expander("Model specification and provenance"):
        st.json(
            {
                "specification": result.config.model_dump(mode="json"),
                "dataset_sha256": result.dataset_sha256,
                "package_versions": result.package_versions,
                "diagnostic_thresholds": result.config.diagnostic_thresholds.model_dump(
                    mode="json"
                ),
            }
        )


def _show_diagnostic_flags(result: ModelResult, category: str) -> None:
    flags = [flag for flag in result.diagnostic_flags if flag.category == category]
    if not flags:
        st.success("No flags in this category.")
        return
    for flag in flags:
        evidence = ", ".join(f"{name}={value}" for name, value in flag.evidence.items())
        message = f"**{flag.code}** — {flag.message}\n\nEvidence: {evidence}"
        if flag.severity == "unstable":
            st.error(message)
        else:
            st.warning(message)


def _format_p_value(p_value: float | None) -> str:
    """Format p-values for display while JSON retains full numeric precision."""
    if p_value is None:
        return "—"
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


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
    if "dataset" not in st.session_state:
        _load_sample()
    frame = st.session_state["dataset"]
    mapping = st.session_state["dataset_mapping"]
    source = st.session_state["dataset_source"]
    section = sidebar_navigation(source, len(frame))
    try:
        if source["kind"] == "upload":
            validate_weekly_dataset(frame, mapping)
        else:
            validate_dataset(
                frame, allow_negative_target_media=section in {"Models", "Adstock Analysis"}
            )
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
    elif section == "Adstock Analysis":
        adstock_page()
    elif section == "Diagnostics":
        diagnostics_page()
    elif section == "AI Assistant":
        assistant()
    else:
        roi()
