import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from media_mix_ai_lab.modeling.interfaces import ModelResult

APP = Path(__file__).resolve().parents[1] / "app.py"
NAVIGATION_INDEX = {
    "Home": 0,
    "Data": 1,
    "Explore": 2,
    "Models": 3,
    "AI Assistant": 4,
    "ROI Analysis": 5,
}


def navigate(app: AppTest, section: str) -> AppTest:
    """Click the same navigation control a user sees in the sidebar."""
    return app.sidebar.button[NAVIGATION_INDEX[section]].click().run()


@pytest.mark.parametrize(
    "section", ["Home", "Data", "Explore", "Models", "AI Assistant", "ROI Analysis"]
)
def test_workbench_sections_render_without_credentials(section):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    navigate(app, section)
    assert not app.exception
    assert not app.sidebar.radio
    assert len(app.sidebar.button) == len(NAVIGATION_INDEX)
    assert app.session_state["active_section"] == section
    assert len(app.session_state["dataset"]) == 104
    assert app.main.title[0].value == (
        "Marketing questions. Measured answers." if section == "Home" else section
    )
    if section == "Explore":
        assert len(app.get("plotly_chart")) == 2
        assert app.metric[0].value == "104"
        app.multiselect[0].set_value([]).run()
        assert not app.exception
        assert len(app.get("plotly_chart")) == 1
    elif section == "Data":
        assert len(app.dataframe[0].value) == 104
        assert not app.get("file_uploader")[0].disabled
        app.main.button[0].click().run()
        assert not app.exception
    elif section == "Models":
        app.selectbox[0].set_value("log_log").run()
        assert not app.exception


def test_models_fit_and_save_a_result():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    navigate(app, "Models")
    next(button for button in app.button if button.label == "Fit model").click().run()

    assert not app.exception
    assert app.session_state["model_results"]
    assert any(metric.label == "Sample size" for metric in app.metric)
    assert any(button.label == "Save result as JSON" for button in app.get("download_button"))
    result = ModelResult.model_validate_json(app.session_state["model_results"][-1])
    assert result.health_summary.flag_count == len(result.diagnostic_flags)
    assert result.health_summary.warning_codes
    assert all(flag.evidence for flag in result.diagnostic_flags)
    assert len(app.dataframe) >= 3
    assert any("Model health:" in warning.value for warning in app.warning)
    evidence = next(
        table.value
        for table in app.dataframe
        if {"t-statistic", "p-value", "Signif."}.issubset(table.value.columns)
    )
    assert "<0.001" in set(evidence["p-value"])
    assert "***" in set(evidence["Signif."])
    accounting = result.observation_accounting
    assert accounting.n_input == accounting.n_fitted == result.sample_size == 104
    assert accounting.n_excluded == 0
    assert any("104 input · 104 fitted · 0 excluded" in item.value for item in app.caption)


def test_models_report_strict_log_domain_errors():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    navigate(app, "Models")
    app.selectbox[0].set_value("log_log").run()
    app.session_state["dataset"].loc[0, "revenue"] = 0
    app.session_state["dataset"].loc[0, "tv_spend"] = 0
    next(button for button in app.button if button.label == "Fit model").click().run()

    assert not app.exception
    assert "revenue: 1 nonpositive value(s)" in app.error[0].value
    assert "tv_spend: 1 nonpositive value" in app.error[0].value
    summaries = [json.loads(element.value) for element in app.get("json")]
    assert any(summary.get("n_eligible") == 103 for summary in summaries)
    assert any(
        len(summary.get("affected_rows", [])) == 1
        and summary["affected_rows"][0]["columns"] == ["revenue", "tv_spend"]
        for summary in summaries
    )
    assert any(element.label == "Review affected source records" for element in app.expander)


def test_models_preview_and_fit_configured_geometric_adstock():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    navigate(app, "Models")
    method = next(
        element for element in app.selectbox if element.label == "tv_spend adstock method"
    )
    method.set_value("geometric").run()
    next(button for button in app.button if button.label == "Fit model").click().run()

    assert not app.exception
    assert len(app.get("plotly_chart")) == 2
    result = ModelResult.model_validate_json(app.session_state["model_results"][-1])
    assert result.config.adstock["tv_spend"].method == "geometric"
    assert "recursive geometric adstock" in result.transformations["tv_spend"]
    assert any("Geometric half-life:" in item.value for item in app.caption)


@pytest.mark.parametrize(
    "history",
    [
        "not-a-number",
        "-1",
        "1,2,3,4,5,6,7,8,9",
    ],
)
def test_models_report_invalid_adstock_history_without_crashing(history):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    navigate(app, "Models")
    method = next(
        element for element in app.selectbox if element.label == "tv_spend adstock method"
    )
    method.set_value("delayed_geometric").run()
    history_input = next(
        element
        for element in app.text_input
        if element.label == "tv_spend prior weekly media values (oldest to newest, optional)"
    )
    history_input.set_value(history).run()

    assert not app.exception
    assert "Invalid adstock settings" in app.error[0].value
    assert next(button for button in app.button if button.label == "Fit model").disabled


def test_invalid_dataset_has_actionable_recovery():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.session_state["dataset"] = app.session_state["dataset"].drop(columns="revenue")
    app.run()
    assert not app.exception
    assert "Missing required columns: revenue" in app.error[0].value
    app.main.button[0].click().run()
    assert not app.exception
    assert not app.error


def test_csv_upload_mapping_provenance_and_exploration():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    navigate(app, "Data")
    content = (
        b"date,sales,search,is_injected_outlier\n"
        b"2024-01-01,10,0,0\n"
        b"2024-01-08,12,2,0\n"
        b"2024-01-15,13,1,0\n"
        b"2024-01-22,16,4,1\n"
    )
    app.get("file_uploader")[0].set_value(("weekly.csv", content, "text/csv")).run()
    assert not app.exception
    app.multiselect[0].set_value(["search"])
    submit = next(button for button in app.button if button.label == "Validate and use dataset")
    submit.click().run()

    assert not app.exception
    assert app.session_state["dataset_source"]["filename"] == "weekly.csv"
    assert app.session_state["dataset_source"]["sha256"]
    assert app.session_state["dataset_mapping"]["target"] == "sales"
    assert app.session_state["dataset_mapping"]["channels"] == ("search",)
    assert len(app.session_state["dataset"]) == 4

    navigate(app, "Explore")
    assert not app.exception
    assert len(app.get("plotly_chart")) == 1

    navigate(app, "Models")
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["model_results"]


def test_navigation_requests_mobile_sidebar_collapse():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.get("html")
    navigate(app, "Data")
    assert not app.exception
    assert any("stSidebarCollapseButton" in h.proto.body for h in app.get("html"))
    app.run()
    assert not app.get("html")


def test_consecutive_navigation_emits_distinct_collapse_scripts():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    navigate(app, "Data")
    first = app.get("html")[0].proto.body
    navigate(app, "Explore")
    assert app.get("html")[0].proto.body != first
