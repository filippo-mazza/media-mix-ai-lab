from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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
    app.button[0].click().run()

    assert not app.exception
    assert app.session_state["model_results"]
    assert any(metric.label == "Sample size" for metric in app.metric)
    assert any(button.label == "Save result as JSON" for button in app.get("download_button"))


def test_models_report_strict_log_domain_errors():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    navigate(app, "Models")
    app.selectbox[0].set_value("log_log").run()
    app.session_state["dataset"].loc[0, "tv_spend"] = 0
    app.button[0].click().run()

    assert not app.exception
    assert "tv_spend: 1 nonpositive value" in app.error[0].value


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
