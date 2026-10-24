from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.mark.parametrize(
    "section", ["Home", "Data", "Explore", "Models", "AI Assistant", "ROI Analysis"]
)
def test_workbench_sections_render_without_credentials(section):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.sidebar.radio[0].set_value(section).run()
    assert not app.exception
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
        app.button[0].click().run()
        assert not app.exception
    elif section == "Models":
        app.selectbox[0].set_value("log_log").run()
        assert not app.exception


def test_invalid_dataset_has_actionable_recovery():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    app.session_state["dataset"] = app.session_state["dataset"].drop(columns="revenue")
    app.run()
    assert not app.exception
    assert "Missing required columns: revenue" in app.error[0].value
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
