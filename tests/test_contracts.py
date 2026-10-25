import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from media_mix_ai_lab.ai.interfaces import SummaryRequest, summarize_column
from media_mix_ai_lab.core.config import AdstockSpec, ModelConfig
from media_mix_ai_lab.data.synthetic import generate_sample_data


def test_model_configuration_round_trip():
    config = ModelConfig(
        kind="log_log",
        adstock={
            "tv_spend": AdstockSpec(
                method="delayed_geometric",
                alpha=0.7,
                max_lag_weeks=8,
                theta=2,
                weight_scaling="sum_to_one",
            )
        },
    )
    assert ModelConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize(
    "settings",
    [
        {"adstock": {"tv_spend": {"method": "geometric", "alpha": 1}}},
        {"adstock": {"tv_spend": {"method": "delayed_geometric", "alpha": 0}}},
        {"adstock": {"tv_spend": {"method": "delayed_geometric", "theta": 9}}},
        {"adstock": {"unknown": {"method": "geometric"}}},
        {"kind": "unknown"},
        {"channels": ()},
        {"channels": ("revenue",)},
        {"controls": ("tv_spend",)},
        {"execute_python": "print(1)"},
        {"adstock": {"tv_spend": {"method": "delayed_geometric", "max_lag_weeks": 13}}},
        {"media_sign_expectations": {"unknown_channel": "positive"}},
        {"media_sign_expectations": {"tv_spend": "impossible"}},
        {"diagnostic_thresholds": {"poor_condition_number": 1}},
    ],
)
def test_model_config_rejects_invalid_settings(settings):
    with pytest.raises(ValidationError):
        ModelConfig(**settings)


def test_summary_uses_real_data_and_preserves_outlier():
    frame = generate_sample_data()
    original = frame.copy(deep=True)
    result = summarize_column(frame, SummaryRequest(column="revenue"))
    assert result.count == 104
    assert result.mean == pytest.approx(frame.revenue.mean())
    assert result.minimum == frame.revenue.min()
    assert result.maximum == frame.revenue.max()
    assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "payload",
    [
        {"column": "__import__('os')"},
        {"column": "revenue", "tool": "execute_python"},
        {"column": "revenue", "code": "print(1)"},
    ],
)
def test_tools_reject_unsupported_commands(payload):
    with pytest.raises(ValidationError):
        SummaryRequest.model_validate(payload)
