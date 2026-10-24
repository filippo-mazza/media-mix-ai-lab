from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from media_mix_ai_lab.data.synthetic import (
    MAX_WEEKLY_MEDIA_INVESTMENT,
    OUTLIER_LIFT,
    OUTLIER_POSITION,
    generate_sample_data,
)
from media_mix_ai_lab.data.validation import validate_dataset


def test_sample_is_reproducible_and_independent():
    first = generate_sample_data()
    second = generate_sample_data()
    assert_frame_equal(first, second)
    first.loc[0, "revenue"] = 0
    assert second.loc[0, "revenue"] > 0
    assert not second.equals(generate_sample_data(seed=43))


def test_sample_schema_frequency_and_outlier():
    frame = generate_sample_data()
    validate_dataset(frame)
    assert len(frame) == 104
    assert (np.diff(frame["date"].to_numpy()) == np.timedelta64(7, "D")).all()
    assert frame["is_injected_outlier"].sum() == 1
    assert frame.loc[OUTLIER_POSITION, "is_injected_outlier"]
    weekly_media = frame[["tv_spend", "search_spend", "social_spend"]].sum(axis=1)
    assert weekly_media.max() <= MAX_WEEKLY_MEDIA_INVESTMENT
    assert weekly_media.max() > 9_500
    baseline = (
        12_000
        + 0.55 * frame.tv_spend
        + 0.85 * frame.search_spend
        + 0.45 * frame.social_spend
        - 90 * (frame.price_index - 100)
        + 1_500 * frame.promotions
        + 2_000 * frame.seasonality
    )
    residual = frame.revenue - baseline
    assert residual.iloc[OUTLIER_POSITION] > OUTLIER_LIFT - 1_500
    assert residual.drop(index=OUTLIER_POSITION).abs().max() < 2_000
    assert frame["revenue"].max() < 35_000


def test_committed_csv_matches_generator():
    sample = Path(__file__).resolve().parents[1] / "data/samples/weekly_marketing.csv"
    frame = pd.read_csv(sample, parse_dates=["date"])
    assert_frame_equal(frame, generate_sample_data())


@pytest.mark.parametrize("kwargs", [{"n_weeks": 65}, {"n_weeks": True}, {"seed": -1}])
def test_generator_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        generate_sample_data(**kwargs)


def test_validation_preserves_all_rows_and_values():
    frame = generate_sample_data()
    original = frame.copy(deep=True)
    validate_dataset(frame)
    assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "nan",
        "infinite",
        "negative",
        "dates",
        "gap",
        "duplicate",
        "promotion",
        "price",
        "string",
        "empty",
    ],
)
def test_validation_rejects_bad_data_without_mutation(problem):
    frame = generate_sample_data()
    if problem == "missing":
        frame = frame.drop(columns="revenue")
    elif problem == "nan":
        frame.loc[0, "revenue"] = np.nan
    elif problem == "infinite":
        frame.loc[0, "tv_spend"] = np.inf
    elif problem == "negative":
        frame.loc[0, "search_spend"] = -1
    elif problem == "dates":
        frame = frame.iloc[::-1].copy()
    elif problem == "gap":
        frame = frame.drop(index=2)
    elif problem == "duplicate":
        frame.loc[1, "date"] = frame.loc[0, "date"]
    elif problem == "promotion":
        frame.loc[0, "promotions"] = 2
    elif problem == "price":
        frame.loc[0, "price_index"] = 0
    elif problem == "string":
        frame["revenue"] = frame["revenue"].astype(str)
    else:
        frame = frame.iloc[:0]
    original = frame.copy(deep=True)
    with pytest.raises(ValueError):
        validate_dataset(frame)
    assert_frame_equal(frame, original)
