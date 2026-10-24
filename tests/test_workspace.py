import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from media_mix_ai_lab.data.workspace import (
    map_weekly_dataset,
    quality_report,
    read_csv_upload,
)


def test_csv_mapping_preserves_values_and_reports_missing_and_zeros():
    content = b"week,sales,search,price\n2024-01-01,10,0,3\n2024-01-08,,2,0\n2024-01-15,12,,4\n"
    frame = read_csv_upload(content)
    original = frame.copy(deep=True)

    mapped, mapping = map_weekly_dataset(frame, "week", "sales", ("search",), ("price",))

    assert len(mapped) == len(frame) == 3
    assert_frame_equal(frame, original)
    assert mapped["week"].dtype == "datetime64[ns]"
    report = quality_report(mapped, mapping).set_index("Column")
    assert report.loc["sales", "Missing values"] == 1
    assert report.loc["sales", "Zero values"] == 0
    assert report.loc["search", "Missing values"] == 1
    assert report.loc["search", "Zero values"] == 1
    assert report.loc["price", "Zero values"] == 1


@pytest.mark.parametrize(
    "content, message",
    [
        (b"", "empty"),
        (b"\xff", "UTF-8"),
        (b"week,sales,sales\n2024-01-01,1,2\n", "unique"),
        (b"week,,sales\n2024-01-01,1,2\n", "empty"),
        (b"week,sales\n", "no data rows"),
    ],
)
def test_csv_reader_rejects_ambiguous_or_empty_input(content, message):
    with pytest.raises(ValueError, match=message):
        read_csv_upload(content)


@pytest.mark.parametrize(
    "dates, message",
    [
        (["bad", "2024-01-08", "2024-01-15"], "invalid dates"),
        (["2024-01-01", "2024-01-15", "2024-01-22"], "regular weekly"),
        (["2024-01-01", "2024-01-01", "2024-01-15"], "unique"),
    ],
)
def test_weekly_mapping_rejects_bad_date_series(dates, message):
    frame = pd.DataFrame({"week": dates, "sales": [1, 2, 3], "search": [4, 5, 6]})
    with pytest.raises(ValueError, match=message):
        map_weekly_dataset(frame, "week", "sales", ("search",))


def test_weekly_mapping_rejects_infinite_values_but_keeps_missing():
    frame = pd.DataFrame(
        {
            "week": pd.date_range("2024-01-01", periods=2, freq="7D"),
            "sales": [1, None],
            "search": [0, float("inf")],
        }
    )
    with pytest.raises(ValueError, match="infinite"):
        map_weekly_dataset(frame, "week", "sales", ("search",))


def test_weekly_mapping_requires_distinct_roles_and_a_channel():
    frame = pd.DataFrame(
        {
            "week": pd.date_range("2024-01-01", periods=2, freq="7D"),
            "sales": [1, 2],
            "search": [3, 4],
        }
    )
    with pytest.raises(ValueError, match="distinct"):
        map_weekly_dataset(frame, "week", "sales", ("search",), ("search",))
    with pytest.raises(ValueError, match="at least one channel"):
        map_weekly_dataset(frame, "week", "sales", ())
