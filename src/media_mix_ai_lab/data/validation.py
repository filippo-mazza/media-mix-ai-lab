"""Validate without silently cleaning, sorting, dropping, or imputing rows."""

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = (
    "revenue",
    "tv_spend",
    "search_spend",
    "social_spend",
    "price_index",
    "promotions",
    "seasonality",
)
REQUIRED_COLUMNS = ("date", *NUMERIC_COLUMNS)


def validate_dataset(frame: pd.DataFrame, allow_negative_target_media: bool = False) -> None:
    """Reject incompatible inputs without mutation; optionally defer negativity to model checks."""
    if frame.empty:
        raise ValueError("The dataset is empty.")
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique.")
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if not pd.api.types.is_datetime64_any_dtype(frame["date"]):
        raise ValueError("date must contain parsed datetime values.")
    dates = frame["date"]
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be nonmissing, unique, and in ascending order.")
    if (
        len(dates) > 1
        and not (
            dates.diff().iloc[1:].to_numpy(dtype="timedelta64[ns]") == np.timedelta64(7, "D")
        ).all()
    ):
        raise ValueError("Dates must have a regular weekly frequency; rows are never filled in.")
    for column in NUMERIC_COLUMNS:
        values = frame[column]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            raise ValueError(f"{column} must be numeric.")
        if not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all():
            raise ValueError(f"{column} must contain finite, nonmissing values.")
    if (
        not allow_negative_target_media
        and (frame[["revenue", "tv_spend", "search_spend", "social_spend"]] < 0).any().any()
    ):
        raise ValueError("Revenue and advertising spend cannot be negative.")
    if (frame["price_index"] <= 0).any():
        raise ValueError("price_index must be positive.")
    if not frame["promotions"].isin([0, 1]).all():
        raise ValueError("promotions must contain 0 or 1.")
