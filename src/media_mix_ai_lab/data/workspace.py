"""CSV import, weekly column mapping, and quality inspection for the data workspace."""

import csv
import io
from typing import TypedDict

import numpy as np
import pandas as pd


class ColumnMapping(TypedDict):
    date: str
    target: str
    channels: tuple[str, ...]
    controls: tuple[str, ...]


def read_csv_upload(content: bytes) -> pd.DataFrame:
    """Read a UTF-8 CSV while rejecting ambiguous or empty headers."""
    if not content:
        raise ValueError("The uploaded CSV is empty.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV files must use UTF-8 encoding.") from error
    try:
        records = csv.reader(io.StringIO(text), strict=True)
        header = next(records, None)
        if not header:
            raise ValueError("The uploaded CSV must include a header row.")
        if any(not name.strip() for name in header):
            raise ValueError("CSV column names cannot be empty.")
        if len(header) != len(set(header)):
            raise ValueError("CSV column names must be unique.")
        for record in records:
            if record and len(record) != len(header):
                raise ValueError(
                    f"CSV record ending on line {records.line_num} has {len(record)} "
                    f"fields; expected {len(header)}."
                )
    except csv.Error as error:
        raise ValueError(f"Could not parse CSV: {error}") from error
    try:
        frame = pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError as error:
        raise ValueError("The uploaded CSV has no data rows.") from error
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not read CSV: {error}") from error
    if frame.empty:
        raise ValueError("The uploaded CSV has no data rows.")
    return frame


def validate_weekly_dataset(frame: pd.DataFrame, mapping: ColumnMapping) -> None:
    """Validate mapped weekly columns without dropping, sorting, or filling rows."""
    if frame.empty:
        raise ValueError("The dataset is empty.")
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique.")
    date_column = mapping["date"]
    numeric_columns = (mapping["target"], *mapping["channels"], *mapping["controls"])
    if not mapping["channels"]:
        raise ValueError("Map at least one channel column.")
    selected_columns = (date_column, *numeric_columns)
    if len(set(selected_columns)) != len(selected_columns):
        raise ValueError("Date, target, channel, and control mappings must be distinct.")
    missing_columns = set(selected_columns) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Mapped columns are missing: {', '.join(sorted(missing_columns))}")
    dates = frame[date_column]
    if not pd.api.types.is_datetime64_any_dtype(dates):
        raise ValueError(f"{date_column} must contain parsed datetime values.")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be nonmissing, unique, and in ascending order.")
    if (
        len(dates) > 1
        and not (
            dates.diff().iloc[1:].to_numpy(dtype="timedelta64[ns]") == np.timedelta64(7, "D")
        ).all()
    ):
        raise ValueError("Dates must have a regular weekly frequency; rows are never filled in.")

    for column in numeric_columns:
        values = frame[column]
        if (
            not pd.api.types.is_numeric_dtype(values)
            or pd.api.types.is_bool_dtype(values)
            or pd.api.types.is_complex_dtype(values)
        ):
            raise ValueError(f"{column} must be numeric.")
        observed = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all():
            raise ValueError(f"{column} must not contain infinite values.")


def map_weekly_dataset(
    frame: pd.DataFrame,
    date_column: str,
    target_column: str,
    channel_columns: tuple[str, ...],
    control_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, ColumnMapping]:
    """Parse and validate selected weekly columns while preserving all source rows."""
    mapping: ColumnMapping = {
        "date": date_column,
        "target": target_column,
        "channels": channel_columns,
        "controls": control_columns,
    }
    if date_column not in frame.columns:
        raise ValueError(f"Date column not found: {date_column}")
    mapped = frame.copy(deep=True)
    try:
        mapped[date_column] = pd.to_datetime(frame[date_column], errors="coerce", format="mixed")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{date_column} could not be parsed as dates.") from error
    if mapped[date_column].isna().any():
        raise ValueError(f"{date_column} contains missing or invalid dates.")
    validate_weekly_dataset(mapped, mapping)
    return mapped, mapping


def quality_report(frame: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    """Summarize missing and zero values without changing the supplied dataset."""
    roles = {
        mapping["date"]: "Date",
        mapping["target"]: "Target",
        **dict.fromkeys(mapping["channels"], "Channel"),
        **dict.fromkeys(mapping["controls"], "Control"),
    }
    rows = []
    for column in frame.columns:
        values = frame[column]
        missing = int(values.isna().sum())
        zeros = (
            int(values.eq(0).fillna(False).sum())
            if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
            else pd.NA
        )
        rows.append(
            {
                "Column": column,
                "Role": roles.get(column, "Unmapped"),
                "Missing values": missing,
                "Missing %": round(100 * missing / len(frame), 2),
                "Zero values": zeros,
            }
        )
    return pd.DataFrame(rows)
