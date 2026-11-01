import re
import warnings

import pandas as pd

from config.settings import HIGH_CARDINALITY_THRESHOLD

_NUMERIC_THRESHOLD = 0.80
_DATETIME_THRESHOLD = 0.80
_BOOL_VALUES = {"true", "false", "yes", "no", "y", "n"}
# ISO 8601 date, optionally with a time and then a timezone (Z or +HH:MM); the
# timezone can only follow a time, so "2020-01-15" never reads as an offset
_ISO_DATETIME = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?P<tz>Z|[+-]\d{2}:\d{2})?)?",
    re.IGNORECASE,
)


def parse_datetimes(series: pd.Series) -> pd.Series:
    """Parse values as datetimes (NaT where unparseable), tolerating mixed formats.

    format="mixed" parses each value individually, so a column mixing ISO
    dates, DD/MM/YYYY and "Jan 15 2020" still parses. Timestamps with different
    UTC offsets ("...Z" and "...+02:00") can't share one timezone-aware dtype:
    pandas 3 raises and pandas 2 returns objects, so they are parsed to UTC
    (ported from data-quality-mcp #14). Used by type detection, stats and
    distributions alike.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # UTC comes before the plain fallback, which infers one format from the
        # first value and would silently null differently formatted ones
        for options in ({"format": "mixed"}, {"format": "mixed", "utc": True}, {}):
            try:
                parsed = pd.to_datetime(series, errors="coerce", **options)
            except (TypeError, ValueError):
                continue
            if pd.api.types.is_datetime64_any_dtype(parsed.dtype):
                return parsed
    return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")


def _all_iso(values: pd.Series) -> bool:
    """Every value is an ISO 8601 date or timestamp that parses, and either all
    of them carry a timezone or none does (a mix is ambiguous)."""
    text = pd.Series(values.astype(str).str.strip().unique())
    if not text.str.fullmatch(_ISO_DATETIME.pattern, case=False).all():
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(text, format="ISO8601", utc=True, errors="coerce")
    if parsed.isna().any():
        return False
    has_tz = text.str.extract(_ISO_DATETIME, expand=True)["tz"].notna()
    return bool(has_tz.all() or not has_tz.any())


def _infer_type(series: pd.Series, iso_dates_typed: bool = False) -> tuple[str, bool]:
    """Return (inferred_type, type_mismatch) for a single series.

    With `iso_dates_typed` (set for CSV files, which have no date type), text
    whose values are all ISO 8601 dates in one timezone convention is not a
    type mismatch.
    """
    dtype = series.dtype

    if pd.api.types.is_bool_dtype(dtype):
        return "boolean", False
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric", False
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime", False

    non_null = series.dropna()
    if len(non_null) == 0:
        return "categorical", False

    # Boolean-like strings — require at least 2 distinct values to avoid false positives
    unique_lower = set(non_null.astype(str).str.strip().str.lower().unique())
    if unique_lower.issubset(_BOOL_VALUES) and len(unique_lower) >= 2:
        return "boolean", True

    # Numeric check
    numeric_parsed = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = float(numeric_parsed.notna().mean())
    if numeric_ratio == 1.0:
        return "numeric", True  # all values parse → stored as string
    if numeric_ratio >= _NUMERIC_THRESHOLD:
        return "mixed", True  # mostly numeric with stray non-numerics

    # Datetime check — only when the column isn't mostly numeric-looking.
    # format="mixed" lets pandas parse each value individually, which handles columns
    # that contain dates in more than one format (e.g. ISO + DD/MM/YYYY + "Jan 15 2020").
    if numeric_ratio < 0.5:
        datetime_ratio = float(parse_datetimes(non_null).notna().mean())
        if datetime_ratio >= _DATETIME_THRESHOLD:
            # CSV has no date type, so ISO text is how a CSV stores dates
            return "datetime", not (iso_dates_typed and _all_iso(non_null))

    # Cardinality-based: categorical vs free text
    # <= threshold: a 50% unique ratio still counts as categorical
    cardinality_ratio = non_null.nunique() / len(non_null)
    if cardinality_ratio <= HIGH_CARDINALITY_THRESHOLD:
        # Detect case-inconsistent labels (e.g. "Male"/"MALE"/"M")
        if non_null.astype(str).str.strip().str.lower().nunique() < non_null.nunique():
            return "categorical", True
        return "categorical", False

    return "text", False


def detect_types(df: pd.DataFrame, iso_dates_typed: bool = False) -> dict:
    """Return inferred type info per column.

    `iso_dates_typed`: the data came from a CSV, so ISO date text is typed.

    Returns:
        {col: {"pandas_dtype": str, "inferred_type": str, "type_mismatch": bool}}
    """
    result = {}
    for col in df.columns:
        inferred_type, type_mismatch = _infer_type(df[col], iso_dates_typed)
        result[col] = {
            "pandas_dtype": str(df[col].dtype),
            "inferred_type": inferred_type,
            "type_mismatch": type_mismatch,
        }
    return result
