import warnings

import pandas as pd


_BOOL_MAP = {
    "true": True, "false": False,
    "yes": True, "no": False,
    "y": True, "n": False,
    "1": True, "0": False,
}


def fix_types(
    df: pd.DataFrame,
    column: str,
    target_type: str,
) -> tuple[pd.DataFrame, dict]:
    """Return a new DataFrame with the column cast to target_type and a log entry.

    target_type:
        "numeric"     — strip commas/currency, then pd.to_numeric (coerce on failure)
        "datetime"    — pd.to_datetime with format='mixed' (coerce on failure);
                        mixed UTC offsets become UTC, and a mix of values with
                        and without a timezone raises ValueError
        "categorical" — pandas Categorical dtype (efficient storage for low-cardinality cols)
        "string"      — object/string, preserving NaN
        "boolean"     — map common boolean strings (true/yes/1 → True, etc.)

    Never modifies the input DataFrame.
    """
    out = df.copy()
    original_dtype = str(out[column].dtype)

    if target_type == "numeric":
        cleaned = (
            out[column].astype(str)
            .str.replace(r"[,$%€£\s]", "", regex=True)
            .replace("nan", float("nan"))
        )
        out[column] = pd.to_numeric(cleaned, errors="coerce")

    elif target_type == "datetime":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # pandas 2 warns about mixed timezones before returning objects
            warnings.simplefilter("ignore", FutureWarning)
            try:
                parsed = pd.to_datetime(out[column], format="mixed", errors="coerce")
            except Exception:
                parsed = None
            # pandas 2 returns an object column for mixed UTC offsets, where
            # pandas 3 raises: both go to the fallback
            if parsed is None or not pd.api.types.is_datetime64_any_dtype(parsed):
                parsed = _to_datetime_fallback(out[column], column)
            out[column] = parsed

    elif target_type == "categorical":
        out[column] = out[column].astype("category")

    elif target_type == "string":
        # Preserve NaN; convert everything else to str
        out[column] = out[column].where(out[column].isna(), out[column].astype(str))

    elif target_type == "boolean":
        out[column] = (
            out[column].astype(str).str.strip().str.lower().map(_BOOL_MAP)
        )

    elif target_type == "lowercase":
        out[column] = (
            out[column]
            .where(out[column].isna(), out[column].astype(str).str.strip().str.lower())
            .astype("category")
        )

    else:
        raise ValueError(f"Unknown target_type '{target_type}'")

    return out, {
        "action": "fix_type",
        "column": column,
        "strategy": f"to_{target_type}",
        "original_dtype": original_dtype,
        "new_dtype": str(out[column].dtype),
        "rows_before": len(df),
        "rows_after": len(out),
        "rows_removed": 0,
    }


def _timezone_awareness(series: pd.Series) -> set[bool]:
    """Whether each distinct value that parses as a timestamp carries a
    timezone (True) or not (False). Values that don't parse are left out:
    they become NaT anyway."""
    awareness = set()
    for value in pd.unique(series.dropna().astype(str).str.strip()):
        try:
            awareness.add(pd.Timestamp(value).tzinfo is not None)
        except (ValueError, TypeError, OverflowError):
            continue
    return awareness


def _to_datetime_fallback(series: pd.Series, column: str) -> pd.Series:
    """The previous fallback, plus a rule for timezone offsets (ported from
    data-quality-mcp #14).

    Timestamps with different UTC offsets can't share one timezone-aware
    dtype, so they are converted to UTC, which keeps every instant. A mix of
    timestamps with and without a timezone is ambiguous, so it raises instead
    of guessing one (the plain fallback would silently null some of them).
    """
    awareness = _timezone_awareness(series)
    if awareness == {True}:
        return pd.to_datetime(series, format="mixed", errors="coerce", utc=True)
    if True not in awareness:
        return pd.to_datetime(series, errors="coerce")
    raise ValueError(
        f"'{column}' mixes timestamps with and without a timezone (a UTC offset "
        "such as +02:00, or Z), so converting it would mean guessing a timezone "
        "for some of them. Give every value an offset, or none, then convert."
    )
