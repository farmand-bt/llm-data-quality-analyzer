import warnings

import pandas as pd

from config.settings import HIGH_CARDINALITY_THRESHOLD

_NUMERIC_THRESHOLD = 0.80
_DATETIME_THRESHOLD = 0.80
_BOOL_VALUES = {"true", "false", "yes", "no", "y", "n"}


def _infer_type(series: pd.Series) -> tuple[str, bool]:
    """Return (inferred_type, type_mismatch) for a single series."""
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
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                try:
                    dt_parsed = pd.to_datetime(non_null, format="mixed", errors="coerce")
                except Exception:
                    dt_parsed = pd.to_datetime(non_null, errors="coerce")
            datetime_ratio = float(dt_parsed.notna().mean())
            if datetime_ratio >= _DATETIME_THRESHOLD:
                return "datetime", True
        except Exception:
            pass

    # Cardinality-based: categorical vs free text
    # <= threshold: a 50% unique ratio still counts as categorical
    cardinality_ratio = non_null.nunique() / len(non_null)
    if cardinality_ratio <= HIGH_CARDINALITY_THRESHOLD:
        # Detect case-inconsistent labels (e.g. "Male"/"MALE"/"M")
        if non_null.astype(str).str.strip().str.lower().nunique() < non_null.nunique():
            return "categorical", True
        return "categorical", False

    return "text", False


def detect_types(df: pd.DataFrame) -> dict:
    """Return inferred type info per column.

    Returns:
        {col: {"pandas_dtype": str, "inferred_type": str, "type_mismatch": bool}}
    """
    result = {}
    for col in df.columns:
        inferred_type, type_mismatch = _infer_type(df[col])
        result[col] = {
            "pandas_dtype": str(df[col].dtype),
            "inferred_type": inferred_type,
            "type_mismatch": type_mismatch,
        }
    return result
