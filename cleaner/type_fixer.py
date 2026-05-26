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
        "datetime"    — pd.to_datetime with format='mixed' (coerce on failure)
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
            try:
                out[column] = pd.to_datetime(out[column], format="mixed", errors="coerce")
            except Exception:
                out[column] = pd.to_datetime(out[column], errors="coerce")

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
        out[column] = out[column].where(
            out[column].isna(),
            out[column].astype(str).str.strip().str.lower(),
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
