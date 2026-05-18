import pandas as pd

from config.settings import OUTLIER_IQR_MULTIPLIER


def handle_outliers(
    df: pd.DataFrame,
    column: str,
    indices: list,
    strategy: str,
) -> tuple[pd.DataFrame, dict]:
    """Return a new DataFrame with outliers treated and a log entry.

    indices: row labels from profiler/outliers.py (iqr_indices or zscore_indices).

    strategies:
        remove      — drop outlier rows entirely
        cap_iqr     — clip values to [Q1 − 1.5×IQR, Q3 + 1.5×IQR] (winsorization)
        replace_nan — replace outlier cell values with NaN (treat as missing later)

    Never modifies the input DataFrame.
    """
    out = df.copy()
    # Only act on indices still present (a previous step might have removed some rows)
    valid_indices = [i for i in indices if i in out.index]
    n_treated = len(valid_indices)
    rows_before = len(out)

    if strategy == "remove":
        out = out.drop(index=valid_indices).reset_index(drop=True)

    elif strategy == "cap_iqr":
        vals = pd.to_numeric(out[column], errors="coerce")
        q1 = float(vals.quantile(0.25))
        q3 = float(vals.quantile(0.75))
        iqr = q3 - q1
        lo = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        hi = q3 + OUTLIER_IQR_MULTIPLIER * iqr
        out[column] = vals.clip(lower=lo, upper=hi)

    elif strategy == "replace_nan":
        out.loc[valid_indices, column] = float("nan")

    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    return out, {
        "action": "handle_outliers",
        "column": column,
        "strategy": strategy,
        "outliers_treated": n_treated,
        "rows_before": rows_before,
        "rows_after": len(out),
        "rows_removed": rows_before - len(out),
    }
