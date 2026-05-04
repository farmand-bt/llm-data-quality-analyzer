import pandas as pd


def _classify_missing_pattern(col: str, df: pd.DataFrame, missing_pct: float) -> str:
    """Heuristic classification of the missing mechanism for a column.

    Returns "none", "MCAR", "MAR", or "MNAR".
    - MNAR: >60% missing (likely structurally absent based on the value itself)
    - MAR: missingness strongly co-occurs with another column's missingness
    - MCAR: no detectable pattern
    """
    if missing_pct == 0:
        return "none"
    if missing_pct > 60:
        return "MNAR"

    missing_mask = df[col].isnull()
    n_missing = missing_mask.sum()
    for other_col in df.columns:
        if other_col == col:
            continue
        other_missing = df[other_col].isnull().sum()
        if other_missing == 0:
            continue
        overlap = int((missing_mask & df[other_col].isnull()).sum())
        if overlap / n_missing > 0.7:
            return "MAR"

    return "MCAR"


def analyze_missing(df: pd.DataFrame) -> dict:
    """Return missing value analysis: overall stats, per-column counts, and patterns.

    Returns:
        {
            "overall_missing_pct": float,
            "total_missing_cells": int,
            "per_column": {col: {"missing_count", "missing_pct", "missing_pattern"}},
            "co_occurrence": {"{col_a}|{col_b}": int},  # rows where both are missing
        }
    """
    total_cells = df.size
    total_missing = int(df.isnull().sum().sum())
    overall_pct = round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0.0

    per_column = {}
    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        missing_pct = round(missing_count / len(df) * 100, 2)
        per_column[col] = {
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "missing_pattern": _classify_missing_pattern(col, df, missing_pct),
        }

    # Pairwise co-occurrence for columns that have any missing values
    missing_cols = [c for c in df.columns if df[c].isnull().any()]
    co_occurrence: dict[str, int] = {}
    if len(missing_cols) > 1:
        null_matrix = df[missing_cols].isnull()
        for i, col_a in enumerate(missing_cols):
            for col_b in missing_cols[i + 1 :]:
                both = int((null_matrix[col_a] & null_matrix[col_b]).sum())
                if both > 0:
                    co_occurrence[f"{col_a}|{col_b}"] = both

    return {
        "overall_missing_pct": overall_pct,
        "total_missing_cells": total_missing,
        "per_column": per_column,
        "co_occurrence": co_occurrence,
    }
