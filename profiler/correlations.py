import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from config.settings import HIGH_CORRELATION_THRESHOLD

# Cap categorical columns for Cramér's V to avoid O(n_cat²) blowing up
_MAX_CAT_COLS = 15


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V association between two categorical series."""
    ct = pd.crosstab(x.dropna(), y.dropna())
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(ct)
    n = int(ct.sum().sum())
    min_dim = min(ct.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def compute_correlations(df: pd.DataFrame, type_info: dict) -> dict:
    """Return Pearson correlation matrix, Cramér's V for categoricals, and high-correlation pairs.

    Returns:
        {
            "pearson": {col_a: {col_b: float | None}},
            "cramers_v": {col_a: {col_b: float}},
            "high_correlation_pairs": [
                {"col_a": str, "col_b": str, "method": str, "value": float}
            ],
        }
    """
    numeric_cols = [
        col for col in df.columns
        if type_info[col]["inferred_type"] in ("numeric", "mixed")
    ]
    cat_cols = [
        col for col in df.columns
        if type_info[col]["inferred_type"] in ("categorical", "boolean")
    ]

    pearson: dict = {}
    cramers: dict = {}
    high_pairs: list[dict] = []

    # --- Pearson (numeric-numeric) ---
    if len(numeric_cols) >= 2:
        num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        matrix = num_df.corr(method="pearson")

        for col_a in numeric_cols:
            pearson[col_a] = {}
            for col_b in numeric_cols:
                val = matrix.loc[col_a, col_b]
                pearson[col_a][col_b] = round(float(val), 4) if not pd.isna(val) else None

        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1 :]:
                val = matrix.loc[col_a, col_b]
                if not pd.isna(val) and abs(val) >= HIGH_CORRELATION_THRESHOLD:
                    high_pairs.append({
                        "col_a": col_a,
                        "col_b": col_b,
                        "method": "pearson",
                        "value": round(float(val), 4),
                    })

    # --- Cramér's V (categorical-categorical) ---
    cat_limited = cat_cols[:_MAX_CAT_COLS]
    if len(cat_limited) >= 2:
        for i, col_a in enumerate(cat_limited):
            cramers[col_a] = {}
            for col_b in cat_limited[i + 1 :]:
                v = _cramers_v(df[col_a], df[col_b])
                cramers[col_a][col_b] = round(v, 4)
                if v >= HIGH_CORRELATION_THRESHOLD:
                    high_pairs.append({
                        "col_a": col_a,
                        "col_b": col_b,
                        "method": "cramers_v",
                        "value": round(v, 4),
                    })

    high_pairs.sort(key=lambda p: abs(p["value"]), reverse=True)

    return {
        "pearson": pearson,
        "cramers_v": cramers,
        "high_correlation_pairs": high_pairs,
    }
