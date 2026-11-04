import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from config.settings import HIGH_CORRELATION_THRESHOLD, MIN_CORRELATION_ROWS

# Cap categorical columns for Cramér's V to avoid O(n_cat²) blowing up
_MAX_CAT_COLS = 15


def corrected_cramers_v(chi2: float, n: int, shape: tuple[int, int]) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013) from a chi-square without
    Yates' correction on an r x c table of n rows. It subtracts what chance
    gives a table of that shape, so unrelated columns come out near 0 even on
    sparse tables, where the plain V can reach 0.9. 0.0 when no association
    remains, or when a side has a different value on every row."""
    rows, cols = shape
    phi2 = max(0.0, chi2 / n - (rows - 1) * (cols - 1) / (n - 1))
    rows_c = rows - (rows - 1) ** 2 / (n - 1)
    cols_c = cols - (cols - 1) ** 2 / (n - 1)
    denominator = min(rows_c - 1, cols_c - 1)
    return float(np.sqrt(phi2 / denominator)) if denominator > 0 else 0.0


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V between two categorical series (0 to 1), over
    the rows where both are present; 0.0 when either has fewer than 2
    categories there. Rows are matched by index label, as pd.crosstab does.

    Ported from data-quality-mcp (Milestone 3.5, Amendment: bias-corrected
    Cramér's V). The plain V, with Yates' correction on 2 x 2 tables, gave
    unrelated many-level columns on few rows a value of 0.9 or more.
    """
    x, y = x.align(y, join="inner")
    return _cramers_v_codes(pd.factorize(x)[0], pd.factorize(y)[0])


def _shared_codes(codes: np.ndarray) -> tuple[np.ndarray, int]:
    """Renumber codes 0..k-1 by the categories that occur, and return k."""
    occurs = np.bincount(codes) > 0
    return (np.cumsum(occurs) - 1)[codes], int(occurs.sum())


def _cramers_v_codes(a: np.ndarray, b: np.ndarray) -> float:
    """_cramers_v from factorize codes of the whole columns (-1 = missing), so
    each column is factorized once rather than once per pair."""
    both = (a >= 0) & (b >= 0)
    if not both.any():
        return 0.0
    # Only the categories on the shared rows, so every row and column of the
    # table has counts (as with pd.crosstab), which chi2_contingency requires
    a, ka = _shared_codes(a[both])
    b, kb = _shared_codes(b[both])
    if ka < 2 or kb < 2:
        return 0.0
    # The table by counting codes (much faster than pd.crosstab)
    table = np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb)
    chi2 = float(chi2_contingency(table, correction=False)[0])
    return corrected_cramers_v(chi2, len(a), table.shape)


def compute_correlations(df: pd.DataFrame, type_info: dict) -> dict:
    """Return Pearson correlation matrix, Cramér's V for categoricals, and high-correlation pairs.

    A pair with fewer than MIN_CORRELATION_ROWS rows holding both values is None
    and never a high pair: two rows always give r = +/-1 (ported from
    data-quality-mcp, Milestone 3.5 D5). Cramér's V is bias-corrected (see
    _cramers_v).

    Returns:
        {
            "pearson": {col_a: {col_b: float | None}},
            "cramers_v": {col_a: {col_b: float | None}},
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
        matrix = num_df.corr(method="pearson", min_periods=MIN_CORRELATION_ROWS)

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
        codes = {col: pd.factorize(df[col])[0] for col in cat_limited}
        for i, col_a in enumerate(cat_limited):
            cramers[col_a] = {}
            for col_b in cat_limited[i + 1 :]:
                shared = int(((codes[col_a] >= 0) & (codes[col_b] >= 0)).sum())
                if shared < MIN_CORRELATION_ROWS:
                    cramers[col_a][col_b] = None
                    continue
                v = _cramers_v_codes(codes[col_a], codes[col_b])
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
