import warnings

import pandas as pd
from scipy import stats

from config.settings import HIGH_CARDINALITY_THRESHOLD


def _normality_test(series: pd.Series) -> tuple[float | None, bool | None]:
    """Shapiro-Wilk for n ≤ 5000, D'Agostino-Pearson for larger samples.

    p ≥ 0.05 → fail to reject normality (treat as normal).
    Returns (p_value, is_normal). Both None when n < 3 or variance is zero.
    """
    vals = pd.to_numeric(series, errors="coerce").dropna()
    n = len(vals)
    if n < 3 or float(vals.std()) == 0:
        return None, None
    try:
        if n <= 5000:
            _, p = stats.shapiro(vals)
        else:
            _, p = stats.normaltest(vals)
        return round(float(p), 6), float(p) >= 0.05
    except Exception:
        return None, None


def _skew_label(skewness: float | None) -> str | None:
    if skewness is None or (isinstance(skewness, float) and pd.isna(skewness)):
        return None
    if abs(skewness) < 0.5:
        return "symmetric"
    if skewness >= 1.0:
        return "highly right-skewed"
    if skewness > 0:
        return "moderately right-skewed"
    if skewness <= -1.0:
        return "highly left-skewed"
    return "moderately left-skewed"


def _kurtosis_label(kurtosis: float | None) -> str | None:
    if kurtosis is None or (isinstance(kurtosis, float) and pd.isna(kurtosis)):
        return None
    if kurtosis > 1.0:
        return "leptokurtic"   # heavy tails, sharply peaked
    if kurtosis < -1.0:
        return "platykurtic"   # light tails, flat
    return "mesokurtic"        # normal-like tails


def _detect_datetime_gaps(series: pd.Series) -> dict:
    """Find gaps significantly larger than the median inter-event interval.

    A gap is flagged when it exceeds 3× the median interval.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            dt = pd.to_datetime(series, format="mixed", errors="coerce").dropna().sort_values()
        except Exception:
            dt = pd.to_datetime(series, errors="coerce").dropna().sort_values()
    if len(dt) < 2:
        return {"max_gap_days": None, "gap_count": 0, "median_gap_days": None}
    diffs_days = dt.diff().dropna().dt.days
    median_gap = float(diffs_days.median())
    threshold = max(1.0, median_gap * 3)
    return {
        "max_gap_days": int(diffs_days.max()),
        "gap_count": int((diffs_days > threshold).sum()),
        "median_gap_days": round(median_gap, 1),
    }


def analyze_distributions(df: pd.DataFrame, type_info: dict, stats_info: dict) -> dict:
    """Return per-column distribution analysis.

    Takes already-computed stats_info to avoid recomputing skewness/kurtosis.

    Returns:
        {col: {
            "normality_pvalue": float | None,
            "is_normal": bool | None,
            "skew_label": str | None,
            "kurtosis_label": str | None,
            "high_cardinality": bool | None,
            "datetime_gaps": dict | None,
        }}
    """
    result = {}
    for col in df.columns:
        inferred = type_info[col]["inferred_type"]
        series = df[col]
        col_stats = stats_info.get(col, {})

        if inferred in ("numeric", "mixed"):
            pval, is_normal = _normality_test(series)
            result[col] = {
                "normality_pvalue": pval,
                "is_normal": is_normal,
                "skew_label": _skew_label(col_stats.get("skewness")),
                "kurtosis_label": _kurtosis_label(col_stats.get("kurtosis")),
                "high_cardinality": None,
                "datetime_gaps": None,
            }

        elif inferred in ("categorical", "boolean"):
            non_null = series.dropna()
            card_ratio = non_null.nunique() / len(non_null) if len(non_null) > 0 else 0.0
            result[col] = {
                "normality_pvalue": None,
                "is_normal": None,
                "skew_label": None,
                "kurtosis_label": None,
                "high_cardinality": card_ratio > HIGH_CARDINALITY_THRESHOLD,
                "datetime_gaps": None,
            }

        elif inferred == "datetime":
            result[col] = {
                "normality_pvalue": None,
                "is_normal": None,
                "skew_label": None,
                "kurtosis_label": None,
                "high_cardinality": None,
                "datetime_gaps": _detect_datetime_gaps(series),
            }

        elif inferred == "text":
            # Text columns are high cardinality by definition — that's why they were
            # classified as text rather than categorical.
            result[col] = {
                "normality_pvalue": None,
                "is_normal": None,
                "skew_label": None,
                "kurtosis_label": None,
                "high_cardinality": True,
                "datetime_gaps": None,
            }

        else:  # unknown / mixed fallback
            result[col] = {
                "normality_pvalue": None,
                "is_normal": None,
                "skew_label": None,
                "kurtosis_label": None,
                "high_cardinality": None,
                "datetime_gaps": None,
            }

    return result
