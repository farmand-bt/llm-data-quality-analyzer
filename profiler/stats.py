import pandas as pd


def _numeric_stats(series: pd.Series) -> dict:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return {
            "mean": None, "median": None, "std": None,
            "min": None, "max": None, "skewness": None,
            "kurtosis": None, "zeros": 0, "unique_count": 0,
        }
    return {
        "mean": round(float(vals.mean()), 4),
        "median": round(float(vals.median()), 4),
        "std": round(float(vals.std()), 4),
        "min": round(float(vals.min()), 4),
        "max": round(float(vals.max()), 4),
        "skewness": round(float(vals.skew()), 4),
        "kurtosis": round(float(vals.kurtosis()), 4),
        "zeros": int((vals == 0).sum()),
        "unique_count": int(vals.nunique()),
    }


def _categorical_stats(series: pd.Series) -> dict:
    non_null = series.dropna()
    if len(non_null) == 0:
        return {"unique_count": 0, "top_values": {}, "cardinality_ratio": 0.0}
    top_vals = non_null.value_counts().head(5)
    return {
        "unique_count": int(non_null.nunique()),
        "top_values": {str(k): int(v) for k, v in top_vals.items()},
        "cardinality_ratio": round(non_null.nunique() / len(non_null), 4),
    }


def _datetime_stats(series: pd.Series) -> dict:
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        dt = series.dropna()
    else:
        try:
            dt = pd.to_datetime(series, format="mixed", errors="coerce").dropna()
        except Exception:
            dt = pd.to_datetime(series, errors="coerce").dropna()
    if len(dt) == 0:
        return {"min_date": None, "max_date": None, "range_days": None}
    return {
        "min_date": str(dt.min()),
        "max_date": str(dt.max()),
        "range_days": int((dt.max() - dt.min()).days),
    }


def _text_stats(series: pd.Series) -> dict:
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return {"unique_count": 0, "avg_length": 0.0, "max_length": 0, "min_length": 0}
    lengths = non_null.str.len()
    return {
        "unique_count": int(non_null.nunique()),
        "avg_length": round(float(lengths.mean()), 2),
        "max_length": int(lengths.max()),
        "min_length": int(lengths.min()),
    }


def compute_stats(df: pd.DataFrame, type_info: dict) -> dict:
    """Return per-column statistics. Contents vary by inferred_type."""
    result = {}
    for col in df.columns:
        inferred = type_info[col]["inferred_type"]
        series = df[col].dropna()

        if inferred in ("numeric", "mixed"):
            result[col] = _numeric_stats(series)
        elif inferred in ("categorical", "boolean"):
            result[col] = _categorical_stats(series)
        elif inferred == "datetime":
            result[col] = _datetime_stats(series)
        elif inferred == "text":
            result[col] = _text_stats(series)
        else:
            result[col] = {}

    return result
