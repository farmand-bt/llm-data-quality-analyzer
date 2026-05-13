import pandas as pd

from config.settings import OUTLIER_IQR_MULTIPLIER, OUTLIER_ZSCORE_THRESHOLD


def detect_outliers(df: pd.DataFrame, type_info: dict) -> dict:
    """Return IQR and Z-score outlier counts, percentages, and row indices per numeric column.

    Only numeric and mixed columns are analysed; all others return zeros.
    The returned indices are integer row labels from df.index — used by cleaner/pipeline.py.

    Returns:
        {col: {
            "iqr_count": int, "iqr_pct": float, "iqr_indices": list[int],
            "zscore_count": int, "zscore_pct": float, "zscore_indices": list[int],
        }}
    """
    result = {}
    n_total = len(df)
    _empty = {"iqr_count": 0, "iqr_pct": 0.0, "iqr_indices": [],
               "zscore_count": 0, "zscore_pct": 0.0, "zscore_indices": []}

    for col in df.columns:
        if type_info[col]["inferred_type"] not in ("numeric", "mixed"):
            result[col] = dict(_empty)
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()

        if len(valid) < 4:
            result[col] = dict(_empty)
            continue

        # --- IQR method ---
        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))
        iqr = q3 - q1

        if iqr == 0:
            iqr_mask = pd.Series(False, index=series.index)
        else:
            lo = q1 - OUTLIER_IQR_MULTIPLIER * iqr
            hi = q3 + OUTLIER_IQR_MULTIPLIER * iqr
            iqr_mask = series.notna() & ((series < lo) | (series > hi))

        # --- Z-score method ---
        mean = float(valid.mean())
        std = float(valid.std())

        if std == 0:
            z_mask = pd.Series(False, index=series.index)
        else:
            z_scores = (series - mean) / std
            z_mask = series.notna() & (z_scores.abs() > OUTLIER_ZSCORE_THRESHOLD)

        iqr_indices = [int(i) for i in iqr_mask[iqr_mask].index]
        zscore_indices = [int(i) for i in z_mask[z_mask].index]

        result[col] = {
            "iqr_count": len(iqr_indices),
            "iqr_pct": round(len(iqr_indices) / n_total * 100, 2),
            "iqr_indices": iqr_indices,
            "zscore_count": len(zscore_indices),
            "zscore_pct": round(len(zscore_indices) / n_total * 100, 2),
            "zscore_indices": zscore_indices,
        }

    return result
