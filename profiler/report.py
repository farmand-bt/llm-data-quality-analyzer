from datetime import datetime

import pandas as pd

from config.settings import COLUMN_WARNING_MISSING_THRESHOLD, QUALITY_GRADE_THRESHOLDS
from profiler.distributions import analyze_distributions
from profiler.missing import analyze_missing
from profiler.stats import compute_stats
from profiler.type_detector import detect_types


def _compute_quality_score(type_info: dict, missing_info: dict) -> int:
    """Composite quality score 0–100.

    Weights (M2):
      - Missing penalty:       up to -40 pts (2 pts per % overall missing, capped)
      - Type mismatch penalty: up to -20 pts (proportional to mismatch ratio)
      - Clean column bonus:    up to  +5 pts (columns with zero issues)

    Duplicate and outlier penalties are added in Milestone 4.
    """
    score = 100.0
    total_cols = len(type_info)

    missing_pct = missing_info["overall_missing_pct"]
    score -= min(40.0, missing_pct * 2.0)

    if total_cols > 0:
        mismatch_cols = sum(1 for v in type_info.values() if v["type_mismatch"])
        score -= (mismatch_cols / total_cols) * 20.0

        clean_cols = sum(
            1
            for col, v in type_info.items()
            if not v["type_mismatch"]
            and missing_info["per_column"][col]["missing_count"] == 0
        )
        score += (clean_cols / total_cols) * 5.0

    return max(0, min(100, round(score)))


def _quality_grade(score: int) -> str:
    for grade, threshold in sorted(QUALITY_GRADE_THRESHOLDS.items(), key=lambda x: -x[1]):
        if score >= threshold:
            return grade
    return "F"


def _column_warnings(col_type: dict, col_missing: dict, col_dist: dict) -> list[str]:
    warnings = []
    threshold_pct = COLUMN_WARNING_MISSING_THRESHOLD * 100
    if col_missing["missing_pct"] > threshold_pct:
        warnings.append(
            f"Over {threshold_pct:.0f}% of values are missing ({col_missing['missing_pct']:.1f}%)"
        )
    if col_type["type_mismatch"]:
        inferred = col_type["inferred_type"]
        dtype = col_type["pandas_dtype"]
        hints = {
            "numeric": "consider converting with pd.to_numeric()",
            "datetime": "consider converting with pd.to_datetime()",
            "boolean": "consider casting to bool",
            "mixed": "column contains mixed types — inspect and clean before casting",
        }
        hint = hints.get(inferred, "consider casting to the correct type")
        warnings.append(
            f"Type mismatch: values appear to be {inferred} but are stored as {dtype} — {hint}"
        )
    if col_dist.get("high_cardinality"):
        warnings.append(
            "High cardinality: column may be an ID or free text rather than a true categorical"
        )
    return warnings


def build_report(df: pd.DataFrame, filename: str) -> dict:
    """Aggregate all profiler sub-module results into a unified report dict.

    This is the central contract consumed by the UI, the cleaner, and the narrator.
    Schema grows across milestones but the top-level shape is stable from M2 onward.
    """
    type_info = detect_types(df)
    missing_info = analyze_missing(df)
    stats_info = compute_stats(df, type_info)
    dist_info = analyze_distributions(df, type_info, stats_info)

    quality_score = _compute_quality_score(type_info, missing_info)

    columns = {}
    for col in df.columns:
        col_dist = dist_info[col]
        columns[col] = {
            "pandas_dtype": type_info[col]["pandas_dtype"],
            "inferred_type": type_info[col]["inferred_type"],
            "type_mismatch": type_info[col]["type_mismatch"],
            "missing_count": missing_info["per_column"][col]["missing_count"],
            "missing_pct": missing_info["per_column"][col]["missing_pct"],
            "missing_pattern": missing_info["per_column"][col]["missing_pattern"],
            "stats": stats_info[col],
            # Outlier fields populated in Milestone 4
            "outliers": {
                "iqr_count": None,
                "zscore_count": None,
                "iqr_indices": [],
                "zscore_indices": [],
            },
            "distribution": {
                "normality_pvalue": col_dist["normality_pvalue"],
                "is_normal": col_dist["is_normal"],
                "skew_label": col_dist["skew_label"],
                "kurtosis_label": col_dist["kurtosis_label"],
                "high_cardinality": col_dist["high_cardinality"],
                "datetime_gaps": col_dist["datetime_gaps"],
            },
            "warnings": _column_warnings(
                type_info[col], missing_info["per_column"][col], col_dist
            ),
        }

    return {
        "dataset": {
            "name": filename,
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
            "overall_missing_pct": missing_info["overall_missing_pct"],
            "quality_score": quality_score,
            "quality_grade": _quality_grade(quality_score),
        },
        "columns": columns,
        # Duplicate and correlation fields populated in Milestone 4
        "duplicates": {"exact_count": None, "exact_pct": None, "sample_indices": []},
        "correlations": {"pearson": {}, "high_correlation_pairs": []},
        "recommendations": [],
        "generated_at": datetime.now().isoformat(),
    }
