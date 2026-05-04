from datetime import datetime

import pandas as pd

from config.settings import QUALITY_GRADE_THRESHOLDS
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


def _column_warnings(col_type: dict, col_missing: dict) -> list[str]:
    warnings = []
    if col_missing["missing_pct"] > 50:
        warnings.append(
            f"Over 50% of values are missing ({col_missing['missing_pct']:.1f}%)"
        )
    if col_type["type_mismatch"]:
        warnings.append(
            f"Type mismatch: stored as {col_type['pandas_dtype']}, "
            f"inferred as {col_type['inferred_type']}"
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

    quality_score = _compute_quality_score(type_info, missing_info)

    columns = {}
    for col in df.columns:
        columns[col] = {
            "pandas_dtype": type_info[col]["pandas_dtype"],
            "inferred_type": type_info[col]["inferred_type"],
            "type_mismatch": type_info[col]["type_mismatch"],
            "missing_count": missing_info["per_column"][col]["missing_count"],
            "missing_pct": missing_info["per_column"][col]["missing_pct"],
            "missing_pattern": missing_info["per_column"][col]["missing_pattern"],
            "stats": stats_info[col],
            # Outlier and distribution fields populated in Milestone 4
            "outliers": {
                "iqr_count": None,
                "zscore_count": None,
                "iqr_indices": [],
                "zscore_indices": [],
            },
            "distribution": {
                "normality_pvalue": None,
                "is_normal": None,
                "skew_label": None,
            },
            "warnings": _column_warnings(type_info[col], missing_info["per_column"][col]),
        }

    return {
        "dataset": {
            "name": filename,
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
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
