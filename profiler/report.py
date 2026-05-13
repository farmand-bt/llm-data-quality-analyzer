from datetime import datetime

import pandas as pd

from config.settings import COLUMN_WARNING_MISSING_THRESHOLD, QUALITY_GRADE_THRESHOLDS
from profiler.correlations import compute_correlations
from profiler.distributions import analyze_distributions
from profiler.duplicates import analyze_duplicates
from profiler.missing import analyze_missing
from profiler.outliers import detect_outliers
from profiler.recommendations import generate_recommendations
from profiler.stats import compute_stats
from profiler.type_detector import detect_types


def _compute_quality_score(
    type_info: dict,
    missing_info: dict,
    dup_info: dict,
    outlier_info: dict,
) -> int:
    """Composite quality score 0–100.

    Weights:
      - Missing penalty:       up to -40 pts (2 pts per % overall missing, capped)
      - Type mismatch penalty: up to -20 pts (proportional to mismatch ratio)
      - Duplicate penalty:     up to -10 pts (1 pt per % duplicates, capped)
      - Outlier penalty:       up to -10 pts (based on avg outlier % across numeric cols)
      - Clean column bonus:    up to  +5 pts
    """
    score = 100.0
    total_cols = len(type_info)

    # Missing
    score -= min(40.0, missing_info["overall_missing_pct"] * 2.0)

    # Type mismatches
    if total_cols > 0:
        mismatch_cols = sum(1 for v in type_info.values() if v["type_mismatch"])
        score -= (mismatch_cols / total_cols) * 20.0

    # Duplicates
    if dup_info.get("exact_pct"):
        score -= min(10.0, dup_info["exact_pct"])

    # Outliers (average IQR outlier % across numeric columns that have any)
    numeric_outlier_pcts = [
        v["iqr_pct"] for v in outlier_info.values() if v.get("iqr_pct", 0) > 0
    ]
    if numeric_outlier_pcts:
        avg_outlier_pct = sum(numeric_outlier_pcts) / len(numeric_outlier_pcts)
        score -= min(10.0, avg_outlier_pct * 2.0)

    # Clean column bonus
    if total_cols > 0:
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


def _column_warnings(
    col_type: dict,
    col_missing: dict,
    col_dist: dict,
    col_outlier: dict,
) -> list[str]:
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

    iqr_count = col_outlier.get("iqr_count", 0) or 0
    if iqr_count > 0:
        iqr_pct = col_outlier.get("iqr_pct", 0) or 0.0
        if iqr_pct > 5:
            warnings.append(
                f"{iqr_count:,} outliers detected ({iqr_pct:.1f}%, IQR method)"
            )

    return warnings


def build_report(df: pd.DataFrame, filename: str) -> dict:
    """Aggregate all profiler sub-module results into a unified report dict.

    Central contract consumed by the UI, the cleaner, and the narrator.
    """
    type_info = detect_types(df)
    missing_info = analyze_missing(df)
    stats_info = compute_stats(df, type_info)
    dist_info = analyze_distributions(df, type_info, stats_info)
    outlier_info = detect_outliers(df, type_info)
    dup_info = analyze_duplicates(df)
    corr_info = compute_correlations(df, type_info)

    quality_score = _compute_quality_score(type_info, missing_info, dup_info, outlier_info)

    columns = {}
    for col in df.columns:
        col_dist = dist_info[col]
        col_out = outlier_info[col]
        columns[col] = {
            "pandas_dtype": type_info[col]["pandas_dtype"],
            "inferred_type": type_info[col]["inferred_type"],
            "type_mismatch": type_info[col]["type_mismatch"],
            "missing_count": missing_info["per_column"][col]["missing_count"],
            "missing_pct": missing_info["per_column"][col]["missing_pct"],
            "missing_pattern": missing_info["per_column"][col]["missing_pattern"],
            "stats": stats_info[col],
            "outliers": {
                "iqr_count": col_out["iqr_count"],
                "iqr_pct": col_out["iqr_pct"],
                "iqr_indices": col_out["iqr_indices"],
                "zscore_count": col_out["zscore_count"],
                "zscore_pct": col_out["zscore_pct"],
                "zscore_indices": col_out["zscore_indices"],
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
                type_info[col],
                missing_info["per_column"][col],
                col_dist,
                col_out,
            ),
        }

    report = {
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
        "duplicates": dup_info,
        "correlations": corr_info,
        "recommendations": [],           # populated below (needs full report)
        "generated_at": datetime.now().isoformat(),
    }

    report["recommendations"] = generate_recommendations(report)
    return report
