"""Prompt builders for the three narrator modes.

Only summary statistics from the report dict are injected — never raw data rows.
"""

_OVERVIEW_TEMPLATE = (
    "You are a data quality analyst. Given the following dataset profile, "
    "write a concise 3-5 sentence executive summary of the data quality. "
    "Focus on the most critical issues and their potential impact.\n\n"
    "Dataset: {name}, {rows:,} rows × {cols} columns "
    "(quality score {quality_score}/100, grade {quality_grade})\n"
    "Missing values: {missing_summary}\n"
    "Exact duplicates: {duplicate_count:,}\n"
    "Type mismatches: {type_issues}\n"
    "Outlier columns: {outlier_summary}\n\n"
    "Summary:"
)

_COLUMN_TEMPLATE = (
    "Given the following column profile, write 2-3 sentences describing "
    "the column's quality and any concerns. Be specific and actionable.\n\n"
    "Column: {name} ({col_type})\n"
    "Stats: {stats}\n"
    "Missing: {missing_pct:.1f}%\n"
    "Outliers (IQR): {outlier_count}\n"
    "Warnings: {warnings}\n\n"
    "Analysis:"
)

_CLEANING_TEMPLATE = (
    "Given the following data quality issues, suggest a prioritized "
    "cleaning strategy. Explain the trade-offs of each recommendation.\n\n"
    "Issues:\n{issues_list}\n\n"
    "Strategy:"
)


def build_overview_prompt(report: dict) -> str:
    ds = report["dataset"]
    cols = report["columns"]

    missing_cols = sorted(
        [(c, v["missing_pct"]) for c, v in cols.items() if v["missing_count"] > 0],
        key=lambda x: -x[1],
    )
    missing_summary = (
        ", ".join(f"{c} ({p:.1f}%)" for c, p in missing_cols[:5])
        if missing_cols
        else "none"
    )

    type_issues_n = sum(1 for v in cols.values() if v["type_mismatch"])
    type_issues = f"{type_issues_n} column(s)" if type_issues_n else "none"

    outlier_cols = sorted(
        [
            (c, v["outliers"]["iqr_count"])
            for c, v in cols.items()
            if v["outliers"]["iqr_count"] > 0
        ],
        key=lambda x: -x[1],
    )
    outlier_summary = (
        ", ".join(f"{c} ({n})" for c, n in outlier_cols[:5])
        if outlier_cols
        else "none detected"
    )

    return _OVERVIEW_TEMPLATE.format(
        name=ds["name"],
        rows=ds["rows"],
        cols=ds["columns"],
        quality_score=ds["quality_score"],
        quality_grade=ds["quality_grade"],
        missing_summary=missing_summary,
        duplicate_count=report["duplicates"]["exact_count"],
        type_issues=type_issues,
        outlier_summary=outlier_summary,
    )


def build_column_prompt(report: dict, col_name: str) -> str:
    info = report["columns"][col_name]
    stats = info.get("stats", {})

    stats_parts = []
    for key in ("mean", "median", "std", "min", "max", "unique_count", "top_value"):
        val = stats.get(key)
        if val is not None:
            stats_parts.append(f"{key}={val}")
    stats_str = ", ".join(stats_parts) if stats_parts else "n/a"

    warnings = "; ".join(info.get("warnings", [])) or "none"

    return _COLUMN_TEMPLATE.format(
        name=col_name,
        col_type=info["inferred_type"],
        stats=stats_str,
        missing_pct=info["missing_pct"],
        outlier_count=info["outliers"]["iqr_count"],
        warnings=warnings,
    )


def build_cleaning_prompt(report: dict) -> str:
    recs = report.get("recommendations", [])
    if not recs:
        issues_list = "No issues detected."
    else:
        lines = []
        for r in recs:
            col_tag = f" [{r['column']}]" if r.get("column") else ""
            lines.append(f"- [{r['severity'].upper()}]{col_tag} {r['message']}")
        issues_list = "\n".join(lines)

    return _CLEANING_TEMPLATE.format(issues_list=issues_list)
