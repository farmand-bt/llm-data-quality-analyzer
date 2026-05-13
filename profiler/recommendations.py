from config.settings import MISSING_THRESHOLD


def generate_recommendations(report: dict) -> list[dict]:
    """Rule-based engine: reads the full report and emits actionable recommendations.

    Each recommendation:
        {"severity": "info"|"warning"|"critical", "category": str,
         "column": str|None, "message": str}

    Results are sorted: critical → warning → info.
    """
    recs: list[dict] = []
    columns = report["columns"]
    dup = report.get("duplicates", {})
    corr = report.get("correlations", {})
    threshold_pct = MISSING_THRESHOLD * 100

    # --- Dataset-level ---

    if dup.get("exact_count"):
        pct = dup["exact_pct"]
        severity = "critical" if pct > 5 else "warning"
        recs.append({
            "severity": severity,
            "category": "duplicate",
            "column": None,
            "message": (
                f"{dup['exact_count']:,} exact duplicate rows detected ({pct:.1f}%) — "
                "consider deduplication before analysis or modeling"
            ),
        })

    for pair in corr.get("high_correlation_pairs", []):
        label = "Pearson r" if pair["method"] == "pearson" else "Cramér's V"
        recs.append({
            "severity": "warning",
            "category": "correlation",
            "column": None,
            "message": (
                f"'{pair['col_a']}' and '{pair['col_b']}' are highly correlated "
                f"({label} = {pair['value']:.3f}) — potential feature redundancy"
            ),
        })

    # --- Per-column ---

    for col, info in columns.items():
        missing_pct = info["missing_pct"]

        if missing_pct >= 50:
            recs.append({
                "severity": "critical",
                "category": "missing",
                "column": col,
                "message": (
                    f"'{col}' has {missing_pct:.1f}% missing values — "
                    "consider dropping this column"
                ),
            })
        elif missing_pct >= threshold_pct:
            recs.append({
                "severity": "warning",
                "category": "missing",
                "column": col,
                "message": (
                    f"'{col}' has {missing_pct:.1f}% missing values — "
                    "consider imputation or dropping"
                ),
            })

        if info["type_mismatch"]:
            recs.append({
                "severity": "warning",
                "category": "type",
                "column": col,
                "message": (
                    f"'{col}' is stored as {info['pandas_dtype']} but inferred as "
                    f"{info['inferred_type']} — consider type conversion"
                ),
            })

        outliers = info.get("outliers", {})
        iqr_count = outliers.get("iqr_count") or 0
        if iqr_count > 0:
            iqr_pct = outliers.get("iqr_pct") or 0.0
            severity = "critical" if iqr_pct > 10 else "warning" if iqr_pct > 3 else "info"
            recs.append({
                "severity": severity,
                "category": "outlier",
                "column": col,
                "message": (
                    f"'{col}' has {iqr_count:,} outliers ({iqr_pct:.1f}%, IQR method) — "
                    "review before modeling or impute/remove"
                ),
            })

    _order = {"critical": 0, "warning": 1, "info": 2}
    recs.sort(key=lambda r: _order.get(r["severity"], 3))
    return recs
