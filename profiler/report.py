from datetime import datetime

import pandas as pd


def build_report(df: pd.DataFrame, filename: str) -> dict:
    """Aggregate all profiler sub-module results into a unified report dict.

    Each milestone adds to this structure. The schema is the central contract
    shared by the UI, the cleaner, and the narrator.
    """
    return {
        "dataset": {
            "name": filename,
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
            "quality_score": None,
            "quality_grade": None,
        },
        "columns": {},
        "duplicates": {"exact_count": None, "exact_pct": None, "sample_indices": []},
        "correlations": {"pearson": {}, "high_correlation_pairs": []},
        "recommendations": [],
        "generated_at": datetime.now().isoformat(),
    }
