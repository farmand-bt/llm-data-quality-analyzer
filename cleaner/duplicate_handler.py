import pandas as pd


def remove_duplicates(
    df: pd.DataFrame,
    keep: str = "first",
) -> tuple[pd.DataFrame, dict]:
    """Return a new DataFrame with exact duplicates removed and a log entry.

    keep:
        "first"  — keep the first occurrence of each duplicate group
        "last"   — keep the last occurrence
        "none"   — remove every copy (no row from a duplicate group is kept)

    Never modifies the input DataFrame.
    """
    out = df.copy()
    rows_before = len(out)

    keep_arg = keep if keep in ("first", "last") else False
    out = out.drop_duplicates(keep=keep_arg).reset_index(drop=True)

    strategy_label = f"keep_{keep}" if keep in ("first", "last") else "remove_all"
    return out, {
        "action": "remove_duplicates",
        "column": None,
        "strategy": strategy_label,
        "rows_before": rows_before,
        "rows_after": len(out),
        "rows_removed": rows_before - len(out),
    }
