import pandas as pd


# Skip near-duplicate detection for wide DataFrames to keep runtime acceptable.
_MAX_COLS_FOR_NEAR_DUP = 50


def analyze_duplicates(df: pd.DataFrame) -> dict:
    """Return exact duplicate count, near-duplicate count, and sample row indices.

    Exact duplicates: rows that are identical across every column.
    Near-duplicates: rows that are identical on all columns except exactly one
                     (i.e. tolerance = 1 differing column).

    Returns:
        {
            "exact_count": int,        # rows that can be dropped (keep first)
            "exact_pct": float,
            "sample_indices": list,    # indices of all rows in duplicate groups
            "near_duplicate_count": int,
            "near_duplicate_pct": float,
            "near_sample_indices": list,
        }
    """
    n = len(df)
    if n == 0:
        return {
            "exact_count": 0, "exact_pct": 0.0, "sample_indices": [],
            "near_duplicate_count": 0, "near_duplicate_pct": 0.0, "near_sample_indices": [],
        }

    # Exact duplicates
    all_dup_mask = df.duplicated(keep=False)   # marks every copy of a duplicated row
    keep_first_mask = df.duplicated(keep="first")  # marks all but the first occurrence
    exact_count = int(keep_first_mask.sum())
    exact_pct = round(exact_count / n * 100, 2)
    sample_indices = [int(i) for i in list(all_dup_mask[all_dup_mask].index)[:50]]
    exact_idx_set = set(all_dup_mask[all_dup_mask].index)

    # Near-duplicates (tolerance = 1 column difference)
    near_dup_indices: set[int] = set()
    if len(df.columns) >= 2 and len(df.columns) <= _MAX_COLS_FOR_NEAR_DUP:
        for col in df.columns:
            other_cols = [c for c in df.columns if c != col]
            near_mask = df.duplicated(subset=other_cols, keep=False)
            new = set(near_mask[near_mask].index) - exact_idx_set
            near_dup_indices.update(new)

    near_count = len(near_dup_indices)
    near_pct = round(near_count / n * 100, 2)
    near_sample = [int(i) for i in list(near_dup_indices)[:50]]

    return {
        "exact_count": exact_count,
        "exact_pct": exact_pct,
        "sample_indices": sample_indices,
        "near_duplicate_count": near_count,
        "near_duplicate_pct": near_pct,
        "near_sample_indices": near_sample,
    }
