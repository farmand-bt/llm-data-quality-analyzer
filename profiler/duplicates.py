import numpy as np
import pandas as pd


# Skip near-duplicate detection for wide DataFrames to keep runtime acceptable.
_MAX_COLS_FOR_NEAR_DUP = 50

# Odd multipliers make each column's hash contribution distinct, so rows whose
# values are merely permuted across columns rarely produce the same key.
_COLUMN_MIX = np.uint64(0x9E3779B97F4A7C15)


def _near_duplicate_rows(df: pd.DataFrame) -> set:
    """Row labels equal to at least one other row on every column but one.

    Same result as running `df.duplicated(subset=<all but c>, keep=False)` for
    each column c, but each cell is hashed once instead of every column being
    re-compared per pass. A row's key for "all columns except c" is the
    (wrapping) sum of its per-column hashes minus column c's term, so each pass
    only deduplicates one uint64 array. Hash equality is only a filter: every
    candidate group is confirmed on the real values, so a hash collision can
    never add a false near-duplicate. Columns and rows are addressed by
    position, so duplicate column names can't select the wrong data.
    """
    n_rows, n_cols = df.shape
    # Array (not scalar) uint64 arithmetic wraps modulo 2**64 silently, which
    # is intended; numpy only warns about overflow for scalar operations.
    mixers = (
        np.arange(n_cols, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    ) * _COLUMN_MIX
    # One preallocated (n, k) matrix filled in place keeps peak memory low.
    terms = np.empty((n_rows, n_cols), dtype=np.uint64)
    for j in range(n_cols):
        hashes = pd.util.hash_pandas_object(df.iloc[:, j], index=False)
        np.multiply(hashes.to_numpy(np.uint64), mixers[j : j + 1], out=terms[:, j])
    total = terms.sum(axis=1, dtype=np.uint64)

    rows = set()
    for j in range(n_cols):
        candidates = np.flatnonzero(
            pd.Series(total - terms[:, j]).duplicated(keep=False).to_numpy()
        )
        if len(candidates) == 0:
            continue
        others = [k for k in range(n_cols) if k != j]
        subset = df.iloc[candidates, others]
        confirmed = candidates[subset.duplicated(keep=False).to_numpy()]
        rows.update(df.index[confirmed])
    return rows


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
        near_dup_indices = _near_duplicate_rows(df) - exact_idx_set

    near_count = len(near_dup_indices)
    near_pct = round(near_count / n * 100, 2)
    # Sorted (not set order) so the sample is deterministic
    near_sample = [int(i) for i in sorted(near_dup_indices)[:50]]

    return {
        "exact_count": exact_count,
        "exact_pct": exact_pct,
        "sample_indices": sample_indices,
        "near_duplicate_count": near_count,
        "near_duplicate_pct": near_pct,
        "near_sample_indices": near_sample,
    }
