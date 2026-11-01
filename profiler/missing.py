"""Missing-value analysis: counts, co-occurrence, and what the gaps go with.

The missing-value pattern is derived from evidence ported from data-quality-mcp
(Milestone 3.5, D1): `missing_related_to` lists the observed columns whose
values go with a column's missingness, using calibrated tests, and
`missing_pattern` labels it:
    "none"    — no missing values
    "MAR"     — related to at least one observed column (see missing_related_to)
    "MCAR"    — tested and no detectable pattern (consistent with random gaps,
                not proof of them)
    "unknown" — missing values, but not tested: too few missing or present
                rows, no numeric or categorical column to compare with, or
                beyond the first MAX_MISSING_PATTERN_COLUMNS columns with gaps
Data alone can't show MNAR (missingness that depends on the missing values
themselves), so that label is no longer given.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import t as t_distribution

from config.settings import (
    MAX_MISSING_ASSOCIATIONS,
    MAX_MISSING_PATTERN_COLUMNS,
    MAX_PATTERN_CARDINALITY,
    MAX_PATTERN_CATEGORICAL_COLUMNS,
    MAX_PATTERN_NUMERIC_COLUMNS,
    MIN_MISSING_PATTERN_ROWS,
    MISSING_ASSOCIATION_MAX_P,
    MISSING_ASSOCIATION_THRESHOLD,
)
from profiler.type_detector import detect_types

_NUMERIC_TYPES = ("numeric", "mixed")
_CATEGORICAL_TYPES = ("categorical", "boolean")
_MIN_CATEGORY_ROWS = 5  # Rarer categories are left out of Cramér's V


def _pearson_p(r: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Two-sided p-value of Pearson r from n rows (t distribution, n - 2 df)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        dof = n - 2
        t = np.abs(r) * np.sqrt(dof / np.maximum(1.0 - r**2, 0.0))
        return np.where(np.abs(r) >= 1.0, 0.0, 2.0 * t_distribution.sf(t, dof))


def corrected_cramers_v(chi2: float, n: int, shape: tuple[int, int]) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013); the plain V reaches 0.3 by
    chance on sparse tables."""
    rows, cols = shape
    phi2 = max(0.0, chi2 / n - (rows - 1) * (cols - 1) / (n - 1))
    rows_c = rows - (rows - 1) ** 2 / (n - 1)
    cols_c = cols - (cols - 1) ** 2 / (n - 1)
    denominator = min(rows_c - 1, cols_c - 1)
    return float(np.sqrt(phi2 / denominator)) if denominator > 0 else 0.0


def _point_biserial(indicators, centred, usable, varies, min_rows):
    """Point-biserial r of every missing indicator (n x t) against every
    numeric column (n x k) at once, each pair over the rows where the column
    has a finite number. Returns r, p and which pairs qualify, all k x t."""
    n = usable.sum(axis=0).astype(float)[:, None]
    k_missing = usable.T.astype(float) @ indicators
    k_present = n - k_missing
    sxx = (centred**2).sum(axis=0)[:, None]
    sxy = centred.T @ indicators  # the centred column sums to 0 on its rows
    with np.errstate(divide="ignore", invalid="ignore"):
        r = sxy / np.sqrt(sxx * (k_missing * k_present / n))
    eligible = (k_missing >= min_rows) & (k_present >= min_rows) & varies[:, None]
    r = np.where(eligible, r, 0.0)
    return r, _pearson_p(r, np.broadcast_to(n, r.shape)), eligible


def _categorical_test(missing: pd.Series, categories: pd.Series, min_rows: int):
    """Bias-corrected Cramér's V and chi-square p (no Yates' correction) over
    the rows where the category is present, leaving out categories with fewer
    than 5 rows; None when the pair doesn't qualify."""
    codes, _ = pd.factorize(categories)  # -1 where the category is missing
    is_missing = missing.to_numpy(dtype=bool)
    present = codes >= 0
    codes, is_missing = codes[present], is_missing[present]
    if len(codes) == 0:
        return None
    k = int(codes.max()) + 1
    table = np.bincount(codes + k * is_missing, minlength=2 * k).reshape(2, k)
    table = table[:, table.sum(axis=0) >= _MIN_CATEGORY_ROWS]
    n_missing, n_present = int(table[1].sum()), int(table[0].sum())
    if table.shape[1] < 2 or n_missing < min_rows or n_present < min_rows:
        return None
    chi2, p = chi2_contingency(table, correction=False)[:2]
    return corrected_cramers_v(float(chi2), n_missing + n_present, table.shape), float(
        p
    )


def missing_associations(df: pd.DataFrame, type_info: dict) -> dict:
    """For each column, the observed columns whose values go with its
    missingness, as [{"column", "method", "value"}], strongest first.

    Each column with missing values is compared with numeric/mixed columns
    (point-biserial r), categorical/boolean columns (bias-corrected Cramér's
    V) and the other columns' missing indicators (positive phi:
    "missing_together"). Text and datetime columns aren't comparison columns.
    A pair needs MIN_MISSING_PATTERN_ROWS missing and present target values
    among the rows it uses; an association is listed when |value| >=
    MISSING_ASSOCIATION_THRESHOLD and p < MISSING_ASSOCIATION_MAX_P / (pairs
    tested for that column). None: not tested; []: tested, nothing found.
    """
    result = {c: None for c in df.columns}
    missing = df.isna()
    targets = [c for c in df.columns if missing[c].any()][:MAX_MISSING_PATTERN_COLUMNS]
    if not targets:
        return result

    numeric_names = [
        c for c in df.columns if type_info[c]["inferred_type"] in _NUMERIC_TYPES
    ][:MAX_PATTERN_NUMERIC_COLUMNS]
    matrix = np.empty((len(df), 0))
    if numeric_names:
        matrix = (
            df[numeric_names]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float, na_value=np.nan, copy=True)
        )
        matrix[~np.isfinite(matrix)] = np.nan
    usable = ~np.isnan(matrix)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        centred = np.where(usable, matrix - np.nanmean(matrix, axis=0), 0.0)
        # Constancy from the raw values: a float mean's rounding error would
        # make a column of 0.1s look as if it varied
        varies = np.nanmax(matrix, axis=0, initial=-np.inf) > np.nanmin(
            matrix, axis=0, initial=np.inf
        )
    categorical_names = [
        c
        for c in df.columns
        if type_info[c]["inferred_type"] in _CATEGORICAL_TYPES
        and df[c].nunique(dropna=True) <= MAX_PATTERN_CARDINALITY
    ][:MAX_PATTERN_CATEGORICAL_COLUMNS]
    indicators = missing[targets].to_numpy(dtype=float)
    n_rows = len(df)
    counts = indicators.sum(axis=0)
    min_rows = MIN_MISSING_PATTERN_ROWS
    both_classes = (counts >= min_rows) & (n_rows - counts >= min_rows)
    r_numeric, p_numeric, numeric_ok = _point_biserial(
        indicators, centred, usable, varies, min_rows
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.atleast_2d(np.corrcoef(indicators, rowvar=False))
    p_phi = _pearson_p(np.nan_to_num(phi), np.full(phi.shape, float(n_rows)))

    for i, target in enumerate(targets):
        tests = [
            (name, "point_biserial", float(r_numeric[j, i]), float(p_numeric[j, i]))
            for j, name in enumerate(numeric_names)
            if name != target and numeric_ok[j, i]
        ]
        for name in categorical_names:
            if name == target:
                continue
            outcome = _categorical_test(missing[target], df[name], min_rows)
            if outcome is not None:
                tests.append((name, "cramers_v", *outcome))
        if both_classes[i]:
            tests.extend(
                (targets[j], "missing_together", float(phi[i, j]), float(p_phi[i, j]))
                for j in range(len(targets))
                if j != i and both_classes[j]
            )
        if not tests:
            continue
        limit = MISSING_ASSOCIATION_MAX_P / len(tests)  # Bonferroni
        listed = sorted(
            (
                (name, method, value)
                for name, method, value, p_value in tests
                if abs(value) >= MISSING_ASSOCIATION_THRESHOLD
                and p_value < limit
                and (method == "point_biserial" or value > 0)
            ),
            key=lambda item: (-abs(item[2]), item[0]),
        )
        result[target] = [
            {"column": name, "method": method, "value": round(value, 3)}
            for name, method, value in listed[:MAX_MISSING_ASSOCIATIONS]
        ]
    return result


def _pattern_label(missing_count: int, related: list | None) -> str:
    if missing_count == 0:
        return "none"
    if related is None:
        return "unknown"
    return "MAR" if related else "MCAR"


def analyze_missing(df: pd.DataFrame, type_info: dict | None = None) -> dict:
    """Return missing value analysis: overall stats, per-column counts and
    patterns, and pairwise co-occurrence.

    `type_info` is detect_types(df); computed here when not given.

    Returns:
        {
            "overall_missing_pct": float,
            "total_missing_cells": int,
            "per_column": {col: {"missing_count", "missing_pct", "missing_pattern",
                                 "missing_related_to"}},
            "co_occurrence": {"{col_a}|{col_b}": int},  # rows where both are missing
        }
    """
    total_cells = df.size
    total_missing = int(df.isnull().sum().sum())
    overall_pct = (
        round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0.0
    )
    related = missing_associations(
        df, type_info if type_info is not None else detect_types(df)
    )

    per_column = {}
    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        missing_pct = round(missing_count / len(df) * 100, 2) if len(df) else 0.0
        per_column[col] = {
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "missing_pattern": _pattern_label(missing_count, related[col]),
            "missing_related_to": related[col],
        }

    # Pairwise co-occurrence for columns that have any missing values
    missing_cols = [c for c in df.columns if df[c].isnull().any()]
    co_occurrence: dict[str, int] = {}
    if len(missing_cols) > 1:
        null_matrix = df[missing_cols].isnull()
        for i, col_a in enumerate(missing_cols):
            for col_b in missing_cols[i + 1 :]:
                both = int((null_matrix[col_a] & null_matrix[col_b]).sum())
                if both > 0:
                    co_occurrence[f"{col_a}|{col_b}"] = both

    return {
        "overall_missing_pct": overall_pct,
        "total_missing_cells": total_missing,
        "per_column": per_column,
        "co_occurrence": co_occurrence,
    }
