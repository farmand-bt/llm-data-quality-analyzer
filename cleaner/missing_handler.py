import pandas as pd


def handle_missing(
    df: pd.DataFrame,
    strategy: str,
    column: str,
    value=None,
    *,
    reset_index: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Return a new DataFrame with missing values handled and a log entry.

    Strategies:
        drop_rows     — drop rows where this column is null
        drop_column   — drop the entire column
        fill_mean     — fill with column mean (numeric coercion applied first)
        fill_median   — fill with column median
        fill_mode     — fill with most frequent value
        fill_constant — fill with the constant passed in `value`
        ffill         — forward fill (useful for time-series)
        bfill         — backward fill

    reset_index=False keeps the original row labels (see CleaningPipeline).

    Never modifies the input DataFrame.
    """
    out = df.copy()
    missing_before = int(out[column].isnull().sum())
    rows_before = len(out)

    if strategy == "drop_rows":
        out = out.dropna(subset=[column])
        if reset_index:
            out = out.reset_index(drop=True)

    elif strategy == "drop_column":
        out = out.drop(columns=[column])

    elif strategy == "fill_mean":
        numeric = _require_numbers(out[column], column, strategy)
        fill_val = numeric.mean()
        if (numeric.dropna() % 1 == 0).all():
            fill_val = round(fill_val)
        out[column] = out[column].fillna(fill_val)

    elif strategy == "fill_median":
        numeric = _require_numbers(out[column], column, strategy)
        fill_val = numeric.median()
        if (numeric.dropna() % 1 == 0).all():
            fill_val = round(fill_val)
        out[column] = out[column].fillna(fill_val)

    elif strategy == "fill_mode":
        mode = out[column].mode()
        if not mode.empty:
            out[column] = out[column].fillna(mode.iloc[0])

    elif strategy == "fill_constant":
        # A categorical column only accepts known labels, so add the fill
        # value as a category first (fillna raises TypeError otherwise)
        if isinstance(out[column].dtype, pd.CategoricalDtype) and (
            value not in out[column].cat.categories
        ):
            out[column] = out[column].cat.add_categories([value])
        out[column] = out[column].fillna(value)

    elif strategy == "ffill":
        out[column] = out[column].ffill()

    elif strategy == "bfill":
        out[column] = out[column].bfill()

    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    missing_after = int(out[column].isnull().sum()) if strategy != "drop_column" else 0
    return out, {
        "action": "handle_missing",
        "column": column,
        "strategy": strategy,
        "rows_before": rows_before,
        "rows_after": len(out),
        "rows_removed": rows_before - len(out),
        "missing_before": missing_before,
        "missing_after": missing_after,
    }


def _require_numbers(series: pd.Series, column: str, strategy: str) -> pd.Series:
    """Numeric view of the column for fill_mean / fill_median.

    A column with no numbers at all (e.g. an empty CSV column, which loads as
    float64 NaN) has no mean, and round(nan) would crash.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.notna().any():
        raise ValueError(
            f"{strategy} needs at least one numeric value, and '{column}' has none. "
            "Use fill_mode, fill_constant or drop_column instead."
        )
    return numeric
