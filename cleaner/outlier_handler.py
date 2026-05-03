# Implemented in Milestone 5
import pandas as pd


def handle_outliers(
    df: pd.DataFrame, column: str, indices: list, strategy: str
) -> tuple[pd.DataFrame, dict]:
    """Return a new DataFrame with outliers treated and a log entry.

    Strategies: remove, cap_iqr. Indices come from profiler/outliers.py.
    Never modifies the input DataFrame.
    """
    raise NotImplementedError
