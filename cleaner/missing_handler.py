# Implemented in Milestone 5
import pandas as pd


def handle_missing(df: pd.DataFrame, strategy: str, column: str) -> tuple[pd.DataFrame, dict]:
    """Return a new DataFrame with missing values handled and a log entry.

    Strategies: drop_rows, fill_mean, fill_median, fill_mode, fill_constant.
    Never modifies the input DataFrame.
    """
    raise NotImplementedError
