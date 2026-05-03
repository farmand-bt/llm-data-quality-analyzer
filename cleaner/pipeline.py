# Implemented in Milestone 5
import pandas as pd


def run_pipeline(df: pd.DataFrame, steps: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Apply a sequence of cleaning steps and return the cleaned DataFrame and full log.

    Each step dict specifies: action, column (if applicable), and strategy.
    Reads outlier row indices from the profiler report — never re-computes them.
    Never modifies the input DataFrame.
    """
    raise NotImplementedError
