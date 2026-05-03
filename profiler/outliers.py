# Implemented in Milestone 4
import pandas as pd


def detect_outliers(df: pd.DataFrame) -> dict:
    """Return IQR and Z-score outlier counts and row indices per numeric column.

    The returned indices are consumed by cleaner/pipeline.py.
    """
    raise NotImplementedError
