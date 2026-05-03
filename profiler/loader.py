import io

import chardet
import pandas as pd


def load_file(uploaded_file, sheet_name=None) -> pd.DataFrame:
    """Load a CSV or Excel uploaded file into a DataFrame.

    For CSV files, encoding and separator are auto-detected.
    For Excel files, sheet_name selects the sheet (defaults to first).
    """
    name = uploaded_file.name

    if name.endswith(".csv"):
        raw = uploaded_file.read()
        encoding = chardet.detect(raw).get("encoding") or "utf-8"
        sample = raw[:4096].decode(encoding, errors="replace")
        sep = ";" if sample.count(";") > sample.count(",") else ","
        return pd.read_csv(
            io.BytesIO(raw),
            encoding=encoding,
            sep=sep,
            on_bad_lines="warn",
        )

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file, sheet_name=sheet_name or 0)

    raise ValueError(
        f"Unsupported file type: '{name}'. Expected .csv, .xlsx, or .xls"
    )
