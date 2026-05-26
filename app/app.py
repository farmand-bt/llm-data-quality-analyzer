import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from app.components import (
    cleaning_panel,
    column_report,
    correlations,
    duplicates,
    llm_insights,
    outliers,
    overview,
    recommendations,
)
from config.settings import MAX_PREVIEW_ROWS, MAX_UPLOAD_SIZE_MB
from profiler.loader import load_file
from profiler.report import build_report

_SAMPLE_DIR = Path(__file__).parent.parent / "sample_data"

_SAMPLE_META = {
    "housing": {
        "file": "housing_messy.csv",
        "label": "🏠 Housing",
        "description": "520 rows · price, sqft, outliers, mixed types, duplicates",
    },
    "titanic": {
        "file": "titanic_messy.csv",
        "label": "🚢 Titanic",
        "description": "911 rows · age/cabin missing, inconsistent Sex labels, dups",
    },
}


@st.cache_data(show_spinner=False)
def _build_report_cached(df: pd.DataFrame, filename: str) -> dict:
    return build_report(df, filename)


st.set_page_config(
    page_title="Data Quality Analyzer",
    layout="wide",
)

st.title("Data Quality Analyzer")
st.caption("Upload a CSV or Excel file to receive a comprehensive data quality report.")

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
    )

    sheet_name = None
    if uploaded_file is not None and uploaded_file.name.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("Select sheet", xl.sheet_names)
        uploaded_file.seek(0)

    st.caption("— or try a built-in sample —")
    s1, s2 = st.columns(2)
    for _key, _meta in _SAMPLE_META.items():
        _col = s1 if _key == "housing" else s2
        if _col.button(_meta["label"], use_container_width=True):
            st.session_state["sample_source"] = _key

# --- Determine active data source ---
sample_source: str | None = st.session_state.get("sample_source")

if uploaded_file is not None:
    # Real upload always takes priority; discard any stale sample selection
    st.session_state.pop("sample_source", None)
    sample_source = None

if uploaded_file is None and sample_source is None:
    st.info(
        "Upload a CSV or Excel file using the sidebar — "
        "or load a built-in sample to try it instantly."
    )
    st.stop()

# --- Load the DataFrame ---
if uploaded_file is not None:
    if uploaded_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        st.error(
            f"File is too large ({uploaded_file.size / 1024**2:.1f} MB). "
            f"Maximum supported size is {MAX_UPLOAD_SIZE_MB} MB."
        )
        st.stop()

    try:
        df = load_file(uploaded_file, sheet_name=sheet_name)
    except Exception as exc:
        st.error(f"Could not load file: {exc}")
        st.stop()

    filename = uploaded_file.name

else:
    meta = _SAMPLE_META[sample_source]
    sample_path = _SAMPLE_DIR / meta["file"]
    try:
        df = pd.read_csv(sample_path)
    except FileNotFoundError:
        st.error(
            f"Sample file not found: `{meta['file']}`. "
            "Run `uv run python scripts/generate_messy_data.py` to regenerate it."
        )
        st.stop()

    filename = meta["file"]
    st.info(f"Using built-in sample: **{meta['label']}** — {meta['description']}")

# --- Edge-case guards ---
if df.empty:
    st.error("The file contains no rows. Please upload a file that has data.")
    st.stop()

if len(df.columns) == 0:
    st.error("The file contains no columns.")
    st.stop()

if len(df) > 100_000:
    st.warning(
        f"**Large dataset:** {len(df):,} rows detected. "
        "Profiling may take a moment — consider uploading a random sample for faster results."  # noqa: E501
    )

# --- Session state management ---
if st.session_state.get("filename") != filename:
    _stale = (
        "cleaned_df", "cleaning_log", "llm_insights", "llm_qa_history", "llm_enabled"
    )
    for key in _stale:
        st.session_state.pop(key, None)

st.session_state["df"] = df
st.session_state["filename"] = filename

# --- Data preview ---
st.subheader("Data Preview")
st.dataframe(df.head(MAX_PREVIEW_ROWS), use_container_width=True, hide_index=True)
st.caption(
    f"{len(df):,} rows × {len(df.columns):,} columns"
    f" — showing first {min(MAX_PREVIEW_ROWS, len(df))}"
)

st.divider()

with st.spinner("Profiling dataset…"):
    report = _build_report_cached(df, filename)

st.session_state["report"] = report

overview.render(report)

llm_insights.render(report)

st.divider()
column_report.render(report)

st.divider()
outliers.render(report)

st.divider()
duplicates.render(report)

st.divider()
correlations.render(report)

st.divider()
recommendations.render(report)

st.divider()
cleaning_panel.render(df, report)
