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

# --- Main area ---
if uploaded_file is None:
    st.info("Upload a file using the sidebar to begin.")
    st.stop()

if uploaded_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
    st.error(
        f"File is too large ({uploaded_file.size / 1024**2:.1f} MB). "
        f"Maximum supported size is {MAX_UPLOAD_SIZE_MB} MB."
    )
    st.stop()

try:
    df = load_file(uploaded_file, sheet_name=sheet_name)
except Exception as e:
    st.error(f"Could not load file: {e}")
    st.stop()

if st.session_state.get("filename") != uploaded_file.name:
    st.session_state.pop("cleaned_df", None)
    st.session_state.pop("cleaning_log", None)
    st.session_state.pop("llm_insights", None)
    st.session_state.pop("llm_qa_history", None)
    st.session_state.pop("llm_enabled", None)

st.session_state["df"] = df
st.session_state["filename"] = uploaded_file.name

st.subheader("Data Preview")
st.dataframe(df.head(MAX_PREVIEW_ROWS), use_container_width=True, hide_index=True)
st.caption(
    f"{len(df):,} rows × {len(df.columns):,} columns"
    f" — showing first {min(MAX_PREVIEW_ROWS, len(df))}"
)

st.divider()

with st.spinner("Profiling dataset…"):
    report = _build_report_cached(df, uploaded_file.name)

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
