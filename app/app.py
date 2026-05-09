import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from app.components import column_report, overview
from config.settings import MAX_PREVIEW_ROWS
from profiler.loader import load_file
from profiler.report import build_report

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

try:
    df = load_file(uploaded_file, sheet_name=sheet_name)
except Exception as e:
    st.error(f"Could not load file: {e}")
    st.stop()

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
    report = build_report(df, uploaded_file.name)

st.session_state["report"] = report

overview.render(report)

st.divider()
column_report.render(report)
