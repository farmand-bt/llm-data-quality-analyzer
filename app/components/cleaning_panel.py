import io

import pandas as pd
import streamlit as st

from cleaner.pipeline import CleaningPipeline
from profiler.report import build_report


@st.cache_data(show_spinner=False)
def _profile_cleaned(df: pd.DataFrame, name: str) -> dict:
    return build_report(df, name)

_MISSING_OPTIONS_ALL = {
    "— skip —": None,
    "Drop rows with missing": "drop_rows",
    "Drop this column": "drop_column",
    "Fill with mean (numeric)": "fill_mean",
    "Fill with median (numeric)": "fill_median",
    "Fill with mode (most frequent)": "fill_mode",
    "Fill with constant": "fill_constant",
    "Forward fill (time-series)": "ffill",
    "Backward fill (time-series)": "bfill",
}

_MISSING_OPTIONS_NON_NUMERIC = {
    k: v for k, v in _MISSING_OPTIONS_ALL.items()
    if v not in ("fill_mean", "fill_median")
}

_TYPE_OPTIONS = {
    "— skip —": None,
    "Convert to numeric": "numeric",
    "Convert to datetime": "datetime",
    "Convert to categorical": "categorical",
    "Convert to string": "string",
}

_OUTLIER_OPTIONS = {
    "— skip —": None,
    "Remove outlier rows": "remove",
    "Cap to IQR bounds (winsorize)": "cap_iqr",
    "Replace with NaN": "replace_nan",
}

_DUP_OPTIONS = {
    "— skip —": None,
    "Keep first occurrence": "first",
    "Keep last occurrence": "last",
    "Remove all copies": "none",
}


def render(df: pd.DataFrame, report: dict) -> None:
    """Render interactive cleaning controls. Original df is never modified."""
    st.subheader("Data Cleaning")
    st.caption(
        "Choose a strategy for each issue and click **Apply Cleaning**. "
        "The original upload is never changed — cleaning always creates a new copy."
    )

    columns = report["columns"]
    dup_info = report.get("duplicates", {})

    missing_cols = {c: v for c, v in columns.items() if v["missing_count"] > 0}
    mismatch_cols = {c: v for c, v in columns.items() if v["type_mismatch"]}
    outlier_cols = {
        c: v for c, v in columns.items()
        if v["inferred_type"] in ("numeric", "mixed")
        and (v["outliers"].get("iqr_count") or 0) > 0
    }
    exact_count = dup_info.get("exact_count") or 0

    # ------------------------------------------------------------------ #
    # Missing values
    # ------------------------------------------------------------------ #
    if missing_cols:
        with st.expander(
            f"Missing Values — {len(missing_cols)} column(s) affected", expanded=True
        ):
            for col, info in missing_cols.items():
                c1, c2 = st.columns([2, 3])
                c1.markdown(
                    f"**{col}**  \n"
                    f"{info['missing_count']:,} missing ({info['missing_pct']:.1f}%)"
                )
                opts = (
                    _MISSING_OPTIONS_ALL
                    if info["inferred_type"] in ("numeric", "mixed")
                    else _MISSING_OPTIONS_NON_NUMERIC
                )
                choice = c2.selectbox(
                    "Strategy",
                    list(opts.keys()),
                    key=f"miss_{col}",
                    label_visibility="collapsed",
                )
                if choice == "Fill with constant":
                    st.text_input(
                        f"Constant value for `{col}`",
                        key=f"miss_{col}_val",
                        placeholder="Enter a value…",
                    )

    # ------------------------------------------------------------------ #
    # Duplicates
    # ------------------------------------------------------------------ #
    if exact_count > 0:
        with st.expander(f"Duplicates — {exact_count:,} exact duplicate rows", expanded=True):
            d1, d2 = st.columns([2, 3])
            d1.markdown(f"**{exact_count:,} rows can be removed**")
            d2.selectbox(
                "Keep option",
                list(_DUP_OPTIONS.keys()),
                key="dup_keep",
                label_visibility="collapsed",
            )

    # ------------------------------------------------------------------ #
    # Type mismatches
    # ------------------------------------------------------------------ #
    if mismatch_cols:
        with st.expander(
            f"Type Mismatches — {len(mismatch_cols)} column(s)", expanded=True
        ):
            for col, info in mismatch_cols.items():
                c1, c2 = st.columns([2, 3])
                c1.markdown(
                    f"**{col}**  \n"
                    f"stored as `{info['pandas_dtype']}`, "
                    f"inferred as `{info['inferred_type']}`"
                )
                c2.selectbox(
                    "Target type",
                    list(_TYPE_OPTIONS.keys()),
                    key=f"type_{col}",
                    label_visibility="collapsed",
                )

    # ------------------------------------------------------------------ #
    # Outliers
    # ------------------------------------------------------------------ #
    if outlier_cols:
        with st.expander(
            f"Outliers — {len(outlier_cols)} column(s) with outliers", expanded=True
        ):
            st.caption(
                "Outlier indices come from the original upload. "
                "If you also drop rows (missing values or duplicates), "
                "the pipeline runs outlier removal first automatically so indices stay valid."
            )
            out_method = st.radio(
                "Use indices from",
                ["IQR", "Z-score"],
                horizontal=True,
                key="outlier_method_clean",
            )
            method_key = "iqr" if out_method == "IQR" else "zscore"
            for col, info in outlier_cols.items():
                count = info["outliers"].get(f"{method_key}_count", 0)
                c1, c2 = st.columns([2, 3])
                c1.markdown(f"**{col}**  \n{count:,} outlier(s) ({out_method})")
                c2.selectbox(
                    "Strategy",
                    list(_OUTLIER_OPTIONS.keys()),
                    key=f"outlier_{col}",
                    label_visibility="collapsed",
                )

    if not (missing_cols or exact_count or mismatch_cols or outlier_cols):
        st.success("No issues detected — nothing to clean.")
        return

    # ------------------------------------------------------------------ #
    # Apply button
    # ------------------------------------------------------------------ #
    st.markdown("")
    if st.button("Apply Cleaning", type="primary", use_container_width=True):
        steps = _collect_steps(report, missing_cols, mismatch_cols, outlier_cols, exact_count)
        if not steps:
            st.warning("No cleaning actions selected — choose at least one strategy above.")
        else:
            pipeline = CleaningPipeline()
            for step in steps:
                pipeline.add_step(step)
            cleaned = pipeline.execute(df)
            st.session_state["cleaned_df"] = cleaned
            st.session_state["cleaning_log"] = pipeline.get_log()

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
    cleaned_df: pd.DataFrame | None = st.session_state.get("cleaned_df")
    cleaning_log: list = st.session_state.get("cleaning_log", [])

    if cleaned_df is not None and cleaning_log:
        _render_before_after(df, cleaned_df, report)
        _render_log(cleaning_log)
        _render_downloads(cleaned_df, report["dataset"]["name"])


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _collect_steps(
    report: dict,
    missing_cols: dict,
    mismatch_cols: dict,
    outlier_cols: dict,
    exact_count: int,
) -> list[dict]:
    steps: list[dict] = []

    for col, info in missing_cols.items():
        choice = st.session_state.get(f"miss_{col}", "— skip —")
        opts = (
            _MISSING_OPTIONS_ALL
            if info["inferred_type"] in ("numeric", "mixed")
            else _MISSING_OPTIONS_NON_NUMERIC
        )
        strategy = opts.get(choice)
        if strategy is None:
            continue
        step: dict = {"action": "handle_missing", "column": col, "strategy": strategy}
        if strategy == "fill_constant":
            raw = st.session_state.get(f"miss_{col}_val", "").strip()
            if not raw:
                st.warning(f"Enter a constant value for **{col}** before applying.")
                continue
            step["value"] = raw
        steps.append(step)

    if exact_count > 0:
        dup_choice = st.session_state.get("dup_keep", "— skip —")
        keep_val = _DUP_OPTIONS.get(dup_choice)
        if keep_val is not None:
            steps.append({"action": "remove_duplicates", "keep": keep_val})

    for col in mismatch_cols:
        choice = st.session_state.get(f"type_{col}", "— skip —")
        target_type = _TYPE_OPTIONS.get(choice)
        if target_type is None:
            continue
        steps.append({"action": "fix_type", "column": col, "target_type": target_type})

    if outlier_cols:
        method_key = "iqr" if st.session_state.get("outlier_method_clean", "IQR") == "IQR" else "zscore"
        for col in outlier_cols:
            choice = st.session_state.get(f"outlier_{col}", "— skip —")
            strategy = _OUTLIER_OPTIONS.get(choice)
            if strategy is None:
                continue
            indices = report["columns"][col]["outliers"].get(f"{method_key}_indices", [])
            steps.append({
                "action": "handle_outliers",
                "column": col,
                "indices": indices,
                "strategy": strategy,
            })

    steps.sort(key=_step_priority)
    return steps


def _step_priority(step: dict) -> int:
    """Sort order: type fixes and fills (0) → outlier remove (1) → row drops (2).

    Keeps profiler-computed outlier indices valid by running outlier removal
    before any step that resets the DataFrame index.
    """
    action = step["action"]
    strategy = step.get("strategy", "")
    if action == "handle_outliers" and strategy == "remove":
        return 1
    if action == "remove_duplicates" or (
        action == "handle_missing" and strategy in ("drop_rows", "drop_column")
    ):
        return 2
    return 0


def _render_before_after(
    orig_df: pd.DataFrame, cleaned_df: pd.DataFrame, report: dict
) -> None:
    st.divider()
    st.subheader("Before vs. After")

    orig_missing = int(orig_df.isnull().sum().sum())
    clean_missing = int(cleaned_df.isnull().sum().sum())
    orig_dups = report.get("duplicates", {}).get("exact_count") or 0
    clean_dups = int(cleaned_df.duplicated(keep=False).sum())
    orig_score = report["dataset"]["quality_score"]
    orig_grade = report["dataset"]["quality_grade"]

    with st.spinner("Recalculating quality score…"):
        cleaned_report = _profile_cleaned(cleaned_df, report["dataset"]["name"])
    clean_score = cleaned_report["dataset"]["quality_score"]
    clean_grade = cleaned_report["dataset"]["quality_grade"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{len(cleaned_df):,}", delta=f"{len(cleaned_df) - len(orig_df):,}")
    c2.metric(
        "Columns", cleaned_df.shape[1],
        delta=cleaned_df.shape[1] - orig_df.shape[1],
    )
    c3.metric(
        "Missing Cells", f"{clean_missing:,}",
        delta=f"{clean_missing - orig_missing:,}",
    )
    c4.metric(
        "Exact Duplicates", f"{clean_dups:,}",
        delta=f"{clean_dups - orig_dups:,}",
    )
    c5.metric(
        "Quality Score",
        f"{clean_score}/100 ({clean_grade})",
        delta=f"{clean_score - orig_score:+d} from {orig_score} ({orig_grade})",
    )

    with st.expander("Preview cleaned dataset (first 10 rows)"):
        st.dataframe(cleaned_df.head(10), use_container_width=True, hide_index=True)


def _render_log(log: list[dict]) -> None:
    st.subheader("Cleaning Log")
    rows = []
    for entry in log:
        rows.append({
            "Action": entry["action"],
            "Column": entry.get("column") or "—",
            "Strategy": entry.get("strategy", "—"),
            "Original Type": entry.get("original_dtype", "—"),
            "New Type": entry.get("new_dtype", "—"),
            "Rows Before": entry.get("rows_before", "—"),
            "Rows After": entry.get("rows_after", "—"),
            "Rows Removed": entry.get("rows_removed", 0),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_downloads(cleaned_df: pd.DataFrame, original_filename: str) -> None:
    st.subheader("Download Cleaned Data")
    stem = original_filename.rsplit(".", 1)[0] if "." in original_filename else original_filename

    dl1, dl2 = st.columns(2)

    with dl1:
        st.download_button(
            label="Download as CSV",
            data=cleaned_df.to_csv(index=False),
            file_name=f"{stem}_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with dl2:
        buf = io.BytesIO()
        cleaned_df.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        st.download_button(
            label="Download as Excel",
            data=buf.getvalue(),
            file_name=f"{stem}_cleaned.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
