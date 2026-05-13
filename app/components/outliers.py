import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render(report: dict) -> None:
    """Render outlier detection results for all numeric columns."""
    st.subheader("Outlier Analysis")

    columns = report["columns"]
    numeric_cols = {
        col: info
        for col, info in columns.items()
        if info["inferred_type"] in ("numeric", "mixed")
    }

    if not numeric_cols:
        st.info("No numeric columns available for outlier analysis.")
        return

    method = st.radio(
        "Detection method",
        ["IQR (1.5×)", "Z-score (3σ)"],
        horizontal=True,
        key="outlier_method",
    )
    use_iqr = method == "IQR (1.5×)"

    if use_iqr:
        st.caption(
            "**IQR (Interquartile Range):** flags values that fall more than 1.5× the "
            "distance between the 25th and 75th percentiles below Q1 or above Q3. "
            "Robust to skewed data and does not assume a normal distribution."
        )
    else:
        st.caption(
            "**Z-score (Standard Score):** flags values more than 3 standard deviations "
            "from the column mean. Works best when the data is approximately normally "
            "distributed; sensitive to extreme values in heavily skewed columns."
        )

    # Summary table
    st.markdown("**Outlier Summary**")
    rows = []
    for col, info in numeric_cols.items():
        o = info["outliers"]
        count = o["iqr_count"] if use_iqr else o["zscore_count"]
        pct = o.get("iqr_pct", 0.0) if use_iqr else o.get("zscore_pct", 0.0)
        rows.append({
            "Column": col,
            "Type": info["inferred_type"],
            "Outliers": count,
            "Outlier %": f"{pct:.1f}%",
        })

    df_summary = pd.DataFrame(rows).sort_values("Outliers", ascending=False)

    styled = (
        df_summary.style
        .map(lambda v: "color: #dc2626;" if isinstance(v, int) and v > 0 else "", subset=["Outliers"])
        .map(lambda v: "color: #dc2626;" if isinstance(v, str) and float(v.rstrip("%")) > 0 else "", subset=["Outlier %"])
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)

    # Per-column box plots
    df_orig = st.session_state.get("df")
    if df_orig is None:
        return

    cols_with_outliers = [
        col for col, info in numeric_cols.items()
        if (info["outliers"]["iqr_count"] if use_iqr else info["outliers"]["zscore_count"]) > 0
    ]

    if not cols_with_outliers:
        st.success("No outliers detected in any numeric column with the selected method.")
        return

    st.markdown("**Box Plots — Outliers Highlighted in Red**")
    for col in cols_with_outliers:
        o = numeric_cols[col]["outliers"]
        outlier_indices = o["iqr_indices"] if use_iqr else o["zscore_indices"]
        count = len(outlier_indices)

        with st.expander(f"{col}  ·  {count} outlier(s)", expanded=False):
            fig = _boxplot_with_outliers(col, df_orig, outlier_indices)
            if fig:
                st.plotly_chart(fig, use_container_width=True)


def _boxplot_with_outliers(
    col: str, df: pd.DataFrame, outlier_indices: list
) -> go.Figure | None:
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(vals) == 0:
        return None

    outlier_set = set(outlier_indices)
    normal_mask = ~df.index.isin(outlier_set)
    normal_vals = pd.to_numeric(df.loc[normal_mask, col], errors="coerce").dropna()
    outlier_vals = pd.to_numeric(
        df.loc[df.index.isin(outlier_set), col], errors="coerce"
    ).dropna()

    # Sample normal points for display to keep charts fast
    show_normal = (
        normal_vals.sample(300, random_state=42)
        if len(normal_vals) > 300
        else normal_vals
    )

    fig = go.Figure()

    fig.add_trace(go.Box(
        y=vals.tolist(),
        name=col,
        boxpoints=False,
        fillcolor="rgba(147, 197, 253, 0.25)",
        line=dict(color="#3b82f6"),
        showlegend=False,
    ))

    if len(show_normal) > 0:
        fig.add_trace(go.Scatter(
            y=show_normal.tolist(),
            x=[col] * len(show_normal),
            mode="markers",
            marker=dict(color="#93c5fd", size=4, opacity=0.5),
            name="Normal",
        ))

    if len(outlier_vals) > 0:
        fig.add_trace(go.Scatter(
            y=outlier_vals.tolist(),
            x=[col] * len(outlier_vals),
            mode="markers",
            marker=dict(
                color="#ef4444", size=9,
                symbol="circle-open", line=dict(width=2, color="#ef4444"),
            ),
            name=f"Outliers ({len(outlier_vals)})",
        ))

    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=40),
        showlegend=True,
        legend=dict(orientation="h", y=1.12),
    )
    return fig
