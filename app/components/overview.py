from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import MISSING_THRESHOLD

_GRADE_COLORS = {
    "A": "#22c55e",
    "B": "#84cc16",
    "C": "#eab308",
    "D": "#f97316",
    "F": "#ef4444",
}
_TYPE_COLORS = {
    "numeric": "#3b82f6",
    "categorical": "#8b5cf6",
    "datetime": "#06b6d4",
    "boolean": "#10b981",
    "text": "#f59e0b",
    "mixed": "#ef4444",
}


def render(report: dict) -> None:
    """Render the dataset-level overview dashboard."""
    dataset = report["dataset"]
    columns = report["columns"]

    _render_score_and_metrics(dataset, columns)
    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Data Type Distribution")
        _render_type_distribution(columns)
    with right:
        st.subheader("Missing Values per Column")
        _render_missing_bar(columns)

    st.divider()
    st.subheader("Missing Value Heatmap")
    df = st.session_state.get("df")
    if df is not None:
        _render_missing_heatmap(df)

    st.divider()
    st.subheader("Column Summary")
    _render_column_table(columns)


def _render_score_and_metrics(dataset: dict, columns: dict) -> None:
    score = dataset["quality_score"]
    grade = dataset["quality_grade"]
    color = _GRADE_COLORS.get(grade, "#6b7280")

    col_badge, col_metrics = st.columns([1, 3])

    with col_badge:
        st.markdown(
            f"""
            <div style="text-align:center; padding:24px 16px; border-radius:12px;
                        background:{color}18; border:2px solid {color};">
                <div style="font-size:3.5rem; font-weight:700; color:{color}; line-height:1;">
                    {grade}
                </div>
                <div style="font-size:1.4rem; font-weight:600; color:{color}; margin-top:4px;">
                    {score} / 100
                </div>
                <div style="font-size:0.8rem; color:#6b7280; margin-top:6px;">
                    Quality Score
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_metrics:
        total_cells = dataset["rows"] * dataset["columns"]
        total_missing = sum(v["missing_count"] for v in columns.values())
        overall_missing_pct = round(total_missing / total_cells * 100, 1) if total_cells > 0 else 0.0
        mismatch_count = sum(1 for v in columns.values() if v["type_mismatch"])
        warn_count = sum(len(v["warnings"]) for v in columns.values())

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rows", f"{dataset['rows']:,}")
        m2.metric("Columns", dataset["columns"])
        m3.metric("Memory", f"{dataset['memory_mb']:.2f} MB")
        m4.metric("Missing Cells", f"{overall_missing_pct}%")
        m5.metric("Type Mismatches", mismatch_count)

        if warn_count:
            st.warning(f"{warn_count} column warning(s) detected — see Column Summary below.")


def _render_type_distribution(columns: dict) -> None:
    type_counts = Counter(v["inferred_type"] for v in columns.values())
    types = list(type_counts.keys())
    counts = [type_counts[t] for t in types]
    colors = [_TYPE_COLORS.get(t, "#6b7280") for t in types]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=types,
            orientation="h",
            marker_color=colors,
            text=counts,
            textposition="outside",
        )
    )
    fig.update_layout(
        height=max(200, len(types) * 45 + 60),
        margin=dict(l=0, r=40, t=10, b=10),
        xaxis_title="Number of Columns",
        yaxis_title=None,
        showlegend=False,
        xaxis=dict(range=[0, max(counts) * 1.3]),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_missing_bar(columns: dict) -> None:
    cols_with_missing = {c: v["missing_pct"] for c, v in columns.items() if v["missing_count"] > 0}
    if not cols_with_missing:
        st.success("No missing values found in the dataset.")
        return

    threshold_pct = MISSING_THRESHOLD * 100
    sorted_items = sorted(cols_with_missing.items(), key=lambda x: -x[1])
    col_names = [c for c, _ in sorted_items]
    pcts = [p for _, p in sorted_items]
    colors = [
        "#ef4444" if p >= threshold_pct else "#f97316" if p >= 10 else "#fbbf24"
        for p in pcts
    ]

    fig = go.Figure(
        go.Bar(
            x=col_names,
            y=pcts,
            marker_color=colors,
            text=[f"{p:.1f}%" for p in pcts],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=max(250, len(col_names) * 30 + 80),
        margin=dict(l=0, r=0, t=10, b=60),
        yaxis_title="Missing %",
        yaxis_range=[0, min(100, max(pcts) * 1.25)],
        xaxis_title=None,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Red = ≥{threshold_pct:.0f}% (critical threshold) · "
        f"Orange = ≥10% · Yellow = <10%"
    )


def _render_missing_heatmap(df: pd.DataFrame, max_rows: int = 300) -> None:
    sample = df.iloc[:max_rows]
    missing_cols = [c for c in sample.columns if sample[c].isnull().any()]

    if not missing_cols:
        st.success("No missing values — heatmap not applicable.")
        return

    null_matrix = sample[missing_cols].isnull().astype(int)
    fig = px.imshow(
        null_matrix.T,
        color_continuous_scale=["#dbeafe", "#ef4444"],
        aspect="auto",
        labels={"x": "Row index", "y": "Column", "color": "Missing"},
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(
        height=max(150, len(missing_cols) * 28 + 60),
        margin=dict(l=0, r=0, t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Red = missing · Showing first {min(max_rows, len(df)):,} rows · "
        f"{len(missing_cols)} of {len(df.columns)} columns have missing values"
    )


def _render_column_table(columns: dict) -> None:
    rows = []
    for col, info in columns.items():
        rows.append(
            {
                "Column": col,
                "Pandas Dtype": info["pandas_dtype"],
                "Inferred Type": info["inferred_type"],
                "Type Mismatch": "⚠ Yes" if info["type_mismatch"] else "✓ No",
                "Missing Count": info["missing_count"],
                "Missing %": f"{info['missing_pct']:.1f}%",
                "Pattern": info.get("missing_pattern", "—"),
                "Warnings": len(info["warnings"]),
            }
        )

    df_table = pd.DataFrame(rows)

    styled = (
        df_table.style
        .map(lambda v: "color: #dc2626;" if isinstance(v, int) and v > 0 else "",
             subset=["Missing Count", "Warnings"])
        .map(lambda v: "color: #dc2626;" if isinstance(v, str) and float(v.rstrip("%")) > 0 else "",
             subset=["Missing %"])
        .map(lambda v: "color: #dc2626;" if v == "⚠ Yes" else "",
             subset=["Type Mismatch"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    with st.expander("What does the Pattern column mean?"):
        st.markdown(
            """
            The **Pattern** column is a heuristic guess at *why* values are missing:

            | Pattern | Meaning |
            |---------|---------|
            | **none** | No missing values — column is complete |
            | **MCAR** | *Missing Completely At Random* — no detectable pattern; missingness appears random and unrelated to any other variable |
            | **MAR** | *Missing At Random* — missingness strongly overlaps with another column also having missing values, suggesting a shared cause |
            | **MNAR** | *Missing Not At Random* — over 60% of values are absent, suggesting the missingness is structural (e.g. a field only filled under certain conditions) |

            These are heuristics, not statistical proofs. Use them as a starting point for investigation.
            """
        )
