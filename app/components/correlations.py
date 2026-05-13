import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import HIGH_CORRELATION_THRESHOLD


def render(report: dict) -> None:
    """Render Pearson heatmap, Cramér's V table, and high-correlation pair list."""
    st.subheader("Correlation Analysis")

    corr = report.get("correlations", {})
    pearson = corr.get("pearson", {})
    cramers = corr.get("cramers_v", {})
    high_pairs = corr.get("high_correlation_pairs", [])

    if not pearson and not cramers:
        st.info("No numeric or categorical columns available for correlation analysis.")
        return

    # Pearson heatmap
    if pearson:
        st.markdown("**Pearson Correlation — Numeric Columns**")
        _render_pearson_heatmap(pearson)

    # High-correlation pairs
    if high_pairs:
        st.markdown(
            f"**Highly Correlated Pairs** "
            f"(absolute value ≥ {HIGH_CORRELATION_THRESHOLD})"
        )
        rows = [
            {
                "Column A": p["col_a"],
                "Column B": p["col_b"],
                "Method": "Pearson r" if p["method"] == "pearson" else "Cramér's V",
                "Value": p["value"],
            }
            for p in high_pairs
        ]
        df_pairs = pd.DataFrame(rows)
        styled = df_pairs.style.map(
            lambda v: "color: #dc2626; font-weight:600;" if isinstance(v, float) and abs(v) >= HIGH_CORRELATION_THRESHOLD else "",
            subset=["Value"],
        )
        st.dataframe(styled, hide_index=True, use_container_width=True)
    elif pearson or cramers:
        st.success(
            f"No highly correlated pairs found (threshold: |r| ≥ {HIGH_CORRELATION_THRESHOLD})."
        )

    # Cramér's V section
    if cramers:
        with st.expander("Cramér's V — Categorical Associations"):
            _render_cramers_table(cramers)


def _render_pearson_heatmap(pearson: dict) -> None:
    cols = list(pearson.keys())
    if len(cols) < 2:
        st.info("Need at least 2 numeric columns for a correlation heatmap.")
        return

    matrix = [
        [
            round(float(pearson[col_a][col_b]), 3)
            if pearson[col_a].get(col_b) is not None
            else 0.0
            for col_b in cols
        ]
        for col_a in cols
    ]

    fig = px.imshow(
        matrix,
        x=cols,
        y=cols,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig.update_layout(
        height=max(300, len(cols) * 45 + 100),
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(title="r"),
    )
    fig.update_traces(textfont_size=10)
    st.plotly_chart(fig, use_container_width=True)


def _render_cramers_table(cramers: dict) -> None:
    rows = [
        {"Column A": col_a, "Column B": col_b, "Cramér's V": v}
        for col_a, pairs in cramers.items()
        for col_b, v in pairs.items()
    ]
    if not rows:
        st.info("No categorical-categorical pairs to display.")
        return

    df_c = pd.DataFrame(rows).sort_values("Cramér's V", ascending=False)
    st.dataframe(df_c, hide_index=True, use_container_width=True)
