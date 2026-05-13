import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde

from config.settings import MISSING_THRESHOLD

_TYPE_COLORS = {
    "numeric": "#3b82f6",
    "categorical": "#8b5cf6",
    "datetime": "#06b6d4",
    "boolean": "#10b981",
    "text": "#f59e0b",
    "mixed": "#ef4444",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render(report: dict) -> None:
    """Render the per-column profiling section with filter, sort, and jump controls."""
    st.subheader("Per-Column Profiling")

    columns = report["columns"]
    df: pd.DataFrame | None = st.session_state.get("df")

    if not columns:
        st.info("No columns to display.")
        return

    filtered, jump_to = _render_controls(columns)

    if not filtered:
        st.warning("No columns match the current filter.")
        return

    for col, info in filtered.items():
        expanded = col == jump_to
        _render_column_card(col, info, df, expanded=expanded)


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def _render_controls(columns: dict) -> tuple[dict, str | None]:
    """Render filter / sort / jump controls. Returns (filtered_columns, jump_to)."""
    all_types = sorted({v["inferred_type"] for v in columns.values()})

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])

    with ctrl1:
        selected_types = st.multiselect(
            "Filter by type", all_types, default=all_types, key="col_type_filter"
        )

    with ctrl2:
        issues_only = st.checkbox("Issues only", key="col_issues_filter")

    with ctrl3:
        sort_by = st.selectbox(
            "Sort by",
            ["Column name", "Missing %", "Warnings", "Type"],
            key="col_sort",
        )

    with ctrl4:
        col_names = list(columns.keys())
        jump_options = ["— All —"] + col_names
        jump_to_raw = st.selectbox("Jump to column", jump_options, key="col_jump")
        jump_to = jump_to_raw if jump_to_raw != "— All —" else None

    # Filter
    filtered = {
        c: v
        for c, v in columns.items()
        if v["inferred_type"] in selected_types
        and (not issues_only or v["type_mismatch"] or v["missing_count"] > 0 or v["warnings"])
    }

    # Sort
    if sort_by == "Missing %":
        filtered = dict(sorted(filtered.items(), key=lambda x: -x[1]["missing_pct"]))
    elif sort_by == "Warnings":
        filtered = dict(sorted(filtered.items(), key=lambda x: -len(x[1]["warnings"])))
    elif sort_by == "Type":
        filtered = dict(sorted(filtered.items(), key=lambda x: x[1]["inferred_type"]))
    else:
        filtered = dict(sorted(filtered.items()))

    # If jump_to is set, bring that column to the front
    if jump_to and jump_to in filtered:
        rest = {c: v for c, v in filtered.items() if c != jump_to}
        filtered = {jump_to: filtered[jump_to], **rest}

    return filtered, jump_to


# ---------------------------------------------------------------------------
# Column card
# ---------------------------------------------------------------------------


def _render_column_card(
    col: str, info: dict, df: pd.DataFrame | None, expanded: bool = False
) -> None:
    has_warnings = bool(info["warnings"])
    missing_str = f" · {info['missing_pct']:.1f}% missing" if info["missing_count"] > 0 else ""
    warn_prefix = "⚠ " if has_warnings else ""
    label = f"{warn_prefix}{col}  ·  {info['inferred_type']}{missing_str}"

    with st.expander(label, expanded=expanded):
        left, right = st.columns([1, 2])

        with left:
            _render_meta(info)
            st.markdown("")
            _render_stats_block(info)
            st.markdown("")
            _render_distribution_info(info)
            if info["warnings"]:
                st.markdown("")
                for w in info["warnings"]:
                    st.warning(w, icon="⚠️")

        with right:
            if df is not None:
                fig = _make_chart(col, info, df)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No chart available for this column.")
            else:
                st.info("Original data not available for charting.")


def _render_meta(info: dict) -> None:
    color = _TYPE_COLORS.get(info["inferred_type"], "#6b7280")
    st.markdown(
        f'<span style="background:{color}22; color:{color}; padding:2px 8px; '
        f'border-radius:4px; font-size:0.8rem; font-weight:600;">'
        f'{info["inferred_type"].upper()}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Pandas dtype:** `{info['pandas_dtype']}`  \n"
        f"**Type mismatch:** {'Yes' if info['type_mismatch'] else 'No'}  \n"
        f"**Missing:** {info['missing_count']:,} rows ({info['missing_pct']:.1f}%)  \n"
        f"**Pattern:** {info.get('missing_pattern', '—')}"
    )

    _pattern = info.get("missing_pattern", "none")
    _pattern_hints = {
        "MCAR": "MCAR — Missing Completely At Random: no detectable pattern",
        "MAR":  "MAR — Missing At Random: missingness overlaps with another column",
        "MNAR": "MNAR — Missing Not At Random: likely structural absence (>60% missing)",
    }
    if _pattern in _pattern_hints:
        st.caption(_pattern_hints[_pattern])

    # Missing severity bar
    pct = info["missing_pct"]
    if pct > 0:
        threshold_pct = MISSING_THRESHOLD * 100
        bar_color = "#ef4444" if pct >= threshold_pct else "#f97316" if pct >= 10 else "#fbbf24"
        st.markdown(
            f'<div style="background:#e5e7eb; border-radius:4px; height:6px; margin:4px 0;">'
            f'<div style="background:{bar_color}; width:{min(pct, 100):.1f}%; '
            f'height:6px; border-radius:4px;"></div></div>',
            unsafe_allow_html=True,
        )


def _render_stats_block(info: dict) -> None:
    s = info["stats"]
    inferred = info["inferred_type"]

    if not s:
        return

    st.markdown("**Key Stats**")

    if inferred in ("numeric", "mixed"):
        rows = []
        for label, key in [
            ("Mean", "mean"), ("Median", "median"), ("Std Dev", "std"),
            ("Min", "min"), ("Max", "max"), ("Zeros", "zeros"), ("Unique", "unique_count"),
        ]:
            val = s.get(key)
            if val is not None:
                fmt = f"{val:,.4g}" if isinstance(val, float) else f"{val:,}"
                rows.append({"Metric": label, "Value": fmt})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    elif inferred in ("categorical", "boolean"):
        rows = [
            {"Metric": "Unique values", "Value": str(s.get("unique_count", "—"))},
            {"Metric": "Cardinality ratio", "Value": f"{s.get('cardinality_ratio', 0):.1%}"},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        top = s.get("top_values", {})
        if top:
            st.markdown("**Top values**")
            top_rows = [{"Value": str(k), "Count": f"{v:,}"} for k, v in list(top.items())[:5]]
            st.dataframe(pd.DataFrame(top_rows), hide_index=True, use_container_width=True)

    elif inferred == "datetime":
        rows = [
            {"Metric": "Min date", "Value": str(s.get("min_date", "—"))},
            {"Metric": "Max date", "Value": str(s.get("max_date", "—"))},
            {"Metric": "Range (days)", "Value": str(s.get("range_days", "—"))},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    elif inferred == "text":
        rows = [
            {"Metric": "Unique values", "Value": str(s.get("unique_count", "—"))},
            {"Metric": "Avg length", "Value": f"{s.get('avg_length', 0):.1f} chars"},
            {"Metric": "Max length", "Value": f"{s.get('max_length', '—')} chars"},
            {"Metric": "Min length", "Value": f"{s.get('min_length', '—')} chars"},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_distribution_info(info: dict) -> None:
    dist = info.get("distribution", {})
    inferred = info["inferred_type"]

    if inferred in ("numeric", "mixed"):
        lines = []
        if dist.get("skew_label"):
            lines.append(f"**Skewness:** {dist['skew_label']}")
        if dist.get("kurtosis_label"):
            lines.append(f"**Kurtosis:** {dist['kurtosis_label']}")
        if dist.get("normality_pvalue") is not None:
            pval = dist["normality_pvalue"]
            verdict = "Normal" if dist["is_normal"] else "Not normal"
            lines.append(f"**Normality test:** {verdict} (p={pval:.4f})")
        if lines:
            st.markdown("**Distribution**")
            st.markdown("  \n".join(lines))

    elif inferred == "datetime":
        gaps = dist.get("datetime_gaps") or {}
        if gaps.get("gap_count", 0) > 0:
            st.markdown("**Distribution**")
            st.markdown(
                f"**Gaps detected:** {gaps['gap_count']}  \n"
                f"**Max gap:** {gaps['max_gap_days']} days  \n"
                f"**Median interval:** {gaps['median_gap_days']} days"
            )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _make_chart(col: str, info: dict, df: pd.DataFrame) -> go.Figure | None:
    inferred = info["inferred_type"]
    series = df[col]

    if inferred in ("numeric", "mixed"):
        return _numeric_chart(series, col)
    if inferred in ("categorical", "boolean"):
        return _categorical_chart(info["stats"].get("top_values", {}), col)
    if inferred == "datetime":
        return _datetime_chart(series, col)
    if inferred == "text":
        return _text_length_chart(series, col)
    return None


def _numeric_chart(series: pd.Series, col: str) -> go.Figure | None:
    vals = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if len(vals) == 0:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=vals,
            histnorm="probability density",
            name="Distribution",
            marker_color="#3b82f6",
            opacity=0.65,
            nbinsx=min(40, max(5, len(vals) // 5)),
        )
    )

    # KDE overlay — requires at least 5 points and non-zero variance
    if len(vals) >= 5 and float(vals.std()) > 0:
        try:
            kde = gaussian_kde(vals)
            x_range = np.linspace(float(vals.min()), float(vals.max()), 300)
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kde(x_range),
                    mode="lines",
                    name="KDE",
                    line=dict(color="#ef4444", width=2),
                )
            )
        except Exception:
            pass

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=40),
        yaxis_title="Density",
        xaxis_title=col,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.04,
    )
    return fig


def _categorical_chart(top_values: dict, col: str) -> go.Figure | None:
    if not top_values:
        return None
    labels = list(top_values.keys())
    counts = list(top_values.values())

    # Horizontal bar so long labels don't get clipped
    fig = go.Figure(
        go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker_color="#8b5cf6",
            text=counts,
            textposition="outside",
        )
    )
    fig.update_layout(
        height=max(200, len(labels) * 40 + 60),
        margin=dict(l=0, r=60, t=10, b=40),
        xaxis_title="Count",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        xaxis_range=[0, max(counts) * 1.25] if counts else None,
    )
    return fig


def _datetime_chart(series: pd.Series, col: str) -> go.Figure | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            dt = pd.to_datetime(series.dropna(), format="mixed", errors="coerce").dropna()
        except Exception:
            dt = pd.to_datetime(series.dropna(), errors="coerce").dropna()
    if len(dt) == 0:
        return None

    range_days = (dt.max() - dt.min()).days
    if range_days > 730:
        period, label = "Y", "Year"
    elif range_days > 60:
        period, label = "M", "Month"
    else:
        period, label = "W", "Week"

    counts = dt.dt.to_period(period).value_counts().sort_index()

    fig = go.Figure(
        go.Bar(
            x=[str(p) for p in counts.index],
            y=counts.values,
            marker_color="#06b6d4",
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=60),
        yaxis_title="Count",
        xaxis_title=label,
        showlegend=False,
    )
    return fig


def _text_length_chart(series: pd.Series, col: str) -> go.Figure | None:
    lengths = series.dropna().astype(str).str.len()
    if len(lengths) == 0:
        return None

    fig = go.Figure(
        go.Histogram(
            x=lengths,
            marker_color="#f59e0b",
            opacity=0.8,
            nbinsx=min(30, lengths.nunique()),
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=40),
        xaxis_title="String length (chars)",
        yaxis_title="Count",
        showlegend=False,
    )
    return fig
