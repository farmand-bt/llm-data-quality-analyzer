import streamlit as st

_SEVERITY_COLORS = {
    "critical": "#dc2626",
    "warning":  "#d97706",
    "info":     "#2563eb",
}


def render(report: dict) -> None:
    """Render the rule-based recommendations sorted by severity."""
    st.subheader("Recommendations")

    recs = report.get("recommendations", [])

    if not recs:
        st.success("No recommendations — dataset looks clean!")
        return

    counts = {s: sum(1 for r in recs if r["severity"] == s) for s in ("critical", "warning", "info")}
    c1, c2, c3 = st.columns(3)
    c1.metric("Critical", counts["critical"])
    c2.metric("Warnings", counts["warning"])
    c3.metric("Info", counts["info"])

    st.markdown("")

    for severity, label in [("critical", "Critical"), ("warning", "Warning"), ("info", "Info")]:
        group = [r for r in recs if r["severity"] == severity]
        if not group:
            continue

        color = _SEVERITY_COLORS[severity]
        st.markdown(f"**{label}** ({len(group)})")
        for rec in group:
            col_tag = f" `{rec['column']}`" if rec.get("column") else ""
            badge = f"[{rec['category'].upper()}]"
            st.markdown(
                f'<div style="border-left:3px solid {color}; padding:6px 12px; '
                f'margin:4px 0 4px 0; background:{color}0d; border-radius:0 4px 4px 0;">'
                f'<span style="color:{color}; font-weight:600;">{badge}</span>'
                f'<span style="color:#374151;">{col_tag} {rec["message"]}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")
