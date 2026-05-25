import streamlit as st

from narrator.client import is_configured
from narrator.narrator import Narrator

_STATE_KEY = "llm_insights"


def render(report: dict) -> None:
    """Render sidebar toggle and AI insights section.

    When the toggle is off (default), no API calls are made and no main-area
    content is rendered. When on, a single "Generate" button triggers all LLM
    calls and caches results in st.session_state.
    """
    with st.sidebar:
        st.divider()
        enabled = st.toggle("Enable AI Insights", value=False, key="llm_enabled")

    if not enabled:
        return

    st.subheader("AI Insights")

    if not is_configured():
        st.info(
            "AI Insights require GWDG credentials. "
            "Copy `.env.example` to `.env`, fill in `GWDG_API_KEY`, "
            "`GWDG_API_BASE`, and `GWDG_MODEL_NAME`, then restart the app."
        )
        return

    cached: dict = st.session_state.get(_STATE_KEY, {})

    col_btn, col_cap = st.columns([1, 3])
    with col_btn:
        generate_clicked = st.button(
            "Generate" if not cached else "Regenerate",
            type="primary",
            use_container_width=True,
        )
    with col_cap:
        if cached:
            st.caption("Insights cached from the last run. Click Regenerate to refresh.")
        else:
            st.caption("Sends only summary statistics to the LLM — never raw data rows.")

    if generate_clicked:
        cached = _generate_all(report)
        st.session_state[_STATE_KEY] = cached

    if not cached:
        return

    if "overview" in cached:
        st.markdown("#### Dataset Summary")
        st.info(cached["overview"])

    col_insights: dict = cached.get("columns", {})
    if col_insights:
        st.markdown("#### Column Analysis")
        for col_name, text in col_insights.items():
            with st.expander(f"**{col_name}**"):
                st.write(text)

    if "cleaning_plan" in cached:
        st.markdown("#### Recommended Cleaning Strategy")
        st.write(cached["cleaning_plan"])


def _generate_all(report: dict) -> dict:
    narrator = Narrator(report)
    result: dict = {}

    issue_cols = [
        col
        for col, info in report["columns"].items()
        if (
            info["missing_count"] > 0
            or info["outliers"]["iqr_count"] > 0
            or info["type_mismatch"]
            or info["warnings"]
        )
    ]

    with st.spinner("Generating dataset overview…"):
        try:
            result["overview"] = narrator.narrate_overview()
        except Exception as exc:
            st.error(f"Overview generation failed: {exc}")

    if issue_cols:
        result["columns"] = {}
        with st.spinner(f"Analysing {len(issue_cols)} column(s)…"):
            for col in issue_cols:
                try:
                    result["columns"][col] = narrator.narrate_column(col)
                except Exception as exc:
                    result["columns"][col] = f"Analysis failed: {exc}"

    with st.spinner("Generating cleaning strategy…"):
        try:
            result["cleaning_plan"] = narrator.narrate_cleaning_plan()
        except Exception as exc:
            st.error(f"Cleaning plan generation failed: {exc}")

    return result
