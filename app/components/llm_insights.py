import streamlit as st

from narrator.client import is_configured
from narrator.narrator import Narrator

_STATE_KEY = "llm_insights"
_QA_KEY = "llm_qa_history"


def render(report: dict) -> None:
    """Render sidebar toggle and AI insights section.

    When the toggle is off (default), no API calls are made and nothing is
    rendered in the main area (not even a divider). When on, a "Generate"
    button triggers the structured insights and a Q&A box lets the user ask
    free-form questions — all cached in st.session_state.
    """
    with st.sidebar:
        st.divider()
        st.markdown("**🤖 AI Insights**")
        st.caption("Generate plain-English analysis of this dataset.")
        enabled = st.toggle("🤖 Enable AI Insights", value=False, key="llm_enabled")

    if not enabled:
        return

    st.divider()
    st.subheader("🤖 AI Insights")

    if not is_configured():
        st.info(
            "AI Insights require LLM API credentials. "
            "Copy `.env.example` to `.env` and set:\n"
            "- `GWDG_API_KEY` — your API key\n"
            "- `GWDG_API_BASE` — your API endpoint "
            "(any OpenAI-compatible provider works, not just GWDG)\n"
            "- `GWDG_MODEL_NAME` — the model name to use\n\n"
            "Then restart the app."
        )
        return

    _render_generated(report)
    st.divider()
    _render_qa(report)


# ---------------------------------------------------------------------------
# Generated insights (overview + column analysis + cleaning strategy)
# ---------------------------------------------------------------------------


def _render_generated(report: dict) -> None:
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
        st.markdown("#### 📊 Dataset Summary")
        st.info(cached["overview"])

    col_insights: dict = cached.get("columns", {})
    if col_insights:
        st.markdown("#### 🔍 Column Analysis")
        for col_name, text in col_insights.items():
            with st.expander(f"**{col_name}**"):
                st.markdown(text)

    if "cleaning_plan" in cached:
        st.markdown("#### 🧹 Recommended Cleaning Strategy")
        st.markdown(cached["cleaning_plan"])


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


# ---------------------------------------------------------------------------
# Free-form Q&A
# ---------------------------------------------------------------------------


def _render_qa(report: dict) -> None:
    st.markdown("#### 💬 Ask a Question")
    st.caption(
        "Ask anything about this dataset. "
        "The LLM answers based on the profile statistics — not raw data rows."
    )

    qa_history: list = st.session_state.get(_QA_KEY, [])

    for entry in qa_history:
        st.markdown(f"**You:** {entry['question']}")
        st.markdown(entry["answer"])
        st.divider()

    q_col, btn_col = st.columns([4, 1])
    with q_col:
        question = st.text_input(
            "Question",
            placeholder="e.g. Which column needs the most urgent attention?",
            key="llm_question_input",
            label_visibility="collapsed",
        )
    with btn_col:
        ask_clicked = st.button("Ask", key="llm_ask_btn", use_container_width=True)

    if ask_clicked:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            narrator = Narrator(report)
            with st.spinner("Thinking…"):
                try:
                    answer = narrator.answer_question(question.strip())
                    st.session_state[_QA_KEY] = qa_history + [
                        {"question": question.strip(), "answer": answer}
                    ]
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not answer: {exc}")
