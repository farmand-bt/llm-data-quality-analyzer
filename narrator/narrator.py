# Implemented in Milestone 6


def generate_insights(report: dict) -> dict:
    """Take the profiler report dict and return plain-English insights via GWDG API.

    Only summary stats are sent to the LLM — never raw DataFrame rows.
    Results should be cached in st.session_state by the caller.
    """
    raise NotImplementedError
