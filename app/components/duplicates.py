import streamlit as st


def render(report: dict) -> None:
    """Render duplicate row analysis with summary metrics and sample rows."""
    st.subheader("Duplicate Analysis")

    dup = report.get("duplicates", {})
    exact_count = dup.get("exact_count") or 0
    exact_pct = dup.get("exact_pct") or 0.0
    near_count = dup.get("near_duplicate_count") or 0
    near_pct = dup.get("near_duplicate_pct") or 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Exact Duplicates", f"{exact_count:,}")
    col2.metric("Duplicate %", f"{exact_pct:.1f}%")
    col3.metric("Near-Duplicates", f"{near_count:,}")
    col4.metric("Near-Dup %", f"{near_pct:.1f}%")

    if exact_count == 0 and near_count == 0:
        st.success("No duplicate rows detected.")
        return

    df = st.session_state.get("df")

    # Exact duplicates
    if exact_count > 0:
        st.markdown(f"**Exact Duplicate Rows** — {exact_count:,} rows can be removed")
        sample_indices = dup.get("sample_indices", [])
        if sample_indices and df is not None:
            sample_df = df.loc[sample_indices]
            st.dataframe(sample_df, use_container_width=True, hide_index=True)
            if len(sample_indices) >= 50:
                st.caption("Showing first 50 rows from duplicate groups.")

    # Near-duplicates
    if near_count > 0:
        with st.expander(
            f"Near-Duplicates — {near_count:,} rows differ in exactly one column",
            expanded=False,
        ):
            st.caption(
                "These rows are identical on all columns except one. "
                "They may represent data entry errors or intentional variations — "
                "review manually before removing."
            )
            near_indices = dup.get("near_sample_indices", [])
            if near_indices and df is not None:
                near_df = df.loc[near_indices]
                st.dataframe(near_df, use_container_width=True, hide_index=True)
