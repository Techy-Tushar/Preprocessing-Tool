import streamlit as st
import pandas as pd

def run_data_explorer():
    st.markdown("""
    <div class="page-title-box">
        <span style="font-size:28px;font-weight:800;">📂 Data Explorer</span>
        <div style="margin-top:6px;font-size:14px;opacity:0.85;">
            Inspect dataset structure, data types, missing values, and quick profiling.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ---- CHECK DATA ----
    if st.session_state.get("clean_df") is None:
        st.warning("⚠ No dataset found. Please upload a dataset from Home.")
        return

    df = st.session_state["clean_df"]

    # --------------------------------------------------------------------
    # SECTION 1 — BASIC OVERVIEW
    # --------------------------------------------------------------------
    st.header("📌 Dataset Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Rows", df.shape[0])
    with c2:
        st.metric("Columns", df.shape[1])
    with c3:
        st.metric("Memory Usage (KB)", round(df.memory_usage().sum() / 1024, 2))

    st.markdown("---")

    # --------------------------------------------------------------------
    # SECTION 2 — COLUMN SUMMARY
    # --------------------------------------------------------------------
    st.header("📌 Column Summary")

    col_summary = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.values,
        "Missing (%)": df.isnull().mean().values * 100,
        "Unique Values": df.nunique().values,
    })

    st.dataframe(col_summary, use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------------------------
    # SECTION 3 — NUMERIC SUMMARY
    # --------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
    if numeric_cols:
        with st.expander("📊 Numeric Columns — Summary Statistics", expanded=False):
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

    # --------------------------------------------------------------------
    # SECTION 4 — CATEGORICAL SUMMARY
    # --------------------------------------------------------------------
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        with st.expander("🔤 Categorical Columns — Unique Counts", expanded=False):
            cat_summary = pd.DataFrame({
                "Column": cat_cols,
                "Unique Values": [df[col].nunique() for col in cat_cols],
                "Missing (%)": [df[col].isnull().mean() * 100 for col in cat_cols]
            })
            st.dataframe(cat_summary, use_container_width=True)

        with st.expander("📑 Categorical Columns — Value Counts", expanded=False):
            col_to_view = st.selectbox("Select a column", cat_cols)
            st.dataframe(df[col_to_view].value_counts().head(50))

    st.markdown("---")

    # --------------------------------------------------------------------
    # SECTION 5 — PREVIEW
    # --------------------------------------------------------------------
    st.header("📌 Preview (first 50 rows)")
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------------------------
    # SECTION 6 — CONTINUE BUTTON
    # --------------------------------------------------------------------
    if st.button("Save & Continue to Fix Missing Values"):
        st.session_state.page_completed["Data Explorer"] = True
        st.session_state.current_page = "Fix Missing Values"
        st.rerun()
