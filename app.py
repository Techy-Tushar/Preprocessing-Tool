# =====================================================
# Smart Data Preprocessing Assistant
# Original Author: Tushar Rathod
# Year: 2025
# GitHub: https://github.com/Techy-Tushar
# License: MIT
# =====================================================


# app.py (fixed - preserves ALL original logic)
import streamlit as st

# ---------------- SAFE GLOBAL PAGE CONFIG ----------------
try:
    st.set_page_config(
        page_title="Preprocessing Tool",
        page_icon="🛠️",
        layout="wide",
    )
except Exception:
    pass

# ---------------- MASTER SESSION KEYS (initialize early) ----------------
st.session_state.setdefault("original_df", None)
st.session_state.setdefault("clean_df", None)
st.session_state.setdefault("page_completed", {})
st.session_state.setdefault("history", [])
st.session_state.setdefault("summaries", {})
st.session_state.setdefault("show_raw_preview", False)
st.session_state.setdefault("current_page", "Home")

if st.session_state.original_df is not None and st.session_state.clean_df is None:
    st.session_state.clean_df = st.session_state.original_df.copy()

# ---------------- THEME ----------------
try:
    from utils.theme import inject_theme
    inject_theme()
except Exception:
    pass

# ---------------- IMPORT PAGES ----------------
from app_pages.home import run_home
from app_pages.p1_Data_Explorer import run_data_explorer
from app_pages.p2_Fix_Missing_Values import run_fix_missing_values
from app_pages.p2b_Fix_Semantic_Cleanup import run_semantic_cleanup
from app_pages.p3_Outlier_Handling import run_outlier_handling
from app_pages.p4_EDA_Core import run_eda_core
from app_pages.p4b_EDA_Exports import run_eda_exports
from app_pages.p5_Encoding_and_Transformation import run_encoding_transformation

# ⭐ NEW — PAGE 6 IMPORT
from app_pages.p6_Download_Center import run_download_center


# ---------------- PAGE ORDER ----------------
PAGE_ORDER = [
    ("Home", "🏠 Home"),
    ("Data Explorer", "📂 Data Explorer"),
    ("Fix Missing Values", "🧹 Fix Missing Values"),
    ("Semantic Cleanup", "🧠 Semantic Cleanup"),
    ("Outlier Handling", "📉 Outlier Handling"),
    ("EDA Core", "📊 EDA — Core"),
    ("EDA Exports", "📥 Export EDA"),
    ("Encoding & Transformation", "🔢 Encoding & Transformation"),

    # ⭐ NEW — PAGE 6 ENTRY
    ("Download Center", "📥 Download Center"),
]


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 📑 Navigation")

    for key, label in PAGE_ORDER:
        if st.session_state.current_page == key:
            st.markdown(f"### 👉 **{label}**")
        else:
            if st.button(label, key=f"nav_{key}"):
                st.session_state.current_page = key
                st.rerun()

    st.markdown("---")

    if st.session_state.get("original_df") is not None:
        if st.button("👁 Show Raw Data", key="raw_preview"):
            st.session_state.show_raw_preview = True
            st.rerun()

    if st.session_state.get("show_raw_preview", False):
        st.markdown("### Raw Data (first 5 rows)")
        if st.session_state.original_df is not None:
            st.dataframe(st.session_state.original_df.head(5))
        else:
            st.info("No raw data available.")
        if st.button("Hide Raw Data", key="hide_raw"):
            st.session_state.show_raw_preview = False
            st.rerun()


# ---------------- ROUTER ----------------
def route():
    page = st.session_state.current_page

    if page == "Home":
        run_home()
    elif page == "Data Explorer":
        run_data_explorer()
    elif page == "Fix Missing Values":
        run_fix_missing_values()
    elif page == "Semantic Cleanup":
        run_semantic_cleanup()
    elif page == "Outlier Handling":
        run_outlier_handling()
    elif page == "EDA Core":
        run_eda_core()
    elif page == "EDA Exports":
        run_eda_exports()
    elif page == "Encoding & Transformation":
        run_encoding_transformation()

    # ⭐ NEW — PAGE 6 ROUTE
    elif page == "Download Center":
        run_download_center()

    else:
        st.error(f"Unknown page: {page}")


route()
