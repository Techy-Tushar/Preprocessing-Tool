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

# ---------------- GLOBAL TITLE POSITION & SPACING (ALL PAGES) ----------------
st.markdown("""
<style>

/* Keep header alive but invisible */
header {
    visibility: hidden;
}

/* Maintain space so sidebar toggle works */
header [data-testid="stToolbar"] {
    visibility: visible;
}

/* Global top padding so titles are not clipped */
.block-container {
    padding-top: 3.2rem !important;
}

/* Unified page title container */
.page-title-box {
    margin-top: 1.8rem !important;
    margin-bottom: 1.4rem !important;
    padding: 18px 22px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(140,80,255,0.18), rgba(60,140,255,0.15));
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR FINAL BALANCE FIX ----------------
st.markdown("""
<style>

/* Sidebar expanded */
section[data-testid="stSidebar"][aria-expanded="true"] {
    width: 210px !important;
}

/* Sidebar collapsed */
section[data-testid="stSidebar"][aria-expanded="false"] {
    width: 0px !important;
}

/* Remove invisible gap */
section[data-testid="stSidebar"][aria-expanded="false"] > div {
    display: none;
}

/* Smooth animation */
section[data-testid="stSidebar"] {
    transition: width 0.25s ease;
}

/* Reduce top & bottom padding of sidebar */
section[data-testid="stSidebar"] > div {
    padding-top: 0.4rem !important;
    padding-bottom: 0.4rem !important;
}

/* Compact buttons (HEIGHT FIX) */
section[data-testid="stSidebar"] button {
    padding: 0.28rem 0.55rem !important;
    font-size: 13px !important;
    line-height: 1.2 !important;
    border-radius: 9px !important;
}

/* Reduce space between nav buttons */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
    gap: 2px !important;
}

/* Compact section headers */
section[data-testid="stSidebar"] h3 {
    margin-top: 4px !important;
    margin-bottom: 4px !important;
    font-size: 15px !important;
}

/* Compact divider */
section[data-testid="stSidebar"] hr {
    margin: 6px 0 !important;
}

/* Disable sidebar scrolling */
section[data-testid="stSidebar"] {
    overflow: hidden !important;
}

/* Smooth transitions */
section[data-testid="stSidebar"] button {
    transition: box-shadow 0.25s ease, background 0.25s ease;
}

/* Active page — MINI TITLE CARD STYLE */
section[data-testid="stSidebar"] button[kind="secondary"] {
    background: linear-gradient(
        135deg,
        rgba(120,90,255,0.55),
        rgba(200,90,150,0.55),
        rgba(90,160,220,0.55)
    ) !important;

    border-radius: 10px;

    /* Soft depth like title card */
    box-shadow:
        0 10px 28px rgba(0, 0, 0, 0.35),
        inset 0 0 0 1px rgba(255,255,255,0.28);

    color: #ffffff !important;
    font-weight: 600;

    position: relative;
    overflow: hidden;
}


/* Gradient border glow — same language as title */
section[data-testid="stSidebar"] button[kind="secondary"]::after {
    content: "";
    position: absolute;
    inset: -1px;
    border-radius: 11px;
    background: linear-gradient(
        135deg,
        rgba(140,80,255,0.9),
        rgba(220,90,150,0.9),
        rgba(90,170,255,0.9)
    );
    z-index: -1;
    filter: blur(6px);
    opacity: 0.7;
}

/* Hover effect (inactive only) */
section[data-testid="stSidebar"] button:hover {
    box-shadow: 0 0 20px rgba(120,120,255,0.35);
}

/* Hide sidebar resize handle */
div[data-testid="stSidebarResizer"] {
    display: none;
}

</style>
""", unsafe_allow_html=True)



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

# ---------------- RESET HELPER ----------------
from utils.state_helpers import reset_pipeline

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
    # --- RESET PIPELINE (always visible)
    st.subheader("⚙ App Controls")

    if st.button("🔄 Reset Pipeline", key="reset_pipeline_btn"):
        st.session_state["confirm_reset"] = True

    if st.session_state.get("confirm_reset", False):
        st.warning("This will clear all progress and uploaded data.")
        col1, col2 = st.columns(2)

        if col1.button("✅ Yes, reset", key="confirm_reset_yes"):
            if st.session_state.get("original_df") is None:
                st.warning("📂 Please upload a dataset first.")
                st.session_state["confirm_reset"] = False
            else:
                reset_pipeline()
                st.rerun()

        if col2.button("❌ Cancel", key="confirm_reset_no"):
            st.session_state["confirm_reset"] = False

    st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)

    st.markdown(
        "<h3 style='margin-top: 8px; margin-bottom: 10px;'>📄 Navigation</h3>",
        unsafe_allow_html=True
    )

    for key, label in PAGE_ORDER:
        is_active = st.session_state.current_page == key

        if st.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
                type="secondary" if is_active else "primary"
        ):
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
