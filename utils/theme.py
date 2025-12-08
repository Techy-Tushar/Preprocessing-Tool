# ------------------------------------------------------------
# theme.py  —  Global CSS for Smart Data Assistant
# ------------------------------------------------------------
import streamlit as st

def inject_theme():
    """Inject all global CSS styles (loaded only once)."""

    css = """
    <style>

    /* -------------------------------------------------------
       GLOBAL PAGE RESET
    ------------------------------------------------------- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* -------------------------------------------------------
       HERO BANNER (Home Page)
    ------------------------------------------------------- */
    .hero {
        background: linear-gradient(90deg, #a855f7, #6366f1, #3b82f6);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 30px;
        box-shadow: 0 0 25px rgba(168, 85, 247, 0.35);
        animation: fadeIn 0.7s ease-in-out;
    }

    .page-header {
        font-size: 42px;
        font-weight: 700;
        color: white;
        letter-spacing: 1px;
    }

    .page-sub {
        font-size: 19px;
        margin-top: 10px;
        color: #f3f3f3;
    }

    /* Premium title box styling */
    .page-title-box {
        margin-top: 24px !important;
        font-size: 32px !important;
        font-weight: 800 !important;
        padding: 22px 28px !important;
        border-radius: 16px !important;
        background: linear-gradient(90deg, #8b5cf6, #3b82f6);
        color: white !important;
        margin-bottom: 20px !important;
        letter-spacing: 0.5px;
    }

    /* DISTINCT PAGE TITLE */
    .page-title-box {
        font-size: 26px !important;
        font-weight: 700 !important;
        padding: 16px 22px !important;
        border-radius: 12px !important;
        background: rgba(99, 102, 241, 0.20);
        border: 1px solid rgba(99, 102, 241, 0.35);
        backdrop-filter: blur(6px);
        color: #e5e5e5 !important;
        margin-top: 6px !important;
        margin-bottom: 20px !important;
        letter-spacing: 0.4px;
        box-shadow: 0 0 16px rgba(99, 102, 241, 0.20);
    }

    /* -------------------------------------------------------
       BUTTON STYLES
    ------------------------------------------------------- */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1, #3b82f6);
        color: white;
        border: none;
        padding: 0.6rem 1.3rem;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 600;
        transition: 0.2s ease-in-out;
        box-shadow: 0px 4px 14px rgba(99, 102, 241, 0.3);
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        transform: translateY(-2px);
        box-shadow: 0px 6px 18px rgba(37, 99, 235, 0.35);
    }

    /* -------------------------------------------------------
       SIDEBAR STYLES
    ------------------------------------------------------- */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }

    section[data-testid="stSidebar"] button {
        margin-top: 4px !important;
        margin-bottom: 4px !important;
        padding-top: 6px !important;
        padding-bottom: 6px !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        margin-top: 6px !important;
        margin-bottom: 6px !important;
    }

    .sidebar-nav {
        display: flex;
        flex-direction: column;
        gap: 6px;
        padding-top: 6px;
        padding-bottom: 6px;
    }

    .sidebar-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 10px;
        border-radius: 10px;
        cursor: pointer;
        color: #e2e8f0;
        transition: transform 220ms ease, box-shadow 220ms ease, background 220ms ease;
        background: rgba(99,102,241,0.06);
        border: 1px solid rgba(99,102,241,0.06);
        font-weight: 600;
    }

    .sidebar-item:hover {
        transform: translateX(4px);
        background: rgba(99,102,241,0.12);
        box-shadow: 0 6px 18px rgba(37,99,235,0.06);
    }

    .sidebar-item.active {
        transform: translateX(6px);
        background: linear-gradient(90deg, rgba(124,58,237,0.18), rgba(59,130,246,0.12));
        box-shadow: 0 8px 26px rgba(99,102,241,0.18);
        border: 1px solid rgba(124,58,237,0.28);
        color: white;
    }

    /* Icon inside sidebar */
    .sidebar-item .icon {
        width: 26px;
        text-align: center;
    }

    /* -------------------------------------------------------
       INFO BOXES
    ------------------------------------------------------- */
    .stAlert {
        border-radius: 12px !important;
        padding: 12px 15px !important;
        font-size: 15px !important;
    }

    /* -------------------------------------------------------
       ANIMATIONS
    ------------------------------------------------------- */
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)
