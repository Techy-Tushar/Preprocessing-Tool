import streamlit as st
import pandas as pd

def run_home():

    # -------------------------------------------------
    # IMPORTANT: MUST NOT set page_config here.
    # It is already set globally in app.py
    # -------------------------------------------------

    # --- PAGE TITLE ---
    st.markdown("""
        <style>
            @keyframes fadeSlideDown {
                0% {opacity: 0; transform: translateY(-12px);}
                100% {opacity: 1; transform: translateY(0);}
            }

            .page-title-box {
                margin-top: 80px;
                padding: 28px 32px;
                border-radius: 16px;
                position: relative;
                background: rgba(255,255,255,0.05);
                backdrop-filter: blur(10px);
                border: 2px solid transparent;
                animation: fadeSlideDown 0.6s ease-out;
            }

            .page-title-box:before {
                content: "";
                position: absolute;
                inset: 0;
                padding: 2px;
                border-radius: 16px;
                background: linear-gradient(135deg, #6a5af9, #ff5f9e, #46c3ff);
                -webkit-mask: linear-gradient(#fff 0 0) content-box,
                               linear-gradient(#fff 0 0);
                -webkit-mask-composite: xor;
                mask-composite: exclude;
                pointer-events: none;
            }

            .page-title-box:after {
                content: "";
                position: absolute;
                inset: -12px;
                border-radius: 20px;
                background: linear-gradient(135deg,
                    rgba(106,90,249,0.5),
                    rgba(255,95,158,0.45),
                    rgba(70,195,255,0.45)
                );
                filter: blur(35px);
                z-index: -1;
            }

            .page-title-text {
                color: #ffffff;
                font-weight: 900;
                font-size: 34px;
                text-shadow: 0 0 10px rgba(255,255,255,0.3);
            }

            .page-subtitle-text {
                color: rgba(255,255,255,0.78);
                font-size: 16px;
                margin-top: 10px;
            }
        </style>

        <div class="page-title-box">
            <div style="display:flex;align-items:center;gap:14px;">
                <span class="page-title-text">🛠️ Preprocessing Tool 🛠️</span>
            </div>
            <div class="page-subtitle-text">
                Upload, analyze, clean, transform and prepare your dataset for modeling.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # -------------------------------
    # FIXED UPLOADER LOGIC (NO FLICKER)
    # -------------------------------
    st.markdown("### Upload CSV / Excel")
    uploaded = st.file_uploader(
        "Upload dataset (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key="uploader_home"
    )

    # 1️⃣ Store uploaded file in session_state so it persists on rerun
    if uploaded is not None:
        st.session_state.uploaded_file_obj = uploaded

    # 2️⃣ Read file only ONCE (when clean_df not created yet)
    if (
        st.session_state.get("uploaded_file_obj") is not None
        and st.session_state.get("clean_df") is None
    ):
        file = st.session_state.uploaded_file_obj

        try:
            if file.name.lower().endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            # Initialize pipeline
            st.session_state.original_df = df.copy()
            st.session_state.clean_df = df.copy()

            st.success("Dataset uploaded successfully! 🎉")

        except Exception as e:
            st.error(f"❌ Failed to read file: {e}")

    # -------------------------------
    # PREVIEW DATA
    # -------------------------------
    if st.session_state.get("clean_df") is not None:
        st.markdown("### 👀 Preview (first 5 rows)")
        st.dataframe(st.session_state.clean_df.head(5), use_container_width=True)
    else:
        st.info("📂 Please upload a dataset above to begin preprocessing.")

    # ---------------------------------
    # QUICK ACTIONS (same as your code)
    # ---------------------------------
    if st.session_state.get("clean_df") is not None:
        st.markdown("---")
        st.markdown("### Quick Actions")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Go to Data Explorer"):
                st.session_state.current_page = "Data Explorer"
                st.rerun()

        with col2:
            if st.button("Reset Pipeline (clear data)"):
                st.session_state.original_df = None
                st.session_state.clean_df = None
                st.session_state.page_completed = {}
                st.session_state.summaries = {}
                st.session_state.uploaded_file_obj = None
                st.success("Pipeline reset.")
                st.rerun()
