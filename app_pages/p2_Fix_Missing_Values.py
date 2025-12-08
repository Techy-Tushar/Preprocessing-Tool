# ---------------------------------------------------------------
# PAGE 2 — FIX MISSING VALUES (FULL REVIEW + FIXED)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# If you have theme injector keep it
try:
    from utils.theme import inject_theme
except Exception:
    def inject_theme():
        return


# ---------------------------------------------------------------
# UNDO HELPER (restored and hardened)
# ---------------------------------------------------------------
def perform_undo():
    if "last_cleaned_df" in st.session_state and st.session_state.last_cleaned_df is not None:
        st.session_state.cleaned_df = st.session_state.last_cleaned_df.copy()
        st.session_state.clean_df = st.session_state.cleaned_df.copy()
        st.session_state.df = st.session_state.cleaned_df.copy()

        st.session_state.cleaned = True
        st.session_state.auto_state = None
        st.session_state.auto_cols_to_drop = []

        for k in ["outlier_working_df", "outlier_prev_df", "outlier_report"]:
            if k in st.session_state:
                del st.session_state[k]

        st.session_state.just_undone = True
        st.success("↩️ Previous version restored successfully!")
        st.rerun()
    else:
        st.info("No previous snapshot found to undo.")


# ---------------------------------------------------------------
# MAIN PAGE FUNCTION
# ---------------------------------------------------------------
def run_fix_missing_values():

    inject_theme()

    st.markdown("""
    <div class="page-title-box">
        <div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:28px;font-weight:800;">🧹 Fix Missing Values</span>
        </div>
        <div style="margin-top:6px;font-size:14px;opacity:0.85;">
            Handle missing values using automatic and manual methods.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    if st.session_state.get("just_undone", False):
        st.success("↩️ Previous version restored successfully!")
        del st.session_state["just_undone"]

    # Dataset priority: clean_df → original_df → raw_df
    if st.session_state.get("clean_df") is not None:
        df = st.session_state["clean_df"].copy()
    elif st.session_state.get("original_df") is not None:
        df = st.session_state["original_df"].copy()
    else:
        if "raw_df" not in st.session_state or st.session_state.raw_df is None:
            st.warning("⚠️ Please upload a dataset from the Home page first.")
            st.stop()
        df = st.session_state.raw_df.copy()

    st.session_state.setdefault("auto_state", None)
    st.session_state.setdefault("auto_cols_to_drop", [])
    st.session_state.setdefault("cleaned", st.session_state.get("cleaned", False))
    st.session_state.setdefault("summaries", {})

    missing_cols = df.columns[df.isnull().any()].tolist()
    missing_perc = (df.isnull().mean() * 100).round(2)

    if len(missing_cols) == 0:
        st.success("🎉 No missing values found in the current dataset!")
        st.session_state.setdefault("page_completed", {})
        st.session_state["page_completed"]["Fix Missing Values"] = True
        st.session_state.clean_df = df.copy()
        st.info("💾 Your dataset is already clean. Use sidebar for the next steps.")
        st.stop()

    st.subheader("📌 Columns With Missing Values")
    missing_summary = pd.DataFrame({
        "Column": missing_cols,
        "Missing Count": df[missing_cols].isnull().sum().values,
        "Missing %": missing_perc[missing_cols].values
    })

    st.dataframe(missing_summary, use_container_width=True)
    st.divider()

    st.subheader("🟦 Step 1: Select Columns to Clean")

    selection_mode = st.radio(
        "",
        [
            "All columns with missing values",
            "Only numerical columns",
            "Only categorical columns",
            "Select specific columns"
        ]
    )

    if selection_mode == "All columns with missing values":
        target_cols = missing_cols
    elif selection_mode == "Only numerical columns":
        target_cols = [c for c in missing_cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(target_cols) == 0:
            st.info("✔ No numeric columns with missing values.")
            st.stop()
    elif selection_mode == "Only categorical columns":
        target_cols = [c for c in missing_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if len(target_cols) == 0:
            st.info("✔ No categorical columns with missing values.")
            st.stop()
    else:
        target_cols = st.multiselect("Select columns:", options=missing_cols, default=missing_cols)

    if not target_cols:
        st.warning("Select at least one column.")
        st.stop()

    st.info(f"✔ Selected Columns: {target_cols}")
    st.divider()

    st.subheader("🟩 Step 2: Choose Cleaning Method")

    method = st.radio("", ["Automatic (Recommended)", "Manual (Smart)"])

    # -----------------------------------------------------------
    # AUTOMATIC CLEANING
    # -----------------------------------------------------------
    if method == "Automatic (Recommended)":

        st.markdown("""
        <div style='background-color:#11161e;padding:15px;border-radius:10px;
                    border:1px solid #333;margin-bottom:10px;color:#fff;'>
            <b>Automatic Cleaning</b><br>
            • Numeric → Median<br>
            • Categorical → Mode<br>
            • Suggest drop if >40% missing
        </div>
        """, unsafe_allow_html=True)

        if st.button("✨ Apply Automatic Cleaning", key="missing_auto_clean"):
            threshold = 40
            st.session_state.auto_cols_to_drop = [
                c for c in target_cols if missing_perc[c] > threshold
            ]
            st.session_state.auto_state = "confirm" if st.session_state.auto_cols_to_drop else "yes"

        if st.session_state.auto_state == "confirm":
            st.warning("⚠️ Columns with >40% missing:")
            st.write(st.session_state.auto_cols_to_drop)

            c1, c2 = st.columns(2)
            with c1:
                yes = st.button("✅ Yes, Drop", key="missing_yes_drop")
            with c2:
                no = st.button("❌ No, Keep", key="missing_no_drop")

            if yes:
                st.session_state.auto_state = "yes"
            if no:
                st.session_state.auto_state = "no"

        # AUTO — YES PATH
        if st.session_state.auto_state == "yes":

            st.session_state.last_cleaned_df = df.copy()
            df_temp = df.copy()
            drop_cols = st.session_state.auto_cols_to_drop or []

            if drop_cols:
                drop_cols = [c for c in drop_cols if c in df_temp.columns]
                df_temp = df_temp.drop(columns=drop_cols)

            filled = {}
            for col in target_cols:
                if col in drop_cols or col not in df_temp.columns:
                    continue

                if pd.api.types.is_numeric_dtype(df_temp[col]):
                    med = df_temp[col].median()
                    df_temp[col] = df_temp[col].fillna(med)
                    filled[col] = {"method": "Median", "value_used": float(med) if pd.notna(med) else None}
                else:
                    mode_val = df_temp[col].mode()[0] if not df_temp[col].mode().empty else ""
                    df_temp[col] = df_temp[col].fillna(mode_val)
                    filled[col] = {"method": "Mode", "value_used": mode_val}

            st.session_state.cleaned_df = df_temp.copy()
            st.session_state.df = df_temp.copy()         # FIX
            st.session_state.clean_df = df_temp.copy()    # FIX
            st.session_state.cleaned = True
            st.session_state["missing_values_handled"] = True
            st.success("🎉 Cleaning Completed!")
            st.subheader("📊 Cleaning Summary")
            st.json({
                "Dropped Columns": drop_cols,
                "Filled Columns": filled,
                "Initial Shape": list(df.shape),
                "Final Shape": list(df_temp.shape),
                "time": datetime.now().isoformat()
            })

            before_df = df[target_cols].head(20)
            after_df = df_temp[target_cols].head(20)

            colA, colB = st.columns(2)
            with colA:
                st.markdown("#### 🟧 Before Cleaning")
                st.dataframe(before_df)
            with colB:
                st.markdown("#### 🟩 After Cleaning")
                st.dataframe(after_df)

            if st.button("↩️ Undo Last Operation", key="missing_undo_1"):
                perform_undo()

            st.markdown("---")
            if st.button("Proceed to Semantic Cleanup →", key="missing_to_semantic_1"):
                if not st.session_state.cleaned:
                    st.warning("⚠️ Handle missing values first.")
                    st.stop()

                st.session_state.clean_df = st.session_state.cleaned_df.copy()
                st.session_state.df = st.session_state.cleaned_df.copy()
                st.session_state.current_page = "Semantic Cleanup"
                st.rerun()

            st.stop()

        # AUTO — NO PATH
        if st.session_state.auto_state == "no":

            st.session_state.last_cleaned_df = df.copy()
            df_temp = df.copy()

            filled = {}
            for col in target_cols:
                if col not in df_temp.columns:
                    continue

                if pd.api.types.is_numeric_dtype(df_temp[col]):
                    med = df_temp[col].median()
                    df_temp[col] = df_temp[col].fillna(med)
                    filled[col] = {"method": "Median", "value_used": float(med)}
                else:
                    mode_val = df_temp[col].mode()[0] if not df_temp[col].mode().empty else ""
                    df_temp[col] = df_temp[col].fillna(mode_val)
                    filled[col] = {"method": "Mode", "value_used": mode_val}

            st.session_state.cleaned_df = df_temp.copy()
            st.session_state.df = df_temp.copy()          # FIX
            st.session_state.clean_df = df_temp.copy()    # FIX
            st.session_state.cleaned = True
            st.session_state["missing_values_handled"] = True

            st.success("🎉 Cleaning Completed!")
            st.subheader("📊 Cleaning Summary")
            st.json({
                "Dropped Columns": [],
                "Filled Columns": filled,
                "Initial Shape": list(df.shape),
                "Final Shape": list(df_temp.shape),
                "time": datetime.now().isoformat()
            })

            before_df = df[target_cols].head(20)
            after_df = df_temp[target_cols].head(20)

            colA, colB = st.columns(2)
            with colA:
                st.markdown("#### 🟧 Before Cleaning")
                st.dataframe(before_df)
            with colB:
                st.markdown("#### 🟩 After Cleaning")
                st.dataframe(after_df)

            if st.button("↩️ Undo Last Operation", key="missing_undo_2"):
                perform_undo()

            st.markdown("---")
            if st.button("Proceed to Semantic Cleanup →", key="missing_to_semantic_2"):
                if not st.session_state.cleaned:
                    st.warning("⚠️ Handle missing values first.")
                    st.stop()

                st.session_state.clean_df = st.session_state.cleaned_df.copy()
                st.session_state.df = st.session_state.cleaned_df.copy()
                st.session_state.current_page = "Semantic Cleanup"
                st.rerun()

            st.stop()

    # -----------------------------------------------------------
    # MANUAL CLEANING MODE
    # -----------------------------------------------------------
    if method == "Manual (Smart)":

        st.subheader("🧰 Step 3: Manual Cleaning")

        num_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in target_cols if not pd.api.types.is_numeric_dtype(df[c])]

        manual_actions = {}
        if num_cols:
            st.markdown("### 🔢 Numerical Columns")
            with st.expander("Numerical Settings", expanded=True):

                for col in num_cols:
                    st.markdown(f"#### {col} — Missing: {int(df[col].isnull().sum())}")

                    manual_actions[col] = st.radio(
                        f"Choose an action for {col}:",
                        [
                            "Fill with Mean",
                            "Fill with Median",
                            "Fill with Mode",
                            "Fill with Zero",
                            "Fill with Custom Value",
                            "Drop Rows with Missing"
                        ],
                        horizontal=True,
                        key=f"num_action_{col}"
                    )

                    if manual_actions[col] == "Fill with Custom Value":
                        st.text_input(
                            f"Enter custom value for {col}:",
                            key=f"custom_val_{col}",
                            placeholder="Example: 0, 1.5, 100"
                        )

                    st.markdown("---")

        if cat_cols:
            st.markdown("### 🔤 Categorical Columns")
            with st.expander("Categorical Settings"):
                for col in cat_cols:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else ""
                    manual_actions[col] = st.radio(
                        f"{col} — Missing: {int(df[col].isnull().sum())} | Mode: {mode_val}",
                        [f"Fill with Mode ({mode_val})"],
                        horizontal=True,
                        key=f"cat_{col}"
                    )
                    st.markdown("---")

        st.divider()

        if st.button("🚀 Apply Manual Cleaning"):

            st.session_state.last_cleaned_df = df.copy()
            df_clean = df.copy()
            filled = {}

            for col, action in manual_actions.items():

                if col in num_cols:

                    if action == "Fill with Mean":
                        val = df_clean[col].mean()
                        df_clean[col] = df_clean[col].fillna(val)
                        filled[col] = {"method": "Mean", "value_used": float(val)}

                    elif action == "Fill with Median":
                        val = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(val)
                        filled[col] = {"method": "Median", "value_used": float(val)}

                    elif action == "Fill with Mode":
                        val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else np.nan
                        df_clean[col] = df_clean[col].fillna(val)
                        filled[col] = {"method": "Mode", "value_used": val}

                    elif action == "Fill with Zero":
                        df_clean[col] = df_clean[col].fillna(0)
                        filled[col] = {"method": "Zero", "value_used": 0}

                    elif action == "Fill with Custom Value":
                        custom_val = st.session_state.get(f"custom_val_{col}")
                        try:
                            numval = float(custom_val)
                            df_clean[col] = df_clean[col].fillna(numval)
                            filled[col] = {"method": "Custom", "value_used": numval}
                        except:
                            df_clean[col] = df_clean[col].fillna(custom_val)
                            filled[col] = {"method": "Custom", "value_used": custom_val}

                    elif action == "Drop Rows with Missing":
                        df_clean = df_clean[df_clean[col].notna()].reset_index(drop=True)
                        filled[col] = {"method": "Drop Rows", "value_used": None}

                else:
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else ""
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    filled[col] = {"method": "Mode", "value_used": mode_val}

            st.session_state.cleaned_df = df_clean.copy()
            st.session_state.clean_df = df_clean.copy()   # FIX
            st.session_state.df = df_clean.copy()         # FIX
            st.session_state.cleaned = True
            st.session_state["missing_values_handled"] = True
            st.success("🎉 Manual Cleaning Completed!")
            st.subheader("📊 Cleaning Summary")
            st.json(filled)

            before_df = df[target_cols].head(20)
            after_df = df_clean[target_cols].head(20)

            colA, colB = st.columns(2)
            with colA:
                st.markdown("#### 🟧 Before Cleaning")
                st.dataframe(before_df)
            with colB:
                st.markdown("#### 🟩 After Cleaning")
                st.dataframe(after_df)

            if st.button("↩️ Undo Last Operation", key="missing_undo_3"):
                perform_undo()

            st.markdown("---")
            if st.button("Proceed to Semantic Cleanup →", key="missing_to_semantic_3"):
                if not st.session_state.cleaned:
                    st.warning("⚠️ Handle missing values first.")
                    st.stop()

                st.session_state.clean_df = st.session_state.df.copy()   # OPTION A
                st.session_state.current_page = "Semantic Cleanup"
                st.rerun()

            st.stop()

    # -----------------------------------------------------------
    # BOTTOM NAVIGATION (OPTION A FIX)
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("Next Step: Semantic Cleanup")

    st.info("""
    Semantic Cleanup fixes:
    - Unit symbols  
    - Date formats  
    - Phone / ID formats  
    - Mixed-type numeric columns  
    - Salary/experience normalization  
    - Pattern inconsistencies  
    """)

    if st.button("Proceed to Semantic Cleanup →", key="missing_to_semantic_4"):
        if not st.session_state.get("cleaned", False):
            st.warning("⚠️ Handle missing values first.")
            st.stop()

        st.session_state.clean_df = df.copy()   # OPTION A
        st.session_state.df = df.copy()         # OPTION A
        st.session_state.current_page = "Semantic Cleanup"
        st.rerun()
