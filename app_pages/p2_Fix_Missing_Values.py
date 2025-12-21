# ===========================================================
# PAGE 2 — FIX MISSING VALUES
# FINAL • STABLE • TRANSPARENT • MESSY-DATA SAFE
# ===========================================================

import streamlit as st
import pandas as pd
import numpy as np


# ===========================================================
# UNDO — SINGLE LEVEL (BACK TO STEP 3)
# ===========================================================

def perform_undo():
    if "step3_df" in st.session_state:
        restored = st.session_state.step3_df.copy()

        st.session_state.df = restored.copy()
        st.session_state.clean_df = restored.copy()

        st.session_state.missing_actions_log = []
        st.session_state.missing_applied = False

        # 🔧 Reset Fix Missing Values summary (Page-6 consistency)
        st.session_state.fix_missing_summary = {
            "numerical": [],
            "categorical": [],
            "mixed": []
        }

        st.rerun()
    else:
        st.info("Nothing to undo yet.")


# ===========================================================
# PREVIEW RENDER HELPER
# ===========================================================

def render_before_after_preview(before_df, after_df, cols):
    """
    Renders before vs after preview.
    ≤4 cols  -> side-by-side
    >4 cols  -> line-by-line
    """
    if not cols:
        return

    if len(cols) <= 4:
        b1, b2 = st.columns(2)

        with b1:
            st.markdown("**Before**")
            st.dataframe(
                before_df[cols],
                width="stretch",
                height=350
            )

        with b2:
            st.markdown("**After**")
            st.dataframe(
                after_df[cols],
                width="stretch",
                height=350
            )

    else:
        st.markdown("### Before")
        st.dataframe(
            before_df[cols],
            width="stretch",
            height=350
        )

        st.markdown("### After")
        st.dataframe(
            after_df[cols],
            width="stretch",
            height=350
        )



# ===========================================================
# MAIN PAGE
# ===========================================================

def run_fix_missing_values():

    # -------------------------------------------------------
    # SESSION STATE INIT (DEFENSIVE)
    # -------------------------------------------------------
    st.session_state.setdefault("missing_actions_log", [])
    st.session_state.setdefault("missing_applied", False)

    # -------------------------------------------------------
    # PAGE-6 SUMMARY STRUCTURE (FIX MISSING VALUES)
    # -------------------------------------------------------
    st.session_state.setdefault("fix_missing_summary", {
        "numerical": [],
        "categorical": [],
        "mixed": []
    })

    # -------------------------------------------------------
    # HEADER
    # -------------------------------------------------------
    st.markdown("""
        <div class="page-title-box">
            <span style="font-size:28px;font-weight:800;">🧹 Fix Missing Values</span>
            <div style="margin-top:6px;font-size:14px;opacity:0.85;">
                Handle missing values safely with preview, undo, and transparency.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
        # -------------------------------------------------------
    # DATASET RESOLUTION (SINGLE SOURCE OF TRUTH)
    # -------------------------------------------------------
    if st.session_state.get("missing_applied"):
        df = st.session_state.clean_df.copy()

    elif "step3_df" in st.session_state:
        df = st.session_state.step3_df.copy()

    elif st.session_state.get("df") is not None:
        df = st.session_state.df.copy()

    elif st.session_state.get("original_df") is not None:
        df = st.session_state.original_df.copy()

    else:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    # -------------------------------------------------------
    # STEP 1 — MISSING VALUES OVERVIEW
    # -------------------------------------------------------
    st.subheader("📊 Step 1: Missing Values Overview")

    missing_summary = []
    total_rows = len(df)

    for col in df.columns:
        miss_count = int(df[col].isna().sum())
        if miss_count > 0:
            missing_summary.append([
                col,
                miss_count,
                round((miss_count / total_rows) * 100, 2)
            ])

    active_missing_cols = [row[0] for row in missing_summary]

    if missing_summary:
        miss_df = pd.DataFrame(
            missing_summary,
            columns=["Column", "Missing Count", "Missing %"]
        )
        st.dataframe(miss_df, width="stretch")
    else:
        st.success("✅ No missing values found in dataset.")

    # -------------------------------------------------------
    # STEP 2 — COLUMN TYPE DIAGNOSIS
    # -------------------------------------------------------
    st.subheader("🧠 Step 2: Column Type Diagnosis")

    numeric_cols, cat_cols, mix_cols = [], [], []
    type_rows = []

    for col in active_missing_cols:
        raw = df[col]
        as_str = raw.astype(str)

        numeric_series = pd.to_numeric(raw, errors="coerce")
        numeric_ratio = numeric_series.notna().mean()
        has_digit_strings = as_str.str.contains(r"\d", regex=True).any()

        if numeric_ratio >= 0.7:
            numeric_cols.append(col)
            detected = "Numeric"
            reason = "100% numeric values"

        elif numeric_ratio <= 0.3 and not has_digit_strings:
            cat_cols.append(col)
            detected = "Categorical"
            reason = "100% categorical values"

        else:
            mix_cols.append(col)
            detected = "Mixed"
            reason = f"{round(numeric_ratio*100,1)}% numeric values with semantic patterns"

        type_rows.append([col, detected, reason])

    if type_rows:
        type_df = pd.DataFrame(
            type_rows,
            columns=["Column", "Detected Type", "Reason"]
        )
        st.dataframe(type_df, width="stretch")
    else:
        st.info("No columns require diagnosis.")

    # -------------------------------------------------------
    # STEP 3 — COLUMN GROUPS & STRATEGY
    # -------------------------------------------------------
    st.subheader("🗂 Step 3: Column Groups & Handling Strategy")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 🟦 Numeric Columns")
        st.write(numeric_cols if numeric_cols else "—")
        st.caption("Mean • Median • Mode • Zero • Custom • Drop rows")

    with c2:
        st.markdown("### 🟨 Categorical Columns")
        st.write(cat_cols if cat_cols else "—")
        st.caption("Mode • Custom • Drop rows")

    with c3:
        st.markdown("### 🟥 Mixed Columns")
        st.write(mix_cols if mix_cols else "—")
        st.caption("Temporary fill → Semantic Cleanup later")

    st.divider()

    if "step3_df" not in st.session_state and active_missing_cols:
        st.session_state.step3_df = df.copy()

    # =======================================================
    # STEP 4 — HANDLE MISSING VALUES
    # =======================================================

    # ======================
    # 4A — NUMERIC
    # ======================
    with st.expander("🟦 Handle Numeric Columns", expanded=False):

        avail = [c for c in numeric_cols if c in active_missing_cols]

        if not avail:
            st.info("No numeric columns pending.")
        else:
            sel = st.multiselect("Select numeric columns", avail, key="num_sel")
            method = st.selectbox(
                "Choose method",
                [
                    "Fill with Mean",
                    "Fill with Median",
                    "Fill with Mode",
                    "Fill with Zero",
                    "Fill with Custom Value",
                    "Drop Rows"
                ],
                key="num_method"
            )

            preview_df = df.copy()
            fill_values = {}

            for col in sel:
                if method == "Drop Rows":
                    preview_df = preview_df[preview_df[col].notna()]
                    continue

                s = pd.to_numeric(preview_df[col], errors="coerce")

                if method == "Fill with Mean":
                    val = s.mean()
                elif method == "Fill with Median":
                    val = s.median()
                elif method == "Fill with Mode":
                    val = preview_df[col].mode().iloc[0]
                elif method == "Fill with Zero":
                    val = 0
                else:
                    val = st.text_input(f"Custom value for {col}", key=f"num_custom_{col}")

                fill_values[col] = val
                preview_df[col] = preview_df[col].fillna(val)

            if fill_values:
                st.markdown("#### 🔍 Values Used for Filling")
                st.dataframe(
                    pd.DataFrame(
                        [{"Column": k, "Value Used": v} for k, v in fill_values.items()]
                    ),
                    width="stretch"
                )

            if sel:
                st.markdown("#### 🔍 Preview (Before vs After)")
                render_before_after_preview(df, preview_df, sel)

            if st.button("Apply Numeric Handling"):
                st.session_state.clean_df = preview_df.copy()
                st.session_state.df = preview_df.copy()
                st.session_state.missing_applied = True

                for col in sel:
                    st.session_state.missing_actions_log.append({
                        "Column": col,
                        "Type": "Numeric",
                        "Action": method,
                        "Value Used": fill_values.get(col)
                    })

                for col in sel:
                    entry = {
                        "column": col,
                        "method": method
                    }
                    if entry not in st.session_state.fix_missing_summary["numerical"]:
                        st.session_state.fix_missing_summary["numerical"].append(entry)

                st.success("Numeric columns handled.")
                st.rerun()

    # ======================
    # 4B — CATEGORICAL
    # ======================
    with st.expander("🟨 Handle Categorical Columns", expanded=False):

        avail = [c for c in cat_cols if c in active_missing_cols]

        if not avail:
            st.info("No categorical columns pending.")
        else:
            sel = st.multiselect("Select categorical columns", avail, key="cat_sel")
            method = st.selectbox(
                "Choose method",
                ["Fill with Mode", "Fill with Custom Value", "Drop Rows"],
                key="cat_method"
            )

            preview_df = df.copy()
            fill_values = {}

            for col in sel:
                if method == "Drop Rows":
                    preview_df = preview_df[preview_df[col].notna()]
                    continue

                if method == "Fill with Mode":
                    val = preview_df[col].mode().iloc[0]
                else:
                    val = st.text_input(f"Custom value for {col}", key=f"cat_custom_{col}")

                fill_values[col] = val
                preview_df[col] = preview_df[col].fillna(val)

            if fill_values:
                st.markdown("#### 🔍 Values Used for Filling")
                st.dataframe(
                    pd.DataFrame(
                        [{"Column": k, "Value Used": v} for k, v in fill_values.items()]
                    ),
                    width="stretch"
                )

            if sel:
                st.markdown("#### 🔍 Preview (Before vs After)")
                render_before_after_preview(df, preview_df, sel)

            if st.button("Apply Categorical Handling"):
                st.session_state.clean_df = preview_df.copy()
                st.session_state.df = preview_df.copy()
                st.session_state.missing_applied = True

                for col in sel:
                    st.session_state.missing_actions_log.append({
                        "Column": col,
                        "Type": "Categorical",
                        "Action": method,
                        "Value Used": fill_values.get(col)
                    })

                for col in sel:
                    entry = {
                        "column": col,
                        "method": method
                    }
                    if entry not in st.session_state.fix_missing_summary["categorical"]:
                        st.session_state.fix_missing_summary["categorical"].append(entry)

                st.success("Categorical columns handled.")
                st.rerun()

    # ======================
    # 4C — MIXED
    # ======================
    with st.expander("🟥 Handle Mixed Columns", expanded=False):

        avail = [c for c in mix_cols if c in active_missing_cols]

        if not avail:
            st.info("No mixed columns pending.")
        else:
            st.warning(
                "Mixed columns are filled temporarily. "
                "Semantic interpretation happens in the next step."
            )

            sel = st.multiselect("Select mixed columns", avail, key="mix_sel")
            method = st.selectbox(
                "Choose method",
                [
                    "Fill with Mean",
                    "Fill with Median",
                    "Fill with Mode",
                    "Fill with Custom Value",
                    "Drop Rows"
                ],
                key="mix_method"
            )

            preview_df = df.copy()
            fill_values = {}

            for col in sel:
                numeric_series = pd.to_numeric(preview_df[col], errors="coerce")

                if method == "Drop Rows":
                    preview_df = preview_df[preview_df[col].notna()]
                    continue

                if method in ["Fill with Mean", "Fill with Median"]:
                    if numeric_series.notna().sum() == 0:
                        st.warning(
                            f"Column '{col}' has non acceptable numeric values. "
                            "Mean/Median cannot be applied. "
                            "Try Mode, Custom Value, or Drop Rows."
                        )
                        continue

                    val = (
                        numeric_series.mean()
                        if method == "Fill with Mean"
                        else numeric_series.median()
                    )

                elif method == "Fill with Mode":
                    val = preview_df[col].mode().iloc[0]

                else:
                    val = st.text_input(f"Custom value for {col}", key=f"mix_custom_{col}")

                fill_values[col] = val
                preview_df[col] = preview_df[col].fillna(val)

            if fill_values:
                st.markdown("#### 🔍 Values Used for Filling")
                st.dataframe(
                    pd.DataFrame(
                        [{"Column": k, "Value Used": v} for k, v in fill_values.items()]
                    ),
                    width="stretch"
                )

            if sel:
                st.markdown("#### 🔍 Preview (Before vs After)")
                render_before_after_preview(df, preview_df, sel)

            if st.button("Apply Mixed Handling"):
                st.session_state.clean_df = preview_df.copy()
                st.session_state.df = preview_df.copy()
                st.session_state.missing_applied = True

                for col in sel:
                    st.session_state.missing_actions_log.append({
                        "Column": col,
                        "Type": "Mixed",
                        "Action": method,
                        "Value Used": fill_values.get(col)
                    })

                for col in sel:
                    entry = {
                        "column": col,
                        "method": method
                    }
                    if entry not in st.session_state.fix_missing_summary["mixed"]:
                        st.session_state.fix_missing_summary["mixed"].append(entry)

                st.success("Mixed columns handled.")
                st.rerun()

    # -------------------------------------------------------
    # STEP 5 — SUMMARY
    # -------------------------------------------------------
    if st.session_state.missing_actions_log:
        st.subheader("📋 Missing Value Handling Summary")
        st.dataframe(
            pd.DataFrame(st.session_state.missing_actions_log),
            width="stretch"
        )

    # -------------------------------------------------------
    # STEP 6 — OVERALL BEFORE vs AFTER
    # -------------------------------------------------------
    if st.session_state.missing_actions_log:
        cols = list({x["Column"] for x in st.session_state.missing_actions_log})
        b1, b2 = st.columns(2)

        with b1:
            st.markdown("**Before (Step-3 Snapshot)**")
            st.dataframe(st.session_state.step3_df[cols], width="stretch", height=300)

        with b2:
            st.markdown("**After (Current Dataset)**")
            st.dataframe(df[cols], width="stretch", height=300)

    # -------------------------------------------------------
    # STEP 7 — UNDO
    # -------------------------------------------------------
    st.divider()
    if st.button("↩ Undo Last Action"):
        perform_undo()

    # -------------------------------------------------------
    # STEP 9 — DOWNLOAD CLEANED DATASET
    # -------------------------------------------------------
    st.subheader("⬇ Download Cleaned Dataset")

    if not st.session_state.missing_applied:
        st.info(
            "ℹ️ Please handle missing values using the options above before downloading "
            "the cleaned dataset."
        )
    else:
        st.caption(
            "This dataset contains all missing values handled and is ready for the next step."
        )

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download cleaned dataset (CSV)",
            data=csv_data,
            file_name="cleaned_missing_values.csv",
            mime="text/csv",
        )

    # -------------------------------------------------------
    # STEP 8 — SEMANTIC CLEANUP INTRO
    # -------------------------------------------------------
    st.subheader("🧠 Next Step: Semantic Cleanup")
    st.markdown(
        "The next step will interpret ranges, units, and encoded values "
        "to convert mixed and categorical columns into clean, analysis-ready formats."
    )

    # -------------------------------------------------------
    # STEP 10 — NAVIGATION
    # -------------------------------------------------------
    if active_missing_cols:
        st.warning("Please handle all missing values before proceeding.")
    else:
        if st.button("Go to Semantic Cleanup"):
            st.session_state.current_page = "Semantic Cleanup"
            st.rerun()
