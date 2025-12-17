# ---------------------------------------------------------------
# PAGE 2 — FIX MISSING VALUES (FINAL, SPEC-COMPLIANT)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np

try:
    from utils.theme import inject_theme
except Exception:
    def inject_theme():
        return


# ---------------------------------------------------------------
# UNDO HELPER
# ---------------------------------------------------------------
def perform_undo():
    if "last_cleaned_df" in st.session_state:
        last = st.session_state.last_cleaned_df.copy()

        st.session_state.clean_df = last.copy()
        st.session_state.cleaned_df = last.copy()
        st.session_state.df = last.copy()

        st.session_state.page_stage = "editing"
        st.session_state.cleaned = False
        st.session_state.missing_summary = {}

        st.session_state.auto_state = None
        st.session_state.auto_cols_to_drop = []

        st.rerun()
    else:
        st.info("No previous state available to undo.")


# ---------------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------------
def run_fix_missing_values():
    st.markdown("""
    <div class="page-title-box">
        <span style="font-size:28px;font-weight:800;">🧹 Fix Missing Values</span>
        <div style="margin-top:6px;font-size:14px;opacity:0.85;">
            Handle missing values safely with preview, undo, and smart suggestions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ---------------- DATASET RESOLUTION ----------------
    if st.session_state.get("clean_df") is not None:
        df = st.session_state.clean_df.copy()
    elif st.session_state.get("df") is not None:
        df = st.session_state.df.copy()
    elif st.session_state.get("original_df") is not None:
        df = st.session_state.original_df.copy()
    elif st.session_state.get("raw_df") is not None:
        df = st.session_state.raw_df.copy()
    else:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    initial_shape = df.shape

    # ---------------- STATE INIT ----------------
    st.session_state.setdefault("page_stage", "editing")
    st.session_state.setdefault("cleaned", False)
    st.session_state.setdefault("missing_summary", {})
    st.session_state.setdefault("auto_state", None)
    st.session_state.setdefault("auto_cols_to_drop", [])

    st.session_state.setdefault("summaries", {})
    st.session_state["summaries"].setdefault("missing", {})

    # ===========================================================
    # EDITING STAGE
    # ===========================================================
    if st.session_state.page_stage == "editing":

        missing_cols = df.columns[df.isnull().any()].tolist()
        missing_pct = (df.isnull().mean() * 100).round(2)

        # ---------- NO MISSING (GUARDED) ----------
        if not missing_cols and not st.session_state.cleaned:
            st.success("🎉 No missing values found.")
            st.session_state.page_stage = "completed"
            st.session_state.cleaned = True
            st.session_state.clean_df = df.copy()
            st.session_state.cleaned_df = df.copy()
            st.session_state.df = df.copy()
            st.rerun()

        # ---------- SUMMARY ----------
        st.subheader("📌 Columns With Missing Values")
        st.dataframe(
            pd.DataFrame({
                "Column": missing_cols,
                "Missing Count": df[missing_cols].isnull().sum(),
                "Missing %": missing_pct[missing_cols]
            }),
            use_container_width=True
        )
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


        # =======================================================
        # STEP 2 — MODE
        # =======================================================
        st.subheader("🟩 Step 2: Choose Cleaning Mode")
        mode = st.radio("", ["Automatic (Recommended)", "Manual (Smart)"])

        # =======================================================
        # AUTOMATIC MODE
        # =======================================================
        if mode == "Automatic (Recommended)":

            st.markdown("""
            **Automatic Rules**
            - Numeric → Median
            - Categorical → Mode
            - Columns with >40% missing require confirmation to drop
            """)

            if st.button("✨ Apply Automatic Cleaning"):
                st.session_state.auto_cols_to_drop = [
                    c for c in target_cols if missing_pct[c] > 40
                ]
                st.session_state.auto_state = (
                    "confirm" if st.session_state.auto_cols_to_drop else "yes"
                )

            if st.session_state.auto_state == "confirm":
                st.warning("Columns with more than 40% missing values:")
                st.write(st.session_state.auto_cols_to_drop)

                c1, c2 = st.columns(2)
                if c1.button("YES – Drop Columns"):
                    st.session_state.auto_state = "yes"
                if c2.button("NO – Keep Columns"):
                    st.session_state.auto_state = "no"

            if st.session_state.auto_state in ["yes", "no"]:

                preview_df = df.copy()
                summary = {}

                if st.session_state.auto_state == "yes":
                    for c in st.session_state.auto_cols_to_drop:
                        preview_df = preview_df.drop(columns=[c])
                        summary[c] = "Dropped column (>40% missing)"

                for c in target_cols:
                    if c not in preview_df.columns:
                        continue
                    if pd.api.types.is_numeric_dtype(preview_df[c]):
                        preview_df[c] = preview_df[c].fillna(preview_df[c].median())
                        summary[c] = "Filled with Median"
                    else:
                        preview_df[c] = preview_df[c].fillna(preview_df[c].mode()[0])
                        summary[c] = "Filled with Mode"

                preview_cols = [c for c in target_cols if c in preview_df.columns]

                b1, b2 = st.columns(2)
                with b1:
                    st.markdown("**Before**")
                    st.dataframe(df[preview_cols].head(20))
                with b2:
                    st.markdown("**After**")
                    st.dataframe(preview_df[preview_cols].head(20))

                if st.button("✅ Confirm & Apply Automatic Cleaning"):
                    st.session_state.last_cleaned_df = df.copy()
                    st.session_state.clean_df = preview_df.copy()
                    st.session_state.cleaned_df = preview_df.copy()
                    st.session_state.df = preview_df.copy()
                    st.session_state.cleaned = True
                    st.session_state.page_stage = "completed"
                    st.session_state.missing_summary = {
                        "mode": "Automatic",
                        "columns": summary
                    }

                    # ---- send missing-value summary to Page 6 ----
                    st.session_state["summaries"]["missing"]["last_action"] = {
                        "mode": "Automatic",
                        "initial_shape": initial_shape,
                        "final_shape": preview_df.shape,
                        "Dropped Columns": [
                            c for c, v in summary.items() if "Dropped" in v
                        ],
                        "Filled Columns": {
                            c: {"method": v}
                            for c, v in summary.items() if "Filled" in v
                        }
                    }
                    st.rerun()

        # =======================================================
        # MANUAL MODE
        # =======================================================
        if mode == "Manual (Smart)":

            st.subheader("🧰 Step 3: Manual Cleaning")

            manual_actions = {}

            for col in target_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    options = [
                        "Fill with Mean",
                        "Fill with Median",
                        "Fill with Mode",
                        "Fill with Zero",
                        "Fill with Custom Value",
                        "Drop Rows with Missing"
                    ]
                else:
                    options = [
                        "Fill with Mode",
                        "Drop Rows with Missing"
                    ]

                manual_actions[col] = st.selectbox(col, options, key=f"manual_{col}")

                if manual_actions[col] == "Fill with Custom Value":
                    st.text_input(f"Custom value for {col}", key=f"custom_{col}")

            preview_df = df.copy()
            summary = {}

            for col, action in manual_actions.items():
                if action == "Drop Rows with Missing":
                    preview_df = preview_df[preview_df[col].notna()]
                    summary[col] = "Dropped rows with missing values"
                elif action == "Fill with Mean":
                    preview_df[col] = preview_df[col].fillna(preview_df[col].mean())
                    summary[col] = "Filled with Mean"
                elif action == "Fill with Median":
                    preview_df[col] = preview_df[col].fillna(preview_df[col].median())
                    summary[col] = "Filled with Median"
                elif action == "Fill with Mode":
                    preview_df[col] = preview_df[col].fillna(preview_df[col].mode()[0])
                    summary[col] = "Filled with Mode"
                elif action == "Fill with Zero":
                    preview_df[col] = preview_df[col].fillna(0)
                    summary[col] = "Filled with Zero"
                elif action == "Fill with Custom Value":
                    preview_df[col] = preview_df[col].fillna(st.session_state.get(f"custom_{col}"))
                    summary[col] = "Filled with Custom Value"

            preview_cols = [c for c in target_cols if c in preview_df.columns]

            b1, b2 = st.columns(2)
            with b1:
                st.markdown("**Before**")
                st.dataframe(df[preview_cols].head(20))
            with b2:
                st.markdown("**After**")
                st.dataframe(preview_df[preview_cols].head(20))

            if st.button("🚀 Apply Manual Cleaning"):
                st.session_state.last_cleaned_df = df.copy()
                st.session_state.clean_df = preview_df.copy()
                st.session_state.cleaned_df = preview_df.copy()
                st.session_state.df = preview_df.copy()
                st.session_state.cleaned = True
                st.session_state.page_stage = "completed"
                st.session_state.missing_summary = {
                    "mode": "Manual",
                    "columns": summary
                }

                # ---- send missing-value summary to Page 6 ----
                st.session_state["summaries"]["missing"]["last_action"] = {
                    "mode": "Manual",
                    "initial_shape": initial_shape,
                    "final_shape": preview_df.shape,
                    "Dropped Columns": [
                        c for c, v in summary.items() if "Dropped" in v
                    ],
                    "Filled Columns": {
                        c: {"method": v}
                        for c, v in summary.items() if "Filled" in v
                    }
                }
                st.rerun()

    # ===========================================================
    # COMPLETED STAGE
    # ===========================================================
    if st.session_state.page_stage == "completed":

        st.subheader("📊 Missing Value Handling Summary")
        st.write(f"**Mode:** {st.session_state.missing_summary.get('mode','')}")

        st.table(
            pd.DataFrame.from_dict(
                st.session_state.missing_summary.get("columns", {}),
                orient="index",
                columns=["Action Taken"]
            )
        )

        if st.button("↩️ Undo Last Operation"):
            perform_undo()

        st.download_button(
            "⬇️ Download Cleaned Dataset",
            data=st.session_state.clean_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_missing_values.csv",
            mime="text/csv"
        )
        st.markdown("**Missing Value Handled. You can download the dataset or go back using undo.**")

        st.markdown("---")
        st.subheader("Next Step: Semantic Cleanup")
        st.info("Semantic Cleanup fixes dirty or inconsistent *values* inside columns — it standardizes numeric/text representations,"
                "removes noise (symbols, units, punctuation), groups similar patterns, and lets you preview & apply safe fixes.")

        remaining = st.session_state.clean_df.columns[
            st.session_state.clean_df.isnull().any()
        ].tolist()

        if st.button("Proceed to Semantic Cleanup →"):
            if remaining:
                st.warning("Please handle missing values for all columns before proceeding.")
            else:
                st.session_state.current_page = "Semantic Cleanup"
                st.rerun()
