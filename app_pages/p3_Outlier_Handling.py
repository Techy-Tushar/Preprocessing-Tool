# ---------------------------------------------------------
# PAGE 3 — OUTLIER HANDLING (SEMANTIC-STYLE + PARAM SLIDERS)
# ---------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------
# THEME SUPPORT
# ---------------------------------------------------------
try:
    from utils.theme import inject_theme
except:
    def inject_theme(): return

# ---------------------------------------------------------
# HELPERS (UNCHANGED CORE LOGIC)
# ---------------------------------------------------------
def safe_to_numeric(s):
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

def iqr_bounds(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr, iqr

def cap_series(series, lower, upper):
    return safe_to_numeric(series).clip(lower=lower, upper=upper)

def plotly_hist_compare(before, after, col):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Before", "After"],
        shared_yaxes=True
    )

    fig.add_trace(
        go.Histogram(
            x=before.dropna(),
            nbinsx=30,
            name="Before",
            opacity=0.75
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Histogram(
            x=after.dropna(),
            nbinsx=30,
            name="After",
            opacity=0.75
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        title_text=f"{col} — Distribution"
    )

    st.plotly_chart(fig, width="stretch")

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plotly_box_compare(before, after, col):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Before", "After"],
        shared_yaxes=True
    )

    # BEFORE (vertical)
    fig.add_trace(
        go.Box(
            y=before.dropna(),
            boxpoints="outliers",
            marker_color="#7aa6c2"
        ),
        row=1,
        col=1
    )

    # AFTER (vertical)
    fig.add_trace(
        go.Box(
            y=after.dropna(),
            boxpoints="outliers",
            marker_color="#1f77b4"
        ),
        row=1,
        col=2
    )

    fig.update_layout(
        title=dict(
            text=f"{col} — Outliers (Before vs After)",
            y=0.98,
            x=0.5,
            xanchor="center",
            yanchor="top"
        ),
        height=400,
        margin=dict(l=40, r=20, t=70, b=40),
        showlegend=False,
        template="plotly_dark",
        yaxis_title=col
    )

    fig.update_yaxes(title_text=col, row=1, col=1)
    fig.update_yaxes(title_text=col, row=1, col=2)

    st.plotly_chart(fig, width="stretch")


def is_binary_numeric(series):
    """
    Returns True if the column contains only binary values (0/1).
    """
    vals = set(series.dropna().unique())
    return vals.issubset({0, 1})

def make_arrow_safe(df):
    """
    Make DataFrame safe for Streamlit Arrow serialization.
    Converts object columns that are fully numeric into numeric dtype.
    Does NOT mutate original dataframe.
    """
    safe_df = df.copy()

    for col in safe_df.columns:
        if safe_df[col].dtype == "object":
            try:
                safe_df[col] = pd.to_numeric(safe_df[col])
            except Exception:
                pass  # keep as object if conversion fails

    return safe_df

# ---------------------------------------------------------
# SESSION INIT
# ---------------------------------------------------------
def init_state():
    st.session_state.setdefault("outlier_queue", {})
    st.session_state.setdefault("outlier_summary", {})
    st.session_state.setdefault("outlier_prev_df", [])
    st.session_state.setdefault("outlier_selected_cols", [])



def render_outlier_intro():
    st.markdown("""
        <div class="page-title-box">
            <span style="font-size:28px;font-weight:800;">📉 Outlier Handling</span>
            <div style="margin-top:6px;font-size:14px;opacity:0.85;">
                Detect and treat extreme values using statistical methods.
                All actions are logged and reversible.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()


    st.markdown("""
        <div style="
            background:#11161e;
            padding:18px;
            border-radius:12px;
            border:1px solid #2a2f38;
            margin-bottom:18px;
            color:#dcdcdc;
            font-size:15px;
            line-height:1.55;
        ">

        <b style="font-size:17px;">📘 How Outliers Are Detected</b><br><br>

        <b>• IQR Method (1.5 × IQR)</b><br>
        For each numeric column, values lying below 
        <b>Q1 − 1.5×IQR</b> or above <b>Q3 + 1.5×IQR</b> are considered outliers.<br>
        This method is robust for <b>real-world, skewed data</b> and is widely used in data preprocessing.

        <br><br>

        <b>• Z-Score Method (|Z| ≥ 3)</b><br>
        In normally distributed data, <b>99.7%</b> of values lie within ±3 standard deviations.<br>
        Values beyond this range are statistically rare and treated as outliers.

        <br><br>

        <b style="color:#a5d6ff;">How columns are selected:</b><br>
        • Only <b>numerical columns</b> are evaluated<br>
        • Only columns with <b>detected outliers (&gt; 0%)</b> are shown for handling<br>
        • Clean columns are skipped automatically to save time

        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------
def run_outlier_handling():
    inject_theme()
    init_state()

    render_outlier_intro()

    df = st.session_state.get("clean_df")
    if df is None:
        st.warning("⚠ Please complete Semantic Cleanup first.")
        return

    # -----------------------------------------------------
    # DATASET PREVIEW
    # -----------------------------------------------------
    st.subheader("Dataset Preview")
    st.dataframe(
        make_arrow_safe(df).head(50),
        width="stretch"
    )

    numeric_cols = []
    binary_cols = []

    for col in df.columns:
        ser = pd.to_numeric(df[col], errors="coerce")

        # skip columns that are fully non-numeric
        if ser.notna().sum() == 0:
            continue

        # separate binary columns
        if is_binary_numeric(ser):
            binary_cols.append(col)
        else:
            numeric_cols.append(col)

    outlier_cols = {}
    for col in numeric_cols:
        s = safe_to_numeric(df[col]).dropna()
        if s.empty:
            continue
        low, high, _ = iqr_bounds(s)
        mask = (s < low) | (s > high)
        pct = round(100 * mask.sum() / len(s), 2)
        if pct > 0:
            outlier_cols[col] = pct

    st.subheader("Detected Numerical Columns")
    st.caption("Columns have Outliers")
    for c, p in outlier_cols.items():
        st.write(f"• **{c}** ({p}%)")

    if not outlier_cols:
        st.success("No outliers detected.")


    # -----------------------------------------------------
    # COLUMN MULTI-SELECTION
    # -----------------------------------------------------
    st.subheader("Select Columns to Handle Outliers")

    # ✅ sanitize previously selected columns
    valid_defaults = [
        c for c in st.session_state.get("outlier_selected_cols", [])
        if c in outlier_cols
    ]

    selected_cols = st.multiselect(
        "Choose columns",
        list(outlier_cols.keys()),
        default=valid_defaults
    )

    st.session_state["outlier_selected_cols"] = selected_cols

    # -----------------------------------------------------
    # EXPANDERS PER COLUMN
    # -----------------------------------------------------
    for col in selected_cols:
        with st.expander(f"▶ {col}", expanded=False):

            ser = safe_to_numeric(df[col])

            st.write(f"**Outlier %:** {outlier_cols[col]}%")

            method = st.selectbox(
                f"Handling Method for {col}",
                [
                    "IQR Capping",
                    "Z-Score Capping",
                    "Remove Outliers",
                    "Replace with Median",
                    "Ignore Column"
                ],
                key=f"method_{col}"
            )

            params = {}

            # -----------------------------
            # CONDITIONAL PARAM SLIDERS
            # -----------------------------
            if method == "IQR Capping":
                k = st.slider(
                    f"IQR Multiplier (k) — {col}",
                    0.5, 3.0, 1.5, 0.1,
                    key=f"k_{col}"
                )
                params["k"] = k

            elif method == "Z-Score Capping":
                z = st.slider(
                    f"Z-Score Threshold — {col}",
                    1.5, 5.0, 3.0, 0.1,
                    key=f"z_{col}"
                )
                params["z"] = z

            # -----------------------------
            # PREVIEW LOGIC
            # -----------------------------
            if method == "IQR Capping":
                low, high, _ = iqr_bounds(ser.dropna(), params["k"])
                after = cap_series(ser, low, high)

            elif method == "Z-Score Capping":
                mu, sd = ser.mean(), ser.std() or 1
                z = params["z"]
                after = cap_series(ser, mu - z * sd, mu + z * sd)

            elif method == "Remove Outliers":
                low, high, _ = iqr_bounds(ser.dropna(), 1.5)
                mask = (ser < low) | (ser > high)
                after = ser[~mask]

            elif method == "Replace with Median":
                after = ser.fillna(ser.median())

            else:
                after = ser.copy()

            st.markdown("### 📊 Distribution (Before vs After)")
            plotly_hist_compare(ser, after, col)

            st.markdown("### 📦 Outliers (Before vs After)")
            plotly_box_compare(ser, after, col)

            # -----------------------------
            # QUEUE ACTION
            # -----------------------------
            if st.button(f"Mark Approach for {col}", key=f"queue_{col}"):
                st.session_state["outlier_queue"][col] = {
                    "method": method,
                    "params": params
                }
                st.success(f"{col} queued → {method}")

    # -----------------------------------------------------
    # QUEUED ACTIONS
    # -----------------------------------------------------
    if st.session_state["outlier_queue"]:
        st.subheader("Queued Outlier Actions")
        for c, cfg in st.session_state["outlier_queue"].items():
            st.write(f"• **{c}** → {cfg['method']}")

        if st.button("Apply Outlier Handling to Selected Columns"):
            st.session_state["outlier_prev_df"].append(df.copy())
            new_df = df.copy()

            progress = st.progress(0)
            total = len(st.session_state["outlier_queue"])

            for i, (col, cfg) in enumerate(st.session_state["outlier_queue"].items(), 1):
                ser = safe_to_numeric(new_df[col])
                method = cfg["method"]
                params = cfg["params"]

                if method == "IQR Capping":
                    low, high, _ = iqr_bounds(ser.dropna(), params["k"])
                    new_df[col] = cap_series(ser, low, high)

                elif method == "Z-Score Capping":
                    mu, sd = ser.mean(), ser.std() or 1
                    z = params["z"]
                    new_df[col] = cap_series(ser, mu - z * sd, mu + z * sd)

                elif method == "Remove Outliers":
                    low, high, _ = iqr_bounds(ser.dropna(), 1.5)
                    mask = (ser < low) | (ser > high)
                    new_df = new_df.loc[~mask].reset_index(drop=True)

                elif method == "Replace with Median":
                    new_df[col] = ser.fillna(ser.median())

                # -------------------------------------------------
                # ✅ ADD THIS LINE FOR PAGE-6 SUMMARY (NO TIMESTAMP)
                # -------------------------------------------------
                st.session_state.setdefault("outlier_report", [])
                st.session_state["outlier_report"].append(
                    f"{col} → {method}"
                )

                st.session_state["outlier_summary"][col] = {
                    "method": method,
                    "params": params
                }

                progress.progress(i / total)

            st.session_state["clean_df"] = new_df
            st.session_state["df"] = new_df
            st.session_state["outlier_queue"].clear()
            st.session_state["outlier_selected_cols"] = []

            st.success("Outlier handling applied successfully.")
            st.rerun()

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------
    if st.session_state["outlier_summary"]:
        st.subheader("Outlier Handling Summary")
        for c, info in st.session_state["outlier_summary"].items():
            param_txt = ", ".join(f"{k}={v}" for k, v in info["params"].items())
            st.write(f"• **{c}** → {info['method']} ({param_txt})")

    # -----------------------------------------------------
    # UNDO
    # -----------------------------------------------------
    if st.session_state["outlier_prev_df"]:
        if st.button("Undo Last Action"):
            last = st.session_state["outlier_prev_df"].pop()
            st.session_state["clean_df"] = last
            st.session_state["df"] = last
            st.session_state["outlier_queue"].clear()
            st.session_state["outlier_selected_cols"] = []


            if st.session_state.get("outlier_report"):
                st.session_state["outlier_report"].pop()


            st.session_state["outlier_summary"].clear()

            st.success("Undo successful. Returned to selection step.")
            st.rerun()

    # -----------------------------------------------------
    # DOWNLOAD + NAVIGATION
    # -----------------------------------------------------
    if st.session_state["outlier_summary"]:
        st.download_button(
            "⬇ Download Cleaned Dataset",
            make_arrow_safe(st.session_state["clean_df"]).to_csv(index=False),
            "outlier_cleaned.csv"
        )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back to Semantic Cleanup"):
            st.session_state["current_page"] = "Semantic Cleanup"
            st.rerun()
    with c2:
        if st.button("Proceed to EDA →"):
            st.session_state["current_page"] = "EDA Core"
            st.rerun()
