# pages/p3_Outlier_Handling.py
# FINAL Outlier Handling Page (Auto + Manual + Summary + Undo + Fixed Navigation)

import pandas as pd
import numpy as np
import math
from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# THEME SUPPORT
# ---------------------------------------------------------
try:
    from utils.theme import inject_theme
except:
    def inject_theme(): return

# ---------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------
def safe_to_numeric(s):
    # coerce to string first to avoid dtype issues, then remove commas
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def iqr_bounds(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper, iqr


def zscore_mask(series, thresh=3.0):
    s = safe_to_numeric(series)
    mu = s.mean()
    sd = s.std(ddof=0)
    if math.isclose(sd, 0.0):
        sd = 1.0
    z = (s - mu) / sd
    return z.abs() > thresh


def cap_series(series, lower=None, upper=None):
    s = safe_to_numeric(series)
    if lower is not None:
        s = s.clip(lower=lower)
    if upper is not None:
        s = s.clip(upper=upper)
    return s


def remove_outliers(df, col, mask):
    return df.loc[~mask].reset_index(drop=True)


def plot_before_after(before_ser, after_ser, title=None):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        axes[0].hist(pd.to_numeric(before_ser.dropna()), bins=30)
        axes[0].set_title("Before")
        axes[1].hist(pd.to_numeric(after_ser.dropna()), bins=30)
        axes[1].set_title("After")
        if title:
            fig.suptitle(title)
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        # plotting shouldn't block processing
        pass

# ---------------------------------------------------------
# OUTLIER STATS
# ---------------------------------------------------------
def compute_outlier_stats(df, numeric_cols, k=1.5):
    stats = {}
    for col in numeric_cols:
        ser = safe_to_numeric(df[col])
        nn = ser.dropna()
        if nn.empty:
            stats[col] = {
                "min": None, "max": None, "lower": None, "upper": None,
                "outlier_count": 0, "outlier_pct": 0.0, "iqr": 0.0
            }
            continue

        lower, upper, iqr = iqr_bounds(nn, k=k)
        mask = (nn < lower) | (nn > upper)
        out_count = int(mask.sum())
        pct = 100.0 * out_count / max(1, len(nn))

        stats[col] = {
            "min": float(nn.min()),
            "max": float(nn.max()),
            "lower": float(lower),
            "upper": float(upper),
            "outlier_count": out_count,
            "outlier_pct": round(pct, 2),
            "iqr": float(iqr)
        }
    return stats

# ---------------------------------------------------------
# SESSION STATE SETUP
# ---------------------------------------------------------
def init_session():
    """
    Initialize all Outlier Handling session state keys.
    SAFE to call multiple times.
    NO recursion.
    """

    st.session_state.setdefault("outlier_configs", {})
    st.session_state.setdefault("outlier_stats_cache", {})
    st.session_state.setdefault("outlier_report", [])
    st.session_state.setdefault("outlier_prev_df", [])
    st.session_state.setdefault("outlier_preview_limit", 200)

    st.session_state.setdefault("auto_params", {
        "k": 1.5,
        "z_thresh": 3.0
    })

    # Navigation / warnings
    st.session_state.setdefault("outlier_page4_warning", False)
    st.session_state.setdefault("outlier_page4_remaining", [])

    # Navigation state
    st.session_state.setdefault("outlier_page4_warning", False)
    st.session_state.setdefault("outlier_page4_remaining", [])

# ---------------------------------------------------------
# MAIN PAGE FUNCTION
# ---------------------------------------------------------
def run_outlier_handling():
    if "outlier_stats_cache" not in st.session_state:
        init_session()


    st.markdown("""
    <div class="page-title-box">
        <span style="font-size:28px;font-weight:800;">📉 Outlier Handling</span>
        <div style="margin-top:6px;font-size:14px;opacity:0.85;">
            Detect and treat outliers using IQR, Z-score, or custom thresholds.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠ Please complete Semantic Cleanup first.")
        st.stop()



    # ---------------------------------------------------------
    # 🔵 ADD EXPLANATION BOX
    # ---------------------------------------------------------
    st.markdown("""
    <div style='background:#11161e;padding:18px;border-radius:12px;
                border:1px solid #2a2f38;margin-bottom:16px;color:#dcdcdc;
                font-size:15px;line-height:1.55;'>

    <b style='font-size:17px;'>📘 IQR and Z-Score</b><br><br>

    <b>• IQR Method (1.5 × IQR)</b><br>
    Values lying more than <b>1.5×IQR</b> outside Q1 or Q3 are statistically rare.<br>
    This cutoff perfectly balances sensitivity and stability, and works best for 
    <b>real-world, skewed, non-normal</b> data.

    <br><br>

    <b>• Z-Score Method (|Z| ≥ 3)</b><br>
    In a normal distribution, <b>99.7%</b> of values lie within ±3 standard deviations.<br>
    So anything beyond <b>|3|</b> is extremely unlikely and considered a <b>true outlier</b>.

    <br><br>

    <b style='color:#a5d6ff;'>In short:</b><br>
    1.5×IQR handles skewed data ✔<br>
    Z-Score 3 handles normally distributed data ✔<br>
    Both thresholds are global statistical standards.
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------
    # NUMERIC COLUMNS
    # ---------------------------------------------
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        st.info("No numeric columns found.")
        return

    # ---------------------------------------------------------
    # CACHED STATS
    # ---------------------------------------------------------
    cache_id = f"{len(df)}-{','.join(numeric_cols)}"
    if cache_id not in st.session_state["outlier_stats_cache"]:
        st.session_state["outlier_stats_cache"][cache_id] = compute_outlier_stats(df, numeric_cols)

    stats = st.session_state["outlier_stats_cache"][cache_id]

    summary_rows = []
    for col in numeric_cols:
        s = stats[col]
        summary_rows.append({
            "Column": col,
            "Min": s["min"],
            "Max": s["max"],
            "Lower": s["lower"],
            "Upper": s["upper"],
            "Outliers": s["outlier_count"],
            "Outlier %": s["outlier_pct"]
        })

    st.subheader("Outlier Summary Table")
    st.dataframe(pd.DataFrame(summary_rows).sort_values("Outlier %", ascending=False), use_container_width=True)
    st.markdown("---")

    # ---------------------------------------------------------
    # MODE SELECTION
    # ---------------------------------------------------------
    mode = st.radio("Choose Mode:", ["Auto Mode (Recommended)", "Manual Mode (Visuable)"], index=0)

    # ---------------------------------------------------------
    # AUTO MODE
    # ---------------------------------------------------------
    if mode.startswith("Auto"):
        st.subheader("Auto Mode")

        k_val = st.number_input("IQR Multiplier (k)", value=st.session_state["auto_params"]["k"], step=0.1)
        z_val = st.number_input("Z-Score Threshold", value=st.session_state["auto_params"]["z_thresh"], step=0.5)

        if st.button("Preview Auto Mode Plan", key="out_preview_auto"):
            st.session_state["auto_params"]["k"] = float(k_val)
            st.session_state["auto_params"]["z_thresh"] = float(z_val)

            plan = []
            for col in numeric_cols:
                pct = stats[col]["outlier_pct"]

                if pct < 5:
                    method = "remove_rows"
                elif pct <= 20:
                    method = "iqr_cap"
                else:
                    method = "z_cap"

                plan.append({"col": col, "method": method})

            st.session_state["auto_plan"] = plan
            st.success("Auto Mode plan ready.")
            st.rerun()

        if st.session_state.get("auto_plan"):
            st.write("### Planned Auto Actions:")
            for p in st.session_state["auto_plan"]:
                st.write(f"• {p['col']} → {p['method']}")

            if st.button("Apply Auto Mode", key="out_apply_auto"):
                st.session_state["outlier_prev_df"].append(df.copy())
                new_df = df.copy()

                for p in st.session_state["auto_plan"]:
                    col = p["col"]; method = p["method"]
                    ser = safe_to_numeric(new_df[col])

                    if method == "remove_rows":
                        lower, upper, _ = iqr_bounds(ser.dropna(), k=k_val)
                        mask = (ser < lower) | (ser > upper)
                        new_df = remove_outliers(new_df, col, mask)

                    elif method == "iqr_cap":
                        lower, upper, _ = iqr_bounds(ser.dropna(), k=k_val)
                        new_df[col] = cap_series(new_df[col], lower, upper)

                    elif method == "z_cap":
                        mu = ser.mean()
                        sd = ser.std()
                        lower = mu - z_val * sd
                        upper = mu + z_val * sd
                        new_df[col] = cap_series(new_df[col], lower, upper)

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                st.session_state["df"] = new_df
                # PIPELINE FIX: ensure master clean_df updated
                st.session_state["clean_df"] = new_df.copy()

                for col in numeric_cols:
                    st.session_state["outlier_report"].append(f"{col} → auto ({ts})")

                st.session_state.pop("auto_plan", None)

                st.success("Auto Mode applied.")
                st.rerun()

    # ---------------------------------------------------------
    # MANUAL MODE  — (UNCHANGED)
    # ---------------------------------------------------------
    if mode.startswith("Manual"):
        st.subheader("Manual Mode")

        for col in numeric_cols:
            with st.expander(f"▶ {col}", expanded=False):

                config = st.session_state["outlier_configs"].get(col, {"method": "iqr_cap", "params": {}})
                method = st.selectbox(
                    f"Method for {col}",
                    ["iqr_cap","z_cap","remove_rows","replace_mean","replace_median","replace_custom"],
                    index=["iqr_cap","z_cap","remove_rows","replace_mean","replace_median","replace_custom"].index(config["method"]) if config and "method" in config else 0
                )

                params = {}

                if method == "iqr_cap":
                    k = st.number_input(f"IQR k for {col}", value=1.5, step=0.1)
                    params["k"] = k

                elif method == "z_cap":
                    zt = st.number_input(f"Z-threshold for {col}", value=3.0, step=0.5)
                    params["z_thresh"] = zt

                elif method == "replace_custom":
                    custom_val = st.text_input(f"Custom value for {col}")
                    params["custom"] = custom_val

                st.session_state["outlier_configs"][col] = {"method": method, "params": params}

                # PREVIEW
                if st.button(f"Preview {col}", key=f"out_preview_{col}"):
                    lim = st.session_state["outlier_preview_limit"]
                    before = df[[col]].head(lim)
                    temp = df.copy()
                    ser = safe_to_numeric(temp[col])

                    if method == "iqr_cap":
                        lower, upper, _ = iqr_bounds(ser.dropna(), k=params["k"])
                        after = cap_series(temp[col], lower, upper)

                    elif method == "z_cap":
                        mu = ser.mean()
                        sd = ser.std()
                        lower = mu - params["z_thresh"] * sd
                        upper = mu + params["z_thresh"] * sd
                        after = cap_series(temp[col], lower, upper)

                    elif method == "remove_rows":
                        lower, upper, _ = iqr_bounds(ser.dropna(), 1.5)
                        mask = (ser < lower) | (ser > upper)
                        temp2 = remove_outliers(temp, col, mask)
                        after = temp2[col].head(lim)

                    elif method == "replace_mean":
                        after = ser.fillna(ser.mean())

                    elif method == "replace_median":
                        after = ser.fillna(ser.median())

                    elif method == "replace_custom":
                        cv = params.get("custom", "")
                        if not str(cv).strip():
                            st.warning("Enter a custom value.")
                            after = ser
                        else:
                            try:
                                after = ser.fillna(float(cv))
                            except:
                                after = ser.fillna(cv)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("#### BEFORE")
                        st.dataframe(before)
                    with c2:
                        st.write("#### AFTER")
                        # ensure after is a Series or DataFrame
                        if isinstance(after, pd.Series):
                            st.dataframe(after.head(lim).to_frame())
                        else:
                            st.dataframe(after.head(lim))

                    try:
                        plot_before_after(before[col], after, title=f"{col} Preview")
                    except:
                        pass

                # APPLY FIX
                if st.button(f"Apply {col}", key=f"out_apply_{col}"):
                    st.session_state["outlier_prev_df"].append(df.copy())
                    new_df = df.copy()
                    ser = safe_to_numeric(new_df[col])

                    try:
                        if method == "iqr_cap":
                            lower, upper, _ = iqr_bounds(ser.dropna(), k=params["k"])
                            new_df[col] = cap_series(new_df[col], lower, upper)

                        elif method == "z_cap":
                            mu = ser.mean()
                            sd = ser.std()
                            lower = mu - params["z_thresh"] * sd
                            upper = mu + params["z_thresh"] * sd
                            new_df[col] = cap_series(new_df[col], lower, upper)

                        elif method == "remove_rows":
                            lower, upper, _ = iqr_bounds(ser.dropna(), 1.5)
                            mask = (ser < lower) | (ser > upper)
                            new_df = remove_outliers(new_df, col, mask)

                        elif method == "replace_mean":
                            new_df[col] = ser.fillna(ser.mean())

                        elif method == "replace_median":
                            new_df[col] = ser.fillna(ser.median())

                        elif method == "replace_custom":
                            cv = params.get("custom", "")
                            if not str(cv).strip():
                                st.error("Custom value empty.")
                                raise Exception("custom_empty")
                            try:
                                new_df[col] = ser.fillna(float(cv))
                            except:
                                new_df[col] = ser.fillna(cv)

                        # persist changes to session and pipeline master
                        st.session_state["df"] = new_df
                        st.session_state["clean_df"] = new_df.copy()

                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state["outlier_report"].append(f"{col} → {method} ({ts})")

                        st.success(f"Applied {method} to {col}")
                        st.rerun()

                    except Exception as e:
                        if str(e) == "custom_empty":
                            pass
                        else:
                            st.error(f"Failed to apply: {e}")

    st.markdown("---")

    # SUMMARY SECTION
    if st.session_state["outlier_report"]:
        st.subheader("Summary of Outlier Actions")
        for entry in st.session_state["outlier_report"]:
            st.write(f"• {entry}")
    else:
        st.info("No outlier actions yet.")

    st.markdown("---")

    # UNDO
    if st.session_state["outlier_prev_df"]:
        if st.button("Undo Last Action", key="out_undo"):
            last = st.session_state["outlier_prev_df"].pop()

            # Restore data
            st.session_state["df"] = last
            st.session_state["clean_df"] = last.copy()

            # Remove last report entry
            if st.session_state["outlier_report"]:
                st.session_state["outlier_report"].pop()

            # 🔄 RESET UI STATE (THIS IS THE KEY FIX)
            st.session_state.pop("auto_plan", None)
            st.session_state["outlier_configs"] = {}
            st.session_state["outlier_page4_warning"] = False
            st.session_state["outlier_page4_remaining"] = []

            st.success("Undo successful. Returned to mode selection.")
            st.rerun()

    # DOWNLOAD SECTION
    if st.session_state["outlier_report"]:
        st.subheader("📥 Download")

        csv_data = st.session_state["df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Cleaned Dataset (CSV)",
            csv_data,
            file_name="outlier_cleaned.csv",
            mime="text/csv",
            key="out_dl_csv"
        )

        txt = "Outlier Report\n\n"
        for i, entry in enumerate(st.session_state["outlier_report"], 1):
            txt += f"{i}. {entry}\n"

        st.download_button(
            "⬇ Download Outlier Report (TXT)",
            txt,
            file_name="outlier_report.txt",
            mime="text/plain",
            key="out_dl_txt"
        )

    st.markdown("---")
    st.write("### Proceed to EDA ")

    # PROCEED BUTTON
    if st.button("Proceed to EDA →", key="out_go_page4"):
        cleaned_cols = {
            entry.split("→")[0].strip()
            for entry in st.session_state["outlier_report"]
            if "→" in entry
        }

        remaining = [c for c in numeric_cols if c not in cleaned_cols]

        st.session_state["outlier_page4_remaining"] = remaining
        st.session_state["outlier_page4_warning"] = True
        st.rerun()

    if st.session_state.get("outlier_page4_warning", False):

        remaining = st.session_state.get("outlier_page4_remaining", [])

        if not remaining:
            st.session_state["outlier_page4_warning"] = False
            st.session_state["current_page"] = "EDA Core"
            st.rerun()

        st.warning("Some numeric columns are not handled yet:")
        for col in remaining:
            st.write(f"- {col}")

        c1, c2 = st.columns(2)

        with c1:
            if st.button("Continue Anyway", key="out_continue_anyway"):
                st.session_state["outlier_page4_warning"] = False
                st.session_state["current_page"] = "EDA Core"
                st.rerun()

        with c2:
            if st.button("Stay Here", key="out_stay_here"):
                st.session_state["outlier_page4_warning"] = False
                st.rerun()
