# p4b_EDA_Exports.py  — FULL (with EDA Export Summary + sliders for hist/box)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import zipfile
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import time


matplotlib.use("Agg")

# ---------------- CONFIG ----------------
MICRO_SLEEP = 0.01
WARN_THRESHOLD = 101
CONFIRM_THRESHOLD = 301

# ---------------- UTIL / CONFIRM ----------------
def confirm_and_maybe_run(count, label="generate"):
    st.info(f"You are about to **{label} {count} file(s)**.")
    st.warning("This may take some time — progress will show in real time.")
    if count >= CONFIRM_THRESHOLD:
        st.error("🚨 HIGH LOAD — weak devices may freeze.")
        return st.button(f"Proceed Anyway ({label})", key=f"big_{label}")
    if count >= WARN_THRESHOLD:
        return st.button(f"Continue ({label})", key=f"cont_{label}")
    return True


def _white_fig(size=(8,5), dpi=140):
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def _png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ---------------- PNG EXPORTERS (Matplotlib / Seaborn; white bg) ----------------

def png_scatter(df, x, y, trend=False):
    fig, ax = _white_fig((8,5), dpi=150)
    d = df[[x,y]].dropna()
    if d.empty:
        ax.text(0.5,0.5,"No data", ha="center", va="center")
    else:
        try:
            ax.scatter(d[x].astype(float), d[y].astype(float), s=18, alpha=0.8)
        except Exception:
            ax.scatter(d[x], d[y], s=18, alpha=0.8)
        if trend:
            try:
                coef = np.polyfit(d[x].astype(float), d[y].astype(float), 1)
                p = np.poly1d(coef)
                xs = np.linspace(d[x].min(), d[x].max(), 200)
                ax.plot(xs, p(xs), color="red", linewidth=1.6)
            except Exception:
                pass
    ax.set_title(f"{x} vs {y}")
    return _png_bytes(fig)


def png_hist(df, col, bins=40):
    fig, ax = _white_fig((8,4), dpi=140)
    d = df[col].dropna()
    if d.empty:
        ax.text(0.5,0.5,"No data", ha="center", va="center")
    else:
        try:
            sns.histplot(d, bins=bins, ax=ax, kde=False, color="#1f77b4")
        except Exception:
            ax.hist(d, bins=bins)
    ax.set_title(f"Histogram — {col}")
    plt.xticks(rotation=0)
    return _png_bytes(fig)


def png_box(df, cat_col, num_col, order=None):
    fig, ax = _white_fig((10,5), dpi=150)
    d = df.copy()
    # If cat_col is None — draw a simple boxplot for numeric column
    if cat_col is None:
        vals = d[num_col].dropna()
        if vals.empty:
            ax.text(0.5,0.5,"No data", ha="center", va="center")
        else:
            try:
                sns.boxplot(y=vals, ax=ax, color="#a6cee3")
            except Exception:
                ax.boxplot(vals)
        ax.set_title(f"Boxplot — {num_col}")
        return _png_bytes(fig)

    if order:
        try:
            d[cat_col] = pd.Categorical(d[cat_col], categories=order, ordered=True)
        except Exception:
            pass
    try:
        sns.boxplot(data=d, x=cat_col, y=num_col, ax=ax, palette="Set3")
    except Exception:
        med = d.groupby(cat_col)[num_col].median().reset_index()
        ax.bar(med[cat_col].astype(str), med[num_col].values)
    ax.set_title(f"{cat_col} vs {num_col} — Boxplot")
    plt.xticks(rotation=45)
    return _png_bytes(fig)


def png_bar(cat_col, val_col, grouped_df):
    fig, ax = _white_fig((10,5), dpi=150)
    g = grouped_df.copy()
    g[cat_col] = g[cat_col].astype(str)
    try:
        sns.barplot(data=g, x=cat_col, y=val_col, ax=ax, palette="tab20")
    except Exception:
        ax.bar(g[cat_col], g[val_col])
    ax.set_title(f"{cat_col} vs {val_col} — Bar")
    plt.xticks(rotation=45)
    for i, v in enumerate(g[val_col]):
        try:
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        except Exception:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    return _png_bytes(fig)


def png_freq(df, col):
    fig, ax = _white_fig((10,5), dpi=150)
    freq = df[col].value_counts().reset_index()
    freq.columns = [col, "count"]
    try:
        sns.barplot(data=freq, x=col, y="count", ax=ax, palette="viridis")
    except Exception:
        ax.bar(freq[col].astype(str), freq["count"].values)
    plt.xticks(rotation=45)
    ax.set_title(f"Frequency — {col}")
    return _png_bytes(fig)

def png_corr(df):
    """
    Clean correlation matrix export:
    - keeps only numeric columns
    - removes bool columns
    - drops constant columns
    - ensures no blank heatmap cells
    """
    # Keep only numeric (int/float), drop bool completely
    df_num = df.select_dtypes(include=["int64", "float64"]).copy()

    # Drop constant columns
    df_num = df_num.loc[:, df_num.nunique() > 1]

    if df_num.empty or len(df_num.columns) < 2:
        fig, ax = _white_fig((8, 4))
        ax.text(0.5, 0.5, "No numeric features for correlation", ha="center", va="center")
        return _png_bytes(fig)

    corr = df_num.corr()

    # Dynamic sizing for clean export
    n = len(corr.columns)
    fig_w = max(8, n * 0.9)
    fig_h = max(6, n * 0.9)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm_r",
        linewidths=0.5,
        linecolor="#dddddd",
        square=True,
        cbar=True,
        annot_kws={"color": "white", "size": 10, "weight": "bold"},
        ax=ax
    )

    ax.set_title("Correlation Matrix", fontsize=18, weight="bold", pad=20)
    plt.tight_layout()
    return _png_bytes(fig)



def png_catcat_heatmap(a, b, cross_df):
    try:
        pivot = cross_df.pivot(index=b, columns=a, values="Count")
    except Exception:
        pivot = cross_df.pivot_table(index=b, columns=a, values="Count", aggfunc="sum", fill_value=0)
    fig, ax = _white_fig((10,6), dpi=140)
    try:
        sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt="g", ax=ax)
    except Exception:
        ax.text(0.5,0.5,"No heatmap", ha="center")
    ax.set_title(f"{a} vs {b} — Heatmap")
    return _png_bytes(fig)

def draw_clean_correlation_heatmap(df, title="Correlation Heatmap"):
    # Select numeric columns only, remove booleans
    numeric_df = df.select_dtypes(include=["int64", "float64"]).copy()

    # Remove constant columns (cause blank heatmap cells)
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

    if numeric_df.empty or len(numeric_df.columns) < 2:
        st.info("Not enough numeric columns available for correlation heatmap.")
        return

    corr = numeric_df.corr()

    # Dynamic size
    n = len(corr.columns)
    fig_w = max(8, n * 0.9)
    fig_h = max(6, n * 0.9)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm_r",
        linewidths=0.3,
        linecolor="black",
        square=True,
        cbar=True,
        annot_kws={"color": "white", "size": 9, "weight": "bold"},
        ax=ax
    )

    ax.set_title(title, fontsize=20, weight="bold", pad=20)
    plt.tight_layout()
    st.pyplot(fig)



# --------------------- PAGE LOGIC ---------------------

def run_eda_exports():
    st.markdown("""
    <div class="page-title-box">
        <span style="font-size:28px;font-weight:800;">📥 EDA Export Center</span>
        <div style="margin-top:6px;font-size:14px;opacity:0.85;">
            Generate previews, export plots, and download analysis assets.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ------------------ DATASET CHECK ------------------
    if "clean_df" not in st.session_state or st.session_state.clean_df is None:
        st.error("⚠ No dataset found. Please upload data from HOME page first.")
        return

    df = st.session_state.clean_df

    # also block empty datasets
    if df is None or df.empty:
        st.error("⚠ Dataset is empty. Please fix missing values first.")
        return

    # make working copy only now (after validations)
    df = df.copy()

    # ---- EDA EXPORT SUMMARY INIT (for Page 6) ----
    st.session_state.setdefault("eda_export_summary", {
        "histograms": [],
        "boxplots": [],
        "bar_plots": [],
        "num_num": [],
        "num_cat": [],
        "cat_cat": [],
        "multi_scatter_count": 0,
        "multi_num_cat_count": 0,
        "multi_cat_cat_count": 0
    })

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

    # persistent cache of generated files (bytes)
    cache = st.session_state.setdefault("p4b_cache", {})

    # helper sort functions
    def sort_bar_df(g, c, n, opt):
        if opt == "Ascending": return g.sort_values(n)
        if opt == "Descending": return g.sort_values(n, ascending=False)
        if opt == "A → Z": return g.sort_values(c)
        if opt == "Z → A": return g.sort_values(c, ascending=False)
        return g

    def sort_box_order(df_, c, n, opt):
        med = df_.groupby(c)[n].median().reset_index()
        if opt == "Ascending": med = med.sort_values(n)
        elif opt == "Descending": med = med.sort_values(n, ascending=False)
        elif opt == "A → Z": med = med.sort_values(c)
        elif opt == "Z → A": med = med.sort_values(c, ascending=False)
        return med[c].tolist()

    # -----------------------------------------
    # SINGLE PREVIEWS
    # -----------------------------------------
    st.header("👀 Single Chart Previews")
    modes = ["Histogram","Boxplot","Scatter","Categorical Bar","Cat→Num","Cat→Cat","Correlation"]
    sel = st.multiselect("Select preview types:", modes, key="p4b_single_modes")
    # HISTOGRAM (multi-style with progress + slider)
    if "Histogram" in sel:

        if "hist_figs" not in st.session_state:
            st.session_state.hist_figs = []
        if "hist_names" not in st.session_state:
            st.session_state.hist_names = []

        hcols = st.multiselect("Histogram columns:", num_cols, key="p4b_hist_cols")

        if st.button("Preview Histograms", key="p4b_hist_preview"):
            st.session_state.hist_figs = []
            st.session_state.hist_names = []

            total = len(hcols)
            if total > 0:
                prog = st.progress(0)
            else:
                prog = None

            for i, col in enumerate(hcols, start=1):
                fig = px.histogram(df, x=col, nbins=40, title=f"Histogram — {col}")
                fig.update_layout(template="plotly_dark")

                st.session_state.hist_figs.append(fig)
                st.session_state.hist_names.append(col)

                cache[f"hist_{col}.png"] = png_hist(df, col)

                # summary
                if col not in st.session_state.eda_export_summary["histograms"]:
                    st.session_state.eda_export_summary["histograms"].append(col)

                if prog is not None:
                    prog.progress(int(i / total * 100))
                    time.sleep(MICRO_SLEEP)

            st.success(f"Generated {total} histograms.")

    # SLIDER VIEWER — HISTOGRAM
    if st.session_state.get("hist_figs"):
        total_hist = len(st.session_state.hist_figs)
        idx_hist = st.slider("View Histogram:", 1, total_hist, 1, key="p4b_hist_slider")
        st.markdown(f"### 📌 {st.session_state.hist_names[idx_hist-1]}")
        st.plotly_chart(st.session_state.hist_figs[idx_hist-1], width='stretch')

    # BOXPLOT (multi-style with progress + slider)
    if "Boxplot" in sel:

        if "box_figs" not in st.session_state:
            st.session_state.box_figs = []
        if "box_names" not in st.session_state:
            st.session_state.box_names = []

        bcols = st.multiselect("Boxplot columns:", num_cols, key="p4b_box_cols")

        if st.button("Preview Boxplots", key="p4b_box_preview"):
            st.session_state.box_figs = []
            st.session_state.box_names = []

            total = len(bcols)
            if total > 0:
                prog = st.progress(0)
            else:
                prog = None

            for i, col in enumerate(bcols, start=1):
                fig = px.box(df, y=col, title=f"Boxplot — {col}")
                fig.update_layout(template="plotly_dark")

                st.session_state.box_figs.append(fig)
                st.session_state.box_names.append(col)

                # PNG for export
                cache[f"box_{col}.png"] = png_box(df, None, col)

                # summary
                if col not in st.session_state.eda_export_summary["boxplots"]:
                    st.session_state.eda_export_summary["boxplots"].append(col)

                if prog is not None:
                    prog.progress(int(i / total * 100))
                    time.sleep(MICRO_SLEEP)

            st.success(f"Generated {total} boxplots.")

    # SLIDER VIEWER — BOXPLOT
    if st.session_state.get("box_figs"):
        total_box = len(st.session_state.box_figs)
        idx_box = st.slider("View Boxplot:", 1, total_box, 1, key="p4b_box_slider")
        st.markdown(f"### 📌 {st.session_state.box_names[idx_box-1]}")
        st.plotly_chart(st.session_state.box_figs[idx_box-1], width='stretch')

    # CATEGORICAL BAR (multi-style with progress + slider)
    if "Categorical Bar" in sel:

        if "catbar_figs" not in st.session_state:
            st.session_state.catbar_figs = []
        if "catbar_names" not in st.session_state:
            st.session_state.catbar_names = []

        ccols = st.multiselect("Categorical Bar columns:", cat_cols, key="p4b_catbar_cols")

        if st.button("Preview Categorical Bars", key="p4b_catbar_preview"):
            st.session_state.catbar_figs = []
            st.session_state.catbar_names = []

            total = len(ccols)
            if total > 0:
                prog = st.progress(0)
            else:
                prog = None

            for i, col in enumerate(ccols, start=1):

                freq = df[col].value_counts().reset_index()
                freq.columns = [col, "Count"]

                fig = px.bar(
                    freq,
                    x=col,
                    y="Count",
                    text="Count",
                    title=f"Frequency — {col}"
                )
                fig.update_layout(template="plotly_dark")

                st.session_state.catbar_figs.append(fig)
                st.session_state.catbar_names.append(col)

                # PNG for export (Cat vs Num bar style)
                cache[f"catbar_{col}.png"] = png_freq(df, col)

                # summary
                label = f"{col} (frequency)"
                if label not in st.session_state.eda_export_summary["bar_plots"]:
                    st.session_state.eda_export_summary["bar_plots"].append(label)

                if prog is not None:
                    prog.progress(int(i / total * 100))
                    time.sleep(MICRO_SLEEP)

            st.success(f"Generated {total} categorical bar charts.")

        # SLIDER VIEWER — CATEGORICAL BAR
    if st.session_state.get("catbar_figs"):
        total_cb = len(st.session_state.catbar_figs)
        idx_cb = st.slider("View Categorical Bar:", 1, total_cb, 1, key="p4b_catbar_slider")
        st.markdown(f"### 📌 {st.session_state.catbar_names[idx_cb - 1]}")
        st.plotly_chart(
            st.session_state.catbar_figs[idx_cb - 1],
            width='stretch'
        )

    # SCATTER (single)
    if "Scatter" in sel:
        c1, c2 = st.columns(2)
        with c1:
            sc_x = st.selectbox("X (numeric):", num_cols, key="p4b_sc_x")
        with c2:
            sc_y = st.selectbox("Y (numeric):", [n for n in num_cols if n != sc_x], key="p4b_sc_y")
        sc_trend = st.checkbox("Add Trendline", key="p4b_sc_trend")
        if st.button("Preview Scatter", key="p4b_sc_preview"):
            try:
                fig = px.scatter(df, x=sc_x, y=sc_y, trendline=("ols" if sc_trend else None), title=f"{sc_x} vs {sc_y}")
            except Exception:
                fig = px.scatter(df, x=sc_x, y=sc_y, title=f"{sc_x} vs {sc_y}")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, width='stretch')
            cache[f"scatter_{sc_x}_{sc_y}.png"] = png_scatter(df, sc_x, sc_y, sc_trend)

            # summary: num↔num
            pair_label = f"{sc_x} ↔ {sc_y}"
            if pair_label not in st.session_state.eda_export_summary["num_num"]:
                st.session_state.eda_export_summary["num_num"].append(pair_label)

    # CAT->NUM (single)
    if "Cat→Num" in sel:
        c1,c2,c3 = st.columns(3)
        with c1:
            cvn_cat = st.selectbox("Category:", cat_cols, key="p4b_cvn_cat")
        with c2:
            cvn_num = st.selectbox("Numeric:", num_cols, key="p4b_cvn_num")
        with c3:
            cvn_mode = st.radio("Chart:", ["Boxplot","Bar"], horizontal=True, key="p4b_cvn_mode")
        cvn_sort = st.selectbox("Sort:", ["No Sort","Ascending","Descending","A → Z","Z → A"], key="p4b_cvn_sort")
        if st.button("Preview Cat→Num", key="p4b_cvn_preview"):
            if cvn_mode == "Boxplot":
                order = sort_box_order(df, cvn_cat, cvn_num, cvn_sort)
                fig = px.box(df, x=cvn_cat, y=cvn_num, category_orders={cvn_cat:order}, title=f"{cvn_cat} vs {cvn_num}")
                cache[f"catnum_box_{cvn_cat}_{cvn_num}.png"] = png_box(df, cvn_cat, cvn_num, order)
            else:
                grouped = df.groupby(cvn_cat)[cvn_num].mean().reset_index()
                grouped = sort_bar_df(grouped, cvn_cat, cvn_num, cvn_sort)
                fig = px.bar(grouped, x=cvn_cat, y=cvn_num, text=cvn_num, title=f"{cvn_cat} vs {cvn_num}")
                cache[f"catnum_bar_{cvn_cat}_{cvn_num}.png"] = png_bar(cvn_cat, cvn_num, grouped)
                # summary: bar plot pair
                bar_label = f"{cvn_cat} vs {cvn_num}"
                if bar_label not in st.session_state.eda_export_summary["bar_plots"]:
                    st.session_state.eda_export_summary["bar_plots"].append(bar_label)

            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, width='stretch')

            # summary: Num→Cat link (num → cat)
            link = f"{cvn_num} → {cvn_cat}"
            if link not in st.session_state.eda_export_summary["num_cat"]:
                st.session_state.eda_export_summary["num_cat"].append(link)

    # CAT->CAT (single)
    if "Cat→Cat" in sel:
        c1,c2 = st.columns(2)
        with c1:
            cvc_a = st.selectbox("Category A:", cat_cols, key="p4b_cvc_a")
        with c2:
            cvc_b = st.selectbox("Category B:", cat_cols, key="p4b_cvc_b")
        if st.button("Preview Cat→Cat", key="p4b_cvc_preview"):
            if cvc_a == cvc_b:
                freq = df[cvc_a].value_counts().reset_index()
                freq.columns = [cvc_a, "Count"]
                fig = px.bar(freq, x=cvc_a, y="Count", title=f"Frequency — {cvc_a}")
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, width='stretch')
                cache[f"catcat_freq_{cvc_a}.png"] = png_freq(df, cvc_a)
            else:
                cross = df.groupby([cvc_a, cvc_b]).size().reset_index(name="Count")
                fig = px.density_heatmap(cross, x=cvc_a, y=cvc_b, z="Count", title=f"{cvc_a} vs {cvc_b}")
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, width='stretch')
                cache[f"catcat_heat_{cvc_a}_{cvc_b}.png"] = png_catcat_heatmap(cvc_a, cvc_b, cross)

            # summary: Cat↔Cat pair
            cat_pair = f"{cvc_a} ↔ {cvc_b}"
            if cat_pair not in st.session_state.eda_export_summary["cat_cat"]:
                st.session_state.eda_export_summary["cat_cat"].append(cat_pair)

    # ---------------- Correlation Preview (Updated) ----------------
    if "Correlation" in sel:

        if st.button("Preview Correlation", key="p4b_corr_preview"):

            # clean numeric columns (remove boolean + constant)
            numeric_df = df.select_dtypes(include=["int64", "float64"]).copy()
            numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

            if numeric_df is None or numeric_df.empty or len(numeric_df.columns) < 2:
                st.warning("⚠ Not enough numeric columns to generate a correlation heatmap.")
            else:
                st.subheader("📊 Correlation Heatmap Preview")
                draw_clean_correlation_heatmap(numeric_df)

                # Save PNG using clean logic
                try:
                    cache["corr_matrix.png"] = png_corr(numeric_df)
                except:
                    cache["corr_matrix.png"] = b""

    st.divider()
    st.header("📦 Advanced Multi-Graph Generator")
    # -------------------------
    # MULTI-SCATTER (slider, persistent)
    # -------------------------
    st.subheader("1️⃣ Multi Scatter")

    # init
    if "ms_figs" not in st.session_state:
        st.session_state.ms_figs = []
    if "ms_names" not in st.session_state:
        st.session_state.ms_names = []
    if "ms_preview_trigger" not in st.session_state:
        st.session_state.ms_preview_trigger = False

    ms_x = st.multiselect("X columns:", num_cols, key="p4b_ms_x")
    ms_y = st.multiselect("Y columns:", num_cols, key="p4b_ms_y")
    ms_allow_same = st.checkbox("Allow X == Y", key="p4b_ms_allow_same")
    ms_trend = st.checkbox("Add Trendline", key="p4b_ms_trend")

    ms_pairs = [(a,b) for a in ms_x for b in ms_y if (ms_allow_same or a!=b)]
    ms_labels = [f"{x} vs {y}" for x,y in ms_pairs]
    ms_sel = st.multiselect("Select pairs:", ms_labels, default=ms_labels, key="p4b_ms_sel")

    # If user changes selection, reset stored previews
    if tuple(ms_sel) != tuple(st.session_state.get("ms_last_sel", ())):
        st.session_state.ms_figs = []
        st.session_state.ms_names = []
        st.session_state.ms_preview_trigger = False
        st.session_state.ms_last_sel = tuple(ms_sel)

    if ms_sel:
        total = len(ms_sel)
        if not st.session_state.get("ms_confirmed", False):
            ok = confirm_and_maybe_run(total, label="Multi-Scatter")
            if ok:
                st.session_state.ms_confirmed = True
            else:
                st.stop()

        if st.button("Preview Multi-Scatter", key="p4b_ms_preview"):
            st.session_state.ms_figs = []
            st.session_state.ms_names = []
            st.session_state.ms_preview_trigger = True

            prog = st.progress(0)
            for i, lbl in enumerate(ms_sel, start=1):
                time.sleep(MICRO_SLEEP)
                x,y = ms_pairs[ms_labels.index(lbl)]

                # PNG for export
                try:
                    cache[f"ms_{x}_{y}.png"] = png_scatter(df, x, y, ms_trend)
                except:
                    cache[f"ms_{x}_{y}.png"] = b""

                # Plotly fig for preview
                try:
                    fig = px.scatter(df, x=x, y=y, trendline=("ols" if ms_trend else None), title=f"{x} vs {y}")
                    fig.update_layout(template="plotly_dark")
                except:
                    fig = px.scatter(df, x=x, y=y, title=f"{x} vs {y}")
                    fig.update_layout(template="plotly_dark")

                st.session_state.ms_figs.append(fig)
                st.session_state.ms_names.append(f"{x} vs {y}")

                prog.progress(int(i/total*100))
            st.success(f"Generated {total} scatter charts.")

            # summary: multi scatter count
            st.session_state.eda_export_summary["multi_scatter_count"] += total

    # Render slider viewer if previews exist
    if st.session_state.get("ms_preview_trigger", False) and st.session_state.ms_figs:
        total = len(st.session_state.ms_figs)
        idx = st.slider("Select chart to view:", 1, total, 1, key="p4b_ms_slider")
        st.markdown(f"### 📌 {st.session_state.ms_names[idx-1]}")
        st.plotly_chart(st.session_state.ms_figs[idx-1], width='stretch')

    st.divider()

    # -------------------------
    # MULTI CAT -> NUM (slider, persistent)
    # -------------------------
    st.subheader("2️⃣ Multi Cat → Num")

    # init
    if "mc_figs" not in st.session_state:
        st.session_state.mc_figs = []
    if "mc_names" not in st.session_state:
        st.session_state.mc_names = []
    if "mc_preview_trigger" not in st.session_state:
        st.session_state.mc_preview_trigger = False

    mc_cats = st.multiselect("Categories:", cat_cols, key="p4b_mc_cats")
    mc_nums = st.multiselect("Numerics:", num_cols, key="p4b_mc_nums")
    mc_pairs = [(c,n) for c in mc_cats for n in mc_nums]
    mc_labels = [f"{c} vs {n}" for c,n in mc_pairs]
    mc_sel = st.multiselect("Select pairs:", mc_labels, default=mc_labels, key="p4b_mc_sel")
    mc_mode = st.radio("Chart Type:", ["Boxplot","Bar"], horizontal=True, key="p4b_mc_mode")
    mc_sort = st.selectbox("Sort:", ["No Sort","Ascending","Descending","A → Z","Z → A"], key="p4b_mc_sort")

    if tuple(mc_sel) != tuple(st.session_state.get("mc_last_sel", ())):
        st.session_state.mc_figs = []
        st.session_state.mc_names = []
        st.session_state.mc_preview_trigger = False
        st.session_state.mc_last_sel = tuple(mc_sel)

    if mc_sel:
        total = len(mc_sel)
        if not st.session_state.get("mc_confirmed", False):
            ok = confirm_and_maybe_run(total, label="Multi Cat→Num")
            if ok:
                st.session_state.mc_confirmed = True
            else:
                st.stop()

        if st.button("Preview Multi Cat→Num", key="p4b_mc_preview"):
            st.session_state.mc_figs = []
            st.session_state.mc_names = []
            st.session_state.mc_preview_trigger = True

            prog = st.progress(0)
            for i, lbl in enumerate(mc_sel, start=1):
                time.sleep(MICRO_SLEEP)
                c,n = mc_pairs[mc_labels.index(lbl)]

                if mc_mode == "Boxplot":
                    order = sort_box_order(df, c, n, mc_sort)
                    try:
                        cache[f"mc_box_{c}_{n}.png"] = png_box(df, c, n, order)
                    except:
                        cache[f"mc_box_{c}_{n}.png"] = b""
                    fig = px.box(df, x=c, y=n, category_orders={c:order}, title=f"{c} vs {n}")
                    fig.update_layout(template="plotly_dark")
                else:
                    g = df.groupby(c)[n].mean().reset_index()
                    g = sort_bar_df(g, c, n, mc_sort)
                    try:
                        cache[f"mc_bar_{c}_{n}.png"] = png_bar(c, n, g)
                    except:
                        cache[f"mc_bar_{c}_{n}.png"] = b""
                    fig = px.bar(g, x=c, y=n, text=n, title=f"{c} vs {n}")
                    fig.update_layout(template="plotly_dark")

                st.session_state.mc_figs.append(fig)
                st.session_state.mc_names.append(f"{c} vs {n}")

                prog.progress(int(i/total*100))
            st.success(f"Generated {total} Cat→Num charts.")

            # summary: multi num→cat count
            st.session_state.eda_export_summary["multi_num_cat_count"] += total

    if st.session_state.get("mc_preview_trigger", False) and st.session_state.mc_figs:
        total = len(st.session_state.mc_figs)
        idx = st.slider("Select chart to view:", 1, total, 1, key="p4b_mc_slider")
        st.markdown(f"### 📌 {st.session_state.mc_names[idx-1]}")
        st.plotly_chart(st.session_state.mc_figs[idx-1], width='stretch')

    st.divider()
    # -------------------------
    # MULTI CAT -> CAT (slider, persistent)
    # -------------------------
    st.subheader("3️⃣ Multi Cat → Cat")

    # init
    if "mcc_figs" not in st.session_state:
        st.session_state.mcc_figs = []
    if "mcc_names" not in st.session_state:
        st.session_state.mcc_names = []
    if "mcc_preview_trigger" not in st.session_state:
        st.session_state.mcc_preview_trigger = False

    mcc_a = st.multiselect("Category A:", cat_cols, key="p4b_mcc_a")
    mcc_b = st.multiselect("Category B:", cat_cols, key="p4b_mcc_b")
    mcc_allow_self = st.checkbox("Allow A == B (frequency)", key="p4b_mcc_allow_self")
    mcc_pairs = [(a,b) for a in mcc_a for b in mcc_b if (mcc_allow_self or a!=b)]
    mcc_labels = [f"{a} vs {b}" for a,b in mcc_pairs]
    mcc_sel = st.multiselect("Select pairs:", mcc_labels, default=mcc_labels, key="p4b_mcc_sel")

    if tuple(mcc_sel) != tuple(st.session_state.get("mcc_last_sel", ())):
        st.session_state.mcc_figs = []
        st.session_state.mcc_names = []
        st.session_state.mcc_preview_trigger = False
        st.session_state.mcc_last_sel = tuple(mcc_sel)

    if mcc_sel:
        total = len(mcc_sel)
        if not st.session_state.get("mcc_confirmed", False):
            ok = confirm_and_maybe_run(total, label="Multi Cat→Cat")
            if ok:
                st.session_state.mcc_confirmed = True
            else:
                st.stop()

        if st.button("Preview Multi Cat→Cat", key="p4b_mcc_preview"):
            st.session_state.mcc_figs = []
            st.session_state.mcc_names = []
            st.session_state.mcc_preview_trigger = True

            prog = st.progress(0)
            for i, lbl in enumerate(mcc_sel, start=1):
                time.sleep(MICRO_SLEEP)
                a,b = mcc_pairs[mcc_labels.index(lbl)]

                if a == b:
                    freq = df[a].value_counts().reset_index()
                    freq.columns = [a, "count"]
                    fig = px.bar(freq, x=a, y="Count", title=f"Freq — {a}")
                    fig.update_layout(template="plotly_dark")
                    try:
                        cache[f"mcc_freq_{a}.png"] = png_freq(df, a)
                    except:
                        cache[f"mcc_freq_{a}.png"] = b""
                else:
                    cross = df.groupby([a,b]).size().reset_index(name="Count")
                    fig = px.density_heatmap(cross, x=a, y=b, z="Count", title=f"{a} vs {b}")
                    fig.update_layout(template="plotly_dark")
                    try:
                        cache[f"mcc_heat_{a}_{b}.png"] = png_catcat_heatmap(a, b, cross)
                    except:
                        cache[f"mcc_heat_{a}_{b}.png"] = b""

                st.session_state.mcc_figs.append(fig)
                st.session_state.mcc_names.append(f"{a} vs {b}")

                prog.progress(int(i/total*100))
            st.success(f"Generated {total} Cat→Cat charts.")

            # summary: multi cat↔cat count
            st.session_state.eda_export_summary["multi_cat_cat_count"] += total

    if st.session_state.get("mcc_preview_trigger", False) and st.session_state.mcc_figs:
        total = len(st.session_state.mcc_figs)
        idx = st.slider("Select chart to view:", 1, total, 1, key="p4b_mcc_slider")
        st.markdown(f"### 📌 {st.session_state.mcc_names[idx-1]}")
        st.plotly_chart(st.session_state.mcc_figs[idx-1], width='stretch')

    st.divider()

    # --- EXPORT / ZIP
    st.header("📥 Export & Download")
    if cache:
        st.subheader("Individual files")
        for fname, content in cache.items():
            if not content:
                continue
            safe_key = f"p4b_dl_{hash(fname)%100000}"
            try:
                st.download_button(f"⬇ {fname}", data=content, file_name=fname, key=safe_key)
            except Exception:
                st.download_button(f"⬇ {fname}", data=content, file_name=f"fallback_{fname}", key=f"p4b_dl_fallback_{hash(fname)%100000}")

    else:
        st.info("No generated files yet. Run previews above.")

    st.subheader("Create ZIP of generated files")
    if st.button("Create ZIP of generated files", key="p4b_zip_create"):
        st.session_state["p4b_zip_requested"] = True
        st.session_state["p4b_zip_confirmed"] = False

    if st.session_state.get("p4b_zip_requested", False):
        files_to_zip = [(f,c) for f,c in cache.items() if c]
        total_files = len(files_to_zip)
        if total_files == 0:
            st.warning("No generated files to zip. Run previews first.")
            st.session_state["p4b_zip_requested"] = False
            st.stop()

        if not st.session_state.get("p4b_zip_confirmed", False):
            ok = confirm_and_maybe_run(total_files, label="create ZIP")
            if ok:
                st.session_state["p4b_zip_confirmed"] = True
            else:
                st.stop()

        buff = BytesIO()
        prog = st.progress(0)
        with st.spinner("Creating ZIP…"):
            with zipfile.ZipFile(buff, "w") as z:
                for i, (fname, content) in enumerate(files_to_zip, start=1):
                    time.sleep(MICRO_SLEEP)
                    try:
                        z.writestr(fname, content)
                    except Exception as e:
                        z.writestr(f"error_{fname}.txt", str(e))
                    prog.progress(int(i/total_files*100))
        buff.seek(0)
        st.download_button("⬇ Download ZIP", data=buff.getvalue(), file_name="EDA_Exports.zip", key="p4b_zip_dl")
        st.success(f"Created ZIP with {total_files} files.")
        st.session_state["p4b_zip_requested"] = False
        st.session_state["p4b_zip_confirmed"] = False

    st.divider()





    st.markdown("### 🔁 Navigation")
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("⬅ Back", key="p4b_nav_back"):
            st.session_state.current_page = "EDA Core"
            st.rerun()
    with colB:
        if st.button("Next ➡", key="p4b_nav_next"):
            st.session_state.step_completed = st.session_state.get("step_completed", {})
            st.session_state.step_completed["p4b_EDA_Exports"] = True
            st.session_state.current_page = "Encoding & Transformation"
            st.rerun()

    st.success("🎉 Export Center Ready! Generate previews first, then download.")
