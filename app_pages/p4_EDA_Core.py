# pages/p4_EDA_Core.py
# Clean EDA Page — Plotly, small-modern cards, stable navigation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go


# Theme import
try:
    from utils.theme import inject_theme
except:
    def inject_theme():
        return


# ---------------- Helpers ----------------
def safe_to_numeric(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")


def sample_for_plot(df, limit=10000):
    n = len(df)
    if n <= limit:
        return df
    return df.sample(frac=limit/n, random_state=42)


def simple_trendline(x, y):
    try:
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if mask.sum() < 2:
            return None, None
        coef = np.polyfit(x[mask], y[mask], 1)
        return float(coef[0]), float(coef[1])
    except:
        return None, None

def is_effectively_numeric(series, threshold=0.9):
    """
    Treat column as numeric if >= threshold values can be converted to numbers
    """
    coerced = pd.to_numeric(series, errors="coerce")
    ratio = coerced.notna().mean()
    return ratio >= threshold



# ---------------- Page Session Init ----------------
def init_eda_session():

    # Ensure page_completed exists
    st.session_state.setdefault("page_completed", {})
    st.session_state.setdefault("eda_preview_limit", 200)



def draw_clean_correlation_heatmap(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Select numeric columns only
    df_num = df.select_dtypes(include=["int64", "float64"]).copy()

    # Remove constant columns (avoid blank areas)
    df_num = df_num.loc[:, df_num.nunique() > 1]

    if df_num.empty:
        st.warning("No valid numeric columns available for correlation heatmap.")
        return

    corr = df_num.corr()

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        cmap="coolwarm",
        linewidths=.5,
        fmt=".2f",
        cbar=True
    )
    st.pyplot(fig)


# ---------------- MAIN PAGE ----------------
def run_eda_core():
    # ---- THEME + PAGE CONFIG ----
    inject_theme()

    try:
        st.set_page_config(
            page_title="EDA — Core",
            page_icon="📊",
            layout="wide"
        )
    except:
        pass

    # ---- PAGE TITLE CARD ----
    st.markdown("""
    <div class="page-title-box">
        <div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:28px;font-weight:800;">📊 EDA — Core Analysis</span>
        </div>
        <div style="margin-top:6px;font-size:14px;opacity:0.85;">
            Explore distributions, detect patterns, and understand relationships between variables.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ---- SAFE DATASET CHECK ----
    df = st.session_state.get("clean_df")
    if df is None:
        st.warning("⚠ Please complete Missing → Semantic → Outlier steps first.")
        st.stop()



    # split cols
    num_cols = []
    cat_cols = []

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        elif is_effectively_numeric(df[c]):
            # force numeric conversion
            df[c] = pd.to_numeric(df[c], errors="coerce")
            num_cols.append(c)
        else:
            cat_cols.append(c)

    # ---------------- Overview Metrics ----------------
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", df.shape[0])
    with c2:
        st.metric("Columns", df.shape[1])
    with c3:
        st.metric("Missing Values", int(df.isna().sum().sum()))

    st.markdown("---")

    # ---------------- Dataset Preview ----------------
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("---")

    # ---------------- Summary Statistics ----------------
    st.subheader("Summary Statistics")

    left, right = st.columns(2)

    with left:
        st.write("### 📌 Numeric Summary")
        if num_cols:
            desc = df[num_cols].describe().T
            desc["skew"] = df[num_cols].skew()
            st.dataframe(desc, use_container_width=True)
        else:
            st.info("No numeric columns found.")

    with right:
        st.write("### 📌 Categorical Summary")
        if cat_cols:
            desc = df[cat_cols].describe().T
            desc["unique"] = df[cat_cols].nunique()
            st.dataframe(desc, use_container_width=True)
        else:
            st.info("No categorical columns found.")

    st.markdown("---")

    # ---------------- Univariate Numeric ----------------
    st.subheader("Univariate Analysis — Numeric Columns")

    if not num_cols:
        st.info("Dataset has no numeric columns.")
    else:
        col = st.selectbox("Select numeric column", ["--select--"] + num_cols, key="num_uni_col")
        if col != "--select--":
            ser = safe_to_numeric(df[col]).dropna()

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Min", round(float(ser.min()), 3))
            with c2: st.metric("Max", round(float(ser.max()), 3))
            with c3: st.metric("Mean", round(float(ser.mean()), 3))
            with c4: st.metric("Std Dev", round(float(ser.std()), 3))

            fig1 = px.histogram(ser, nbins=40, title=f"Histogram — {col}", template="plotly_white")
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)



            fig2 = px.box(ser.to_frame(col), y=col, points="outliers",
                          title=f"Boxplot — {col}", template="plotly_white")
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)



            skew = float(df[col].skew()) if col in df.columns else 0.0
            if abs(skew) > 1:
                st.warning(f"High skew: {round(skew,2)}")
            elif abs(skew) > 0.5:
                st.info(f"Medium skew: {round(skew,2)}")
            else:
                st.success(f"Low skew: {round(skew,2)}")

    st.markdown("---")

    # ---------------- Univariate Categorical ----------------
    st.subheader("Univariate Analysis — Categorical Columns")

    if not cat_cols:
        st.info("Dataset has no categorical columns.")
    else:
        ccol = st.selectbox("Select categorical column", ["--select--"] + cat_cols, key="cat_uni_col")
        if ccol != "--select--":
            vc = df[ccol].fillna("<<NA>>").astype(str)
            ct = vc.value_counts().reset_index()
            ct.columns = [ccol, "count"]
            ct["pct"] = (ct["count"] / ct["count"].sum() * 100).round(2)

            st.dataframe(ct.head(50), use_container_width=True)

            fig = px.bar(
                ct.head(20), x=ccol, y="count",
                title=f"Top {ccol} categories",
                template="plotly_white"
            )
            fig.update_layout(height=420, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            # no summary needed for cat univariate

    st.markdown("---")

    # ---------------- Numeric → Categorical ----------------
    st.subheader("Numeric → Categorical Relationship")

    if not num_cols or not cat_cols:
        st.info("Requires one numeric & one categorical column.")
    else:
        num_sel = st.selectbox("Numeric column", ["--select--"] + num_cols, key="numcat_num")
        cat_sel = st.selectbox("Categorical column", ["--select--"] + cat_cols, key="numcat_cat")

        if num_sel != "--select--" and cat_sel != "--select--":
            tmp = df[[num_sel, cat_sel]].dropna()
            tmp[cat_sel] = tmp[cat_sel].astype(str)

            mean_table = tmp.groupby(cat_sel)[num_sel].mean().to_frame("mean")

            sort_choice = st.selectbox(
                "Sort Categories By",
                ["None", "A → Z", "Z → A", "Mean Asc", "Mean Desc"],
                key="numcat_sort"
            )

            if sort_choice == "A → Z":
                mean_table = mean_table.sort_index()
            elif sort_choice == "Z → A":
                mean_table = mean_table.sort_index(ascending=False)
            elif sort_choice == "Mean Asc":
                mean_table = mean_table.sort_values("mean")
            elif sort_choice == "Mean Desc":
                mean_table = mean_table.sort_values("mean", ascending=False)

            st.dataframe(mean_table.head(50), use_container_width=True)

            plot_type = st.selectbox("Plot Type", ["Box Plot", "Bar Plot"], key="numcat_plot_type")
            top_n = st.number_input("Top categories to display", 3, 50, 10, key="numcat_topn")

            top_cats = tmp[cat_sel].value_counts().head(top_n).index.tolist()
            plot_df = tmp[tmp[cat_sel].isin(top_cats)]

            if plot_type == "Box Plot":
                fig = px.box(
                    plot_df, x=cat_sel, y=num_sel,
                    category_orders={cat_sel: top_cats},
                    points="outliers", template="plotly_white",
                    title=f"{num_sel} across {cat_sel}"
                )
                fig.update_layout(height=450, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                mean_vals = plot_df.groupby(cat_sel)[num_sel].mean().reindex(top_cats)
                fig = px.bar(
                    mean_vals.reset_index(), x=cat_sel, y=num_sel,
                    template="plotly_white",
                    title=f"Mean {num_sel} across {cat_sel}"
                )
                fig.update_layout(height=420, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)



    st.markdown("---")

    # ---------------- Numeric → Numeric ----------------
    st.subheader("Numeric → Numeric Relationship (Scatter Plot)")

    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns.")
    else:
        x_col = st.selectbox("X-axis", ["--select--"] + num_cols, key="scatter_x")
        y_col = st.selectbox("Y-axis", ["--select--"] + num_cols, key="scatter_y")

        if x_col != "--select--" and y_col != "--select--":
            if x_col == y_col:
                st.warning("Please select two **different** numeric columns.")
            else:
                data = sample_for_plot(df[[x_col, y_col]]).dropna()

                fig = px.scatter(
                    data, x=x_col, y=y_col,
                    title=f"{y_col} vs {x_col}",
                    opacity=0.75, template="plotly_white"
                )

                xv = safe_to_numeric(data[x_col]).to_numpy()
                yv = safe_to_numeric(data[y_col]).to_numpy()
                slope, intercept = simple_trendline(xv, yv)

                if slope is not None:
                    xs = np.linspace(np.nanmin(xv), np.nanmax(xv), 100)
                    ys = slope * xs + intercept
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines", name="Trendline",
                        line=dict(color="red")
                    ))

                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)



    st.markdown("---")

    st.subheader("📊 Correlation Heatmap")
    draw_clean_correlation_heatmap(df)



    # Correlation insights below (no summary needed)
    num_df = df.select_dtypes(include=["int64", "float64"]).copy()
    num_df = num_df.loc[:, num_df.nunique() > 1]
    corr = num_df.corr()

    # ---------------- Correlation Insight Cards ----------------
    st.subheader("Correlation Insights")

    if num_cols:
        # Use only columns present in corr to avoid KeyError
        valid_cols = [c for c in num_cols if c in corr.columns]

        pairs = []
        for i in range(len(valid_cols)):
            for j in range(i + 1, len(valid_cols)):
                a = valid_cols[i]
                b = valid_cols[j]

                if a in corr.index and b in corr.columns:
                    v = corr.loc[a, b]
                    pairs.append((a, b, v))

        # Sort by strongest absolute correlation
        sorted_pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

        strongest_pos = next((p for p in sorted_pairs if p[2] > 0), None)
        strongest_neg = next((p for p in sorted_pairs if p[2] < 0), None)

        col1, col2 = st.columns(2)

        card_html = lambda title, label, corr_val, color: f"""
            <div style="
                background-color:#fff;
                color:#0b0b0b;
                border-radius:8px;
                padding:12px;
                margin-bottom:8px;
                box-shadow:0 1px 3px rgba(0,0,0,0.12);
                border-left:6px solid {color};
            ">
                <div style="font-size:13px; color:#333; margin-bottom:4px;">{title}</div>
                <div style="font-size:16px; font-weight:700;">{label}</div>
                <div style="font-size:18px; font-weight:700; margin-top:4px; color:{color};">
                    {corr_val}
                </div>
            </div>
        """

        with col1:
            st.markdown("### 🟩 Strongest Positive")
            if strongest_pos:
                a, b, v = strongest_pos
                st.markdown(
                    card_html("Positive Correlation", f"{a} ↔ {b}", f"+{round(v, 3)}", "#1fb954"),
                    unsafe_allow_html=True
                )
            else:
                st.info("No positive correlation found.")

        with col2:
            st.markdown("### 🟥 Strongest Negative")
            if strongest_neg:
                a, b, v = strongest_neg
                st.markdown(
                    card_html("Negative Correlation", f"{a} ↔ {b}", f"{round(v, 3)}", "#ff3b30"),
                    unsafe_allow_html=True
                )
            else:
                st.info("No negative correlation found.")

    else:
        st.info("No numeric columns available for correlation.")

    st.markdown("---")

    # ---------------- Quick Insights ----------------
    st.subheader("Quick Insights (Auto-Generated)")

    insights = []
    insights.append(f"Dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")

    if cat_cols:
        uniq = df[cat_cols].nunique().sort_values(ascending=False)
        insights.append(
            f"Most diverse categorical column: **{uniq.index[0]}** with **{uniq.iloc[0]}** unique values."
        )

    if num_cols:
        skewed = [c for c in num_cols if abs(df[c].skew()) > 1]
        if skewed:
            insights.append(f"Highly skewed numeric columns: **{', '.join(skewed)}**.")

    if len(insights) <= 1:
        insights.append("No significant statistical patterns detected in this dataset.")

    for it in insights:
        st.markdown(f"""
            <div style="
                background-color:#fff;
                color:#0b0b0b;
                border-radius:8px;
                padding:10px;
                margin-bottom:10px;
                box-shadow:0 1px 3px rgba(0,0,0,0.08);
            ">{it}</div>
        """, unsafe_allow_html=True)

    st.markdown("---")


    # ---------------- Navigation Buttons ----------------
    col_prev, col_next = st.columns([1, 1])

    with col_prev:
        if st.button("⬅️ Back"):
            st.session_state.current_page = "Outlier Handling"
            st.rerun()

    with col_next:
        if st.button("Next ➡️"):
            st.session_state.current_page = "EDA Exports"
            st.rerun()

    st.session_state.page_completed["p4_EDA_Core"] = True

    return
