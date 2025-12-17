# ------------------------------------------------------
# PAGE 5 — Encoding & Transformation (Final Full File)
# ------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

from utils.theme import inject_theme


# ======================================================
# 🔧 INITIALIZE PAGE-5 STATE VARIABLES
# ======================================================
def init_page5_state():
    """Initialize Page-5 flags + storage."""
    st.session_state.setdefault("p5_summary", [])
    st.session_state.setdefault("p5_skew_handled", {})
    st.session_state.setdefault("p5_last_msg", "")
    st.session_state.setdefault("p5_pca_preview", None)
    st.session_state.setdefault("p5_pca_meta", {})

    # PROGRESS FLAGS
    st.session_state.setdefault("p5_encoding_done", False)
    st.session_state.setdefault("p5_skew_done", False)
    st.session_state.setdefault("p5_corr_done", False)
    st.session_state.setdefault("p5_pca_done", False)

    # NAVIGATION FLAGS (same pattern as Semantic & Outlier pages)
    st.session_state.setdefault("p5_page6_warning", False)
    st.session_state.setdefault("p5_page6_pending", [])


# ======================================================
# 🔧 SAFE TRANSFORM HELPERS
# ======================================================
def safe_log(series):
    s = series.astype(float)
    if s.min() <= 0:
        s += abs(s.min()) + 1
    return np.log1p(s)


def safe_sqrt(series):
    s = series.astype(float)
    if s.min() < 0:
        s += abs(s.min())
    return np.sqrt(s)


def safe_reciprocal(series):
    s = series.astype(float)
    if s.min() <= 0:
        s += abs(s.min()) + 1
    return 1 / (s + 1e-6)


def apply_skew_transform(series, method):
    if method == "Log":
        return safe_log(series)
    elif method == "Square-root":
        return safe_sqrt(series)
    elif method == "Reciprocal":
        return safe_reciprocal(series)
    return series  # Do nothing


# ======================================================
# 1️⃣ ENCODING SECTION
# ======================================================
def section_encoding(df):
    st.header("1️⃣ Encoding")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        st.info("No categorical columns detected.")
        return df

    enc_col = st.selectbox("Select a column to encode:", ["--select--"] + cat_cols)

    enc_method = st.radio(
        "Choose encoding method:",
        ["Label Encoding", "One-Hot Encoding", "Manual Mapping"],
        horizontal=True
    )

    manual_map = {}

    if enc_col != "--select--" and enc_method == "Manual Mapping":
        st.subheader("Manual Mapping")
        uniq_vals = df[enc_col].dropna().unique().tolist()
        for val in uniq_vals:
            new_val = st.text_input(f"Map '{val}' →", key=f"map_{enc_col}_{val}")
            manual_map[val] = new_val if new_val != "" else val

    # PREVIEW
    if enc_col != "--select--":
        st.markdown("### 🔍 Preview (Before → After)")
        before = df[[enc_col]].head(50)

        if enc_method == "Label Encoding":
            le = LabelEncoder()
            after = before.copy()
            after[enc_col] = le.fit_transform(after[enc_col].astype(str))

        elif enc_method == "One-Hot Encoding":
            # Force ints so Streamlit doesn't show checkboxes
            after = pd.get_dummies(before[enc_col], prefix=enc_col).astype(int).head(15)

        else:
            after = before.copy()
            after[enc_col] = after[enc_col].map(lambda x: manual_map.get(x, x))

        c1, c2 = st.columns(2)
        c1.dataframe(before, use_container_width=True)
        c2.dataframe(after, use_container_width=True)

    if st.button("Apply Encoding"):
        if enc_col == "--select--":
            st.warning("Please select a column.")
            return df

        if enc_method == "Label Encoding":
            df[enc_col] = LabelEncoder().fit_transform(df[enc_col].astype(str))
            msg = f"{enc_col} encoded using Label Encoding"

        elif enc_method == "One-Hot Encoding":
            dummies = pd.get_dummies(df[enc_col], prefix=enc_col).astype(int)
            df = df.drop(columns=[enc_col])
            df = pd.concat([df, dummies], axis=1)
            msg = f"{enc_col} encoded using One-Hot Encoding"

        else:
            df[enc_col] = df[enc_col].map(lambda x: manual_map.get(x, x))
            msg = f"{enc_col} encoded using Manual Mapping."

        st.success(msg)
        st.session_state["p5_summary"].append(msg)
        st.session_state["p5_encoding_done"] = True

        return df

    return df


# ======================================================
# 2️⃣ SKEWNESS HANDLING
# ======================================================
def section_skewness(df):
    st.header("2️⃣ Skewness Correction")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in num_cols if df[c].nunique() > 2]

    if not candidates:
        st.info("No suitable numeric columns for skew correction.")
        return df

    col = st.selectbox("Select numeric column:", ["--select--"] + candidates)

    if col == "--select--":
        return df

    series = df[col].dropna()
    skew_val = float(series.skew())

    st.write(f"📌 Current Skewness of **{col}** = `{skew_val:.3f}`")

    # Recommendation expander
    with st.expander("Transformation Recommendation"):
        st.write("""
### How transformations are recommended:
- **Skew > 2.0** → Log or Reciprocal  
- **1.0 < Skew ≤ 2.0** → Square-root  
- **Skew < 1.0** → Usually no transformation needed  

### About methods:
- **Log:** compresses large outliers  
- **Square-root:** mild compression  
- **Reciprocal:** strongest compression  
        """)

    method = st.radio(
        "Choose method:",
        ["Log", "Square-root", "Reciprocal", "Do nothing"],
        horizontal=True
    )

    # Compute transformed series (for preview)
    transformed = apply_skew_transform(series, method)

    st.markdown("### 📊 Distribution Preview (Before → After)")
    c1, c2 = st.columns(2)

    series_plot = series.rename("Before")
    transformed_plot = transformed.rename("After")

    c1.plotly_chart(
        px.histogram(series_plot, nbins=40),
        key=f"skew_before_{col}",
        use_container_width=True
    )
    c2.plotly_chart(
        px.histogram(transformed_plot, nbins=40),
        key=f"skew_after_{col}",
        use_container_width=True
    )

    # VALUE PREVIEW TABLE (Before → After)
    st.markdown("### 🔍 Value Preview (Before → After)")
    before_vals = series.rename("Before").reset_index(drop=True)
    after_vals = transformed.rename("After").reset_index(drop=True)

    preview_df = pd.DataFrame({
        "Before": before_vals.head(50),
        "After": after_vals.head(50)
    })
    st.dataframe(preview_df, use_container_width=True)

    if st.button("Apply Skewness Fix"):
        if method == "Do nothing":
            msg = f"Skewness checked for {col}; no transformation applied."
        else:
            df[col] = apply_skew_transform(df[col], method)
            msg = f"Applied **{method}** transform on **{col}**."

        st.success(msg)
        st.session_state["p5_summary"].append(msg)
        st.session_state["p5_skew_done"] = True
        st.session_state["p5_skew_handled"][col] = method

        return df

    return df
# ======================================================
# 3️⃣ CORRELATION HANDLING
# ======================================================
def section_correlation(df):
    st.header("3️⃣ Correlation Handling")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.info("Not enough numeric columns for correlation.")
        return df

    corr = df[num_cols].corr()

    strong = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c1, c2 = num_cols[i], num_cols[j]
            val = corr.loc[c1, c2]
            if abs(val) >= 0.70:
                strong.append((c1, c2, val))

    if not strong:
        st.success("No high-correlation pairs found (|corr| ≥ 0.70).")
        return df

    st.info("Only pairs with |correlation| ≥ **0.70** require action.")

    labels = [f"{a} ↔ {b} ({val:.3f})" for a, b, val in strong]
    sel = st.selectbox("Select a high-correlation pair:", labels)

    idx = labels.index(sel)
    a, b, val = strong[idx]

    action = st.radio(
        "Choose action:",
        ["Keep both", "Drop one", "Combine features"],
        horizontal=True
    )

    drop_col = None
    combine = None

    if action == "Drop one":
        drop_col = st.selectbox("Select column to drop:", [a, b])

    if action == "Combine features":
        combine = st.radio(
            "Choose combination:",
            ["Add", "Average", "Difference"]
        )

    if st.button("Apply Correlation Fix"):
        if action == "Keep both":
            msg = f"Kept both {a} and {b}."

        elif action == "Drop one":
            df = df.drop(columns=[drop_col])
            msg = f"Dropped **{drop_col}** due to high correlation."

        else:
            if combine == "Add":
                newc = f"{a}_{b}_add"
                df[newc] = df[a] + df[b]
            elif combine == "Average":
                newc = f"{a}_{b}_avg"
                df[newc] = (df[a] + df[b]) / 2
            else:
                newc = f"{a}_{b}_diff"
                df[newc] = df[a] - df[b]
            msg = f"Created new feature **{newc}** from {a} and {b}."

        st.success(msg)
        st.session_state["p5_summary"].append(msg)
        st.session_state["p5_corr_done"] = True

        return df

    return df


# ======================================================
# 4️⃣ PCA — Dimensionality Reduction (Manual + Variance Mode)
# ======================================================
def section_pca(df):
    st.header("4️⃣ PCA — Dimensionality Reduction")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.info("No numeric columns available for PCA.")
        return df

    # 1) Select columns
    pca_cols = st.multiselect("Select numeric columns for PCA:", num_cols)
    if not pca_cols:
        return df

    if len(pca_cols) < 2:
        st.warning("⚠ PCA requires at least 2 numeric columns.")
        return df

    X = df[pca_cols].astype(float)

    # 2) PCA MODE SELECTION
    pca_mode = st.radio(
        "Select PCA Mode:",
        ["Manual Component Selection", "Variance Target Selection"],
        horizontal=True
    )

    # 3) Pre-processing options
    c1, c2 = st.columns(2)
    with c1:
        scale = st.checkbox("Standardize before PCA", value=True)
    with c2:
        drop_original = st.checkbox("Drop original selected columns after PCA", value=False)

    X_scaled = StandardScaler().fit_transform(X) if scale else X.values

    # Fit PCA with maximum components once
    pca_full = PCA(n_components=len(pca_cols))
    pca_full.fit(X_scaled)

    explained_full = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained_full) * 100.0  # in %

    # MODE 1 — MANUAL
    if pca_mode == "Manual Component Selection":
        k = st.slider(
            "Number of PCA components:",
            min_value=1,
            max_value=len(pca_cols),
            value=min(2, len(pca_cols))
        )

        total_var = cumulative[k - 1]

        with st.expander("ℹ️ Explained Variance (What each component captures)"):

            ev_df = pd.DataFrame({
                "Component": [f"PCA{i+1}" for i in range(len(explained_full))],
                "Explained Variance (%)": np.round(explained_full * 100, 2),
                "Cumulative Variance (%)": np.round(cumulative, 2)
            })

            st.write(f"Variance retained with {k} components: **{total_var:.2f}%**")
            st.dataframe(ev_df, use_container_width=True)

            fig = px.bar(
                ev_df.head(k),
                x="Component",
                y="Explained Variance (%)",
                title="Explained Variance per PCA Component"
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📘 PCA Preview (Before → After)"):

            pca_temp = PCA(n_components=k)
            comp_values = pca_temp.fit_transform(X_scaled)

            comp_df = pd.DataFrame(
                comp_values,
                columns=[f"PCA{i+1}" for i in range(k)],
                index=df.index
            )

            st.write("### Before (Original Features)")
            st.dataframe(df[pca_cols].head(), use_container_width=True)

            st.write("### After (PCA Components)")
            st.dataframe(comp_df.head(), use_container_width=True)

            st.session_state["p5_pca_preview"] = comp_df
            st.session_state["p5_pca_meta"] = {
                "cols": pca_cols,
                "k": k,
                "total_var": total_var,
                "drop_original": drop_original,
                "mode": "manual"
            }

    # MODE 2 — VARIANCE TARGET SELECTION
    else:
        target_var = st.slider(
            "Target Variance (%):",
            min_value=50,
            max_value=95,
            value=85
        )

        achievable = cumulative[-1] >= target_var

        if not achievable:
            st.error(
                f"❌ The selected columns cannot achieve **{target_var}%** variance.\n"
                f"Maximum possible variance is **{cumulative[-1]:.2f}%**.\n"
                f"➡️ Please select more columns or lower the target variance."
            )
            return df

        k = int(np.argmax(cumulative >= target_var)) + 1
        total_var = cumulative[k - 1]

        st.success(
            f"Auto-selected **{k} components** to achieve "
            f"**{total_var:.2f}%** variance (target: {target_var}%)."
        )

        with st.expander("ℹ️ Explained Variance (What each component captures)"):

            ev_df = pd.DataFrame({
                "Component": [f"PCA{i+1}" for i in range(len(explained_full))],
                "Explained Variance (%)": np.round(explained_full * 100, 2),
                "Cumulative Variance (%)": np.round(cumulative, 2)
            })

            st.dataframe(ev_df, use_container_width=True)

            fig = px.bar(
                ev_df.head(k),
                x="Component",
                y="Explained Variance (%)",
                title="Explained Variance per PCA Component"
            )

            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📘 PCA Preview (Before → After)"):

            pca_temp = PCA(n_components=k)
            comp_values = pca_temp.fit_transform(X_scaled)

            comp_df = pd.DataFrame(
                comp_values,
                columns=[f"PCA{i+1}" for i in range(k)],
                index=df.index
            )

            st.write("### Before")
            st.dataframe(df[pca_cols].head(), use_container_width=True)

            st.write("### After")
            st.dataframe(comp_df.head(), use_container_width=True)

            st.session_state["p5_pca_preview"] = comp_df
            st.session_state["p5_pca_meta"] = {
                "cols": pca_cols,
                "k": k,
                "target_var": target_var,
                "total_var": total_var,
                "drop_original": drop_original,
                "mode": "variance"
            }

    # APPLY PCA
    if st.button("Apply PCA"):

        comp_df = st.session_state.get("p5_pca_preview")
        meta = st.session_state.get("p5_pca_meta", {})

        if comp_df is None:
            st.error("Please generate PCA preview before applying.")
            return df

        used_cols = meta.get("cols", pca_cols)
        used_k = meta.get("k")
        used_total = meta.get("total_var")
        used_drop = meta.get("drop_original", False)
        mode = meta.get("mode", "manual")

        # Drop originals if chosen
        if used_drop:
            df = df.drop(columns=used_cols)

        # Add PCA components
        for col in comp_df.columns:
            df[col] = comp_df[col]

        # Messages
        if mode == "manual":
            msg = (
                f"PCA applied (Manual Mode): {used_k} components "
                f"(variance retained: {used_total:.2%})."
            )
        else:
            target_var = meta.get("target_var")
            msg = (
                f"PCA applied (Variance Mode): Auto-selected {used_k} components "
                f"to reach {used_total:.2f}% variance "
                f"(target: {target_var}%)."
            )

        st.success(msg)
        st.session_state["p5_summary"].append(msg)
        st.session_state["p5_pca_done"] = True

        return df

    return df
def section_summary_and_navigation(df):
    st.header("Summary")

    encoding_done = st.session_state.get("p5_encoding_done", False)
    skew_done = st.session_state.get("p5_skew_done", False)
    corr_done = st.session_state.get("p5_corr_done", False)
    pca_done = st.session_state.get("p5_pca_done", False)

    summary_present = encoding_done or skew_done or corr_done or pca_done

    with st.expander("View Summary"):
        if not summary_present:
            st.info("No transformations were performed on this page.")
        else:

            # -----------------------------------------------------------
            # 1️⃣ ENCODING  (forced first)
            # -----------------------------------------------------------
            if encoding_done:
                st.markdown("### 1️⃣ Encoding")
                for line in st.session_state["p5_summary"]:
                    if "encoded" in line.lower():
                        st.markdown(f"- {line}")
                st.markdown("---")

            # -----------------------------------------------------------
            # 2️⃣ SKEWNESS  (forced second)
            # -----------------------------------------------------------
            if skew_done:
                st.markdown("### 2️⃣ Skewness Correction")
                for col, method in st.session_state["p5_skew_handled"].items():
                    st.markdown(f"- {col} → {method}")
                st.markdown("---")

            # -----------------------------------------------------------
            # 3️⃣ CORRELATION  (forced third)
            # -----------------------------------------------------------
            if corr_done:
                st.markdown("### 3️⃣ Correlation Handling")
                for s in st.session_state["p5_summary"]:
                    if "dropped" in s.lower() or "feature" in s.lower():
                        st.markdown(f"- {s}")
                st.markdown("---")

            # -----------------------------------------------------------
            # 4️⃣ PCA  (forced last)
            # -----------------------------------------------------------
            if pca_done:
                st.markdown("### 4️⃣ PCA")

                meta = st.session_state.get("p5_pca_meta", {})
                cols = meta.get("cols", [])
                mode = meta.get("mode", None)
                k = meta.get("k", None)
                total_var = meta.get("total_var", None)
                target_var = meta.get("target_var", None)

                if cols:
                    st.markdown(f"- Selected columns: {', '.join(cols)}")

                if mode == "manual":
                    st.markdown(
                        f"- PCA (Manual Mode): {k} components "
                        f"(variance retained: {total_var:.2f}%)"
                    )
                elif mode == "variance":
                    st.markdown(
                        f"- PCA (Variance Mode): Target = {target_var}%, "
                        f"Achieved = {total_var:.2f}% "
                        f"({k} components)"
                    )

                st.markdown("---")

            st.success("All performed transformations are listed above in correct sequence.")

    # ===================== NAVIGATION BAR =====================
    st.markdown("---")
    c1, c2 = st.columns(2)

    # BACK BUTTON (same as other pages)
    with c1:
        if st.button("⬅ Back to EDA Exports"):
            st.session_state["current_page"] = "EDA Exports"
            st.rerun()

    # NEXT BUTTON — only sets warning + pending, then reruns
    with c2:
        if st.button("Next ➡ Download Center"):
            flags = {
                "Encoding": encoding_done,
                "Skewness": skew_done,
                "Correlation": corr_done,
                "PCA": pca_done,
            }

            pending = [name for name, done in flags.items() if not done]

            st.session_state["p5_page6_pending"] = pending
            st.session_state["p5_page6_warning"] = True
            st.rerun()

    # ===================== WARNING / SKIP HANDLER =====================
    if st.session_state.get("p5_page6_warning", False):
        pending = st.session_state.get("p5_page6_pending", [])

        # CASE 1: nothing pending → auto go to Download Center
        if not pending:
            st.session_state["p5_page6_warning"] = False
            st.session_state["current_page"] = "Download Center"
            st.rerun()

        # CASE 2: some steps pending → show warning + Skip / Stay
        st.warning("Some steps on this page are still pending:")

        for step in pending:
            st.write(f"- {step}")

        cA, cB = st.columns(2)

        with cA:
            if st.button("Stay on this page", key="p5_stay_here"):
                st.session_state["p5_page6_warning"] = False
                st.rerun()

        with cB:
            if st.button("Continue anyway → Download Center", key="p5_continue_anyway"):
                st.session_state["p5_page6_warning"] = False
                st.session_state["current_page"] = "Download Center"
                st.rerun()


# ======================================================
# PAGE 5 MAIN FUNCTION
# ======================================================
def run_encoding_transformation():
    st.markdown("""
    <div class="page-title-box">
        <span style="font-size:28px;font-weight:800;">🔢 Encoding & Transformation</span>
        <div style="margin-top:6px;font-size:14px;opacity:0.85;">
            Encode categorical features, correct skewness, handle correlation, and apply PCA.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    df = st.session_state.get("clean_df")
    if df is None:
        st.warning("⚠ No dataset found. Please upload data first.")
        st.stop()

    df = df.copy()
    init_page5_state()

    # Call sections in order
    df = section_encoding(df)
    st.markdown("---")

    df = section_skewness(df)
    st.markdown("---")

    df = section_correlation(df)
    st.markdown("---")

    df = section_pca(df)
    st.markdown("---")

    section_summary_and_navigation(df)

    # Save updated df
    st.session_state["clean_df"] = df.copy()
