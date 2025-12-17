# ---------------------------------------------------------------
# PAGE 6 — DOWNLOAD CENTER (Final, with EDA Core + Exports Summary)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime
import plotly.express as px
import textwrap
import base64

# ReportLab for PDF (simple, readable)
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# ===================================================================
# 🔵 HELPERS: DF → download formats
# ===================================================================
def df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel(df: pd.DataFrame) -> io.BytesIO:
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer


def df_to_json(df: pd.DataFrame) -> bytes:
    return df.to_json(orient="records").encode("utf-8")


# ===================================================================
# 🔵 (OPTIONAL) WRAP TEXT HELPER (still useful if needed later)
# ===================================================================
def wrap_text(text: str, width: int = 100) -> str:
    """
    Wrap long lines into multiple shorter lines.
    (Kept in case we need it for any other export.)
    """
    if not text:
        return ""
    safe = text.replace("\t", "    ")
    wrapped_lines = textwrap.wrap(safe, width=width)
    if not wrapped_lines:
        return safe
    return "\n".join(wrapped_lines)


# ===================================================================
# 🔵 HELPER: Build TEXT SUMMARY (Pages 2 → 5 + EDA Core + Exports)
# ===================================================================
def build_text_summary() -> str:
    """
    Text summary for all preprocessing steps.
    This drives:
      - Overall Summary expander
      - HTML Summary export
      - TXT / PDF Summary
      - Full Report textual section
    """
    lines = []
    now_str = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    lines.append("PREPROCESSING TOOL — SUMMARY REPORT")
    lines.append(f"Generated at: {now_str}")
    lines.append("=" * 60)
    lines.append("")

    # ---------- DATASET INFO ----------
    df = st.session_state.get("clean_df")
    if df is not None:
        lines.append("DATASET OVERVIEW")
        lines.append("-" * 60)
        lines.append(f"- Final Shape : {df.shape[0]} rows × {df.shape[1]} columns")
    else:
        lines.append("DATASET OVERVIEW")
        lines.append("-" * 60)
        lines.append("- No dataset found in session.")
    lines.append("")

    summaries = st.session_state.get("summaries", {})

    # ---------- PAGE 2 — MISSING VALUES ----------
    mv_info = summaries.get("missing", {}).get("last_action")

    lines.append("PAGE 2 — Fix Missing Values")
    lines.append("-" * 60)
    if mv_info:
        init_shape = mv_info.get("Initial Shape") or mv_info.get("initial_shape")
        final_shape = mv_info.get("Final Shape") or mv_info.get("final_shape")
        dropped = mv_info.get("Dropped Columns", [])
        filled_cols = mv_info.get("Filled Columns", {}) or mv_info.get("filled", {})

        if init_shape and final_shape:
            lines.append(f"- Initial Shape : {init_shape[0]} × {init_shape[1]}")
            lines.append(f"- Final Shape   : {final_shape[0]} × {final_shape[1]}")

        if dropped:
            lines.append(f"- Dropped Columns : {', '.join(dropped)}")

        if filled_cols:
            lines.append("- Filled Columns:")
            for col, meta in filled_cols.items():
                method = meta.get("method", "Unknown")
                lines.append(f"    - {col}: {method}")
    else:
        lines.append("- No missing-value operations recorded.")
    lines.append("")

    # ---------- PAGE 2.5 — SEMANTIC CLEANUP ----------
    semantic_log = st.session_state.get("semantic_log", [])

    lines.append("PAGE 2.5 — Semantic Cleanup")
    lines.append("-" * 60)
    if semantic_log:
        lines.append("- Semantic cleanup actions:")
        for entry in semantic_log:
            lines.append(f"    - {entry}")
    else:
        lines.append("- No semantic cleanup actions recorded.")
    lines.append("")

    # ---------- PAGE 3 — OUTLIER HANDLING ----------
    outlier_report = st.session_state.get("outlier_report", [])

    lines.append("PAGE 3 — Outlier Handling")
    lines.append("-" * 60)
    if outlier_report:
        lines.append("- Outlier actions applied:")
        for entry in outlier_report:
            lines.append(f"    - {entry}")
    else:
        lines.append("- No outlier actions recorded.")
    lines.append("")



    # ---------- PAGE 5 — ENCODING & TRANSFORMATION ----------
    p5_summaries = st.session_state.get("p5_summary", [])
    skew = st.session_state.get("p5_skew_handled", {})
    pca_meta = st.session_state.get("p5_pca_meta", {})

    lines.append("PAGE 5 — Encoding & Transformation")
    lines.append("-" * 60)

    # Split p5_summaries into categories so order is fixed:
    enc_ops = []
    corr_ops = []

    for s in p5_summaries or []:
        low = s.lower()
        if "encoded" in low:
            enc_ops.append(s)
        elif "dropped" in low or "feature" in low:
            # correlation handling / feature engineering
            corr_ops.append(s)
        else:
            # skewness / other messages are ignored here
            # because we have a dedicated skewness block below
            pass

    # 1️⃣ Encoding
    if enc_ops:
        lines.append("- Encoding operations:")
        for s in enc_ops:
            lines.append(f"    - {s}")

    # 2️⃣ Skewness
    if skew:
        lines.append("- Skewness corrections:")
        for col, method in skew.items():
            lines.append(f"    - {col}: {method}")

    # 3️⃣ Correlation / Feature Engineering
    if corr_ops:
        lines.append("- Correlation / feature engineering actions:")
        for s in corr_ops:
            lines.append(f"    - {s}")

    # 4️⃣ PCA
    if pca_meta:
        mode = pca_meta.get("mode", "manual")
        k = pca_meta.get("k")
        total_var = pca_meta.get("total_var")
        target_var = pca_meta.get("target_var")

        if mode == "manual":
            lines.append(
                f"- PCA (Manual Mode): {k} components, "
                f"variance retained ≈ {total_var:.2f}%"
            )
        else:
            lines.append(
                f"- PCA (Variance Mode): target = {target_var}%, "
                f"achieved ≈ {total_var:.2f}% with {k} components"
            )

    if not (enc_ops or skew or corr_ops or pca_meta):
        lines.append("- No encoding / transformation actions recorded.")

    lines.append("")
    lines.append("=" * 60)
    lines.append("END OF SUMMARY")
    return "\n".join(lines)


# ===================================================================
# 🔵 HELPER: HTML SUMMARY (full HTML file for export)
# ===================================================================
def build_html_summary(df: pd.DataFrame) -> str:
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    text = build_text_summary()

    html = f"""
    <html>
    <head>
        <meta charset="utf-8" />
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 20px;
            }}
            h1 {{
                color: #1a73e8;
            }}
            pre {{
                background: #f5f5f5;
                padding: 12px;
                border-radius: 8px;
                white-space: pre-wrap;
            }}
        </style>
    </head>
    <body>
        <h1>Preprocessing Tool — Summary Report</h1>
        <p>Generated at: {timestamp}</p>
        <h2>Dataset Overview</h2>
        <p><b>{df.shape[0]}</b> rows × <b>{df.shape[1]}</b> columns</p>
        <h2>Details</h2>
        <pre>{text}</pre>
    </body>
    </html>
    """
    return html





# ===================================================================
# 🔵 HELPER: FULL HTML REPORT (with EDA charts)
# ===================================================================
def build_full_html_report(df: pd.DataFrame) -> str:
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    text_summary = build_text_summary()
    eda_charts_html = build_eda_charts_html(df)

    html = f"""
    <html>
    <head>
        <meta charset="utf-8" />
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 20px;
            }}
            h1 {{
                color: #1a73e8;
            }}
            h2 {{
                margin-top: 28px;
                color: #1a73e8;
            }}
            .box {{
                background: #f0f3f8;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }}
            pre {{
                background: #f5f5f5;
                padding: 12px;
                border-radius: 8px;
                white-space: pre-wrap;
            }}
        </style>
    </head>
    <body>
        <h1>Full Report — Preprocessing Tool</h1>
        <p>Generated at: {timestamp}</p>

        <h2>Dataset Preview (Top 50 rows)</h2>
        <div class="box">
            {df.head(50).to_html(index=False)}
        </div>

        <h2>Full Summary (Pages 2 → 5, EDA Core & Exports)</h2>
        <div class="box">
            <pre>{text_summary}</pre>
        </div>

        {eda_charts_html}
    </body>
    </html>
    """
    return html


# ===================================================================
# 🔵 SIMPLE REPORTLAB PDF SUMMARY
# ===================================================================
def generate_pdf_summary(text_summary: str) -> bytes:
    """
    Build a simple, readable PDF using ReportLab.
    One line per paragraph with basic spacing.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    for line in text_summary.split("\n"):
        clean = line.replace("\u2014", "-")
        if clean.strip() == "":
            story.append(Spacer(1, 8))
        else:
            story.append(Paragraph(clean, styles["Normal"]))
            story.append(Spacer(1, 4))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ===================================================================
# PAGE 6 — MAIN FUNCTION
# ===================================================================
def run_download_center():
    st.markdown("""
    <div class="page-title-box">
        <span style="font-size:28px;font-weight:800;">📥 Download Center</span>
        <div style="margin-top:6px;font-size:14px;opacity:0.85;">
            Export cleaned datasets, summaries, and full preprocessing reports.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    df = st.session_state.get("clean_df")
    if df is None:
        st.warning("⚠ No dataset found. Please upload data first.")
        st.stop()

    # ---- init Page-6 state for persistent full report ----
    st.session_state.setdefault("p6_full_report_ready", False)
    st.session_state.setdefault("p6_full_report_html", "")

    # =========================================================
    # 1️⃣ DATASET PREVIEW + DOWNLOAD DATASET
    # =========================================================
    st.subheader("Dataset Preview (Top 50 rows)")
    st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
    st.write(f"Generated at: **{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}**")
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("### Download Cleaned Dataset")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download CSV",
            data=df_to_csv(df),
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )
    with c2:
        st.download_button(
            "Download Excel",
            data=df_to_excel(df),
            file_name="cleaned_dataset.xlsx",
            mime="application/vnd.ms-excel"
        )
    with c3:
        st.download_button(
            "Download JSON",
            data=df_to_json(df),
            file_name="cleaned_dataset.json",
            mime="application/json"
        )

    st.markdown("---")

    # =========================================================
    # 2️⃣ OVERALL SUMMARY (Expander)
    # =========================================================
    with st.expander("Summary of All Applied Transformations"):
        text_summary = build_text_summary()
        st.markdown(text_summary.replace("\n", "<br>"), unsafe_allow_html=True)

    st.markdown("---")

    # =========================================================
    # 3️⃣ EXPORT SUMMARY OPTIONS (HTML / TXT / JSON / PDF)
    # =========================================================
    st.header("Export Summary")

    st.markdown("""
    **Formats Available:**  
    - **HTML Summary** — (Nicely formatted, readable)  
    - **TXT Summary** — (Plain text, console-style)  
    - **JSON Summary** — (Full machine-readable metadata)  
    - **PDF Summary** — (Simple PDF — only textual summary)
    """)

    # Build summaries
    text_summary = build_text_summary()
    html_summary = build_html_summary(df)

    summaries = st.session_state.get("summaries", {})
    json_payload = {
        "generated_at": datetime.now().isoformat(),
        "dataset_shape": list(df.shape),
        "missing_values": summaries.get("missing", {}),
        "semantic_cleanup": st.session_state.get("semantic_log", []),
        "outliers": st.session_state.get("outlier_report", []),
        "encoding_transformation": {
            "steps": st.session_state.get("p5_summary", []),
            "skewness": st.session_state.get("p5_skew_handled", {}),
            "pca_meta": st.session_state.get("p5_pca_meta", {}),
        },
    }
    json_summary = json.dumps(json_payload, indent=2)

    # Download buttons (summary exports)
    st.download_button(
        "Download HTML Summary",
        data=html_summary,
        file_name="summary.html",
        mime="text/html"
    )

    st.download_button(
        "Download TXT Summary",
        data=text_summary,
        file_name="summary.txt",
        mime="text/plain"
    )

    st.download_button(
        "Download JSON Summary",
        data=json_summary,
        file_name="summary.json",
        mime="application/json"
    )

    # ---- PDF Summary (ReportLab) ----
    pdf_buffer = generate_pdf_summary(text_summary)

    st.download_button(
        "Download PDF Summary",
        data=pdf_buffer,
        file_name="summary.pdf",
        mime="application/pdf"
    )

    st.markdown("---")


    # =========================================================
    # 5️⃣ NAVIGATION
    # =========================================================
    c1, c2 = st.columns(2)

    with c1:
        if st.button("⬅ Back to Page 5"):
            st.session_state["current_page"] = "Encoding & Transformation"
            st.rerun()

    with c2:
        if st.button("Finish / Go to Home"):
            st.session_state["current_page"] = "Home"
            st.rerun()
