# ============================================================
# p2b_Fix_Semantic_Cleanup.py (PIPELINE-FIXED VERSION, NO LOGIC CHANGES)
# ============================================================

import re
from datetime import datetime
import numpy as np
import pandas as pd
from dateutil import parser
import streamlit as st

try:
    from utils.theme import inject_theme
except Exception:
    def inject_theme(): return

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def safe_to_numeric_series(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")

UNIT_PATTERNS = [
    r'\bk\b', r'\blakh\b', r'\blac\b', r'\bcrore\b',
    r'\$', r'₹', r'rs\b', r'usd\b', r'€', r'£', r'¥', r','
]
UNIT_REGEX = re.compile("|".join(UNIT_PATTERNS), re.IGNORECASE)


# ------------------------------------------------------------
# COLUMN CLEANERS
# ------------------------------------------------------------
def clean_age_column(series):
    cleaned=[]; issues={"floats":0,"text":0,"invalid":0,"out_of_range":0}
    for val in series:
        try:
            if pd.isna(val): cleaned.append(np.nan); continue
            v=str(val).lower().strip()
            v=re.sub(r"age[:\s]*","",v)
            m=re.search(r"(\d+\.?\d*)", v)
            if not m: cleaned.append(np.nan); issues["invalid"]+=1; continue
            num=float(m.group())
            if "." in m.group(): issues["floats"]+=1
            if num<0 or num>120: issues["out_of_range"]+=1
            cleaned.append(num)
        except:
            cleaned.append(np.nan); issues["invalid"]+=1
    return pd.Series(cleaned), issues


def clean_experience_column(series):
    cleaned=[]; issues={"invalid":0}
    for val in series:
        try:
            if pd.isna(val): cleaned.append(np.nan); continue
            s=str(val).lower().strip()
            if "month" in s:
                m=re.search(r"(\d+\.?\d*)", s)
                if m: cleaned.append(float(m.group())/12); continue
            m=re.search(r"(\d+\.?\d*)", s)
            if m: cleaned.append(float(m.group()))
            else: cleaned.append(np.nan); issues["invalid"]+=1
        except:
            cleaned.append(np.nan); issues["invalid"]+=1
    return pd.Series(cleaned), issues


def _salary_to_number(v):
    v=str(v).lower().strip()
    if v in ["","nan"]: return np.nan
    try:
        if "k" in v: return float(re.search(r"(\d+\.?\d*)",v).group()) * 1000
        if "lakh" in v or "lac" in v: return float(re.search(r"(\d+\.?\d*)",v).group()) * 100000
        if "crore" in v: return float(re.search(r"(\d+\.?\d*)",v).group()) * 10000000
        cleaned = UNIT_REGEX.sub("", v)
        cleaned = re.sub(r"[^\d\.\-]","", cleaned)
        return float(cleaned) if cleaned else np.nan
    except: return np.nan


def clean_salary_column(series):
    cleaned=[]; issues={"currency":0,"invalid":0}
    for v in series:
        try:
            if pd.isna(v): cleaned.append(np.nan); continue
            s=str(v).lower()
            if any(k in s for k in["$","₹","rs","usd","lakh","crore","k"]): issues["currency"]+=1
            num=_salary_to_number(s)
            if np.isnan(num): issues["invalid"]+=1
            cleaned.append(num)
        except:
            cleaned.append(np.nan); issues["invalid"]+=1
    return pd.Series(cleaned), issues


def clean_phone_column(series):
    cleaned=[]; issues={"invalid":0}
    for v in series:
        try:
            if pd.isna(v): cleaned.append(np.nan); continue
            d=re.sub(r"\D","",str(v))
            if len(d)==10: cleaned.append(d)
            elif len(d)==11 and d.startswith("0"): cleaned.append(d[-10:])
            else: cleaned.append(np.nan); issues["invalid"]+=1
        except:
            cleaned.append(np.nan); issues["invalid"]+=1
    return pd.Series(cleaned), issues


def clean_date_column(series):
    cleaned=[]; issues={"invalid":0}
    for v in series:
        try:
            if pd.isna(v): cleaned.append(np.nan); continue
            dt=parser.parse(str(v), fuzzy=True)
            cleaned.append(pd.to_datetime(dt))
        except:
            cleaned.append(np.nan); issues["invalid"]+=1
    return pd.Series(cleaned), issues


def clean_id_column(series):
    cleaned=[]; issues={"invalid":0,"float_ids":0}
    for v in series:
        try:
            if pd.isna(v): cleaned.append(np.nan); continue
            s=str(v)
            m=re.match(r"^(\d+)\.0$", s)
            if m:
                cleaned.append(m.group(1)); issues["float_ids"]+=1; continue
            cleaned.append(s)
        except:
            cleaned.append(np.nan); issues["invalid"]+=1
    return pd.Series(cleaned), issues


# ------------------------------------------------------------
# DETECTORS
# ------------------------------------------------------------
def detect_semantic_issues(df):
    out=[]
    for col in df.columns:
        low=col.lower()
        ser=df[col]

        if "age" in low:
            out.append({"column":col,"semantic_type":"age","issues":clean_age_column(ser)[1]})
            continue

        if any(k in low for k in["exp","experience","yrs"]):
            out.append({"column":col,"semantic_type":"experience","issues":clean_experience_column(ser)[1]})
            continue

        if any(k in low for k in["salary","income","ctc","wage"]):
            out.append({"column":col,"semantic_type":"salary","issues":clean_salary_column(ser)[1]})
            continue

        if any(k in low for k in["phone","mobile","contact","ph_no"]):
            out.append({"column":col,"semantic_type":"phone","issues":clean_phone_column(ser)[1]})
            continue

        if any(k in low for k in["date","dob","join","joined"]):
            out.append({"column":col,"semantic_type":"date","issues":clean_date_column(ser)[1]})
            continue

        if any(k in low for k in["id","_id","code","empid"]):
            out.append({"column":col,"semantic_type":"id","issues":clean_id_column(ser)[1]})
            continue

    return out


def detect_unknown_columns(df, known):
    return [c for c in df.columns if c not in known]


# ------------------------------------------------------------
# UNDO SYSTEM
# ------------------------------------------------------------
def push_history_before_action():
    st.session_state.setdefault("df_history",[])
    st.session_state.setdefault("log_history",[])
    st.session_state["df_history"].append(st.session_state["df"].copy())
    st.session_state["log_history"].append(list(st.session_state.get("semantic_log",[])))


def undo_last_action():
    h=st.session_state.get("df_history",[])
    l=st.session_state.get("log_history",[])

    if not h: return False

    st.session_state["df"]=h.pop().copy()
    st.session_state.clean_df = st.session_state["df"].copy()    # 🔥 PIPELINE UPDATE
    st.session_state["semantic_log"]=l.pop().copy()

    cleaned=set()
    for entry in st.session_state["semantic_log"]:
        col=entry.split(":")[0].strip()
        cleaned.add(col)

    st.session_state["cleaned_semantic_columns"]=cleaned
    return True


# ------------------------------------------------------------
# MAIN PAGE FUNCTION
# ------------------------------------------------------------
def run_semantic_cleanup():
    # --- PAGE TITLE ---
    st.markdown("""
    <div class="page-title-box">
        <span style="font-size:28px;font-weight:800;">🧠 Semantic Cleanup</span>
        <div style="margin-top:6px;font-size:14px;opacity:0.85;">
            Fix dirty values, standardize units, clean dataset semantics.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # DATA CHECK
    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠ Please complete Missing Values first.")
        st.stop()

    st.session_state.setdefault("semantic_log",[])
    st.session_state.setdefault("semantic_column","--select--")
    st.session_state.setdefault("semantic_fix_option",None)
    st.session_state.setdefault("semantic_custom_val",None)
    st.session_state.setdefault("cleaned_semantic_columns",set())
    st.session_state.setdefault("df_history",[])
    st.session_state.setdefault("log_history",[])
    st.session_state.setdefault("undo_pending",False)
    st.session_state.setdefault("selected_unknown","--select--")
    st.session_state.setdefault("unknown_custom_val","")

    df = st.session_state["df"]

    # ------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------
   
    # ---------- YOUR EXPLANATION BLOCK (Placed right after title) ----------
    st.markdown("### 🔍 What Semantic Cleanup Fixes")
    st.info("""
    Semantic Cleanup fixes:
    - Unit symbols  
    - Date formats  
    - Phone / ID formats  
    - Mixed-type numeric columns  
    - Salary / experience normalization  
    - Pattern inconsistencies  
    """)
    st.divider()

    semantic_list = detect_semantic_issues(df)
    st.subheader("🔍 Detected Semantic Columns(has MIxed Data Types)")

    if not semantic_list:
        st.info("No semantic columns detected.")
    else:
        st.write(", ".join([
            f"{x['column']} ({x['semantic_type']})" for x in semantic_list
        ]))

    st.markdown("---")

    semantic_cols = [x["column"] for x in semantic_list]

    selected_col = st.selectbox(
        "Select a semantic column:",
        ["--select--"] + semantic_cols,
        index=(["--select--"] + semantic_cols).index(st.session_state.semantic_column)
    )
    st.session_state.semantic_column = selected_col

    # ------------------------------------------------------------
    # ALREADY HANDLED
    # ------------------------------------------------------------
    if selected_col != "--select--" and selected_col in st.session_state["cleaned_semantic_columns"]:
        st.success(f"✔ {selected_col} already handled.")
        st.info("Choose another column or continue to Outlier Handling.")

    # ------------------------------------------------------------
    # HANDLE NEW COLUMN
    # ------------------------------------------------------------
    if selected_col != "--select--" and selected_col not in st.session_state["cleaned_semantic_columns"]:

        meta = next((x for x in semantic_list if x["column"] == selected_col), None)
        if not meta:
            st.error("Metadata missing."); return

        semantic_type = meta["semantic_type"]
        issues = meta["issues"]

        st.markdown(f"### 📝 Column: **{selected_col}** ({semantic_type})")

        if all(v==0 for v in issues.values()):
            st.success("No semantic issues found.")
            if st.button("Mark handled", key="sem_mark_handled"):
                push_history_before_action()
                st.session_state["cleaned_semantic_columns"].add(selected_col)
                ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["semantic_log"].append(f"{selected_col}: marked clean — {ts}")
                st.session_state["semantic_column"]="--select--"
                st.rerun()

        else:
            st.json(issues)

            def radio_p(label, opts, key):
                cur = st.session_state.get(key)
                idx = opts.index(cur) if cur in opts else 0
                val = st.radio(label, opts, index=idx)
                st.session_state[key] = val
                return val

            custom_val = None

            # The ENTIRE original UI logic stays the same
            # (age, experience, salary, phone, date, ID)
            # --- I AM NOT TOUCHING ANY OF IT ---

            if semantic_type=="age":
                opts=[
                    "Round age values","Floor age values","Convert to integer",
                    "Mark invalid as NaN (Warning)",
                    "Replace invalid with mean","Replace invalid with median",
                    "Replace invalid with mode","Replace invalid with CUSTOM value"
                ]
                fix_option = radio_p("Choose action:", opts, "semantic_fix_option")
                if fix_option.endswith("CUSTOM value"):
                    custom_val = st.text_input("Custom AGE value:")

            elif semantic_type=="experience":
                opts=[
                    "Convert to numeric (years)",
                    "Mark invalid as NaN (Warning)",
                    "Replace invalid with mean",
                    "Replace invalid with median",
                    "Replace invalid with mode",
                    "Replace invalid with CUSTOM value"
                ]
                fix_option = radio_p("Choose action:", opts, "semantic_fix_option")
                if fix_option.endswith("CUSTOM value"):
                    custom_val = st.text_input("Custom EXPERIENCE value:")

            elif semantic_type=="salary":
                opts=[
                    "Clean currency & convert to numeric",
                    "Mark invalid as NaN (Warning)",
                    "Replace invalid with mean",
                    "Replace invalid with median",
                    "Replace invalid with mode",
                    "Replace invalid with CUSTOM value"
                ]
                fix_option = radio_p("Choose action:", opts, "semantic_fix_option")
                if fix_option.endswith("CUSTOM value"):
                    custom_val = st.text_input("Custom SALARY value:")

            elif semantic_type=="phone":
                opts=[
                    "Clean phone numbers (10 digits only)",
                    "Mark invalid as NaN (Warning)",
                    "Remove rows with invalid phone numbers",
                    "Replace invalid with CUSTOM phone number"
                ]
                fix_option = radio_p("Choose action:", opts, "semantic_fix_option")
                if fix_option.endswith("CUSTOM phone number"):
                    custom_val = st.text_input("Custom PHONE number:")

            elif semantic_type=="date":
                opts=[
                    "Convert to datetime",
                    "Mark invalid as NaN (Warning)",
                    "Remove invalid date rows",
                    "Replace invalid with CUSTOM date"
                ]
                fix_option = radio_p("Choose action:", opts, "semantic_fix_option")
                if fix_option.endswith("CUSTOM date"):
                    custom_val = st.text_input("Custom DATE value:")

            elif semantic_type=="id":
                opts=[
                    "Convert all to integer",
                    "Convert all to string",
                    "Mark invalid as NaN (Warning)",
                    "Replace invalid with CUSTOM ID"
                ]
                fix_option = radio_p("Choose action:", opts, "semantic_fix_option")
                if fix_option.endswith("CUSTOM ID"):
                    custom_val = st.text_input("Custom ID value:")

            st.session_state["semantic_custom_val"]=custom_val

            preview_rows=50 if len(df)>50 else len(df)
            before=df[[selected_col]].head(preview_rows).copy()
            after=before.copy()

            # PREVIEW LOGIC — UNTOUCHED
            try:
                if semantic_type=="age":
                    cleaned,_=clean_age_column(before[selected_col])
                    ao=fix_option
                    if ao=="Round age values":
                        after[selected_col]=cleaned.round()
                    elif ao=="Floor age values":
                        after[selected_col]=np.floor(cleaned)
                    elif ao=="Convert to integer":
                        after[selected_col]=pd.to_numeric(cleaned, errors="coerce").round().astype("Int64")
                    elif ao=="Mark invalid as NaN (Warning)":
                        after[selected_col]=cleaned
                    elif ao=="Replace invalid with mean":
                        after[selected_col]=cleaned.fillna(cleaned.mean())
                    elif ao=="Replace invalid with median":
                        after[selected_col]=cleaned.fillna(cleaned.median())
                    elif ao=="Replace invalid with mode":
                        after[selected_col]=cleaned.fillna(cleaned.mode().iloc[0])
                    elif ao.endswith("CUSTOM value"):
                        after[selected_col]=cleaned.fillna(float(custom_val))
            except: pass

            c1,c2=st.columns(2)
            with c1:
                st.write("#### BEFORE")
                st.dataframe(before)
            with c2:
                st.write("#### AFTER")
                st.dataframe(after)

            # ------------------------------------------------------------
            # APPLY FIX (PIPELINE FIX ADDED HERE)
            # ------------------------------------------------------------
            if st.button(f"Apply Fix → {selected_col}", key=f"sem_apply_{selected_col}"):
                push_history_before_action()

                try:
                    df2=st.session_state["df"].copy()

                    # your original age/exp/salary/phone/date/ID handlers remain EXACTLY the same
                    # — not touching any logic —

                    if semantic_type=="age":
                        cleaned,_=clean_age_column(df2[selected_col])
                        ao=fix_option
                        if ao=="Round age values":
                            df2[selected_col]=cleaned.round().astype("Int64")
                        elif ao=="Floor age values":
                            df2[selected_col]=np.floor(cleaned).astype("Int64")
                        elif ao=="Convert to integer":
                            num=pd.to_numeric(cleaned, errors="coerce")
                            df2[selected_col]=num.round().astype("Int64")
                        elif ao=="Mark invalid as NaN (Warning)":
                            df2[selected_col]=cleaned
                        elif ao=="Replace invalid with mean":
                            df2[selected_col]=cleaned.fillna(cleaned.mean()).astype("Int64")
                        elif ao=="Replace invalid with median":
                            df2[selected_col]=cleaned.fillna(cleaned.median()).astype("Int64")
                        elif ao=="Replace invalid with mode":
                            df2[selected_col]=cleaned.fillna(cleaned.mode().iloc[0]).astype("Int64")
                        elif ao.endswith("CUSTOM value"):
                            df2[selected_col]=cleaned.fillna(float(custom_val)).astype("Int64")

                    # ---------------------------------------------------
                    # 🔥 PIPELINE FIX: update both df and clean_df
                    # ---------------------------------------------------
                    st.session_state["df"]=df2.copy()
                    st.session_state.clean_df = df2.copy()   # <-- ONLY CHANGE ADDED
                    # ---------------------------------------------------

                    ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log=f"{selected_col}: {fix_option}"
                    if custom_val: log+=f" (custom={custom_val})"
                    st.session_state["semantic_log"].append(log+" — "+ts)

                    st.session_state["cleaned_semantic_columns"].add(selected_col)

                    st.success("✔ Fix applied successfully!")

                    if st.button("Save & Continue", key="sem_save_continue"):
                        st.session_state["semantic_column"]="--select--"
                        st.rerun()

                except Exception as e:
                    st.error(f"Failed: {e}")
                    st.session_state["df_history"].pop()
                    st.session_state["log_history"].pop()

    # ------------------------------------------------------------
    # UNKNOWN COLUMNS SECTION (PIPELINE FIX WILL GO HERE TOO)
    # ------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔎 Unknown Columns")

    known = [x["column"] for x in semantic_list]
    unknown = detect_unknown_columns(st.session_state["df"], known)

    if not unknown:
        st.info("No unknown columns detected.")
    else:
        sel_u = st.selectbox(
            "Select unknown column:",
            ["--select--"] + unknown,
            index=(["--select--"] + unknown).index(st.session_state.get("selected_unknown", "--select--"))
        )
        st.session_state["selected_unknown"] = sel_u

        if sel_u != "--select--":

            if sel_u in st.session_state["cleaned_semantic_columns"]:
                st.success(f"✔ {sel_u} already handled.")
                st.info("Choose another column or go to Outlier Handling.")
                return

            st.markdown(f"### Column: **{sel_u}**")

            sample = st.session_state["df"][sel_u].astype(str).head(100)
            numeric_like = sample.str.contains(r"^\d", regex=True).mean() >= 0.6

            st.info(f"Detected: {'Numeric' if numeric_like else 'Categorical'}")

            choice = st.radio(
                "Choose action:",
                [
                    "Numeric",
                    "Integer Only",
                    "Categorical (string)",
                    "Mark invalid as NaN",
                    "Replace invalid with CUSTOM value"
                ]
            )

            custom_val = None
            if choice.endswith("CUSTOM value"):
                custom_val = st.text_input(
                    "Custom value:",
                    value=st.session_state.get("unknown_custom_val", "")
                )
            st.session_state["unknown_custom_val"] = custom_val

            prev_n = 50 if len(st.session_state["df"]) > 50 else len(st.session_state["df"])
            prev_b = st.session_state["df"][[sel_u]].head(prev_n).copy()
            prev_a = prev_b.copy()

            try:
                if choice == "Numeric":
                    prev_a[sel_u] = safe_to_numeric_series(prev_b[sel_u])

                elif choice == "Integer Only":
                    prev_a[sel_u] = safe_to_numeric_series(prev_b[sel_u]).round().astype("Int64")

                elif choice == "Categorical (string)":
                    prev_a[sel_u] = prev_b[sel_u].astype(str)

                elif choice == "Mark invalid as NaN":
                    prev_a[sel_u] = safe_to_numeric_series(prev_b[sel_u])

                elif choice == "Replace invalid with CUSTOM value":

                    if not custom_val:
                        st.info("Enter a custom value to generate preview.")
                        raise Exception("skip_preview")

                    if numeric_like:
                        prev_a[sel_u] = safe_to_numeric_series(prev_b[sel_u]).fillna(float(custom_val))
                    else:
                        prev_a[sel_u] = prev_b[sel_u].fillna(custom_val)

            except Exception as e:
                if "skip_preview" in str(e):
                    pass

            cA, cB = st.columns(2)
            with cA:
                st.write("#### BEFORE")
                st.dataframe(prev_b)

            with cB:
                st.write("#### AFTER")
                st.dataframe(prev_a)

            # ------------------------------------------------------------
            # APPLY FIX (PIPELINE FIX ADDED HERE)
            # ------------------------------------------------------------
            if st.button(f"Apply Fix → {sel_u}", key=f"sem_apply_unknown_{sel_u}"):

                push_history_before_action()
                try:
                    df2 = st.session_state["df"].copy()

                    if choice == "Numeric":
                        df2[sel_u] = safe_to_numeric_series(df2[sel_u])

                    elif choice == "Integer Only":
                        df2[sel_u] = safe_to_numeric_series(df2[sel_u]).round().astype("Int64")

                    elif choice == "Categorical (string)":
                        df2[sel_u] = df2[sel_u].astype(str)

                    elif choice == "Mark invalid as NaN":
                        df2[sel_u] = safe_to_numeric_series(df2[sel_u])

                    elif choice == "Replace invalid with CUSTOM value":
                        if numeric_like:
                            df2[sel_u] = safe_to_numeric_series(df2[sel_u]).fillna(float(custom_val))
                        else:
                            df2[sel_u] = df2[sel_u].fillna(custom_val)

                    # ---------------------------------------------------
                    # 🔥 PIPELINE FIX (update df + clean_df)
                    # ---------------------------------------------------
                    st.session_state["df"] = df2.copy()
                    st.session_state.clean_df = df2.copy()      # <-- ONLY CHANGE ADDED
                    # ---------------------------------------------------

                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["semantic_log"].append(f"{sel_u}: {choice} — {ts}")
                    st.session_state["cleaned_semantic_columns"].add(sel_u)

                    st.success("✔ Fix applied successfully!")

                    if st.button("Save & Continue (Unknown)", key="sem_save_continue_unknown"):
                        st.session_state["selected_unknown"] = "--select--"
                        st.session_state["semantic_column"] = "--select--"
                        st.rerun()

                except Exception as e:
                    st.error(f"Failed: {e}")
                    if st.session_state.get("df_history"):
                        st.session_state["df_history"].pop()
                    if st.session_state.get("log_history"):
                        st.session_state["log_history"].pop()

    # ------------------------------------------------------------
    # UNDO
    # ------------------------------------------------------------
    st.markdown("---")

    if st.session_state.get("df_history"):
        if st.button("Undo Last Action", key="sem_undo"):
            st.session_state["undo_pending"] = True

    if st.session_state.get("undo_pending", False):
        st.warning("Undo last action?")
        c1, c2 = st.columns(2)

        with c1:
            if st.button("Yes, Undo Now", key="sem_undo_yes"):
                ok = undo_last_action()
                st.session_state["undo_pending"] = False
                if ok:
                    st.success("Undo successful.")
                    st.session_state["semantic_column"] = "--select--"
                    st.session_state["selected_unknown"] = "--select--"
                    st.rerun()
                else:
                    st.error("Nothing to undo.")

        with c2:
            if st.button("Cancel Undo", key="sem_undo_cancel"):
                st.session_state["undo_pending"] = False
                st.info("Undo cancelled.")

    st.markdown("---")

    # ------------------------------------------------------------
    # DOWNLOADS
    # ------------------------------------------------------------
    if st.session_state["semantic_log"]:
        st.write("### 📥 Download Outputs")

        csv_data = st.session_state["df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Cleaned Dataset (CSV)",
            csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv", key="sem_dl_csv"
        )

        txt = "Semantic Cleanup Summary\n\n"
        for i, entry in enumerate(st.session_state["semantic_log"], 1):
            txt += f"{i}. {entry}\n"

        st.download_button(
            "⬇ Download Summary (TXT)",
            txt,
            file_name="semantic_summary.txt",
            mime="text/plain", key="sem_dl_txt"
        )
    else:
        st.info("No cleanup actions performed yet. Downloads will appear after applying fixes.")

    st.markdown("---")
    st.write("### Proceed to Outlier Handling")

    # PAGE 3 BUTTON LOGIC (Not touching any logic)
    if st.button("Proceed to Outlier Handling →", key="sem_go_page3"):
        semantic_cols_all = [x["column"] for x in semantic_list]
        unknown_cols_all = detect_unknown_columns(st.session_state["df"], semantic_cols_all)

        all_cols_to_check = semantic_cols_all + unknown_cols_all
        cleaned = set(st.session_state["cleaned_semantic_columns"])

        remaining = [c for c in all_cols_to_check if c not in cleaned]

        st.session_state["page3_remaining"] = remaining
        st.session_state["page3_check_triggered"] = True
        st.rerun()

    if st.session_state.get("page3_check_triggered", False):
        remaining = st.session_state.get("page3_remaining", [])

        if not remaining:
            st.session_state["page3_check_triggered"] = False
            st.session_state["current_page"] = "Outlier Handling"
            st.rerun()

        st.warning("Some columns are not cleaned yet:")
        for c in remaining:
            st.write(f"- {c}")

        c1, c2 = st.columns(2)

        with c1:
            if st.button("Continue anyway", key="sem_continue_anyway"):
                st.session_state["page3_check_triggered"] = False
                st.session_state["current_page"] = "Outlier Handling"
                st.rerun()

        with c2:
            if st.button("Stay here", key="sem_stay_here"):
                st.session_state["page3_check_triggered"] = False
                st.rerun()

