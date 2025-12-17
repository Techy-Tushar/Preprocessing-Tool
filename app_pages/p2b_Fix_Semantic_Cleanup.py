# p2b_Fix_Semantic_Cleanup.py
# ===========================================================
# Semantic Cleanup — Multi-column batch cleanup (final)
# - Multi-column expanders (first expanded) with pattern grouping
# - Per-pattern action selection + checkbox to include in batch
# - Live preview (top 10) for each pattern/action
# - Collected Actions Summary (live)
# - Apply All with progress bar and optional micro-sleep
# - Undo resets to df state right after missing-values page
# - Uses st.rerun() and preserves header/info box as requested
# ===========================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import time
from datetime import datetime

# ---------------------------
# Utilities (pure functions)
# ---------------------------
WORDS_TO_NUM = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
    "six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,
    "twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
    "sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,
    "twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,
    "seventy":70,"eighty":80,"ninety":90,"hundred":100
}
NULL_LIKE = {"", "nan", "null", "na", "none", "n/a"}

def word_to_num(text):
    if not text: return None
    s = str(text).lower().strip()
    if s in WORDS_TO_NUM:
        return WORDS_TO_NUM[s]
    parts = re.split(r"[\s\-]+", s)
    total = 0
    used = False
    for p in parts:
        if p in WORDS_TO_NUM:
            if p == "hundred":
                if total == 0:
                    total = 100
                else:
                    total = total * 100
            else:
                total += WORDS_TO_NUM[p]
            used = True
        else:
            return None
    return int(total) if used else None

def remove_special_chars_keep_digits(s):
    if s is None: return ""
    s = str(s)
    s = re.sub(r"[,\s₹$€£¥]", "", s)
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return m.group(0) if m else ""

def extract_first_number_from_string(s):
    if s is None: return None
    s_str = str(s).strip()
    digits = remove_special_chars_keep_digits(s_str)
    if digits:
        try:
            return float(digits)
        except:
            pass
    first = s_str.split()[0].lower() if s_str.split() else ""
    wn = word_to_num(first)
    if wn is not None:
        return float(wn)
    return None

# Classification helpers
def is_numeric_like(v):
    if v is None: return False
    s = str(v).strip()
    if s.lower() in NULL_LIKE: return False
    s2 = re.sub(r"[,\s₹$€£¥]", "", s)
    return bool(re.match(r"^[-+]?\d+(\.\d+)?$", s2))

def analyze_column_basic(series, sample_n=300):
    s = series.dropna().astype(str).head(sample_n)
    total = len(s)
    if total == 0:
        return "Unknown", "No usable values"
    cnt_num = sum(is_numeric_like(x) for x in s)
    pct_num = cnt_num / total
    pct_text = 1 - pct_num
    if pct_num >= 0.7:
        return "Numeric", f"✔ {round(pct_num*100)}% numeric-like"
    elif pct_text >= 0.7:
        return "Categorical", f"✔ {round(pct_text*100)}% text-like"
    else:
        return "Mixed", f"⚠ Mixed: {round(pct_num*100)}% numeric-like"

# Pattern detection / grouping
RANGE_RE = re.compile(r"^\s*\d+\.?\d*\s*[-–]\s*\d+\.?\d*\s*$")
COMPARE_RE = re.compile(r"^\s*(<=|>=|<|>)\s*\+?\d+\.?\d*\s*$")
PURE_NUM_RE = re.compile(r"^\s*\d+(\.\d+)?\s*$")
UNIT_PATTERN = ["year", "yr", "month", "mo", "day", "d"]

def canonical_keyword_core(val):
    if val is None: return ""
    s = re.sub(r"[^A-Za-z]", "", str(val)).lower().strip()
    return s

def detect_pattern_key(val):
    if val is None: return "missing"
    s = str(val).strip()
    s_low = s.lower()
    if s_low in NULL_LIKE:
        return "missing"
    # exact pure numbers
    if PURE_NUM_RE.match(s_low):
        return "pure_numbers"
    # ranges like 10-20
    if RANGE_RE.match(s_low):
        return "range_pattern"
    # comparisons like <10, <=5, >20, >=3 or trailing +
    if COMPARE_RE.match(s_low) or s_low.endswith("+"):
        return "comparison"
    # unit patterns (years/months/days)
    for u in UNIT_PATTERN:
        if re.search(rf"\b\d+\.?\d*\s*{u}s?\b", s_low):
            if "year" in u or "yr" in u: return "X years"
            if "month" in u or "mo" in u: return "X months"
            if "day" in u or u == "d": return "X days"
    # currency-like
    if re.search(r"[\₹\$\£\€]|k\b|lakh|lac|crore", s_low):
        return "currency_like"
    # mixed text+number
    if re.search(r"\d", s) and re.search(r"[A-Za-z]", s):
        return "mixed_text_number"
    # spacing issues
    if s != s.strip():
        return "leading_trailing_spaces"
    if re.search(r"\s{2,}", s):
        return "multiple_inner_spaces"
    # special characters
    if re.search(r"[^\w\s]", s):
        return "special_characters_present"
    # pure text
    if re.search(r"[A-Za-z]", s) and not re.search(r"\d", s):
        return "pure_text"
    return "other"

def group_by_pattern(series):
    total = len(series)
    vc = series.astype(str).value_counts(dropna=False)
    groups = {}
    for val, cnt in vc.items():
        key = detect_pattern_key(val)
        groups.setdefault(key, {"values": [], "count": 0})
        groups[key]["values"].append(val)
        groups[key]["count"] += int(cnt)

    # keyword-core grouping for textual groups
    cores = {}
    for k in ["pure_text", "special_characters_present", "leading_trailing_spaces", "multiple_inner_spaces"]:
        if k in groups:
            for v in list(groups[k]["values"]):
                core = canonical_keyword_core(v)
                if not core: continue
                cores.setdefault(core, set()).add(v)

    for core, vals in list(cores.items()):
        if len(vals) >= 2:
            group_name = f"keyword:{core}"
            groups.setdefault(group_name, {"values": [], "count": 0})
            for v in vals:
                prev_key = detect_pattern_key(v)
                if v in groups.get(prev_key, {}).get("values", []):
                    groups[prev_key]["values"].remove(v)
                    groups[prev_key]["count"] -= vc.get(str(v), 0)
                groups[group_name]["values"].append(v)
                groups[group_name]["count"] += vc.get(str(v), 0)

    keys_to_delete = [k for k,meta in groups.items() if meta["count"] <= 0 or not meta["values"]]
    for k in keys_to_delete:
        groups.pop(k, None)
    for k in list(groups.keys()):
        groups[k]["pct"] = groups[k]["count"] / max(1, total)
        groups[k]["values"] = sorted(groups[k]["values"], key=lambda v: -vc.get(str(v), 0))
    ordered = sorted(groups.items(), key=lambda x: -x[1]["pct"])
    return ordered

# ---------------------------
# Action Labels (with bracket descriptions)
# ---------------------------
def ACTION_LABELS():
    return {
        # generic
        "keep": "keep (no change)",
        "replace_custom": "custom (replace with custom value)",

        # numeric extraction
        "to_int": "to_int (convert to integer)",
        "to_float": "to_float (convert to float)",
        "extract_numeric": "extract_numeric (extract numeric part)",

        # range options
        "range_keep": "range_keep (keep as-is)",
        "range_min": "range_min (min — keep minimum value)",
        "range_max": "range_max (max — keep maximum value)",
        "range_avg": "range_avg (average — convert range to mean)",
        "range_custom": "range_custom (custom — replace with custom value)",
        "range_mixed": "range_mixed (mixed_range — fill each row with random value inside range)",

        # comparison
        "comparison_keep": "comparison_keep (keep as-is)",
        "comparison_ignore_sign": "comparison_ignore_sign (ignore sign — convert to numeric)",
        "comparison_ceiling_floor": "comparison_ceiling_floor (ceiling/floor based on sign)",

        # special characters
        "special_keep": "special_keep (keep as-is)",
        "special_remove_special": "special_remove_special (remove special characters)",
        "special_first": "special_first (extract first numeric value)",
        "special_last": "special_last (extract last numeric value)",
        "special_arith": "special_arith (arithmetic between extracted numbers)",

        # text operations
        "remove_numbers": "remove_numbers (remove digits from text)",
        "remove_special_characters": "remove_special_characters (remove punctuation & symbols)",
        "convert_to_lowercase": "lowercase (convert text to lowercase)",
        "convert_to_uppercase": "uppercase (convert text to uppercase)",
        "strip_spaces": "strip_spaces (trim leading & trailing spaces)",
        "remove_all_spaces": "remove_all_spaces (remove every space)"
    }

# ---------------------------
# Transformations (extended)
# ---------------------------
def transform_series_for_group(series, group_values_set, action, custom_val=None, extra=None):
    """
    series: pd.Series OR pd.Series subset (with original index)
    group_values_set: iterable of original string values to transform
    action: key from ACTION_LABELS
    custom_val: optional custom replacement
    extra: dict for additional params (special_arith: {"op": "+","order":"a_b"/"b_a"})
    """
    # We accept sub-series (indexed). Work on copy to preserve index.
    s = series.copy()

    mask = s.astype(str).isin(set(group_values_set))
    if not mask.any():
        return s

    # helpers
    def parse_first_and_second_nums(text):
        text = "" if text is None else str(text)
        nums = re.findall(r"[-+]?\d+\.?\d*", text)
        if not nums:
            return (None, None)
        if len(nums) == 1:
            return (float(nums[0]), None)
        return (float(nums[0]), float(nums[1]))

    def parse_range_vals(text):
        text = "" if text is None else str(text)
        m = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)", text)
        if not m:
            nums = re.findall(r"\d+\.?\d*", text)
            if len(nums) >= 2:
                return (float(nums[0]), float(nums[1]))
            if len(nums) == 1:
                return (float(nums[0]), float(nums[0]))
            return (None, None)
        return (float(m.group(1)), float(m.group(2)))

    rng = np.random.default_rng()

    # Range actions
    if action == "range_keep":
        return s

    if action == "range_min":
        def _min(v):
            a, b = parse_range_vals(v)
            if a is None: return np.nan
            return int(a) if float(a).is_integer() else a
        s.loc[mask] = s.loc[mask].apply(_min)
        return s

    if action == "range_max":
        def _max(v):
            a, b = parse_range_vals(v)
            if b is None: return np.nan
            return int(b) if float(b).is_integer() else b
        s.loc[mask] = s.loc[mask].apply(_max)
        return s

    if action == "range_avg":
        def _avg(v):
            a, b = parse_range_vals(v)
            if a is None or b is None: return np.nan
            avg = (a + b) / 2.0
            return int(round(avg)) if float(avg).is_integer() else avg
        s.loc[mask] = s.loc[mask].apply(_avg)
        return s

    if action == "range_custom":
        s.loc[mask] = custom_val
        return s

    if action == "range_mixed":
        def _mixed(v):
            a, b = parse_range_vals(v)
            if a is None or b is None:
                return np.nan
            amin = int(math.ceil(min(a, b)))
            bmax = int(math.floor(max(a, b)))
            if amin > bmax:
                return int(amin)
            return int(rng.integers(amin, bmax + 1))
        s.loc[mask] = s.loc[mask].apply(_mixed)
        return s

    # Comparison actions
    if action == "comparison_keep":
        return s

    if action == "comparison_ignore_sign":
        def _ign(v):
            a, _ = parse_first_and_second_nums(v)
            if a is None: return np.nan
            return int(a) if float(a).is_integer() else a
        s.loc[mask] = s.loc[mask].apply(_ign)
        return s

    if action == "comparison_ceiling_floor":
        def _cf(v):
            txt = "" if v is None else str(v).strip()
            a, _ = parse_first_and_second_nums(txt)
            if a is None: return np.nan
            if txt.startswith("<=") or txt.startswith(">="):
                return int(a)
            if txt.startswith("<"):
                val = int(a) - 1
                return val
            if txt.startswith(">"):
                return int(a) + 1
            if txt.endswith("+"):
                return int(a) + 1
            # fallback
            return int(a)
        s.loc[mask] = s.loc[mask].apply(_cf)
        return s

    # Special characters actions
    if action == "special_keep":
        return s

    if action == "special_remove_special":
        def _rm(v):
            return re.sub(r"[^\dA-Za-z\s]", "", str(v))
        s.loc[mask] = s.loc[mask].apply(_rm)
        return s

    if action == "special_first":
        def _first(v):
            a, b = parse_first_and_second_nums(v)
            if a is None: return np.nan
            return int(a) if float(a).is_integer() else a
        s.loc[mask] = s.loc[mask].apply(_first)
        return s

    if action == "special_last":
        def _last(v):
            nums = re.findall(r"[-+]?\d+\.?\d*", str(v))
            if not nums: return np.nan
            val = float(nums[-1])
            return int(val) if float(val).is_integer() else val
        s.loc[mask] = s.loc[mask].apply(_last)
        return s

    if action == "special_arith":
        op = extra.get("op") if extra else "+"
        order = extra.get("order") if extra else "a_b"
        def _arith(v):
            nums = re.findall(r"[-+]?\d+\.?\d*", str(v))
            if len(nums) < 2:
                return np.nan
            a = float(nums[0]); b = float(nums[1])
            first, second = (a, b) if order == "a_b" else (b, a)
            try:
                if op == "+":
                    res = first + second
                elif op == "-":
                    res = first - second
                elif op == "*":
                    res = first * second
                elif op == "/":
                    if second == 0:
                        return np.nan
                    res = first / second
                else:
                    return np.nan
                return int(res) if float(res).is_integer() else res
            except:
                return np.nan
        s.loc[mask] = s.loc[mask].apply(_arith)
        return s

    # Fallback numeric actions (extract_first_number based)
    def _extract_num(v):
        num = extract_first_number_from_string(v)
        return num

    if action in {"to_int", "to_float", "extract_numeric"}:
        def _num_action(v):
            num = _extract_num(v)
            if num is None:
                return np.nan
            if action == "to_int":
                try:
                    return int(round(num))
                except:
                    return np.nan
            if action == "to_float":
                try:
                    return float(num)
                except:
                    return np.nan
            if action == "extract_numeric":
                return num
        s.loc[mask] = s.loc[mask].apply(_num_action)
        return s

    # Text operations
    if action == "remove_numbers":
        s.loc[mask] = s.loc[mask].apply(lambda v: re.sub(r"\d+", "", str(v)))
        return s
    if action == "remove_special_characters":
        s.loc[mask] = s.loc[mask].apply(lambda v: re.sub(r"[^\w\s]", "", str(v)))
        return s
    if action == "convert_to_lowercase":
        s.loc[mask] = s.loc[mask].apply(lambda v: str(v).lower())
        return s
    if action == "convert_to_uppercase":
        s.loc[mask] = s.loc[mask].apply(lambda v: str(v).upper())
        return s
    if action == "strip_spaces":
        s.loc[mask] = s.loc[mask].apply(lambda v: str(v).strip())
        return s
    if action == "remove_all_spaces":
        s.loc[mask] = s.loc[mask].apply(lambda v: re.sub(r"\s+", "", str(v)))
        return s
    if action == "replace_custom":
        s.loc[mask] = custom_val
        return s

    return s

# ---------------------------
# Preview / Apply / Undo helpers
# ---------------------------
def push_history_before_action():
    st.session_state.setdefault("df_history", [])
    st.session_state["df_history"].append(st.session_state["df"].copy())

def preview_before_after(df, col, group_values, action, custom_val=None, extra=None, sample_n=50):
    """
    Return aligned before and after dataframes (samples).
    Operates on masked sub-series so BEFORE/AFTER alignment is preserved.
    """
    if not group_values:
        return pd.DataFrame(columns=[col]), pd.DataFrame(columns=[col])

    mask_idx = df[col].astype(str).isin(set(group_values))
    before = df.loc[mask_idx, [col]].head(sample_n).copy().reset_index(drop=True)

    # transform only the masked sub-series
    sub_series = df.loc[mask_idx, col].copy()
    transformed_sub = transform_series_for_group(sub_series, group_values, action, custom_val, extra)

    # build after aligned with before (preserve row order via head(sample_n))
    after = transformed_sub.head(sample_n).to_frame(name=col).reset_index(drop=True)
    return before, after

def apply_transform_and_log(col, group_values, action, custom_val=None, extra=None):
    """
    Apply transformation for a specific column & specific group_values (subset).
    This function is used by Apply All to apply each selected pattern.
    """
    push_history_before_action()
    df2 = st.session_state["df"].copy()

    sub_idx = df2[col].astype(str).isin(set(group_values))
    if not sub_idx.any():
        # nothing to do
        return False

    sub_series = df2.loc[sub_idx, col].copy()
    transformed_sub = transform_series_for_group(sub_series, group_values, action, custom_val, extra)
    df2.loc[sub_idx, col] = transformed_sub.values

    st.session_state["df"] = df2.copy()
    st.session_state["clean_df"] = df2.copy()

    action_labels = ACTION_LABELS()
    label = action_labels.get(action, action)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{col}: {label} — {ts}"
    st.session_state.setdefault("semantic_log", []).append(entry)

    st.session_state.setdefault("handled_semantic_groups", {}).setdefault(col, set()).update(set(group_values))
    st.session_state.setdefault("handled_semantic_columns", set()).add(col)
    st.session_state["last_action_msg"] = entry

    return True

def undo_reset_to_before_semantic():
    """
    Reset df to the state right after Missing Values page (saved on page load).
    """
    if st.session_state.get("df_before_semantic") is not None:
        st.session_state["df"] = st.session_state["df_before_semantic"].copy()
        st.session_state["clean_df"] = st.session_state["df"].copy()
        st.session_state["semantic_log"] = []
        st.session_state["df_history"] = []
        st.session_state["handled_semantic_groups"] = {}
        st.session_state["handled_semantic_columns"] = set()
        st.session_state["semantic_batch_actions"] = {}
        st.session_state["last_action_msg"] = "Reset to pre-semantic state."
        st.success("Reset successful. All semantic changes removed.")
        return True
    else:
        st.warning("No backup found to reset to.")
        return False

def is_column_fully_handled(col):
    """
    A column is fully handled when all remaining pattern groups are clean.
    Clean groups = pure_numbers, pure_text.
    Anything else means the column still needs cleanup.
    """
    df = st.session_state["df"]
    groups = group_by_pattern(df[col])
    clean_groups = {"pure_numbers", "pure_text"}
    for gkey, meta in groups:
        if gkey not in clean_groups:
            return False
    return True

# ---------------------------
# UI pieces: intro and main page logic
# ---------------------------
def render_intro_box():
    st.markdown("""
    <div class="page-title-box">
        <span style="font-size:28px;font-weight:800;">🧠 Semantic Cleanup</span>
        <div style="margin-top:6px;font-size:14px;opacity:0.85;">
            Clean inconsistent values, normalize patterns, and standardize semantics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()


def run_semantic_cleanup_page(df):
    # show last action message (persistent) if present
    last_msg = st.session_state.get("last_action_msg")
    if last_msg:
        st.success(last_msg)

    # Sidebar summary (always visible)
    with st.sidebar:
        st.markdown("### ✅ Semantic Cleanup — Summary")
        logs = st.session_state.get("semantic_log", [])
        if logs:
            for entry in reversed(logs[-50:]):
                st.write(entry)
        else:
            st.info("No actions performed yet.")
        st.markdown("---")
        st.write("Handled columns:")
        handled_cols = st.session_state.get("handled_semantic_columns", set())
        if handled_cols:
            for c in sorted(handled_cols):
                done = "✔" if is_column_fully_handled(c) else "○"
                # last action for this column
                last_entry = ""
                for l in reversed(logs):
                    if l.startswith(c + ":"):
                        last_entry = l
                        break
                st.write(f"{done} {c} — {last_entry.split(' — ')[0].replace(c+': ','') if last_entry else ''}")
        else:
            st.write("— none —")

    # Step 1: classification table
    st.subheader("Step 1 — Column classification")
    rows = []
    for c in df.columns:
        t, reason = analyze_column_basic(df[c])
        rows.append({"Column": c, "Detected Type": t, "Reason": reason})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.markdown("---")

    # initialize batch actions structure
    st.session_state.setdefault("semantic_batch_actions", {})  # format: {col: {gkey: {"action":..., "custom":..., "extra":..., "values": [...]}}}
    # FIX: Safe initialization (avoid boolean evaluation of DataFrame)
    if "df_before_semantic" not in st.session_state:
        st.session_state["df_before_semantic"] = st.session_state["df"].copy()

    total_columns = len(df.columns)
    # Step 2: multi-column expanders
    st.subheader("Step 2 — Multi-column cleanup (choose actions per pattern)")
    for i, col in enumerate(df.columns):
        expanded = True if i == 0 else False  # first open only
        with st.expander(f"Column: {col}", expanded=expanded):
            st.write("Unique values (sample):")
            uniques = df[col].astype(str).unique().tolist()
            st.write(f"Total unique values: {len(uniques)} — showing first 200")
            st.code(uniques[:200])

            st.markdown("**Frequency (top values)**")
            freq = df[col].astype(str).value_counts(dropna=False).to_frame("Count")
            freq["Percent"] = (freq["Count"] / len(df)) * 100
            st.dataframe(freq.head(200), use_container_width=True)

            st.markdown("**Detected pattern groups (Highest to lowest)**")
            groups = group_by_pattern(df[col])
            if not groups:
                st.info("No groups detected for this column.")
                continue

            # sort groups already returned ordered by pct desc
            for gkey, meta in groups:
                pct = round(meta["pct"] * 100, 3)
                st.write(f"**{gkey}** — {pct}% — {meta['count']} rows")
                st.write(meta["values"][:8])

                # build UI for each group: action dropdown, preview, checkbox to include
                # prepare action options based on gkey
                if "range" in gkey:
                    action_options = [
                        "range_keep (keep as-is)",
                        "range_min (min — keep minimum value)",
                        "range_max (max — keep maximum value)",
                        "range_avg (average — convert range to mean)",
                        "range_custom (custom — replace with custom value)",
                        "range_mixed (mixed_range — fill each row with random value inside range)"
                    ]
                elif "comparison" in gkey or gkey == "comparison":
                    action_options = [
                        "comparison_keep (keep as-is)",
                        "comparison_ignore_sign (ignore sign — convert to numeric)",
                        "comparison_ceiling_floor (ceiling/floor based on sign)"
                    ]
                elif "special" in gkey or "multiple_inner_spaces" in gkey or "leading_trailing_spaces" in gkey:
                    action_options = [
                        "special_keep (keep as-is)",
                        "special_remove_special (remove special characters)",
                        "special_first (extract first numeric value)",
                        "special_last (extract last numeric value)",
                        "special_arith (arithmetic between extracted numbers)"
                    ]
                else:
                    action_options = [
                        "keep (no change)",
                        "to_int (convert to integer)",
                        "to_float (convert to float)",
                        "extract_numeric (extract numeric part)",
                        "remove_numbers (remove digits from text)",
                        "remove_special_characters (remove punctuation & symbols)",
                        "convert_to_lowercase (lowercase)",
                        "convert_to_uppercase (uppercase)",
                        "strip_spaces (strip_spaces)",
                        "remove_all_spaces (remove_all_spaces)",
                        "replace_custom (custom (replace with custom value))"
                    ]

                action_choice_display = st.selectbox(
                    f"Action for group {gkey} in {col}",
                    action_options,
                    index=0,
                    key=f"action_choice_{col}_{gkey}"
                )

                # derive action key (first token until space or "(")
                action_key = action_choice_display.split()[0]

                custom_val = None
                extra = {}

                if action_key == "range_custom" or action_key == "replace_custom":
                    custom_val = st.text_input(f"Custom replacement value for {col} / {gkey}:", value="", key=f"custom_{col}_{gkey}")

                if action_key == "special_arith":
                    st.markdown("Arithmetic options:")
                    op = st.selectbox("Operation", ["+", "-", "*", "/"], index=0, key=f"op_{col}_{gkey}")
                    order_sel = st.selectbox("Number order", ["first → second", "second → first"], index=0, key=f"order_{col}_{gkey}")
                    order = "a_b" if order_sel.startswith("first") else "b_a"
                    extra = {"op": op, "order": order}

                if action_key == "range_mixed":
                    st.caption("Range mixed will fill each row with a row-wise different random integer inside the detected range.")

                # prepare preview (top 10)
                group_values = meta["values"]
                before_df, after_df = preview_before_after(st.session_state["df"], col, group_values, action_key, custom_val, extra, sample_n=50)

                c1, c2 = st.columns(2)
                with c1:
                    st.write("BEFORE (sample, top 50)")
                    st.dataframe(before_df, use_container_width=True)
                with c2:
                    st.write("AFTER (sample, top 50)")
                    st.dataframe(after_df, use_container_width=True)

                # include checkbox to add to batch
                batch_key = st.checkbox(f"Include this group '{gkey}' for APPLY ALL", key=f"include_{col}_{gkey}")

                # update session semantic_batch_actions based on checkbox
                batch = st.session_state.setdefault("semantic_batch_actions", {})
                if batch_key:
                    col_actions = batch.setdefault(col, {})
                    col_actions[gkey] = {"action": action_key, "custom": custom_val, "extra": extra, "values": list(group_values)}
                    st.success(f"Queued: {col} → {gkey} → {action_key}")
                else:
                    # if exists and unchecked, remove from batch
                    if col in batch and gkey in batch[col]:
                        del batch[col][gkey]
                        # if column has no remaining groups, remove col entry
                        if not batch[col]:
                            del batch[col]

                st.markdown("---")

    st.markdown("---")
    # Collected Actions Summary (live)
    st.subheader("Collected Actions Summary (selected groups)")
    actions = st.session_state.get("semantic_batch_actions", {})
    if not actions:
        st.info("No groups selected for batch apply yet.")
    else:
        for col, groups in actions.items():
            st.write(f"**{col}**")
            for gk, info in groups.items():
                act = info.get("action")
                label = ACTION_LABELS().get(act, act)
                st.write(f" → {gk} : {label}")
    st.markdown("---")

    # Apply All button + progress
    st.subheader("Apply selected changes (batch)")

    if st.button("Apply All Selected"):
        actions = st.session_state.get("semantic_batch_actions", {})
        if not actions:
            st.warning("No actions selected. Select at least one group to apply.")
        else:
            # create steps count
            total_steps = sum(len(groups) for groups in actions.values())
            if total_steps == 0:
                st.warning("No groups selected.")
            else:
                st.info("Applying semantic cleanup… this may take a few seconds.")
                progress = st.progress(0)
                step = 0
                # apply each selected group
                for col, groups in list(actions.items()):
                    for gk, info in list(groups.items()):
                        action = info.get("action")
                        custom = info.get("custom")
                        extra = info.get("extra", {})
                        values = info.get("values", [])
                        # apply
                        apply_transform_and_log(col, values, action, custom, extra)
                        step += 1
                        progress.progress(int(step / total_steps * 100))
                        # small sleep to keep UI responsive & show progress (tunable)
                        time.sleep(0.02)
                st.success("Batch apply complete.")
                # build final summary (immediate) and show
                final_summary = st.session_state.get("semantic_log", [])[-(total_steps):]
                st.markdown("### Final Summary (last applied actions)")
                for e in final_summary:
                    st.write(e)
                # after apply, clear batch (user applied)
                st.session_state["semantic_batch_actions"] = {}
                # update handled columns (already done in apply)
                st.rerun()

    if st.button("Undo (reset to pre-semantic state)"):
        ok = undo_reset_to_before_semantic()
        if ok:
            st.rerun()

    st.markdown("---")
    # Final summary before navigation (also appears after apply as requested)
    st.subheader("Final Summary (latest actions)")
    logs = st.session_state.get("semantic_log", [])
    if logs:
        # show last 50
        for entry in reversed(logs[-50:]):
            st.write(entry)
    else:
        st.write("— none —")

        # ----------------------------------------------------
        # 🔍 BEFORE vs AFTER DATASET PREVIEW (TOP 20 ROWS)
        # ----------------------------------------------------
    st.subheader("🔍 Apply Semantic cleanup and Preview Dataset(Top 50 rows)")

    old_df = st.session_state.get("df_before_semantic")
    new_df = st.session_state.get("df")

    if isinstance(old_df, pd.DataFrame) and isinstance(new_df, pd.DataFrame):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### BEFORE Cleanup")
            st.dataframe(old_df.head(50), use_container_width=True)

        with c2:
            st.markdown("### AFTER Cleanup")
            st.dataframe(new_df.head(50), use_container_width=True)

    else:
        st.info("Preview not available yet. Apply some semantic cleanup first.")

    st.markdown("---")

    # ----------------------------------------------------
    # DOWNLOAD CLEANED CSV
    # ----------------------------------------------------
    st.subheader("📥 Download Cleaned Dataset")

    clean_df = st.session_state.get("df")

    if isinstance(clean_df, pd.DataFrame):
        csv_data = clean_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            help="Download the cleaned dataframe after semantic cleanup"
        )
    else:
        st.info("Dataset not available yet.")

    st.markdown("---")
    # ----------------------------------------------------
    # END DOWNLOAD BLOCK
    # ----------------------------------------------------

    # ----------------------------------------------------
    # END PREVIEW BLOCK
    # ----------------------------------------------------

    st.markdown("---")
    # Navigation (unchanged logic)
    st.subheader("Navigation")
    if st.button("Proceed to Outlier Handling →"):
        all_cols = list(st.session_state["df"].columns)
        fully_handled = [c for c in all_cols if is_column_fully_handled(c)]
        partially_handled = [c for c in all_cols if c in st.session_state.get("handled_semantic_columns", set()) and c not in fully_handled]
        not_handled = [c for c in all_cols if c not in st.session_state.get("handled_semantic_columns", set())]

        if len(st.session_state.get("handled_semantic_columns", set())) == 0:
            st.warning("You did not perform any semantic cleanup. Stay here or Continue anyway?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Stay here"):
                    st.info("Staying on this page.")
            with c2:
                if st.button("Continue anyway"):
                    st.session_state["current_page"] = "Outlier Handling"
                    st.rerun()

        elif len(partially_handled) > 0 or len(not_handled) > 0:
            st.warning(
                f"Semantic cleanup is incomplete.\n"
                f"Partially handled: {len(partially_handled)} | Not handled: {len(not_handled)}"
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Stay and finish"):
                    st.info("Stay here to complete semantic cleanup.")
            with c2:
                if st.button("Continue anyway"):
                    st.session_state["current_page"] = "Outlier Handling"
                    st.rerun()

        else:
            st.success("All columns fully cleaned. Proceeding to Outlier Handling.")
            st.session_state["current_page"] = "Outlier Handling"
            st.rerun()

# ---------------------------
# PUBLIC wrapper (keeps old behavior & name)
# ---------------------------
def run_semantic_cleanup():

    # Page header (keep as-is)
    st.markdown(
        """
        <div class="page-title-box">
            <div style="display:flex;align-items:center;gap:12px;">
                <span style="font-size:28px;font-weight:800;"> 🧠 Semantic Cleanup</span>
            </div>
            <div style="margin-top:6px;font-size:14px;opacity:0.85;">
                Fix inconsistent values, remove noise, and standardize patterns before Outlier Handling.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    # Short info box (keep as-is)
    st.markdown(
        """
        <div style="
            background-color:#0E2A47;
            padding:16px 18px;
            border-radius:10px;
            margin-bottom:25px;
            color:#e4ecf5;
            font-size:15px;
            line-height:1.55;
        ">
            <b>What this page fixes:</b>
            <ul style="margin-top:8px;">
                <li>Mixed values like <code>12 years</code>, <code>3 months</code>, <code>5+</code></li>
                <li>Spacing, casing, punctuation, and symbol inconsistencies</li>
                <li>Hidden patterns such as ranges, comparisons, keyword variants</li>
                <li>Extracting & converting numeric information</li>
                <li>Cleaning text to a uniform, usable structure</li>
            </ul>
            <span style="color:#b8c6d9;">
                Makes your data clean, consistent, and ready for Outlier Handling & EDA.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load df (unchanged logic)
    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠ Please complete Fix Missing Values first (Page 2).")
        st.stop()

    # Initialize session state (only if not present)
    st.session_state.setdefault("semantic_log", [])
    st.session_state.setdefault("df_history", [])
    st.session_state.setdefault("handled_semantic_groups", {})
    st.session_state.setdefault("handled_semantic_columns", set())
    st.session_state.setdefault("semantic_dtype_choice", {})
    st.session_state.setdefault("clean_df", df.copy())
    st.session_state.setdefault("semantic_batch_actions", {})
    if "df_before_semantic" not in st.session_state:
        st.session_state["df_before_semantic"] = df.copy()

    run_semantic_cleanup_page(df)

# allow direct run for debugging (safe)
if __name__ == "__main__":
    run_semantic_cleanup()
