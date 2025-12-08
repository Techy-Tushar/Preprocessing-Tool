# -------------------------------------------------------------
# utils/state_helpers.py
# Core persistence, undo system, change tracking
# -------------------------------------------------------------
import streamlit as st
from copy import deepcopy
import time


# -------------------------------------------------------------
# SAVE CURRENT STATE INTO UNDO STACK
# -------------------------------------------------------------
def save_undo(label="change"):
    """
    Save a snapshot of DF + change history before applying a modification.
    Used for Undo functionality.
    """
    if "undo_stack" not in st.session_state:
        st.session_state.undo_stack = []

    snapshot = {
        "timestamp": time.time(),
        "label": label,
        "df": deepcopy(st.session_state.clean_df),
        "page_changes": deepcopy(st.session_state.page_changes)
    }

    st.session_state.undo_stack.append(snapshot)


# -------------------------------------------------------------
# UNDO LAST CHANGE
# -------------------------------------------------------------
def undo_last_change():
    """
    Reverts the last DF change, restoring old state + change history.
    """
    if not st.session_state.undo_stack:
        return False, "No previous actions to undo."

    snapshot = st.session_state.undo_stack.pop()

    # Restore dataframe
    st.session_state.clean_df = deepcopy(snapshot["df"])

    # Restore per-page changes
    st.session_state.page_changes = deepcopy(snapshot["page_changes"])

    return True, f"Undo successful: {snapshot['label']}"


# -------------------------------------------------------------
# RECORD CHANGE IN PAGE HISTORY
# -------------------------------------------------------------
def record_change(page_id, col=None, action=None, details=None):
    """
    Store one operation performed by user on a specific page.
    """
    st.session_state.page_changes[page_id].append({
        "timestamp": time.time(),
        "column": col,
        "action": action,
        "details": details
    })


# -------------------------------------------------------------
# GET SUMMARY OF PRIOR CHANGES FOR PAGE
# -------------------------------------------------------------
def get_page_summary(page_id):
    """
    Returns HTML summary list for change history of a page.
    Perfect for showing user previous changes when revisiting a page.
    """
    changes = st.session_state.page_changes.get(page_id, [])
    if not changes:
        return ""

    html = "<ul>"
    for c in changes:
        col = c.get("column", "Unnamed Column")
        action = c.get("action", "Unknown Action")
        html += f"<li><b>{col}</b> — {action}</li>"
    html += "</ul>"

    return html


# -------------------------------------------------------------
# SAFE APPLY DF MODIFICATION (UNDO + RECORD)
# -------------------------------------------------------------
def apply_change(page_id, col, action, func, details=None):
    """
    Safely apply a DF modification:
    ✔ save snapshot for undo
    ✔ apply change using provided function
    ✔ record change history
    """
    save_undo(f"{page_id}: {action} on {col}")

    # Run the given transformation function on DF
    st.session_state.clean_df = func(st.session_state.clean_df)

    # Log the action
    record_change(page_id, col, action, details)

    return True


# -------------------------------------------------------------
# CHECK IF PAGE HAS CHANGES (FOR WARNINGS)
# -------------------------------------------------------------
def page_has_changes(page_id):
    """
    Returns True if the page has any saved changes.
    """
    return len(st.session_state.page_changes.get(page_id, [])) > 0
