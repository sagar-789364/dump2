import streamlit as st
from pages.login import login_page
from pages.service_selection import service_selection_page
from pages.Usecase_1.app_uc1 import main as uc1_main
from pages.Usecase_2.app_uc2 import main as uc2_main

def initialize_app_state():
    """Initialize all required session state variables"""
    state_variables = {
        "logged_in": False,
        "username": "",
        "selected_service": None,
        "page_reload_counter": 0,
        "last_interaction": None,
        "form_submitted": False,
        "current_page": None,
        "sidebar_state": "collapsed",
        "page_history": [],
        "sidebar_disabled": False  # Add sidebar_disabled state
    }
    
    for var, default_value in state_variables.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

def update_page_history(page_name):
    """Push the page to history if it's not the same as the last one."""
    if "page_history" not in st.session_state:
        st.session_state.page_history = []
    if not st.session_state.page_history or st.session_state.page_history[-1] != page_name:
        st.session_state.page_history.append(page_name)

def main():
    # Initialize state first
    initialize_app_state()

    # Determine if sidebar should be disabled
    if not st.session_state["logged_in"] or st.session_state["selected_service"] is None:
        st.session_state["sidebar_disabled"] = True
    else:
        st.session_state["sidebar_disabled"] = False

    # Set sidebar state accordingly
    if st.session_state["sidebar_disabled"]:
        st.set_page_config(
            page_title="Accounting Research Application",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        # Hide sidebar with CSS
        st.markdown("""
            <style>
            [data-testid="stSidebar"] { display: none !important; }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.set_page_config(
            page_title="Accounting Research Application",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    # Use direct session state access
    if not st.session_state["logged_in"]:
        st.session_state["current_page"] = "login"
        update_page_history("login")
        login_page()
    elif st.session_state["selected_service"] is None:
        st.session_state["current_page"] = "service_selection"
        update_page_history("service_selection")
        service_selection_page()
    elif st.session_state["selected_service"] == "research":
        st.session_state["current_page"] = "research"
        update_page_history("research")
        uc1_main()
    elif st.session_state["selected_service"] == "memo":
        st.session_state["current_page"] = "memo"
        update_page_history("memo")
        uc2_main()

if __name__ == "__main__":
    main()
