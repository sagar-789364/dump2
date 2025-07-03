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
        "sidebar_disabled": False,
        "app_initialized": False
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

def configure_page_layout():
    """Configure page layout based on current state"""
    # Determine if sidebar should be disabled
    if not st.session_state.get("logged_in", False) or st.session_state.get("selected_service") is None:
        st.session_state["sidebar_disabled"] = True
    else:
        st.session_state["sidebar_disabled"] = False

    # Set sidebar state accordingly
    if st.session_state.get("sidebar_disabled", True):
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

def determine_current_page():
    """Determine which page to display based on session state"""
    if not st.session_state.get("logged_in", False):
        return "login"
    elif st.session_state.get("selected_service") is None:
        return "service_selection"
    elif st.session_state.get("selected_service") == "research":
        return "research"
    elif st.session_state.get("selected_service") == "memo":
        return "memo"
    else:
        return "login"  # Default fallback

def render_current_page(page_name):
    """Render the appropriate page based on page name"""
    # Update current page and history
    st.session_state["current_page"] = page_name
    update_page_history(page_name)
    
    # Render the appropriate page
    if page_name == "login":
        login_page()
    elif page_name == "service_selection":
        service_selection_page()
    elif page_name == "research":
        uc1_main()
    elif page_name == "memo":
        uc2_main()
    else:
        # Fallback to login if unknown page
        st.session_state["current_page"] = "login"
        st.session_state["logged_in"] = False
        st.session_state["selected_service"] = None
        login_page()

def main():
    """Main application entry point with session state management"""
    # Initialize state first
    initialize_app_state()
    
    # Configure page layout only once
    if not st.session_state.get("app_initialized", False):
        configure_page_layout()
        st.session_state["app_initialized"] = True
    
    # Determine current page based on session state
    current_page = determine_current_page()
    
    # Render the current page
    render_current_page(current_page)

if __name__ == "__main__":
    main()