import streamlit as st
import base64

def initialize_header_state():
    """Initialize header-related session state variables"""
    header_state_variables = {
        "back_btn_clicked": False,
        "header_logo_loaded": False,
        "header_css_loaded": False,
        "encoded_logo": None
    }
    
    for var, default_value in header_state_variables.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

def get_header_css():
    """Return custom CSS styles for the header (cached in session state)"""
    if not st.session_state.get("header_css_loaded", False):
        st.session_state["header_css"] = """            
        <style>
        /* Hide only the background and border of the default header, but keep the sidebar icon visible */
        header[data-testid="stHeader"] {
            background: transparent;
            box-shadow: none;
        }
        
        /* Hide the title text in the header, but keep the hamburger icon */
        header[data-testid="stHeader"] .st-emotion-cache-18ni7ap {
            display: none !important;
        }

        /* Fixed header */
        .header-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 80px;
            background-color: #004b87;
            padding: 10px 20px;
            z-index: 1000;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
        }

        /* Style for the header logo */
        .header-logo {
            height: 150px;
            width: auto;
            margin-right: 20px;
            margin-left: 50px;
            border: none;
            background: transparent;
        }
        </style>
        """
        st.session_state["header_css_loaded"] = True
    
    return st.session_state["header_css"]

def load_header_logo():
    """Load and encode the header logo (cached in session state)"""
    if not st.session_state.get("header_logo_loaded", False):
        try:
            with open("static/TheLogo.png", "rb") as image_file:
                st.session_state["encoded_logo"] = base64.b64encode(image_file.read()).decode()
        except FileNotFoundError:
            st.session_state["encoded_logo"] = None
        
        st.session_state["header_logo_loaded"] = True
    
    return st.session_state.get("encoded_logo")

def create_header_with_logo():
    """Create the header with logo using session state"""
    encoded_logo = load_header_logo()
    
    if encoded_logo:
        st.markdown(f'''
        <div class="header-container" id="custom-header">
            <div class="header-logo-section">
                <img src="data:image/png;base64,{encoded_logo}" class="header-logo" alt="Logo">
            </div>
            <div class="header-content-section">
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="header-container" id="custom-header">
            <div class="header-logo-section">
                <!-- Logo not found -->
            </div>
            <div class="header-content-section">
            </div>
        </div>
        ''', unsafe_allow_html=True)

def handle_back_button():
    """Handle back button click using session state"""
    if st.session_state.get("back_btn_clicked", False):
        # Only proceed if there's a page to go back to
        if "page_history" in st.session_state and len(st.session_state.page_history) > 1:
            # Remove current page
            st.session_state.page_history.pop()
            prev_page = st.session_state.page_history[-1]
            st.session_state.current_page = prev_page
            
            # Update relevant session state based on previous page
            if prev_page == "service_selection":
                st.session_state.selected_service = None
            elif prev_page == "login":
                st.session_state.logged_in = False
                st.session_state.selected_service = None
        
        # Reset button state
        st.session_state["back_btn_clicked"] = False
        st.rerun()

def render_header():
    """Apply header CSS and render the header with logo"""
    initialize_header_state()
    st.markdown(get_header_css(), unsafe_allow_html=True)
    create_header_with_logo()

def render_back_button():
    """Render a back button in the header and handle navigation logic using session state"""
    initialize_header_state()
    
    # Only show if there is a previous page
    if "page_history" in st.session_state and len(st.session_state.page_history) > 1:
        # Use button with callback instead of form
        if st.button("← Back", key="back_button"):
            st.session_state["back_btn_clicked"] = True
    
    # Handle back button click
    handle_back_button()