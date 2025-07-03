import streamlit as st
import base64

def get_header_css():
    """Return custom CSS styles for the header (from login page)"""
    return """            
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

def create_header_with_logo():
    """Create the header with logo"""
    try:
        with open("static/TheLogo.png", "rb") as image_file:
            encoded_logo = base64.b64encode(image_file.read()).decode()
        st.markdown(f'''
        <div class="header-container" id="custom-header">
            <div class="header-logo-section">
                <img src="data:image/png;base64,{encoded_logo}" class="header-logo" alt="Logo">
            </div>
            <div class="header-content-section">
            </div>
        </div>
        ''', unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown('''
        <div class="header-container" id="custom-header">
            <div class="header-logo-section">
                <!-- Logo not found -->
            </div>
            <div class="header-content-section">
            </div>
        </div>
        ''', unsafe_allow_html=True)

def render_header():
    """Apply header CSS and render the header with logo"""
    st.markdown(get_header_css(), unsafe_allow_html=True)
    create_header_with_logo()


def render_back_button():
    """Render a back button in the header and handle navigation logic."""
    # Only show if there is a previous page
    if "page_history" in st.session_state and len(st.session_state.page_history) > 1:
        # Use form to avoid Streamlit button rerun issues
        with st.form("back_btn_form", clear_on_submit=True):
            back_clicked = st.form_submit_button(
                "← Back", 
            )
            # Style the form to display inline
            st.markdown(
                '<style>[data-testid="stForm"]{display:inline; margin:0; padding:0;}</style>',
                unsafe_allow_html=True
            )
            if back_clicked:
                # Remove current page
                st.session_state.page_history.pop()
                prev_page = st.session_state.page_history[-1]
                st.session_state.current_page = prev_page
                # Optionally update selected_service for main app
                if prev_page == "service_selection":
                    st.session_state.selected_service = None
                elif prev_page == "login":
                    st.session_state.logged_in = False
                st.rerun()
