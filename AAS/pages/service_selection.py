import streamlit as st
from pages.header import render_header

def get_service_selection_css():
    return """
    <style>
    /* Button styling to match container design */
    .stButton > button {
        background: white !important;
        border-radius: 15px !important;
        padding: 30px !important;
        width: 100% !important;
        text-align: center !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        cursor: pointer !important;
        border: 2px solid #e0e0e0 !important;
        height: auto !important;
        display: block !important;
    }
    .stButton > button:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2) !important;
        border-color: #004b87 !important;
    }
    .service-content {
        text-align: center;
        padding: 10px;
    }
    .service-icon {
        font-size: 48px;
        margin-bottom: 20px;
        color: #004b87;
    }
    .service-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #004b87;
    }
    .service-description {
        font-size: 16px;
        color: #666;
        line-height: 1.5;
        margin-bottom: 20px;
    }
    .page-header {
        text-align: center;
        padding: 10px;
        margin-bottom: 10px;
    }
    .page-header h1 {
        color: #004b87;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .page-header p {
        color: #666;
        font-size: 18px;
        max-width: 600px;
        margin: 0 auto;
    }
    .service-columns {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-top: 50px;
        padding: 20px;
    }
    </style>
    """

def initialize_service_selection_state():
    """Initialize service selection related session state variables"""
    service_state_variables = {
        "research_btn_clicked": False,
        "memo_btn_clicked": False,
        "switch_service_btn_clicked": False,
        "logout_btn_clicked": False,
        "service_content_loaded": False,
        "service_content": None
    }
    
    for var, default_value in service_state_variables.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

def get_service_content():
    """Get service content from session state or initialize it"""
    if not st.session_state.get("service_content_loaded", False):
        st.session_state["service_content"] = {
            "research": {
                "icon": "🔍",
                "title": "Research",
                "description": "Access our comprehensive accounting research tools and guidance. "
                             "Get instant answers to your accounting queries with AI-powered assistance."
            },
            "memo": {
                "icon": "📝",
                "title": "Memo Generation",
                "description": "Generate detailed accounting memorandums automatically. "
                             "Streamline your documentation process with our intelligent memo generator."
            }
        }
        st.session_state["service_content_loaded"] = True
    
    return st.session_state["service_content"]

def handle_research_service_selection():
    """Handle research service selection"""
    if st.session_state.get("research_btn_clicked", False):
        st.session_state["selected_service"] = "research"
        st.session_state["research_btn_clicked"] = False  # Reset the button state
        st.rerun()  # Only rerun when service is actually selected

def handle_memo_service_selection():
    """Handle memo service selection"""
    if st.session_state.get("memo_btn_clicked", False):
        st.session_state["selected_service"] = "memo"
        st.session_state["memo_btn_clicked"] = False  # Reset the button state
        st.rerun()  # Only rerun when service is actually selected

def handle_switch_service():
    """Handle switching between services"""
    if st.session_state.get("switch_service_btn_clicked", False):
        current_service = st.session_state.get("selected_service")
        if current_service == "research":
            st.session_state["selected_service"] = "memo"
        elif current_service == "memo":
            st.session_state["selected_service"] = "research"
        st.session_state["switch_service_btn_clicked"] = False  # Reset the button state
        st.rerun()

def handle_logout():
    """Handle logout functionality"""
    if st.session_state.get("logout_btn_clicked", False):
        # Reset all relevant session state
        st.session_state["logged_in"] = False
        st.session_state["selected_service"] = None
        st.session_state["current_page"] = "login"
        st.session_state["logout_btn_clicked"] = False  # Reset the button state
        st.rerun()

def service_selection_page():
    """Render the service selection page with session state management"""
    # Initialize session state for service selection
    initialize_service_selection_state()
    
    if "selected_service" not in st.session_state:
        st.session_state.selected_service = None
    
    # Apply CSS
    st.markdown(get_service_selection_css(), unsafe_allow_html=True)
    
    # Render header
    render_header()
    
    # Get service content from session state
    services = get_service_content()
    
    # Page header
    st.markdown("""
    <div class="page-header">
        <h1>Welcome to Accounting Services</h1>
        <p>Select a service to continue</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Service selection columns
    st.markdown('<div class="service-columns">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        research_content = f"""
        <div class="service-content">
            <div class="service-icon">{services['research']['icon']}</div>
            <div class="service-title">{services['research']['title']}</div>
            <div class="service-description">{services['research']['description']}</div>
        </div>
        """
        st.markdown(research_content, unsafe_allow_html=True)
        
        # Research service button
        if st.button("Select Research Service", key="research_service_btn", use_container_width=True):
            st.session_state["research_btn_clicked"] = True
    
    with col2:
        memo_content = f"""
        <div class="service-content">
            <div class="service-icon">{services['memo']['icon']}</div>
            <div class="service-title">{services['memo']['title']}</div>
            <div class="service-description">{services['memo']['description']}</div>
        </div>
        """
        st.markdown(memo_content, unsafe_allow_html=True)
        
        # Memo service button
        if st.button("Select Memo Generation", key="memo_service_btn", use_container_width=True):
            st.session_state["memo_btn_clicked"] = True
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle service selections - only rerun when necessary
    handle_research_service_selection()
    handle_memo_service_selection()
    
    # Sidebar controls (only show when sidebar is not disabled)
    if not st.session_state.get("sidebar_disabled", False):
        with st.sidebar:
            # Switch Service button - only show if a service is already selected
            if st.session_state.get("selected_service") in ("research", "memo"):
                if st.button("Switch Service", key="switch_service_btn"):
                    st.session_state["switch_service_btn_clicked"] = True
            
            # Logout button
            if st.button("Logout", key="logout_btn"):
                st.session_state["logout_btn_clicked"] = True
    
    # Handle sidebar actions
    handle_switch_service()
    handle_logout()