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

@st.cache_data
def get_service_content():
    """Cache service content to prevent unnecessary recomputation"""
    return {
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

def handle_service_selection(service: str):
    """Handle service selection without immediate rerun"""
    if "service_selection_pending" not in st.session_state:
        st.session_state.service_selection_pending = None
    
    st.session_state.service_selection_pending = service
    st.session_state.selected_service = service

def service_selection_page():
    """Render the service selection page with optimized performance"""
    # Initialize session state for service selection
    if "selected_service" not in st.session_state:
        st.session_state.selected_service = None
    
    # Page configuration
    # st.set_page_config(
    #     page_title="Service Selection",
    #     layout="wide",
    #     initial_sidebar_state="collapsed"
    # )
    
    # Apply CSS
    st.markdown(get_service_selection_css(), unsafe_allow_html=True)
    
    # Render header
    render_header()
    
    # Get cached service content
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
        if st.button("Select Research Service", key="research", use_container_width=True):
            handle_service_selection("research")
    
    with col2:
        memo_content = f"""
        <div class="service-content">
            <div class="service-icon">{services['memo']['icon']}</div>
            <div class="service-title">{services['memo']['title']}</div>
            <div class="service-description">{services['memo']['description']}</div>
        </div>
        """
        st.markdown(memo_content, unsafe_allow_html=True)
        if st.button("Select Memo Generation", key="memo", use_container_width=True):
            handle_service_selection("memo")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle service selection and navigation
    if st.session_state.get("service_selection_pending"):
        st.rerun()
    
    # Logout and Switch Service buttons in sidebar
    if not st.session_state.get("sidebar_disabled", False):
        with st.sidebar:
            # Switch Service button toggles between research and memo
            if st.session_state.get("selected_service") in ("research", "memo"):
                if st.button("Switch Service"):
                    if st.session_state["selected_service"] == "research":
                        st.session_state["selected_service"] = "memo"
                    else:
                        st.session_state["selected_service"] = "research"
                    st.rerun()
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.selected_service = None
                st.session_state.current_page = "login"
                st.rerun()
