import streamlit as st
import base64
from datetime import datetime
from pages.header import render_header

def get_login_css():
    """Return custom CSS styles for login page (minimalistic, professional, with glossy animated greeting)"""
    return """
    <style>
    .main-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        padding: 20px;
        margin-top: -80px;
        padding-top: 80px;
    }
    .login-container {
        width: 100%;
        max-width: 420px;
        padding: 48px 36px 36px 36px;
        border-radius: 18px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        background: #fafbfc;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 32px;
        border: 1px solid #e6e8eb;
    }
    .greeting-text {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: 0.5px;
        font-family: 'Segoe UI', 'Arial', sans-serif;
        text-align: center;
        width: 100%;
        margin-bottom: 0px;
        background: linear-gradient(90deg, #003580, #2453d1, #0e2236, #b2eaff, #00b6ff, #7a2ff7, #ff4fa3, #003580);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        animation: glossy-gradient 6s linear infinite;
    }
    @keyframes glossy-gradient {
        0% { background-position: 0% 50%; }
        25% { background-position: 50% 50%; }
        50% { background-position: 100% 50%; }
        75% { background-position: 50% 50%; }
        100% { background-position: 0% 50%; }
    }
    .user-icon {
        width: 180px;
        height: 180px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 8px auto;
        /* No background, no border, no shadow */
        background: none;
        border: none;
        box-shadow: none;
    }
    .user-icon img {
        width: 160px;
        height: 160px;
        object-fit: contain;
    }
    .form-container {
        width: 100%;
        max-width: 320px;
        display: flex;
        flex-direction: column;
        align-items: stretch;
        gap: 18px;
        margin: 0 auto;
    }
    .form-container .stTextInput,
    .form-container .stTextInput > div,
    .form-container .stTextInput input {
        width: 100% !important;
        margin: 0 !important;
        box-sizing: border-box;
    }
    .form-container .stTextInput input {
        width: 100% !important;
        text-align: center;
        padding: 14px 20px;
        font-size: 16px;
        border: 1.5px solid #d1d5db;
        border-radius: 8px;
        background: #fff;
        transition: border-color 0.2s;
        box-sizing: border-box;
    }
    .form-container .stTextInput input:focus {
        border-color: #004b87;
        outline: none;
        box-shadow: 0 0 0 2px rgba(0, 75, 135, 0.07);
    }
    .form-container .stButton {
        width: 100% !important;
        margin: 0 !important;
    }
    .form-container .stButton > div {
        width: 100% !important;
        margin: 0 !important;
    }
    .form-container .stButton button {
        width: 100% !important;
        height: 48px !important;
        background: linear-gradient(90deg, #004b87 60%, #0072ce 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: background 0.2s !important;
        margin: 0 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .form-container .stButton button:hover {
        background: #003a6b !important;
    }
    .stAlert {
        text-align: center;
        max-width: 320px;
        margin: 0 auto;
    }
    .spacer {
        height: 18px;
        width: 100%;
        display: block;
    }
    @media (max-width: 768px) {
        .login-container {
            max-width: 98vw;
            padding: 24px 8px;
        }
        .greeting-text {
            font-size: 22px;
        }
        .user-icon {
            width: 100px;
            height: 100px;
        }
        .user-icon img {
            width: 80px;
            height: 80px;
        }
    }
    </style>
    """

def load_logo_base64():
    """Load and encode the user icon"""
    try:
        with open("static/user.png", "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        return None

def initialize_login_state():
    """Initialize login-related session state variables"""
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = ""
    if "login_time" not in st.session_state:
        st.session_state["login_time"] = None

def handle_login(username: str) -> bool:
    """Handle login logic"""
    if username.strip():
        st.session_state["username"] = username.strip()
        st.session_state["logged_in"] = True
        st.session_state["login_time"] = datetime.now().isoformat()
        return True
    return False

def get_greeting():
    """Return Good Morning/Afternoon/Evening based on local time."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good Morning"
    elif 12 <= hour < 17:
        return "Good Afternoon"
    else:
        return "Good Evening"

def login_page():
    """Render the login page"""
    # st.set_page_config(
    #     page_title="Login - Accounting Research Chatbot",
    #     layout="wide",
    #     initial_sidebar_state="collapsed"
    # )
    initialize_login_state()
    st.markdown(get_login_css(), unsafe_allow_html=True)
    render_header()
    col_left, col_center, col_right = st.columns([1.5, 1, 1.5])
    with col_center:
        greeting = get_greeting()
        st.markdown(f'<div class="greeting-text">{greeting}</div>', unsafe_allow_html=True)
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        encoded_user = load_logo_base64()
        if encoded_user:
            st.markdown(f'''
                <div class="user-icon">
                    <img src="data:image/png;base64,{encoded_user}" alt="User Icon">
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
                <div class="user-icon">
                    <span style="font-size: 80px; color: #004b87;">👤</span>
                </div>
            ''', unsafe_allow_html=True)
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            key="login_username",
            label_visibility="collapsed"
        )
        st.button("Login", use_container_width=True, key="login_btn")
        if st.session_state.get("login_btn"):
            if handle_login(username):
                # Fetch previous sessions for this user and set in session state
                try:
                    from pages.Usecase_1.app_uc1 import fetch_all_sessions_from_memory
                    st.session_state["all_sessions"] = fetch_all_sessions_from_memory()
                except Exception as e:
                    st.session_state["all_sessions"] = []
                st.rerun()
            else:
                st.error("Please enter a username")
        st.markdown('</div>', unsafe_allow_html=True)

def check_login_status():
    """Check if user is logged in"""
    initialize_login_state()
    return st.session_state["logged_in"]
