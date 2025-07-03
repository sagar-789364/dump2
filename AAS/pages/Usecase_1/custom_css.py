def get_custom_css():
    """Return custom CSS styles (excluding header)"""
    return """            
    <style>
    /* Change sidebar icon (hamburger) color to white in header */
    header[data-testid="stHeader"] svg {
        color: #fff !important;
        fill: #fff !important;
    }
    /* When sidebar is open, set icon color to black */
    [data-testid="stSidebar"] ~ div [data-testid="stHeader"] svg {
        color: #000 !important;
        fill: #000 !important;
    }

    /* Spacer to push content below header */
    .spacer {
        height: 50px;
    }

    /* Adjust main content when sidebar is expanded */
    [data-testid="stSidebar"] {
        margin-top: 80px;
        z-index: 999;
    }

    /* Shrink main content when sidebar is expanded */
    @media (min-width: 768px) {
        .main-content {
            transition: margin-left 0.3s ease;
        }
        .sidebar-expanded .main-content {
            margin-left: 300px;
        }
    }

    /* Style for the logo image */
    .logo-container img {
        max-height: 50px;
        width: auto;
    }

    /* Active session styling - light grey background */
    .active-session {
        background-color: #e8e8e8 !important;
        border-radius: 5px;
        padding: 5px;
        margin-bottom: 2px !important;
    }

    /* Style for active session button */
    .active-session button[data-testid="baseButton-secondary"] {
        background-color: #e8e8e8 !important;
        border: 1px solid #d0d0d0 !important;
        color: #333333 !important;
    }

    .active-session button[data-testid="baseButton-secondary"]:hover {
        background-color: #d8d8d8 !important;
    }

    /* Reduce spacing between session buttons */
    .sidebar-session-container {
        margin-bottom: 5px !important;
    }

    /* Reduce margin for session buttons */
    .sidebar-session-container button {
        margin-bottom: 0px !important;
    }

    /* Custom session separator */
    .session-separator {
        margin: 5px 0 !important;
        border-top: 1px solid #ddd;
        height: 1px;
    }

    /* Override default streamlit button margins in sidebar */
    .stSidebar .stButton > button {
        margin-bottom: 5px !important;
    }
    </style>
    """
