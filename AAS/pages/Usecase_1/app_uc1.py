import streamlit as st
from datetime import datetime
import uuid
import PyPDF2
from pages.Usecase_1.rag_sk_uc1 import (
    generate_guidance_response_sync,
    generate_document_sync,
    get_author_options_from_index,
    get_framework_options_from_index,
    get_accounting_standards_options_from_index,
    get_topics_options_from_index,
    )
from pages.Usecase_1.sk_plugins_uc1 import AzureConfig, ConversationMemoryPlugin
from pages.header import render_header
from pages.Usecase_1.upload_files import (
    dropdown_with_add, 
    render_upload_section, 
    perform_file_upload, 
    clear_upload_form
    )
from pages.Usecase_1.custom_css import get_custom_css
from pages.Usecase_1.helper_fn import (
    generate_session_name, 
    sort_sessions_by_last_updated, 
    update_session_data,
    render_conversation_item,
    clear_input_fields,
    handle_query_submission
    )

# Add imports for Azure Search
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import os
import json
import importlib.util
import tempfile
import os
import base64

# =================== REUSABLE FUNCTIONS ===================

def fetch_all_sessions_from_memory():
    """Fetch all conversations from persistent memory and reconstruct sessions."""
    # Check if we already have sessions loaded to avoid unnecessary API calls
    if "sessions_loaded" not in st.session_state:
        st.session_state["sessions_loaded"] = False
    
    if st.session_state["sessions_loaded"] and "all_sessions" in st.session_state:
        return st.session_state["all_sessions"]
    
    config = AzureConfig()
    memory_plugin = ConversationMemoryPlugin(config)
    import asyncio

    async def fetch():
        kernel = None  # Not used by retrieve_all_conversations
        arguments = {}
        # Add user filter if logged in
        if "username" in st.session_state and st.session_state["username"]:
            arguments["user"] = st.session_state["username"]
        result = await memory_plugin.retrieve_all_conversations(kernel, arguments)
        data = result
        if hasattr(result, "to_string"):
            data = result.to_string()
        try:
            conversations = []
            if isinstance(data, str):
                data = json.loads(data)
            conversations = data.get("conversations", [])
        except Exception:
            conversations = []
        # Group conversations by session (using authors+scenario+query as a composite key)
        sessions = {}
        for conv in conversations:
            # Use authors+scenario as a session key (customize as needed)
            session_key = (
                ",".join(conv.get("authors", []))
                + "|"
                + conv.get("scenario", "")[:30]
            )
            if session_key not in sessions:
                sessions[session_key] = {
                    "session_id": session_key,
                    "session_name": generate_session_name(conv),
                    "created_timestamp": conv.get("timestamp", ""),
                    "last_updated": conv.get("timestamp", ""),
                    "conversations": [],
                }
            sessions[session_key]["conversations"].append(conv)
            # Update last_updated
            sessions[session_key]["last_updated"] = conv.get("timestamp", "")
        # Sort sessions by last_updated
        session_list = list(sessions.values())
        session_list.sort(
            key=lambda x: x.get("last_updated", x["created_timestamp"]), reverse=True
        )
        return session_list

    sessions = asyncio.run(fetch())
    st.session_state["all_sessions"] = sessions
    st.session_state["sessions_loaded"] = True
    return sessions


def get_dropdown_options():
    """Get dropdown options using session state instead of caching"""
    # Initialize options_loaded flag
    if "options_loaded" not in st.session_state:
        st.session_state["options_loaded"] = False
    
    # Check if options need to be reloaded
    if "reload_dropdown_options" in st.session_state and st.session_state["reload_dropdown_options"]:
        st.session_state["options_loaded"] = False
        st.session_state["reload_dropdown_options"] = False
    
    # Load options if not already loaded
    if not st.session_state["options_loaded"]:
        try:
            st.session_state["author_options"] = get_author_options_from_index()
            st.session_state["framework_options"] = get_framework_options_from_index()
            st.session_state["accounting_standards_options"] = get_accounting_standards_options_from_index()
            st.session_state["topics_options"] = get_topics_options_from_index()
            st.session_state["options_loaded"] = True
        except Exception as e:
            st.error(f"Error loading dropdown options: {str(e)}")
            # Set default empty options
            st.session_state["author_options"] = []
            st.session_state["framework_options"] = []
            st.session_state["accounting_standards_options"] = []
            st.session_state["topics_options"] = []
    
    return {
        "author_options": st.session_state.get("author_options", []),
        "framework_options": st.session_state.get("framework_options", []),
        "accounting_standards_options": st.session_state.get("accounting_standards_options", []),
        "topics_options": st.session_state.get("topics_options", [])
    }


def initialize_session_state():
    """Initialize all session state variables without triggering reruns"""
    base_states = {
        "all_sessions": [],
        "current_session_id": None,
        "current_conversations": [],
        "author_select": [],
        "scenario_input": "",
        "question_input": "",
        "reset_counter": 0,
        "sidebar_expanded": False,
        "input_preserve_key": 0,
        "orchestrators": {},
        "last_cleared": False,
        "show_upload_section": False,
        "memory_loaded": False,
        "options_loaded": False,
        "sessions_loaded": False,
        "show_upload_page": False,
        "upload_in_progress": False,
        "upload_success": False,
        "should_rerun": False
    }
    
    # Initialize without triggering reruns
    for key, default_value in base_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def save_current_session():
    """Save current session if it has conversations"""
    if not st.session_state.current_conversations or not st.session_state.current_session_id:
        return
    
    # Check if session already exists
    session_exists = any(
        s["session_id"] == st.session_state.current_session_id 
        for s in st.session_state.all_sessions
    )
    
    if session_exists:
        update_session_data(st.session_state.current_session_id, st.session_state.current_conversations)
    else:
        # Create new session entry
        latest_conv = st.session_state.current_conversations[-1]
        session_data = {
            "session_id": st.session_state.current_session_id,
            "session_name": generate_session_name(latest_conv),
            "created_timestamp": st.session_state.current_conversations[0]["timestamp"],
            "last_updated": latest_conv["timestamp"],
            "conversations": st.session_state.current_conversations.copy()
        }
        st.session_state.all_sessions.append(session_data)
    
    sort_sessions_by_last_updated()


def create_new_session():
    """Create new session without unnecessary reruns"""
    if st.session_state.current_conversations:
        save_current_session()
    
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.current_conversations = []
    clear_input_fields()


def add_current_session_to_sidebar():
    """Add current session to sidebar when first conversation is created"""
    if not st.session_state.current_conversations or not st.session_state.current_session_id:
        return
    
    # Check if session already exists in sidebar
    session_exists = any(
        s["session_id"] == st.session_state.current_session_id 
        for s in st.session_state.all_sessions
    )
    
    if not session_exists:
        latest_conv = st.session_state.current_conversations[-1]
        session_data = {
            "session_id": st.session_state.current_session_id,
            "session_name": generate_session_name(latest_conv),
            "created_timestamp": st.session_state.current_conversations[0]["timestamp"],
            "last_updated": latest_conv["timestamp"],
            "conversations": st.session_state.current_conversations.copy()
        }
        st.session_state.all_sessions.append(session_data)
        sort_sessions_by_last_updated()


def switch_to_session(session_id):
    """Switch to a specific session and load its conversations from memory"""
    save_current_session()
    # Load selected session from all_sessions
    for session in st.session_state.all_sessions:
        if session["session_id"] == session_id:
            st.session_state.current_session_id = session_id
            st.session_state.current_conversations = session["conversations"].copy()
            clear_input_fields()
            st.session_state["should_rerun"] = True
            break


def render_sidebar_sessions():
    """Render the sessions in the sidebar"""
    with st.sidebar:
        st.session_state.sidebar_expanded = True
        st.markdown("### 💬 Sessions")
        
        # New Session Button
        if st.button("➕ New Session", key="new_session_btn"):
            create_new_session()
            st.session_state["should_rerun"] = True
            st.rerun()
        
        st.markdown("---")
        
        # Load sessions if not already loaded
        if not st.session_state.get("sessions_loaded", False):
            st.session_state["all_sessions"] = fetch_all_sessions_from_memory()
        
        # Display all sessions (ordered by most recent activity)
        if st.session_state.all_sessions:
            for i, session in enumerate(st.session_state.all_sessions):
                is_active = session["session_id"] == st.session_state.current_session_id
                
                # Create session button with active styling
                st.markdown('<div class="sidebar-session-container">', unsafe_allow_html=True)
                
                if is_active:
                    st.markdown('<div class="active-session">', unsafe_allow_html=True)
                
                if st.button(
                    f"📋 {session['session_name']}", 
                    key=f"session_btn_{session['session_id']}",
                    type="secondary"
                ):
                    switch_to_session(session["session_id"])
                    st.session_state["should_rerun"] = True
                    st.rerun()
                
                if is_active:
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add minimal separator between sessions
                if i < len(st.session_state.all_sessions) - 1:
                    st.markdown('<div class="session-separator"></div>', unsafe_allow_html=True)
        
        # Show current session status
        if st.session_state.current_session_id:
            if not any(s["session_id"] == st.session_state.current_session_id for s in st.session_state.all_sessions):
                st.markdown("---")
                st.markdown("**Current Session:** New (unsaved)")
                st.markdown(f"**Conversations:** {len(st.session_state.current_conversations)}")
        else:
            st.info("No active session. Submit a query to start.")


def upload_file_toggle():
    """Sidebar toggle between Upload Files and Research modes."""
    with st.sidebar:
        if st.session_state.get("show_upload_page", False):
            toggle_label = "🔍 Research"
        else:
            toggle_label = "📤 Upload Files"
        
        if st.button(toggle_label, key="toggle_upload_page_btn"):
            st.session_state["show_upload_page"] = not st.session_state.get("show_upload_page", False)
            st.session_state["show_upload_section"] = st.session_state["show_upload_page"]
            st.rerun()


def research_section(options):
    """Render the research section with form"""
    col1, col_sep, col2 = st.columns([0.99, 0.02, 0.99])
    
    with col1:
        st.subheader("Ask a New Question")
        with st.form(key='query_form'):
            selected_framework = st.selectbox(
                "Framework:",
                options["framework_options"],
                index=None,
                placeholder="Select Framework",
                key="query_framework"
            )
            selected_accounting_standards = st.selectbox(
                "Accounting Standards:",
                options["accounting_standards_options"],
                index=None,
                placeholder="Select Accounting Standards",
                key="query_accounting_standards"
            )
            selected_topics = st.selectbox(
                "Topics:",
                options["topics_options"],
                index=None,
                placeholder="Select Topics",
                key="query_topics"
            )
            selected_authors = st.multiselect(
                "Select Author:",
                options["author_options"],
                key=f"author_select_{st.session_state.input_preserve_key}"
            )
            scenario = st.text_area(
                "Research Scenario (Context):", 
                height=100,
                key=f"scenario_input_{st.session_state.input_preserve_key}"
            )
            question = st.text_area(
                "Specific Question:", 
                height=100,
                key=f"question_input_{st.session_state.input_preserve_key}"
            )
            
            col1_btn, col_sep, col2_btn = st.columns([0.99, 0.02, 0.99])
            with col1_btn:
                clear_button = st.form_submit_button("Clear inputs", on_click=clear_input_fields)
            with col2_btn:
                submitted = st.form_submit_button("Submit Query")
                if submitted:
                    with st.spinner("Generating response..."):
                        success = handle_query_submission(
                            selected_framework,
                            selected_accounting_standards,
                            selected_topics,
                            selected_authors,
                            scenario,
                            question
                        )
                    if success:
                        st.session_state["show_response"] = True
                        add_current_session_to_sidebar()
                    else:
                        st.session_state["show_response"] = False

    with col2:
        if not st.session_state.current_conversations:
            st.info("No conversations yet. Submit a query to get started.")
        else:
            for i, conversation in enumerate(reversed(st.session_state.current_conversations)):
                render_conversation_item(conversation, i, is_latest=(i == 0))


# =================== MAIN APPLICATION ===================
def main():
    """Main application function"""
    initialize_session_state()
    render_header()

    # Get dropdown options using session state
    options = get_dropdown_options()

    st.markdown(get_custom_css(), unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center;'>Accounting Research Chatbot</h1>",
        unsafe_allow_html=True
    )
    
    # Render upload toggle
    upload_file_toggle()
    
    # Render main content based on mode
    if st.session_state.get("show_upload_page", False):
        render_upload_section(None, options["framework_options"], 
                              options["accounting_standards_options"], 
                              options["topics_options"], 
                              options["author_options"])
    else:
        research_section(options)
    
    # Render sidebar sessions
    render_sidebar_sessions()
    
    # Save current session
    save_current_session()
    
    # Render service controls in sidebar
    with st.sidebar:
        st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
        if st.button("Switch Service", key="switch_service_sidebar_btn"):
            st.session_state.selected_service = None
            st.rerun()
        
        if st.button("Logout", key="logout_sidebar_btn"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state["sidebar_collapsed"] = True
            st.rerun()
    
    # Handle sidebar collapse
    if st.session_state.get("sidebar_collapsed"):
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"]{
                min-width: 0rem;
                width: 0rem;
                overflow-x: hidden;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    
    # Apply main content styling
    main_class = "main-content sidebar-expanded" if st.session_state.sidebar_expanded else "main-content"
    st.markdown(f'<div class="{main_class}"></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()