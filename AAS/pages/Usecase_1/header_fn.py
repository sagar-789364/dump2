from datetime import datetime
import streamlit as st
import uuid
from pages.Usecase_1.rag_sk_uc1 import (
    generate_guidance_response_sync,
    generate_document_sync
)

def generate_session_name(conversation):
    """Generate a session name based on conversation data"""
    # Always display authors as a list, separated by ', '
    authors = conversation.get("authors", [])
    if isinstance(authors, str):
        author_list = [a.strip() for a in authors.split(",") if a.strip()]
    else:
        author_list = authors
    author_str = ", ".join(author_list)
    scenario_keywords = " ".join(conversation.get("scenario", "").split()[:2])

    # Use 'question' directly (always present)
    question = conversation.get("question", "")
    if question:
        question_keywords = " ".join(question.split()[:2])
    else:
        question_keywords = "No Question"

    combined_keywords = f"{scenario_keywords}... | {question_keywords}...".strip()

    timestamp_str = conversation.get("timestamp", "")
    timestamp_obj = None
    if timestamp_str:
        try:
            # Try ISO 8601 format with 'Z'
            if timestamp_str.endswith("Z"):
                timestamp_obj = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
            else:
                timestamp_obj = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                # Try fromisoformat (Python 3.11+ handles 'Z' as UTC)
                timestamp_obj = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except Exception:
                timestamp_obj = None

    if timestamp_obj is None:
        # Fallback to current timestamp if parsing fails
        timestamp_obj = datetime.now()

    date_str = timestamp_obj.strftime("%m/%d/%Y")
    time_str = timestamp_obj.strftime("%H:%M")

    return f"{author_str} | {combined_keywords} | {date_str} | {time_str}"


def sort_sessions_by_last_updated():
    """Sort sessions by last_updated timestamp (most recent first)"""
    st.session_state.all_sessions.sort(
        key=lambda x: x.get("last_updated", x["created_timestamp"]), 
        reverse=True
    )

def update_session_data(session_id, conversations):
    """Update existing session with latest conversation data"""
    for i, session in enumerate(st.session_state.all_sessions):
        if session["session_id"] == session_id:
            latest_conv = conversations[-1]
            st.session_state.all_sessions[i]["session_name"] = generate_session_name(latest_conv)
            st.session_state.all_sessions[i]["last_updated"] = latest_conv["timestamp"]
            st.session_state.all_sessions[i]["conversations"] = conversations.copy()
            break


def clear_input_fields():
    """Clear input fields without triggering multiple reruns"""
    st.session_state.framework_select = None
    st.session_state.accounting_standards_select = None
    st.session_state.topics_select = None
    st.session_state.author_select = []
    st.session_state.scenario_input = ""
    st.session_state.question_input = ""
    st.session_state.input_preserve_key += 1
    
    # Force clear the form inputs
    if f"framework_select_{st.session_state.input_preserve_key-1}" in st.session_state:
        del st.session_state[f"framework_select_{st.session_state.input_preserve_key-1}"]
    if f"accounting_standards_select_{st.session_state.input_preserve_key-1}" in st.session_state:
        del st.session_state[f"accounting_standards_select_{st.session_state.input_preserve_key-1}"]
    if f"topics_select_{st.session_state.input_preserve_key-1}" in st.session_state:
        del st.session_state[f"topics_select_{st.session_state.input_preserve_key-1}"]
    if f"author_select_{st.session_state.input_preserve_key-1}" in st.session_state:
        del st.session_state[f"author_select_{st.session_state.input_preserve_key-1}"]
    if f"scenario_input_{st.session_state.input_preserve_key-1}" in st.session_state:
        del st.session_state[f"scenario_input_{st.session_state.input_preserve_key-1}"]
    if f"question_input_{st.session_state.input_preserve_key-1}" in st.session_state:
        del st.session_state[f"question_input_{st.session_state.input_preserve_key-1}"]


def generate_document_for_conversation(conversation, index):
    """Generate document for a specific conversation using the orchestrator"""
    try:
        conversation_id = f"{st.session_state.current_session_id}_{index}"
        if conversation_id in st.session_state.orchestrators:
            orchestrator = st.session_state.orchestrators[conversation_id]
            filename = f"conversation_{index}.docx"
            # Pass filename as before; orchestrator uses last_response_data which already includes scenario, query, response
            doc_bytes = generate_document_sync(orchestrator, filename)
            return doc_bytes
        else:
            st.warning("Document generation failed: Orchestrator not found.")
            return None
    except Exception as e:
        st.error(f"Error generating document: {str(e)}")
        return None


def handle_query_submission(selected_framework, selected_accounting_standards, selected_topics, selected_authors, scenario, question):
    """Handle query submission without unnecessary reruns"""
    if not (selected_authors and scenario and question):
        st.warning("Please fill in all fields before submitting.")
        return False

    try:
        framework = selected_framework
        accounting_standards = selected_accounting_standards
        topics = selected_topics

        response, orchestrator = generate_guidance_response_sync(
            selected_authors, scenario, question, 
            st.session_state["username"], framework, 
            accounting_standards, topics
        )

        if response.startswith("Unable to Answer"):
            st.error(response)
            return False

        if not st.session_state.current_session_id:
            st.session_state.current_session_id = str(uuid.uuid4())

        new_conversation = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "authors": selected_authors,
            "scenario": scenario,
            "question": question,
            "response": response,
            "user": st.session_state["username"],
            "framework": framework,
            "accounting_standards": accounting_standards,
            "topics": topics
        }

        st.session_state.current_conversations.append(new_conversation)
        conversation_index = len(st.session_state.current_conversations) - 1
        conversation_id = f"{st.session_state.current_session_id}_{conversation_index}"
        st.session_state.orchestrators[conversation_id] = orchestrator

        # Local import to avoid circular import
        from pages.Usecase_1.app_uc1 import add_current_session_to_sidebar
        add_current_session_to_sidebar()

        st.session_state.question_input = ""
        return True

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return False

def render_conversation_item(conversation, index, is_latest=False):
    """Render a single conversation item"""
    authors = conversation.get("authors", [])
    if isinstance(authors, str):
        author_list = [a.strip() for a in authors.split(",") if a.strip()]
    else:
        author_list = authors
    author_str = ", ".join(author_list)
    framework = conversation.get("framework", "")
    accounting_standards = conversation.get("accounting_standards", "")
    topics = conversation.get("topics", "")

    if is_latest:
        row1, row2 = st.columns([0.85, 0.15])
        with row1:
            st.markdown(f"### 🕒 {author_str}")
        with row2:
            doc_bytes = generate_document_for_conversation(conversation, len(st.session_state.current_conversations) - 1 - index)
            if doc_bytes:
                st.download_button(
                    label="📥",
                    data=doc_bytes,
                    file_name=f"conversation_{index}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"latest_download_{index}"
                )
            else:
                st.button("📥", disabled=True, key=f"latest_download_disabled_{index}")
        st.markdown(f"👤 **User:**")
        st.markdown(f"**Framework:** {framework}")
        st.markdown(f"**Accounting Standards:** {accounting_standards}")
        st.markdown(f"**Topics:** {topics}")
        st.markdown(f"**Author(s):** {author_str}")
        st.markdown(f"**Scenario:** {conversation['scenario']}")
        st.markdown(f"**Question:** {conversation['question']}")
        st.markdown("---")
        st.markdown(f"🤖 **Chatbot Response:**")
        response_parts = conversation["response"].split("## ")
        for part in response_parts:
            if part.strip():
                lines = part.split("\n", 1)
                if len(lines) == 2:
                    title, content = lines
                    if title.strip() == "Sources:":
                        with st.expander("## Sources"):
                            st.markdown(content)
                    else:
                        st.markdown(f"## {title}")
                        st.markdown(content)
        st.markdown("---")
    else:
        scenario_keywords = " ".join(conversation["scenario"].split()[:5]) + "...";
        question_keywords = " ".join(conversation["question"].split()[:5]) + "...";
        title = f"📂 {author_str} | {scenario_keywords} | {question_keywords}";
        with st.expander(title):
            row1, row2 = st.columns([0.85, 0.15])
            with row1:
                st.markdown(f"🕒 **Timestamp:** {conversation['timestamp']}")
            with row2:
                doc_bytes = generate_document_for_conversation(conversation, len(st.session_state.current_conversations) - 1 - index)
                if doc_bytes:
                    st.download_button(
                        label="📥",
                        data=doc_bytes,
                        file_name=f"conversation_{index}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key=f"conversation_download_{index}"
                    )
                else:
                    st.button("📥", disabled=True, key=f"conversation_download_disabled_{index}")
            st.markdown(f"👤 **User**")
            st.markdown(f"**Author(s):** {author_str}")
            st.markdown(f"**Framework:** {framework}")
            st.markdown(f"**Accounting Standards:** {accounting_standards}")
            st.markdown(f"**Topics:** {topics}")
            st.markdown(f"**Scenario:** {conversation['scenario']}")
            st.markdown(f"**Question:** {conversation['question']}")
            st.markdown("🤖 **Chatbot Response:**")
            st.markdown(conversation["response"])
