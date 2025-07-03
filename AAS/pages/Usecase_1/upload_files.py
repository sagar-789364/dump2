import streamlit as st
import tempfile
import os
import importlib.util
import PyPDF2

def initialize_upload_session_state():
    """Initialize all session state variables for upload functionality"""
    upload_state_keys = [
        "upload_framework", "upload_accounting_standards", "upload_topics", "upload_author",
        "add_framework", "add_accounting_standards", "add_topics", "add_author",
        "new_frameworks", "new_accounting_standards", "new_topics", "new_authors",
        "upload_in_progress", "upload_success", "show_upload_section",
        "reload_dropdown_options", "file_upload_completed"
    ]
    
    for key in upload_state_keys:
        if key not in st.session_state:
            if key.startswith("add_") or key in ["upload_in_progress", "upload_success", "show_upload_section", "reload_dropdown_options", "file_upload_completed"]:
                st.session_state[key] = False
            elif key.startswith("new_"):
                st.session_state[key] = []
            else:
                st.session_state[key] = ""

def dropdown_with_add(label, options, state_key, add_key, new_values_key):
    """
    Create a dropdown with option to add custom values using session state management.
    """
    # Initialize state keys if they don't exist
    initialize_upload_session_state()
    
    # Check if we're in adding mode
    is_adding_new = st.session_state.get(add_key, False)

    if not is_adding_new:
        # Merge options with any new values added in this session
        merged_options = options + [v for v in st.session_state[new_values_key] if v not in options]
        options_display = merged_options + ["Add new..."]
        
        # Get current index for selectbox
        current_value = st.session_state.get(state_key, "")
        current_index = 0
        if current_value and current_value in options_display:
            current_index = options_display.index(current_value)
        
        selected = st.selectbox(
            label,
            options_display,
            index=current_index,
            placeholder=f"Select {label}",
            key=f"{state_key}_dropdown"
        )
        
        if selected == "Add new...":
            st.session_state[add_key] = True
            st.session_state[state_key] = ""
            st.rerun()
            return ""
        elif selected and selected != "Add new...":
            st.session_state[state_key] = selected
            return selected
        else:
            return st.session_state.get(state_key, "")
    else:
        # Show text input for custom value
        col1, col2 = st.columns([9, 1])
        with col1:
            new_val = st.text_input(
                f"Enter new {label.lower()}:",
                key=f"{state_key}_text_input",
                placeholder=f"Type custom {label.lower()} here...",
                value=st.session_state.get(state_key, "")
            )
        with col2:
            cancel_clicked = st.button("✗", key=f"cancel_{state_key}", help="Cancel and go back to dropdown")
        
        # Handle new value addition
        if new_val and new_val.strip():
            new_val_clean = new_val.strip()
            if new_val_clean not in st.session_state[new_values_key]:
                st.session_state[new_values_key].append(new_val_clean)
            st.session_state[state_key] = new_val_clean
            st.session_state[add_key] = False
            st.rerun()
            return new_val_clean
        
        # Handle cancel
        if cancel_clicked:
            st.session_state[add_key] = False
            st.session_state[state_key] = ""
            st.rerun()
            return ""
            
        return st.session_state.get(state_key, "")

def render_upload_section(uploaded_file, framework_options, accounting_standards_options, topics_options, author_options):
    """Render the upload section with session state management."""
    
    # Initialize session state
    initialize_upload_session_state()
    
    # Only render if upload section should be shown
    if not st.session_state.get("show_upload_section", False):
        return
    
    st.markdown("---")
    st.markdown("<h3>Document Metadata</h3>", unsafe_allow_html=True)

    # All dropdowns using session state
    upload_framework = dropdown_with_add(
        "Framework", 
        framework_options, 
        "upload_framework", 
        "add_framework", 
        "new_frameworks"
    )
    
    upload_accounting_standards = dropdown_with_add(
        "Accounting Standards", 
        accounting_standards_options, 
        "upload_accounting_standards", 
        "add_accounting_standards", 
        "new_accounting_standards"
    )
    
    upload_topics = dropdown_with_add(
        "Topics", 
        topics_options, 
        "upload_topics", 
        "add_topics", 
        "new_topics"
    )
    
    upload_author = dropdown_with_add(
        "Author/User Type", 
        author_options, 
        "upload_author", 
        "add_author", 
        "new_authors"
    )

    st.markdown("<h3>Upload PDF Files</h3>", unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file to upload", 
        type=["pdf", "docx", "txt"], 
        key="file_upload"
    )

    # Update session state with uploaded file info
    if uploaded_file:
        st.success(f"File selected: {uploaded_file.name}")
        st.session_state["current_uploaded_file"] = uploaded_file
    
    # Check if upload is ready
    upload_ready = (
        uploaded_file and 
        upload_framework and 
        upload_accounting_standards and 
        upload_topics and 
        upload_author
    )

    # Upload button logic with session state
    upload_button_disabled = not upload_ready or st.session_state.get("upload_in_progress", False)
    
    if st.button("🚀 Start Upload", disabled=upload_button_disabled, use_container_width=True):
        if upload_ready:
            st.session_state["upload_in_progress"] = True
            st.session_state["upload_success"] = False
            st.rerun()

    # Handle upload process
    if st.session_state.get("upload_in_progress", False) and upload_ready and uploaded_file:
        perform_file_upload(
            uploaded_file, 
            upload_framework, 
            upload_accounting_standards, 
            upload_topics, 
            upload_author
        )

    # Show success message
    if st.session_state.get("upload_success", False):
        st.success("🎉 File uploaded and indexed successfully! You can now search for content from this document.")
        # Reset success state after showing message
        if st.button("Upload Another File", key="upload_another"):
            reset_upload_session_state()
            st.rerun()

    # Show status of required fields
    if not upload_ready and not st.session_state.get("upload_in_progress", False):
        missing_fields = []
        if not uploaded_file:
            missing_fields.append("File")
        if not upload_framework:
            missing_fields.append("Framework")
        if not upload_accounting_standards:
            missing_fields.append("Accounting Standards")
        if not upload_topics:
            missing_fields.append("Topics")
        if not upload_author:
            missing_fields.append("Author/User Type")
        if missing_fields:
            st.warning(f"Please complete: {', '.join(missing_fields)}")

def perform_file_upload(uploaded_file, framework, accounting_standards, topics, author):
    """Handle the actual file upload process with session state management"""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    try:
        # Save uploaded file to temp location
        with open(temp_path, "wb") as file_writer:
            file_writer.write(uploaded_file.getbuffer())
        
        # Import the upload module
        spec = importlib.util.spec_from_file_location(
            "pdf_image_embedd_azure_index",
            os.path.join(os.path.dirname(__file__), "pdf_image_embedd_azure_index.py")
        )
        pdf_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pdf_module)
        
        output_file = os.path.splitext(temp_path)[0] + "_extracted.txt"
        
        # Get number of pages for progress tracking
        num_pages = 10
        if uploaded_file.name.lower().endswith(".pdf"):
            try:
                reader = PyPDF2.PdfReader(temp_path)
                num_pages = len(reader.pages)
            except Exception:
                pass
        
        # Inject CSS for spinning emoji
        st.markdown(
            """
            <style>
            .spin-emoji {
                display: inline-block;
                animation: spin 2.5s linear infinite;
                font-size: 1.5em;
                vertical-align: middle;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Progress tracking
        progress_text = st.empty()
        
        def progress_callback(page_idx, total_pages):
            percent = int((page_idx + 1) / total_pages * 100)
            progress_text.markdown(
                f"<span class='spin-emoji'>⏳</span> <span style='font-weight:bold;'>Processing page {page_idx + 1} of {total_pages}...</span> "
                f"<span style='font-weight:bold;'>{percent}%</span>",
                unsafe_allow_html=True
            )
        
        # Initial progress display
        progress_text.markdown(
            f"<span class='spin-emoji'>⏳</span> <span style='font-weight:bold;'>Processing page 1 of {num_pages}...</span> <span style='font-weight:bold;'>0%</span>",
            unsafe_allow_html=True
        )
        
        # Process the file
        try:
            pdf_module.process_images_to_indexes_and_upload(
                temp_path,
                output_file,
                framework=framework,
                accounting_standards=accounting_standards,
                topics=topics,
                author=author,
                progress_callback=progress_callback
            )
        except TypeError:
            # Fallback for modules that don't support progress callback
            pdf_module.process_images_to_indexes_and_upload(
                temp_path,
                output_file,
                framework=framework,
                accounting_standards=accounting_standards,
                topics=topics,
                author=author
            )
            
            # Manual progress simulation
            for i in range(num_pages):
                percent = int((i + 1) / num_pages * 100)
                progress_text.markdown(
                    f"<span class='spin-emoji'>⏳</span> <span style='font-weight:bold;'>Processing page {i + 1} of {num_pages}...</span> "
                    f"<span style='font-weight:bold;'>{percent}%</span>",
                    unsafe_allow_html=True
                )
        
        # Show completion
        progress_text.markdown(
            f"<span style='font-size:1.5em;'>✅</span> <span style='font-weight:bold;color:green;'>Upload complete!</span> <span style='font-weight:bold;color:green;'>100%</span>",
            unsafe_allow_html=True
        )
        
        # Update session state
        st.session_state["upload_in_progress"] = False
        st.session_state["upload_success"] = True
        st.session_state["file_upload_completed"] = True
        st.session_state["reload_dropdown_options"] = True
        
        # Clean up temp files
        cleanup_temp_files(temp_path, output_file)
        
    except Exception as e:
        st.error(f"❌ Upload failed: {str(e)}")
        st.session_state["upload_in_progress"] = False
        st.session_state["upload_success"] = False
        cleanup_temp_files(temp_path, output_file)

def cleanup_temp_files(temp_path, output_file):
    """Clean up temporary files"""
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(output_file):
            os.remove(output_file)
    except Exception as e:
        st.warning(f"Could not clean up temporary files: {str(e)}")

def reset_upload_session_state():
    """Reset upload-related session state for new upload"""
    upload_keys_to_reset = [
        "upload_framework", "upload_accounting_standards", "upload_topics", "upload_author",
        "add_framework", "add_accounting_standards", "add_topics", "add_author",
        "upload_in_progress", "upload_success", "file_upload_completed",
        "current_uploaded_file"
    ]
    
    for key in upload_keys_to_reset:
        if key in st.session_state:
            if key.startswith("add_") or key in ["upload_in_progress", "upload_success", "file_upload_completed"]:
                st.session_state[key] = False
            else:
                st.session_state[key] = ""

def clear_upload_form():
    """Clear all upload form fields and states - kept for backward compatibility"""
    reset_upload_session_state()
    
    # Clear any remaining UI-specific keys
    current_preserve_key = st.session_state.get('input_preserve_key', 0)
    for field in ["upload_framework", "upload_accounting_standards", "upload_topics", "upload_author"]:
        dropdown_key = f"{field}_dropdown_{current_preserve_key}"
        text_input_key = f"{field}_text_input_{current_preserve_key}"
        if dropdown_key in st.session_state:
            del st.session_state[dropdown_key]
        if text_input_key in st.session_state:
            del st.session_state[text_input_key]

def render_header_global():
    """Render the header at the very top, always consistent across reruns and page switches."""
    import streamlit as st
    from pages.header import render_header
    render_header()

def get_upload_status():
    """Get current upload status from session state"""
    return {
        "in_progress": st.session_state.get("upload_in_progress", False),
        "success": st.session_state.get("upload_success", False),
        "completed": st.session_state.get("file_upload_completed", False),
        "show_section": st.session_state.get("show_upload_section", False)
    }

def set_upload_section_visibility(visible):
    """Set upload section visibility"""
    st.session_state["show_upload_section"] = visible
    if not visible:
        reset_upload_session_state()