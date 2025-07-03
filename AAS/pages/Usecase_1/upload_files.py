import streamlit as st
import tempfile
import os
import importlib.util
import PyPDF2

def dropdown_with_add(label, options, state_key, add_key, new_values_key):
    """
    Create a dropdown with option to add custom values. Rerun UI when 'Add new...' is selected to update widget.
    """
    # Initialize state keys if they don't exist
    if add_key not in st.session_state:
        st.session_state[add_key] = False
    if state_key not in st.session_state:
        st.session_state[state_key] = ""
    if new_values_key not in st.session_state:
        st.session_state[new_values_key] = []

    is_adding_new = st.session_state.get(add_key, False)

    if not is_adding_new:
        # Merge options with any new values added in this session
        merged_options = options + [v for v in st.session_state[new_values_key] if v not in options]
        options_display = merged_options + ["Add new..."]
        selected = st.selectbox(
            label,
            options_display,
            index=options_display.index(st.session_state[state_key]) if st.session_state[state_key] in options_display else None,
            placeholder=f"Select {label}",
            key=f"{state_key}_dropdown"
        )
        if selected == "Add new...":
            st.session_state[add_key] = True
            st.session_state[state_key] = ""
            st.rerun()  # Rerun to update widget from dropdown to text input
            return ""
        elif selected and selected != "Add new...":
            st.session_state[state_key] = selected
            return selected
        else:
            return ""
    else:
        # Show text input for custom value with cancel button integrated
        col1, col2 = st.columns([9, 1])
        with col1:
            new_val = st.text_input(
                f"Enter new {label.lower()}:",
                key=f"{state_key}_text_input",
                placeholder=f"Type custom {label.lower()} here..."
            )
        with col2:
            cancel_clicked = st.button("✗", key=f"cancel_{state_key}", help="Cancel and go back to dropdown")
        # Auto-save when user enters value (no explicit save button needed)
        if new_val and new_val.strip():
            new_val_clean = new_val.strip()
            if new_val_clean not in st.session_state[new_values_key]:
                st.session_state[new_values_key].append(new_val_clean)
            st.session_state[state_key] = new_val_clean
            st.session_state[add_key] = False
            return new_val_clean
        if cancel_clicked:
            st.session_state[add_key] = False
            st.session_state[state_key] = ""
            st.rerun()
        return st.session_state.get(state_key, "")
    return st.session_state.get(state_key, "")

def render_upload_section(uploaded_file, framework_options, accounting_standards_options, topics_options, author_options):
    """Render the upload section with all widgets and upload button outside any form (no st.form used)."""
    if st.session_state["show_upload_section"]:
        st.markdown("---")
        st.markdown("<h3>Document Metadata</>", unsafe_allow_html=True)

        # All dropdowns and custom value logic outside any form
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
            "author/User Type", 
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

        if uploaded_file:
            st.success(f"File selected: {uploaded_file.name}")
        upload_ready = (
            uploaded_file and 
            upload_framework and 
            upload_accounting_standards and 
            upload_topics and 
            upload_author
        )

        # --- Add upload_in_progress state ---
        if "upload_in_progress" not in st.session_state:
            st.session_state["upload_in_progress"] = False

        # Upload button (no form)
        if st.button("🚀 Start Upload", disabled=not upload_ready or st.session_state["upload_in_progress"], use_container_width=True):
            if upload_ready:
                st.session_state["upload_in_progress"] = True
                st.session_state["upload_success"] = False
                st.rerun()  # Rerun to immediately disable the button

        # If upload_in_progress, start the upload process
        if st.session_state.get("upload_in_progress", False):
            if upload_ready and uploaded_file:
                perform_file_upload(
                    uploaded_file, 
                    upload_framework, 
                    upload_accounting_standards, 
                    upload_topics, 
                    upload_author
                )
                st.session_state["upload_in_progress"] = False
                st.session_state["upload_success"] = True
                # Do NOT rerun here, just show success and let user continue

        # Show success message if upload was successful
        if st.session_state.get("upload_success", False):
            st.success("🎉 File uploaded and indexed successfully! Dropdowns will refresh if needed.")
            # Here you should reload dropdown options from your persistent source if needed
            # For example: framework_options = load_framework_options()
            # (Assume this is handled in the main app logic)

        # Show status of required fields
        if not upload_ready:
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
                missing_fields.append("author/User Type")
            if missing_fields:
                st.warning(f"Please complete: {', '.join(missing_fields)}")

def perform_file_upload(uploaded_file, framework, accounting_standards, topics, author):
    """Handle the actual file upload process"""
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
        # Inject CSS for spinning emoji (slower)
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
        progress_text = st.empty()
        def progress_callback(page_idx, total_pages):
            percent = int((page_idx + 1) / total_pages * 100)
            progress_text.markdown(
                f"<span class='spin-emoji'>⏳</span> <span style='font-weight:bold;'>Processing page {page_idx + 1} of {total_pages}...</span> "
                f"<span style='font-weight:bold;'>{percent}%</span>",
                unsafe_allow_html=True
            )
        # Initial spinner
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
        st.success(f"🎉 File '{uploaded_file.name}' uploaded and indexed successfully!")
        clear_upload_form()
        st.session_state["show_upload_section"] = False
        st.session_state["upload_success"] = True
        st.session_state["reload_dropdown_options"] = True  # Ensure dropdowns reload after upload
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(output_file):
            os.remove(output_file)
        # st.rerun() removed so success message remains visible
    except Exception as e:
        st.error(f"❌ Upload failed: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def clear_upload_form():
    """Clear all upload form fields and states"""
    upload_keys = [
        "upload_framework", "upload_accounting_standards", "upload_topics", "upload_author",
        "add_framework", "add_accounting_standards", "add_topics", "add_author",
        "file_upload"
    ]
    current_preserve_key = st.session_state.get('input_preserve_key', 0)
    for field in ["upload_framework", "upload_accounting_standards", "upload_topics", "upload_author"]:
        dropdown_key = f"{field}_dropdown_{current_preserve_key}"
        text_input_key = f"{field}_text_input_{current_preserve_key}"
        if dropdown_key in st.session_state:
            del st.session_state[dropdown_key]
        if text_input_key in st.session_state:
            del st.session_state[text_input_key]
    for key in upload_keys:
        if key in st.session_state:
            del st.session_state[key]

def render_header_global():
    """Render the header at the very top, always consistent across reruns and page switches."""
    import streamlit as st
    from pages.header import render_header
    render_header()
