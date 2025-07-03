import streamlit as st
import os
from io import StringIO
from contextlib import redirect_stdout
import asyncio
import base64

from pages.Usecase_2.main_uc2 import config, pipeline
from pages.header import render_header


def get_custom_css():
    return """     
    <style>       
    /* ...existing non-header CSS... */
    </style>
    """


def initialize_session_state():
    """Initialize all session state variables for UC2"""
    # UI state management
    if "uc2_base_filename" not in st.session_state:
        st.session_state.uc2_base_filename = "accounting_memorandum"
    
    # Processing state
    if "uc2_processing" not in st.session_state:
        st.session_state.uc2_processing = False
    
    if "uc2_progress" not in st.session_state:
        st.session_state.uc2_progress = 0
    
    if "uc2_status" not in st.session_state:
        st.session_state.uc2_status = ""
    
    # Output file tracking
    if "uc2_last_docx" not in st.session_state:
        st.session_state.uc2_last_docx = None
    
    if "uc2_last_pdf" not in st.session_state:
        st.session_state.uc2_last_pdf = None
    
    # Generation results
    if "uc2_generation_complete" not in st.session_state:
        st.session_state.uc2_generation_complete = False
    
    if "uc2_generation_error" not in st.session_state:
        st.session_state.uc2_generation_error = None
    
    # Button states
    if "uc2_generate_clicked" not in st.session_state:
        st.session_state.uc2_generate_clicked = False


def convert_docx_to_pdf(docx_path, pdf_path):
    """Convert DOCX to PDF with better error handling"""
    try:
        # Try multiple conversion methods
        conversion_success = False
        
        # Method 1: docx2pdf (Windows/Linux with LibreOffice)
        try:
            from docx2pdf import convert
            convert(docx_path, pdf_path)
            conversion_success = True
            print(f"✅ PDF converted successfully using docx2pdf: {pdf_path}")
        except Exception as e:
            print(f"❌ docx2pdf conversion failed: {e}")
        
        # Method 2: python-docx2txt + reportlab (fallback)
        if not conversion_success:
            try:
                from docx import Document
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.lib.units import inch
                
                # Extract text from DOCX
                doc = Document(docx_path)
                text_content = []
                for para in doc.paragraphs:
                    if para.text.strip():
                        text_content.append(para.text)
                
                # Create PDF
                pdf_doc = SimpleDocTemplate(pdf_path, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                for text in text_content:
                    p = Paragraph(text, styles['Normal'])
                    story.append(p)
                    story.append(Spacer(1, 0.2*inch))
                
                pdf_doc.build(story)
                conversion_success = True
            except Exception as e:
                print(f"❌ reportlab conversion failed: {e}")
        
        return conversion_success
        
    except Exception as e:
        print(f"❌ All PDF conversion methods failed: {e}")
        return False


def render_pdf_preview(pdf_path):
    """Render PDF preview in the UI"""
    try:
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()
                if len(pdf_data) > 0:
                    st.markdown("**📄 PDF Preview:**")
                    try:
                        base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                        pdf_display = f"""
                        <iframe src="data:application/pdf;base64,{base64_pdf}" 
                                width="100%" height="600px"></iframe>
                        """
                        st.markdown(pdf_display, unsafe_allow_html=True)
                    except Exception:
                        st.info("PDF preview not available in browser.")
                    
                    return True
                else:
                    st.error("PDF file is empty.")
                    return False
        else:
            st.error(f"PDF file not found: {pdf_path}")
            return False
    except Exception as e:
        st.error(f"Error rendering PDF preview: {e}")
        return False


def update_progress(val, msg):
    """Update progress bar and status message"""
    st.session_state.uc2_progress = val
    st.session_state.uc2_status = msg


def handle_filename_change():
    """Handle filename input changes"""
    # Reset generation state when filename changes
    st.session_state.uc2_generation_complete = False
    st.session_state.uc2_generation_error = None
    st.session_state.uc2_last_docx = None
    st.session_state.uc2_last_pdf = None


def handle_generate_click():
    """Handle generate button click"""
    st.session_state.uc2_generate_clicked = True
    st.session_state.uc2_processing = True
    st.session_state.uc2_progress = 0
    st.session_state.uc2_status = "Initializing memorandum generation pipeline..."
    st.session_state.uc2_generation_complete = False
    st.session_state.uc2_generation_error = None


async def generate_memorandum_async():
    """Asynchronous memorandum generation"""
    try:
        update_progress(10, "Initializing memorandum generation pipeline...")

        # Step 1: Extract all transaction data in parallel
        update_progress(20, "Extracting transaction data...")
        parties_task = pipeline.retrieval_plugin.extract_transaction_parties()
        price_task = pipeline.retrieval_plugin.extract_purchase_price_details()
        terms_task = pipeline.retrieval_plugin.extract_key_terms()
        parties_data, price_data, terms_data = await asyncio.gather(
            parties_task, price_task, terms_task
        )
        update_progress(30, "Extracted parties, price, and terms data.")

        # Step 2: Extract entity details
        update_progress(40, "Extracting entity details...")
        entity_details = await pipeline.response_plugin.extract_entity_details(parties_data, price_data)
        update_progress(50, "Extracted entity details.")

        # Step 3: Generate memo sections
        update_progress(55, "Generating memo sections...")
        header_purpose_task = pipeline.response_plugin.generate_enhanced_memo_header_and_purpose(entity_details)
        background_task = pipeline.response_plugin.generate_enhanced_background_section(entity_details, parties_data, price_data)
        key_terms_task = pipeline.response_plugin.generate_key_terms_table(terms_data)
        docs_reviewed_task = pipeline.response_plugin.generate_documents_reviewed_section(parties_data, price_data, terms_data)
        header_purpose, background, key_terms, docs_reviewed = await asyncio.gather(
            header_purpose_task, background_task, key_terms_task, docs_reviewed_task
        )
        update_progress(60, "Generated enhanced memo sections.")

        # Step 4: Analyze all 9 accounting issues
        update_progress(70, "Analyzing all 9 accounting issues (this may take several minutes)...")
        all_issue_analyses = await pipeline.response_plugin.analyze_all_accounting_issues(
            price_data, parties_data, price_data, terms_data
        )
        update_progress(80, "Completed all issue analyses.")

        # Step 5: Generate dynamic literature section and executive summary
        update_progress(85, "Generating dynamic literature section and executive summary...")
        literature_task = pipeline.response_plugin.generate_dynamic_literature_section(all_issue_analyses)
        executive_summary_task = pipeline.response_plugin.generate_executive_summary(all_issue_analyses)
        literature, executive_summary = await asyncio.gather(
            literature_task, executive_summary_task
        )
        update_progress(90, "Generated dynamic literature and executive summary.")

        # Step 6: Generate final DOCX document
        update_progress(95, "Generating final Word document...")
        docx_filename = st.session_state.uc2_base_filename + ".docx"
        pdf_filename = st.session_state.uc2_base_filename + ".pdf"
        
        result = pipeline.doc_plugin.generate_complete_memorandum(
            memo_header_purpose=header_purpose,
            background=background,
            key_terms=key_terms,
            literature=literature,
            documents_reviewed=docs_reviewed,
            all_issue_analyses=all_issue_analyses,
            executive_summary=executive_summary,
            filename=docx_filename,
            pdf_filename=pdf_filename
        )
        
        # Force PDF conversion if not already done
        if isinstance(result, dict) and result.get('docx_path'):
            docx_path = result['docx_path']
            pdf_path = result.get('pdf_path')
            
            if not pdf_path or not os.path.exists(pdf_path):
                update_progress(98, "Converting to PDF...")
                pdf_path = docx_path.replace('.docx', '.pdf')
                if convert_docx_to_pdf(docx_path, pdf_path):
                    result['pdf_path'] = pdf_path
                else:
                    st.error("⚠️ PDF conversion failed. Only DOCX will be available.")
        
        update_progress(100, "Memorandum generated successfully!")
        
        # Update session state with results
        if isinstance(result, dict):
            st.session_state.uc2_last_docx = result.get('docx_path')
            st.session_state.uc2_last_pdf = result.get('pdf_path')
        else:
            st.session_state.uc2_last_docx = docx_filename
            st.session_state.uc2_last_pdf = pdf_filename
        
        st.session_state.uc2_generation_complete = True
        st.session_state.uc2_processing = False
        
        return result
        
    except Exception as e:
        st.session_state.uc2_generation_error = str(e)
        st.session_state.uc2_processing = False
        update_progress(0, f"Error: {str(e)}")
        raise e


def render_indexes_section():
    """Render the indexes information section"""
    st.header("Available Indexes")
    st.subheader("Transaction Documentation Indexes:")
    st.info(f"""
    • Guidance Files Index ({config.guidance_index})
    - Contains Big 4 guidance and examples

    • Transactions Documents Index ({config.agreement_index})
    - Contains purchase agreements

    • Valuation Index ({config.valuation_index})
    - Contains valuation reports

    • Financial Index ({config.financial_index})
    - Contains Excel balance sheets
    """)


def render_progress_section():
    """Render progress bar and status"""
    if st.session_state.uc2_processing or st.session_state.uc2_generation_complete:
        progress_bar = st.progress(st.session_state.uc2_progress)
        status_text = st.text(st.session_state.uc2_status)
        
        if st.session_state.uc2_generation_error:
            st.error(f"Generation failed: {st.session_state.uc2_generation_error}")


def render_output_settings():
    """Render output settings section"""
    st.header("Output Settings")
    
    # Filename input with callback to handle changes
    base_filename = st.text_input(
        "Enter output filename (without extension)",
        value=st.session_state.uc2_base_filename,
        help="Enter the desired base name for the output files (no extension)",
        on_change=handle_filename_change,
        key="uc2_filename_input"
    )
    
    # Update session state if filename changed
    if base_filename != st.session_state.uc2_base_filename:
        st.session_state.uc2_base_filename = base_filename
        handle_filename_change()
    
    is_valid_filename = bool(base_filename.strip())
    
    return is_valid_filename


def render_generation_button(is_valid_filename):
    """Render generation button and handle click"""
    if is_valid_filename:
        # Generate button
        generate_btn = st.button(
            "Generate Memorandum", 
            disabled=st.session_state.uc2_processing,
            key="uc2_generate_btn",
            on_click=handle_generate_click
        )
        
        # Process generation if button was clicked and not already processing
        if st.session_state.uc2_generate_clicked and st.session_state.uc2_processing:
            with st.spinner("Generating memorandum. This may take several minutes..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    result = loop.run_until_complete(generate_memorandum_async())
                    
                    loop.close()
                    
                    # Reset the clicked state
                    st.session_state.uc2_generate_clicked = False
                    
                    st.success("Memorandum generated successfully!")
                    
                except Exception as e:
                    st.session_state.uc2_processing = False
                    st.session_state.uc2_generate_clicked = False
                    st.error(f"Error during processing: {str(e)}")
    else:
        st.info("Please enter a valid output filename")


def render_output_preview():
    """Render output preview and download buttons"""
    if st.session_state.uc2_generation_complete and st.session_state.uc2_last_docx:
        if os.path.exists(st.session_state.uc2_last_docx):
            # Always try to show PDF preview
            if st.session_state.uc2_last_pdf and os.path.exists(st.session_state.uc2_last_pdf):
                render_pdf_preview(st.session_state.uc2_last_pdf)
            else:
                pdf_path = st.session_state.uc2_last_docx.replace('.docx', '.pdf')
                if convert_docx_to_pdf(st.session_state.uc2_last_docx, pdf_path):
                    st.session_state.uc2_last_pdf = pdf_path
                    render_pdf_preview(pdf_path)
                else:
                    st.error("❌ PDF preview is unavailable. PDF conversion failed.")
                    st.info("💡 The DOCX file is ready for download below.")
            
            # Download buttons side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 Download DOCX",
                    data=open(st.session_state.uc2_last_docx, "rb"),
                    file_name=os.path.basename(st.session_state.uc2_last_docx),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            
            with col2:
                if st.session_state.uc2_last_pdf and os.path.exists(st.session_state.uc2_last_pdf):
                    st.download_button(
                        label="📄 Download PDF",
                        data=open(st.session_state.uc2_last_pdf, "rb"),
                        file_name=os.path.basename(st.session_state.uc2_last_pdf),
                        mime="application/pdf"
                    )


def create_streamlit_ui():
    """Create the main Streamlit UI"""
    # Initialize session state
    initialize_session_state()
    
    # Apply CSS and render header
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    render_header()
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center;'>Accounting Memorandum Generator</h1>",
        unsafe_allow_html=True
    )

    # Two columns layout for indexes and output settings (1:1 ratio)
    col1, col2 = st.columns(2)

    with col1:
        render_indexes_section()
        render_progress_section()

    with col2:
        is_valid_filename = render_output_settings()
        render_output_preview()

    # Generate button (full width)
    render_generation_button(is_valid_filename)


def main():
    """Main function"""
    create_streamlit_ui()
    
    # Add logout and switch service buttons in sidebar
    with st.sidebar:
        if st.session_state.get("selected_service") in ("research", "memo"):
            if st.button("Switch Service", key="switch_service_btn_uc2"):
                if st.session_state["selected_service"] == "memo":
                    st.session_state["selected_service"] = "research"
                else:
                    st.session_state["selected_service"] = "memo"
                st.rerun()
        
        if st.button("Logout", key="logout_btn_uc2"):
            st.session_state["logged_in"] = False
            st.session_state["selected_service"] = None
            # Clear UC2 specific session state
            for key in list(st.session_state.keys()):
                if key.startswith("uc2_"):
                    del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()