import streamlit as st
import os
from io import StringIO
from contextlib import redirect_stdout

from pages.Usecase_2.main_uc2 import config, pipeline
from pages.header import render_header


def get_custom_css():
    return """     
    <style>       
    /* ...existing non-header CSS... */
    </style>
    """

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
    try:
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()
                if len(pdf_data) > 0:
                    st.markdown("**📄 PDF Preview:**")
                    try:
                        import base64
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

def create_streamlit_ui():
    # st.set_page_config(page_title="Accounting Memorandum Generator", layout="wide")
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
        # Progress bar in left column, limited width
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

    with col2:
        st.header("Output Settings")
        base_filename = st.text_input(
            "Enter output filename (without extension)",
            value="accounting_memorandum",
            help="Enter the desired base name for the output files (no extension)"
        )
        is_valid_filename = bool(base_filename.strip())

        # Output preview and download section
        preview_placeholder = st.empty()
        download_col1, download_col2 = st.columns(2)
        docx_download_placeholder = download_col1.empty()
        pdf_download_placeholder = download_col2.empty()

    debug_container = st.empty()

    # Session state for progress
    if "progress" not in st.session_state:
        st.session_state.progress = 0
    if "status" not in st.session_state:
        st.session_state.status = ""
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "last_docx" not in st.session_state:
        st.session_state.last_docx = None
    if "last_pdf" not in st.session_state:
        st.session_state.last_pdf = None

    if is_valid_filename:
        generate_btn = st.button("Generate Memorandum", disabled=st.session_state.processing, key="generate_btn")
        # Progress bar and status in left column
        with col1:
            progress_bar = progress_placeholder.progress(st.session_state.progress)
            status_text = status_placeholder.text(st.session_state.status)

        if generate_btn and not st.session_state.processing:
            st.session_state.processing = True
            st.session_state.progress = 0
            st.session_state.status = "Initializing memorandum generation pipeline..."
            output_buffer = StringIO()
            try:
                import asyncio

                def update_progress(val, msg):
                    st.session_state.progress = val
                    st.session_state.status = msg
                    progress_placeholder.progress(val)
                    status_placeholder.text(msg)

                with st.spinner("Generating memorandum. This may take several minutes..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    update_progress(10, "Initializing memorandum generation pipeline...")

                    # Step 1: Extract all transaction data in parallel
                    update_progress(20, "Extracting transaction data...")
                    parties_task = pipeline.retrieval_plugin.extract_transaction_parties()
                    price_task = pipeline.retrieval_plugin.extract_purchase_price_details()
                    terms_task = pipeline.retrieval_plugin.extract_key_terms()
                    parties_data, price_data, terms_data = loop.run_until_complete(
                        asyncio.gather(parties_task, price_task, terms_task)
                    )
                    update_progress(30, "Extracted parties, price, and terms data.")

                    # Step 2: Extract entity details
                    update_progress(40, "Extracting entity details...")
                    entity_details = loop.run_until_complete(
                        pipeline.response_plugin.extract_entity_details(parties_data, price_data)
                    )
                    update_progress(50, "Extracted entity details.")

                    # Step 3: Generate memo sections
                    update_progress(55, "Generating memo sections...")
                    header_purpose_task = pipeline.response_plugin.generate_enhanced_memo_header_and_purpose(entity_details)
                    background_task = pipeline.response_plugin.generate_enhanced_background_section(entity_details, parties_data, price_data)
                    key_terms_task = pipeline.response_plugin.generate_key_terms_table(terms_data)
                    docs_reviewed_task = pipeline.response_plugin.generate_documents_reviewed_section(parties_data, price_data, terms_data)
                    header_purpose, background, key_terms, docs_reviewed = loop.run_until_complete(
                        asyncio.gather(header_purpose_task, background_task, key_terms_task, docs_reviewed_task)
                    )
                    update_progress(60, "Generated enhanced memo sections.")

                    # Step 4: Analyze all 9 accounting issues
                    update_progress(70, "Analyzing all 9 accounting issues (this may take several minutes)...")
                    all_issue_analyses = loop.run_until_complete(
                        pipeline.response_plugin.analyze_all_accounting_issues(price_data, parties_data, price_data, terms_data)
                    )
                    update_progress(80, "Completed all issue analyses.")

                    # Step 5: Generate dynamic literature section and executive summary
                    update_progress(85, "Generating dynamic literature section and executive summary...")
                    literature_task = pipeline.response_plugin.generate_dynamic_literature_section(all_issue_analyses)
                    executive_summary_task = pipeline.response_plugin.generate_executive_summary(all_issue_analyses)
                    literature, executive_summary = loop.run_until_complete(
                        asyncio.gather(literature_task, executive_summary_task)
                    )
                    update_progress(90, "Generated dynamic literature and executive summary.")

                    # Step 6: Generate final DOCX document
                    update_progress(95, "Generating final Word document...")
                    pdf_filename = base_filename + ".pdf"
                    docx_filename = base_filename + ".docx"
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
                            pdf_path = st.session_state.last_docx.replace('.docx', '.pdf')
                            if convert_docx_to_pdf(docx_path, pdf_path):
                                result['pdf_path'] = pdf_path
                            else:
                                st.error("⚠️ PDF conversion failed. Only DOCX will be available.")
                    
                    output_buffer.write(str(result) + "\n")
                    update_progress(100, "Memorandum generated successfully!")

                    loop.close()

                st.session_state.processing = False

                # Save last generated docx/pdf path from backend result
                if isinstance(result, dict):
                    st.session_state.last_docx = result.get('docx_path')
                    st.session_state.last_pdf = result.get('pdf_path')
                else:
                    st.session_state.last_docx = docx_filename
                    st.session_state.last_pdf = pdf_filename

            except Exception as e:
                st.session_state.processing = False
                st.error(f"Error during processing: {str(e)}")
            # finally:
            #     with debug_container:
            #         st.subheader("Debug Output")
            #         st.text(output_buffer.getvalue())
        else:
            # Show progress and status if processing
            with col1:
                progress_placeholder.progress(st.session_state.progress)
                status_placeholder.text(st.session_state.status)

        # Output preview and download in right column - PDF ONLY
        with col2:
            if st.session_state.last_docx and os.path.exists(st.session_state.last_docx):
                # Always try to show PDF preview
                if st.session_state.last_pdf and os.path.exists(st.session_state.last_pdf):
                    render_pdf_preview(st.session_state.last_pdf)
                else:
                    pdf_path = st.session_state.last_docx.replace('.docx', '.pdf')
                    if convert_docx_to_pdf(st.session_state.last_docx, pdf_path):
                        st.session_state.last_pdf = pdf_path
                        render_pdf_preview(pdf_path)
                    else:
                        st.error("❌ PDF preview is unavailable. PDF conversion failed.")
                        st.info("💡 The DOCX file is ready for download below.")
                # Download buttons side by side
                with st.container():
                    dcol1, dcol2 = st.columns(2)
                    with dcol1:
                        docx_download_placeholder.download_button(
                            label="📄 Download DOCX",
                            data=open(st.session_state.last_docx, "rb"),
                            file_name=os.path.basename(st.session_state.last_docx),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    with dcol2:
                        if st.session_state.last_pdf and os.path.exists(st.session_state.last_pdf):
                            pdf_download_placeholder.download_button(
                                label="📄 Download PDF",
                                data=open(st.session_state.last_pdf, "rb"),
                                file_name=os.path.basename(st.session_state.last_pdf),
                                mime="application/pdf"
                            )
    else:
        st.info("Please enter a valid output filename (must end with .docx)")

def main():
    # st.set_page_config(page_title="Accounting Memorandum Generator", layout="wide")
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
            st.rerun()

if __name__ == "__main__":
    main()
