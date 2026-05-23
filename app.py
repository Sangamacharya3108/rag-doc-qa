import streamlit as st
import rag_engine

st.set_page_config(page_title="Document Q&A", layout="centered")

st.title("Document Q&A")
st.caption("Ask questions about your uploaded PDF document and get grounded answers.")

# Initialize session state variables
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

# Sidebar for PDF upload
with st.sidebar:
    st.header("Document Upload")
    
    # We use a dynamic key to reset the file uploader when resetting the state
    uploader_key = "pdf_uploader_active" if not st.session_state.document_processed else "pdf_uploader_inactive"
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key=uploader_key)
    
    if uploaded_file is not None and not st.session_state.document_processed:
        with st.spinner("Processing document..."):
            text = rag_engine.extract_text_from_pdf(uploaded_file)
            if not text.strip():
                st.error("Could not extract text from this PDF")
            else:
                chunks = rag_engine.chunk_text(text)
                if not chunks:
                    st.error("Could not extract text from this PDF")
                else:
                    index, chunks_list = rag_engine.build_vector_store(chunks)
                    st.session_state.vector_index = index
                    st.session_state.chunks = chunks_list
                    st.session_state.document_processed = True
                    st.success(f"Processed successfully! {len(chunks_list)} chunks created.")
                    st.rerun()

    if st.session_state.document_processed:
        if st.button("Upload New Document"):
            st.session_state.vector_index = None
            st.session_state.chunks = []
            st.session_state.chat_history = []
            st.session_state.document_processed = False
            st.rerun()

# Main area logic
if not st.session_state.document_processed:
    st.info("Please upload a PDF document in the sidebar to get started.")
else:
    # Render chat history
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(content)
            
    # User query input
    if query := st.chat_input("Ask a question about the document..."):
        with st.chat_message("user"):
            st.write(query)
            
        with st.spinner("Thinking..."):
            relevant_chunks = rag_engine.retrieve_relevant_chunks(
                query, st.session_state.vector_index, st.session_state.chunks
            )
            answer = rag_engine.generate_answer(
                query, relevant_chunks, st.session_state.chat_history
            )
            
        with st.chat_message("assistant"):
            st.write(answer)
            
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("assistant", answer))
        st.rerun()
