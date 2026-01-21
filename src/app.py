import streamlit as st
import os
import shutil
from query import query_rag, generate_suggestions
from ingest import ingest_documents

DATA_PATH = "data"

st.title("📚 RAG Bot")

# Initialize session state for suggestions
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # Clear existing data
                if os.path.exists(DATA_PATH):
                    shutil.rmtree(DATA_PATH)
                os.makedirs(DATA_PATH)
                
                # Save uploaded files
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(DATA_PATH, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Run ingestion
                status = ingest_documents()
                st.success(status)
                st.session_state.processed = True
                
                # Generate suggestions
                st.session_state.suggestions = generate_suggestions()
        else:
            st.warning("Please upload at least one file.")

st.write("Ask questions about your uploaded documents!")

# Display suggestions
if st.session_state.suggestions:
    st.caption("Suggested Questions:")
    cols = st.columns(len(st.session_state.suggestions))
    for i, question in enumerate(st.session_state.suggestions):
        if cols[i].button(question, key=f"sugg_{i}"):
            st.session_state.clicked_query = question

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input or clicked suggestion
if prompt := (st.chat_input("What would you like to know?") or st.session_state.get("clicked_query")):
    # Clear clicked query state so it doesn't persist
    if "clicked_query" in st.session_state:
        del st.session_state.clicked_query

    # Check if processed
    if not st.session_state.get("processed", False) and not os.path.exists("faiss_index"):
         st.error("Please upload and process documents first!")
    else:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            try:
                response = query_rag(prompt)
            except Exception as e:
                response = f"An error occurred: {e}"

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
