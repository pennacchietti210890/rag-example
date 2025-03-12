import base64
import time

import requests
import streamlit as st

# Set page config at the very beginning
st.set_page_config(
    page_title="Doc QA", page_icon="📄", layout="wide", initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
    <style>
        body { background-color: #0e1117; color: white; }
        .stTextInput>div>div>input { background-color: #1e2229; color: white; font-size: 16px; }
        .stButton>button { background-color: #4CAF50; color: white; font-size: 16px; }
        .stMarkdown { font-size: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# Sidebar
with st.sidebar:
    st.header("🔧 Model Settings")
    model_type = st.radio(
        "Choose a model:", ("Local (Llama 3.2 - 3B)", "Cloud (Llama 3 - 70B via Groq)")
    )
    model_map = {
        "Local (Llama 3.2 - 3B)": "local",
        "Cloud (Llama 3 - 70B via Groq)": "groq",
    }
    selected_model = model_map[model_type]

    st.header("🎚️ Generation Parameters")
    with st.expander("LLM Settings", expanded=False):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in the output. Higher values make the output more random, lower values make it more deterministic.",
        )
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.1,
            help="Controls diversity via nucleus sampling. Lower values make the output more focused.",
        )
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=2000,
            value=200,
            step=50,
            help="Maximum number of tokens to generate in the response.",
        )

    st.markdown("---")  # Visual separator

    with st.expander("RAG Settings", expanded=False):
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=2000,
            value=500,
            step=50,
            help="Size of text chunks for document processing. Larger chunks provide more context but may be less precise.",
        )
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=200,
            value=50,
            step=10,
            help="Number of overlapping tokens between chunks. Higher overlap helps maintain context across chunks.",
        )
        num_chunks = st.slider(
            "Number of Chunks",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Number of most relevant chunks to retrieve. More chunks provide broader context but may include less relevant information.",
        )

    st.header("ℹ️ How It Works")
    st.write(
        "1️⃣ Upload a document(PDF).\n"
        "2️⃣ Ask specific questions about the document.\n"
        "3️⃣ Receive AI-generated responses based on the document's content."
    )
    st.info("🔍 Ensure that the uploaded document is a pdf file.")

# Main content
st.title("📄 Doc Q&A - example RAG application")
st.markdown("**Upload a document (PDF) and ask questions about its content!**")

# File Upload Section
st.header("🗂 Upload a Document (PDF)")
uploaded_file = st.file_uploader(
    "Choose a PDF file", type=["pdf"], help="Upload a document for analysis"
)

if uploaded_file and (st.session_state.uploaded_file_name != uploaded_file.name):
    with st.spinner("Uploading and processing file..."):
        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf",
                )
            }
            # Add RAG parameters to the upload request
            params = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "num_chunks": num_chunks,
            }
            response = requests.post(
                "http://localhost:8000/upload/",
                files=files,
                params=params,  # Pass RAG parameters as query parameters
            )

            if response.status_code == 200:
                st.session_state.session_id = response.json().get("session_id")
                st.session_state.uploaded_file_name = uploaded_file.name
                st.success("✅ File uploaded and indexed successfully!")
            else:
                st.error(f"❌ Failed to upload file: {response.text}")
        except Exception as e:
            st.error(f"❌ Error uploading file: {str(e)}")

# Chat-Style Q&A Section
st.header("💬 Chat with the Report")
query = st.text_input(
    "Enter your question:", help="Type your question about the uploaded document"
)
submit_button = st.button("Ask")

if submit_button and query:
    if not st.session_state.session_id:
        st.error("❌ Please upload a document first.")
    else:
        with st.spinner("Fetching answer..."):
            try:
                response = requests.post(
                    "http://localhost:8000/query/",
                    json={
                        "query": query,
                        "model_type": selected_model,
                        "session_id": st.session_state.session_id,
                        # LLM parameters
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                        # RAG parameters
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "num_chunks": num_chunks,
                    },
                )

                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer found.")
                    st.session_state.chat_history.append((query, answer))
                else:
                    st.error(f"❌ Failed to fetch answer: {response.text}")
            except Exception as e:
                st.error(f"❌ Error fetching answer: {str(e)}")

# Display chat history
for question, answer in reversed(st.session_state.chat_history):
    st.markdown(f"**📝 Question:** {question}")
    st.markdown(f"**💡 Answer:** {answer}")
    st.markdown("---")
