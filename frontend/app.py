import base64
import time

import requests
import streamlit as st

# Set page config at the very beginning
st.set_page_config(
    page_title="Doc QA",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
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
    st.header("🔧 Model Selection")
    model_type = st.radio(
        "Choose a model:", ("Local (Llama 3.2 - 3B)", "Cloud (Llama 3 - 70B via Groq)")
    )
    model_map = {
        "Local (Llama 3.2 - 3B)": "local",
        "Cloud (Llama 3 - 70B via Groq)": "groq",
    }
    selected_model = model_map[model_type]

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
                "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
            }
            response = requests.post("http://localhost:8000/upload/", files=files)

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