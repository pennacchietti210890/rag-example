import base64
import time
import os

import requests
import streamlit as st

# Get backend URL from environment variable or use default
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Initialize encryption
encryption_key = None

# Simple XOR cipher for encryption/decryption
def xor_encrypt_decrypt(data: str, key: str) -> str:
    """Encrypt or decrypt data using XOR with the given key"""
    # Convert strings to bytes
    data_bytes = data.encode()
    # Create a repeating key of the same length as data
    key_bytes = (key * (len(data_bytes) // len(key) + 1))[:len(data_bytes)].encode()
    # XOR operation
    result_bytes = bytes(a ^ b for a, b in zip(data_bytes, key_bytes))
    # Return base64 encoded result
    return base64.b64encode(result_bytes).decode()

def fetch_encryption_key():
    """Fetch encryption key from backend"""
    global encryption_key
    if not encryption_key:
        try:
            response = requests.get(f"{BACKEND_URL}/encryption-key")
            if response.status_code == 200:
                encryption_key = response.json().get("encryption_key")
                st.session_state["encryption_key"] = encryption_key
                return True
            else:
                st.error("Failed to fetch encryption key from server")
                return False
        except Exception as e:
            st.error(f"Failed to fetch encryption key: {str(e)}")
            return False
    return True

def encrypt_api_key(api_key):
    """Encrypt API key using XOR cipher"""
    if not encryption_key:
        if not fetch_encryption_key():
            return None
    
    try:
        # For debugging
        st.session_state["last_encrypted"] = xor_encrypt_decrypt(api_key, encryption_key)
        return xor_encrypt_decrypt(api_key, encryption_key)
    except Exception as e:
        st.error(f"Failed to encrypt API key: {str(e)}")
        return None

# Fetch encryption key at startup
fetch_encryption_key()

# Set page config at the very beginning
st.set_page_config(
    page_title="RAG Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
        body { background-color: #0e1117; color: white; }
        .stTextInput>div>div>input { background-color: #1e2229; color: white; font-size: 16px; }
        .stButton>button { background-color: #4CAF50; color: white; font-size: 16px; }
        .stMarkdown { font-size: 16px; }
        .passage-box {
            background-color: #1e2229;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
            font-size: 12px;
            font-style: italic;
        }
        .prompt-box {
            background-color: #1e2229;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
            font-family: monospace;
            white-space: pre-wrap;
            font-size: 12px;
            font-style: italic;
        }
        .section-header {
            font-size: 1.2em;
            margin-top: 2em;
            margin-bottom: 1em;
        }
        .debug-header {
            font-size: 1em;
            color: #888;
            margin-bottom: 1em;
        }
        .pdf-note {
            font-size: 0.8em;
            color: #888;
            margin-top: 2em;
            padding-top: 1em;
            border-top: 1px solid #333;
        }
        .highlight-1 { background-color: rgba(255, 255, 0, 0.3); }  /* Bright yellow */
        .highlight-2 { background-color: rgba(255, 255, 255, 0.3); }  /* White */
        .highlight-3 { background-color: rgba(0, 255, 0, 0.3); }  /* Bright green */
        .highlight-4 { background-color: rgba(255, 165, 0, 0.3); }  /* Orange */
        .highlight-5 { background-color: rgba(0, 255, 255, 0.3); }  /* Cyan */
        .prompt-section {
            white-space: pre-wrap;
            display: block;
            margin-bottom: 0.5em;
        }
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
if "available_models" not in st.session_state:
    st.session_state.available_models = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "models_fetched" not in st.session_state:
    st.session_state.models_fetched = False
if "last_prompt_sections" not in st.session_state:
    st.session_state.last_prompt_sections = None
if "last_passages" not in st.session_state:
    st.session_state.last_passages = None

def fetch_available_models(api_key: str = None):
    """Fetch available models from the backend if not already fetched"""
    if not st.session_state.models_fetched and api_key:
        try:
            # Encrypt the API key
            encrypted_api_key = encrypt_api_key(api_key)
            if not encrypted_api_key:
                st.warning("Encryption failed, trying direct API key")
                encrypted_api_key = api_key  # Fallback to direct API key
                
            response = requests.get(f"{BACKEND_URL}/models/", params={"encrypted_api_key": encrypted_api_key})
            if response.status_code == 200:
                st.session_state.available_models = response.json().get("models", [])
                st.session_state.models_fetched = True
                st.success("Successfully fetched models!")
            else:
                st.error(f"Failed to fetch models. Please check your API key. Status: {response.status_code}, Response: {response.text}")
                st.session_state.available_models = []
                st.session_state.models_fetched = True
        except Exception as e:
            st.error(f"Failed to fetch available models: {str(e)}")
            st.session_state.available_models = []
            st.session_state.models_fetched = True

# Sidebar
with st.sidebar:
    st.header("🔧 Model Settings")
    
    # Add secure API key input
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key. This will be encrypted before being sent to the backend.",
        key="groq_api_key"
    )

    # Fetch models when API key is provided
    if groq_api_key:
        fetch_available_models(groq_api_key)
    
    if st.session_state.available_models:
        selected_model = st.selectbox(
            "Select Groq Model",
            options=st.session_state.available_models,
            key="model_select",
            help="Choose from available Groq models"
        )
    else:
        if groq_api_key:
            st.error("No Groq models available. Please check your API key.")
        else:
            st.warning("Please enter your Groq API key to see available models.")
        selected_model = None

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

    st.markdown('<div class="pdf-note">🔍 Ensure that the uploaded document is a pdf file.</div>', unsafe_allow_html=True)

# Create two columns for the main content and right sidebar
main_col, right_sidebar = st.columns([3, 1])

with main_col:
    # Main content
    st.title("🤖 RAG Playground")
    st.markdown("**Upload a document (PDF) and ask questions about its content!**")

    # File Upload Section
    st.markdown('<div class="section-header">🗂 Upload a Document (PDF)</div>', unsafe_allow_html=True)
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
                data = {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "num_chunks": num_chunks,
                }
                response = requests.post(
                    f"{BACKEND_URL}/upload/",
                    files=files,
                    data=data,  # Pass RAG parameters as form data
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
    st.markdown('<div class="section-header">💬 Chat with the Report</div>', unsafe_allow_html=True)
    query = st.text_input(
        "Enter your question:", help="Type your question about the uploaded document"
    )
    submit_button = st.button("Ask")

    if submit_button and query:
        if not st.session_state.session_id:
            st.error("❌ Please upload a document first.")
        elif not groq_api_key:
            st.error("❌ Please provide your Groq API key.")
        else:
            with st.spinner("Fetching answer..."):
                try:
                    # Encrypt the API key
                    encrypted_api_key = encrypt_api_key(groq_api_key)
                    if not encrypted_api_key:
                        st.warning("Encryption failed, trying direct API key")
                        encrypted_api_key = groq_api_key  # Fallback to direct API key
                    
                    response = requests.post(
                        f"{BACKEND_URL}/query/",
                        json={
                            "query": query,
                            "model_name": selected_model,
                            "session_id": st.session_state.session_id,
                            "encrypted_api_key": encrypted_api_key,
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
                        data = response.json()
                        answer = data.get("answer", "No answer found.")
                        # Update session state with latest prompt sections and passages
                        st.session_state.last_prompt_sections = data.get("prompt_sections", [])
                        st.session_state.last_passages = data.get("retrieved_passages", [])
                        # Store Q&A pair
                        st.session_state.chat_history.append({
                            "question": query,
                            "answer": answer
                        })
                    else:
                        st.error(f"❌ Failed to fetch answer: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error fetching answer: {str(e)}")

    # Display chat history
    for qa_pair in reversed(st.session_state.chat_history):
        st.markdown(f"**📝 Question:** {qa_pair['question']}")
        st.markdown(f"**💡 Answer:** {qa_pair['answer']}")
        st.markdown("---")

# Right Sidebar
with right_sidebar:
    #st.markdown('<div class="debug-header">🔍 Debug Information</div>', unsafe_allow_html=True)
    
    # Add encryption debug info
    #with st.expander("🔐 Encryption Debug", expanded=False):
    #    st.write("Encryption Key Status:", "Available" if encryption_key else "Not Available")
    #    if "encryption_key" in st.session_state:
    #        # Only show first few characters for security
    #        st.write("Key (first 5 chars):", st.session_state["encryption_key"][:5] + "...")
    #    
    #    if "last_encrypted" in st.session_state:
    #        # Only show first few characters for security
    #        st.write("Last encrypted value (first 10 chars):", st.session_state["last_encrypted"][:10] + "...")
    
    # Show the most recent prompt sections if available
    if st.session_state.last_prompt_sections:
        with st.expander("📝 Last Prompt", expanded=True):
            prompt_html = ""
            for i, section in enumerate(st.session_state.last_prompt_sections):
                if i == 0:  # System prompt
                    prompt_html += f'<div class="prompt-section">{section}</div>'
                elif i < len(st.session_state.last_prompt_sections) - 1:  # Passages
                    highlight_class = f"highlight-{i}" if i <= 5 else "highlight-1"
                    prompt_html += f'<div class="prompt-section {highlight_class}">{section}</div>'
                else:  # Question and answer
                    prompt_html += f'<div class="prompt-section">{section}</div>'
            st.markdown(f'<div class="prompt-box">{prompt_html}</div>', unsafe_allow_html=True)
