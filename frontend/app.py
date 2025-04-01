import base64
import time
import os
import io

import requests
import streamlit as st
import fitz  # PyMuPDF for PDF handling
from streamlit_pdf_viewer import pdf_viewer  # Import the PDF viewer component

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
    key_bytes = (key * (len(data_bytes) // len(key) + 1))[: len(data_bytes)].encode()
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
        st.session_state["last_encrypted"] = xor_encrypt_decrypt(
            api_key, encryption_key
        )
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
            font-size: 1.8em;
            margin-top: 2em;
            margin-bottom: 1em;
            font-weight: 600;
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
        
        /* UPDATED styles for UI color changes */
        /* Force main area to have gray background */
        .main {
            background-color: #222222 !important;
        }
        
        /* Force sidebar to have black background */
        [data-testid="stSidebar"] {
            background-color: #0e1117 !important;
        }
        
        /* Target all major sections within the main area to ensure gray background */
        .css-18e3th9, .css-1d391kg, .css-12oz5g7 {
            background-color: #222222 !important;
        }
        
        /* Make sidebar content explicitly black */
        .sidebar .sidebar-content, [data-testid="stSidebar"] > div {
            background-color: #0e1117 !important;
        }
        
        /* Ensure content is centrally aligned */
        .block-container {
            max-width: 1000px !important;
            padding-left: 5% !important;
            padding-right: 5% !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        
        /* Center content when sidebar is collapsed */
        [data-testid="stSidebar"][aria-expanded="false"] + .main .block-container {
            padding-left: 5% !important;
            padding-right: 5% !important;
        }
        
        /* Center the content in columns */
        .row-widget.stHorizontal {
            justify-content: center !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        
        /* Apply styling to stApp to ensure full coverage */
        .stApp {
            background-color: #222222 !important;
        }
        
        /* Hide the default Streamlit header with deploy button and menu */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Adjust top padding to account for hidden header */
        .main .block-container {
            padding-top: 1rem !important;
        }
        
        /* Custom styling for the question input field - bigger, more rounded, slightly darker gray */
        [data-testid="stTextInput"] > div > div > input,
        [data-testid="stTextArea"] > div > div > textarea {
            background-color: #2a2a2a !important; /* Slightly darker than #252525 background */
            color: white !important;
            font-size: 16px !important;
            border-radius: 18px !important; /* More rounded corners */
            border: 1px solid #333 !important;
            padding: 20px !important;
            min-height: 70px !important;
            width: 100% !important;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1) !important; /* Subtle inner shadow for depth */
            transition: all 0.2s ease !important;
        }
        
        /* Add subtle hover effect */
        [data-testid="stTextArea"] > div > div > textarea:hover {
            background-color: #303030 !important;
            border-color: #444 !important;
        }
        
        /* Add focus effect */
        [data-testid="stTextArea"] > div > div > textarea:focus {
            border-color: #000000 !important; /* Changed from #555 to pure black */
            box-shadow: 0 0 0 2px rgba(0,0,0,0.15) !important; /* Changed shadow to be darker */
            outline: none !important;
        }
        
        /* Additional fix to prevent any red outlines on focus for all input elements */
        *:focus {
            outline-color: black !important;
            box-shadow: none !important;
        }
        
        /* Override any default Streamlit focus styles */
        [data-testid="stTextArea"] > div {
            border: none !important;
            box-shadow: none !important;
        }
        
        [data-testid="stTextArea"] > div > div > textarea:focus-visible {
            outline: none !important;
            border-color: #000000 !important;
        }
        
        /* Make the text start at the top of the input box rather than middle */
        [data-testid="stTextInput"] > div > div,
        [data-testid="stTextArea"] > div > div {
            align-items: flex-start !important;
        }
        
        /* Custom styling for the Ask button - gray and wider */
        .stButton > button {
            background-color: var(--gray-700, #2e3440) !important;
            color: white !important;
            border: 1px solid #333 !important;
            border-radius: 10px !important;
            font-size: 16px !important;
            font-weight: bold !important;
            min-width: 150px !important;
            padding: 10px 20px !important;
            transition: all 0.3s ease !important;
        }
        
        /* Hover effect for the Ask button */
        .stButton > button:hover {
            background-color: #252525 !important;
            border-color: #555 !important;
        }
        
        /* Specific style for the chat header */
        .chat-header {
            font-size: 28px;
            margin-top: 1em;
            margin-bottom: 1.2em;
            font-weight: 600;
            letter-spacing: 0.38px;
        }
        
        /* Style for emoji in headers to align with text */
        .header-emoji {
            font-size: 28px;
            vertical-align: middle;
            margin-right: 8px;
        }
        
        /* NEW: Reduce the size of sidebar elements */
        [data-testid="stSidebar"] {
            padding-top: 0.5rem !important;
        }
        
        /* Make sidebar text smaller */
        [data-testid="stSidebar"] .stMarkdown, 
        [data-testid="stSidebar"] .stMarkdown p, 
        [data-testid="stSidebar"] .stMarkdown h1, 
        [data-testid="stSidebar"] .stMarkdown h2, 
        [data-testid="stSidebar"] .stMarkdown h3 {
            font-size: 14px !important;
        }
        
        /* Make sidebar section headers smaller */
        [data-testid="stSidebar"] .section-header {
            font-size: 16px !important;
            margin-top: 1em !important;
            margin-bottom: 0.5em !important;
        }
        
        /* Make sidebar input fields and buttons smaller */
        [data-testid="stSidebar"] [data-testid="stTextInput"] > div > div > input,
        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] [data-testid="stSelectbox"],
        [data-testid="stSidebar"] .stSlider {
            font-size: 12px !important;
            padding: 8px !important;
            min-height: auto !important;
        }
        
        /* Reduce spacing between sidebar elements */
        [data-testid="stSidebar"] > div > div > div > div > div {
            margin-bottom: 0.5rem !important;
        }
        
        /* Make sidebar headers smaller */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] h4 {
            font-size: 16px !important;
            margin-top: 0.75em !important;
            margin-bottom: 0.5em !important;
        }
        
        /* Adjust sidebar header emoji */
        [data-testid="stSidebar"] .header-emoji {
            font-size: 16px !important;
        }
        
        /* Fix right sidebar positioning to prevent overlap */
        .row-widget > [data-testid="column"]:nth-child(2) {
            padding-left: 15px !important;
            margin-left: 10px !important;
        }
        
        /* Last Prompt section in right sidebar - better positioning */
        [data-testid="column"] > div > [data-testid="stExpander"] {
            margin-left: 15px !important;
            max-width: 95% !important;
            float: right !important;
        }
        
        /* NEW: Extreme right sidebar positioning */
        /* Create a fixed position right sidebar */
        .rightmost-sidebar {
            position: fixed !important;
            top: 0 !important;
            right: 0 !important;
            width: 350px !important;
            height: 100vh !important;
            background-color: #0e1117 !important;
            padding: 20px !important;
            z-index: 999 !important;
            overflow-y: auto !important;
            box-shadow: -3px 0 10px rgba(0,0,0,0.2) !important;
            border-left: 1px solid #333 !important;
        }
        
        /* Style for prompt box in extreme right sidebar */
        .rightmost-sidebar .prompt-box {
            background-color: #1a1a1a !important;
            border-radius: 5px !important;
            padding: 10px !important;
            margin-top: 10px !important;
            border-left: 3px solid #3a3a3a !important;
            font-size: 11px !important;
            max-height: 80vh !important;
            overflow-y: auto !important;
        }
        
        /* Style for the expander in extreme right sidebar */
        .rightmost-sidebar .st-expander {
            border: none !important;
            background-color: transparent !important;
        }
        
        /* Adjust the main content area to make room for the extreme right sidebar */
        [data-testid="column"]:nth-child(1) {
            margin-right: 350px !important;
            width: calc(100% - 350px) !important;
        }
        
        /* Hide the original right sidebar column */
        [data-testid="column"]:nth-child(2) {
            display: none !important;
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
if "passage_page_map" not in st.session_state:
    st.session_state.passage_page_map = []

# Initialize RAG parameters with default values
# These need to be defined before the sidebar where file upload happens
chunk_size = 500
chunk_overlap = 50
num_chunks = 3
distance_metric = "l2"  # Default to L2 distance


def fetch_available_models(api_key: str = None):
    """Fetch available models from the backend if not already fetched"""
    if not st.session_state.models_fetched and api_key:
        try:
            # Encrypt the API key
            encrypted_api_key = encrypt_api_key(api_key)
            if not encrypted_api_key:
                st.warning("Encryption failed, trying direct API key")
                encrypted_api_key = api_key  # Fallback to direct API key

            response = requests.get(
                f"{BACKEND_URL}/models/",
                params={"encrypted_api_key": encrypted_api_key},
            )
            if response.status_code == 200:
                st.session_state.available_models = response.json().get("models", [])
                st.session_state.models_fetched = True
                st.success("Successfully fetched models!")
            else:
                st.error(
                    f"Failed to fetch models. Please check your API key. Status: {response.status_code}, Response: {response.text}"
                )
                st.session_state.available_models = []
                st.session_state.models_fetched = True
        except Exception as e:
            st.error(f"Failed to fetch available models: {str(e)}")
            st.session_state.available_models = []
            st.session_state.models_fetched = True


# Sidebar
with st.sidebar:
    # Add document upload section to the top of the sidebar
    st.markdown(
        '<div class="section-header">🗂 Upload a Document (PDF)</div>',
        unsafe_allow_html=True,
    )
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
                    "distance_metric": distance_metric,
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

    st.markdown("---")  # Visual separator

    st.header("🔧 Model Settings")

    # Add secure API key input
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key. This will be encrypted before being sent to the backend.",
        key="groq_api_key",
    )

    # Fetch models when API key is provided
    if groq_api_key:
        fetch_available_models(groq_api_key)

    if st.session_state.available_models:
        selected_model = st.selectbox(
            "Select Groq Model",
            options=st.session_state.available_models,
            key="model_select",
            help="Choose from available Groq models",
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

    # Add RAG options radio button
    rag_option = st.radio(
        "Select RAG Mode:",
        options=["RAG", "Self-RAG"],
        index=0,  # Default to RAG
        help="Choose retrieval mode: RAG (standard retrieval), or Self-RAG (model decides when to retrieve).",
    )

    # Initialize rag_enabled in session state if not already present
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = True

    # Update rag_enabled based on selection
    st.session_state.rag_enabled = rag_option in ["RAG", "Self-RAG"]

    # Only show RAG settings if RAG or Self-RAG is selected
    if st.session_state.rag_enabled:
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
            
            # Add distance metric selection dropdown
            distance_metric = st.selectbox(
                "Distance Metric",
                options=[
                    "l2",  # Euclidean distance (L2)
                    "ip",  # Inner product (Dot product)
                    "cosine",  # Cosine similarity
                    "hamming",  # Hamming distance
                ],
                index=0,  # Default to L2
                format_func=lambda x: {
                    "l2": "L2 Distance (Euclidean)",
                    "ip": "Dot Product",
                    "cosine": "Cosine Similarity",
                    "hamming": "Hamming Distance"
                }.get(x, x),
                help="Distance metric used for similarity search: L2 (Euclidean) measures direct distance between vectors, Dot Product measures vector alignment, Cosine Similarity measures angle between vectors (normalized), and Hamming Distance measures binary differences."
            )
    else:
        # Set default values when RAG is disabled
        chunk_size = 500
        chunk_overlap = 50
        num_chunks = 3
        distance_metric = "l2"  # Default distance metric

    st.markdown(
        '<div class="pdf-note">🔍 Ensure that the uploaded document is a pdf file.</div>',
        unsafe_allow_html=True,
    )

# Create two columns for the main content and right sidebar
main_col, right_sidebar = st.columns([3, 1])

with main_col:
    # Main content
    # st.title("RAG Playground")
    # st.markdown("**Upload a document (PDF) and ask questions about its content!**")

    # Chat-Style Q&A Section
    st.markdown(
        '<div class="chat-header"><span class="header-emoji"></span>Chat with your Document</div>',
        unsafe_allow_html=True,
    )

    # Create a nice layout for the chat input and button
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_area(
            "Enter your question:",
            help="Type your question about the uploaded document",
            placeholder="Ask a question about the document...",
            label_visibility="collapsed",
            height=100,
        )
    with col2:
        submit_button = st.button("Ask", use_container_width=True)

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
                            "rag_enabled": st.session_state.rag_enabled,
                            "rag_mode": rag_option.lower(),
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "num_chunks": num_chunks,
                            "distance_metric": distance_metric,
                        },
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "No answer found.")
                        # Update session state with latest prompt sections and passages
                        st.session_state.last_prompt_sections = data.get(
                            "prompt_sections", []
                        )
                        st.session_state.last_passages = data.get(
                            "retrieved_passages", []
                        )
                        # Reset the highlighted state to ensure new passages are highlighted
                        st.session_state.passages_highlighted = False
                        # Store Q&A pair
                        st.session_state.chat_history.append(
                            {"question": query, "answer": answer}
                        )
                    else:
                        st.error(f"❌ Failed to fetch answer: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error fetching answer: {str(e)}")

    # Display chat history
    for qa_pair in reversed(st.session_state.chat_history):
        st.markdown(f"**📝 Question:** {qa_pair['question']}")
        st.markdown(f"**💡 Answer:** {qa_pair['answer']}")
        st.markdown("---")

    # Add PDF viewer below chat history
    if uploaded_file and st.session_state.session_id:
        st.markdown(
            '<div class="chat-header"><span class="header-emoji">📄</span>Document with Highlighted Passages</div>',
            unsafe_allow_html=True,
        )

        # Store the PDF content in session state if not already present
        if "pdf_content" not in st.session_state and uploaded_file:
            st.session_state.pdf_content = uploaded_file.getvalue()

        # Initialize current passage index if not already present
        if "current_passage_index" not in st.session_state:
            st.session_state.current_passage_index = 0

        # If we have passages from the last query, highlight them in the PDF
        if "last_passages" in st.session_state and st.session_state.last_passages:
            # Only process highlighting if we haven't already for these passages
            if (
                "passages_highlighted" not in st.session_state
                or not st.session_state.passages_highlighted
            ):
                with st.spinner("Highlighting passages in the document..."):
                    try:
                        # Load the PDF
                        pdf_content = st.session_state.pdf_content
                        pdf_file = fitz.open(stream=pdf_content, filetype="pdf")

                        # Map of passage index to page number
                        passage_page_map = []

                        # Use different colors for different passages
                        colors = [
                            (1, 1, 0),  # Yellow
                            (1, 0.7, 0.7),  # Light Red
                            (0.7, 1, 0.7),  # Light Green
                            (0.7, 0.7, 1),  # Light Blue
                            (1, 0.7, 1),  # Light Purple
                        ]

                        # Process each passage
                        for i, passage in enumerate(st.session_state.last_passages):
                            # Skip empty passages
                            if not passage or not passage.strip():
                                passage_page_map.append(None)
                                continue

                            # Get color for this passage
                            color = colors[i % len(colors)]

                            # Clean up the passage
                            clean_passage = " ".join(passage.split())

                            # Track if this passage was found
                            passage_found = False
                            passage_page = None

                            # Try different approaches to extract representative text for searching
                            # First try with a longer phrase (first 5-8 words)
                            words = clean_passage.split()
                            phrases_to_try = []

                            if len(words) >= 5:
                                # Try with first 5-8 words
                                phrase_length = min(5, len(words))
                                phrases_to_try.append(" ".join(words[:phrase_length]))

                                # Also try with the middle 5-8 words if the passage is long enough
                                if len(words) >= 10:
                                    mid_start = len(words) // 2 - 3
                                    mid_end = mid_start + min(8, len(words) - mid_start)
                                    phrases_to_try.append(
                                        " ".join(words[mid_start:mid_end])
                                    )

                            # If no substantial phrases, just use the full passage
                            if not phrases_to_try and clean_passage:
                                phrases_to_try.append(clean_passage)

                            # Try each phrase until we find a match
                            for phrase in phrases_to_try:
                                if passage_found:
                                    break

                                # Search for the phrase in each page
                                for page_num in range(len(pdf_file)):
                                    if passage_found:
                                        break

                                    page = pdf_file[page_num]
                                    try:
                                        text_instances = page.search_for(phrase)

                                        if text_instances:  # If found matches
                                            # Add highlight for the first instance
                                            highlight = page.add_highlight_annot(
                                                text_instances[0]
                                            )
                                            highlight.set_colors(stroke=color)
                                            highlight.update()

                                            # Record the page for this passage
                                            passage_found = True
                                            passage_page = page_num
                                            break
                                    except Exception:
                                        continue

                            # Add the page number to the passage_page_map
                            passage_page_map.append(passage_page)

                        # Save the highlighted PDF to memory
                        output_buffer = io.BytesIO()
                        pdf_file.save(output_buffer)
                        pdf_file.close()

                        # Update the PDF content with highlighted version
                        st.session_state.pdf_content = output_buffer.getvalue()

                        # Store the passage_page_map in session state
                        st.session_state.passage_page_map = passage_page_map

                        # Mark as highlighted
                        st.session_state.passages_highlighted = True

                        # Set current passage index to the first valid page
                        valid_passages = [
                            (i, p)
                            for i, p in enumerate(passage_page_map)
                            if p is not None
                        ]
                        if valid_passages:
                            # Get the first passage with a valid page
                            st.session_state.current_passage_index = valid_passages[0][
                                0
                            ]
                            # Set the current page to the first highlighted passage
                            st.session_state.current_pdf_page = valid_passages[0][1]

                    except Exception as e:
                        st.warning(f"Could not highlight passages in PDF: {str(e)}")

        # Display the PDF using the streamlit_pdf_viewer component
        st.markdown("### PDF Document")

        try:
            # Get the current page from session state
            current_page = 0
            if "current_pdf_page" in st.session_state:
                current_page = st.session_state.current_pdf_page

            # Add extra check to make sure we go to the first passage page
            if (
                "passage_page_map" in st.session_state
                and st.session_state.passage_page_map
                and "current_passage_index" in st.session_state
            ):
                idx = st.session_state.current_passage_index
                if (
                    0 <= idx < len(st.session_state.passage_page_map)
                    and st.session_state.passage_page_map[idx] is not None
                ):
                    current_page = st.session_state.passage_page_map[idx]
                    # Make sure to update the current_pdf_page in session state
                    st.session_state.current_pdf_page = current_page

            # Use the streamlit_pdf_viewer component

            # Filter out None values from passage_page_map and add 1 to convert to 1-indexed
            pages_to_render = [
                x + 1 for x in st.session_state.passage_page_map if x is not None
            ]

            pdf_viewer(
                st.session_state.pdf_content,
                pages_to_render=pages_to_render,
                height=600,
            )

        except Exception as e:
            st.error(f"Error displaying PDF: {str(e)}")

            # Fallback to base64 iframe if the component fails
            try:
                base64_pdf = base64.b64encode(st.session_state.pdf_content).decode(
                    "utf-8"
                )
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                st.info(
                    "Using fallback PDF viewer. For better experience, please install the streamlit-pdf-viewer component."
                )
            except Exception as e2:
                st.error(f"Failed to display PDF with fallback method: {str(e2)}")

        # Add a download button for the highlighted PDF
        st.download_button(
            label="📥 Download Highlighted PDF",
            data=st.session_state.pdf_content,
            file_name=f"highlighted_{st.session_state.uploaded_file_name}"
            if "uploaded_file_name" in st.session_state
            else "highlighted_document.pdf",
            mime="application/pdf",
        )

# Remove the custom div approach and use pure HTML/JavaScript injection to create the extreme right sidebar
right_sidebar_html = """
<div class="rightmost-sidebar">
    <h4 style="color: white; font-size: 16px; margin-bottom: 10px;">📝 Last Prompt</h4>
    <div class="prompt-box">
"""

# Add the prompt sections to the HTML if available
if st.session_state.last_prompt_sections:
    for i, section in enumerate(st.session_state.last_prompt_sections):
        if i == 0:  # System prompt
            right_sidebar_html += f'<div class="prompt-section">{section}</div>'
        elif i < len(st.session_state.last_prompt_sections) - 1:  # Passages
            highlight_class = f"highlight-{i}" if i <= 5 else "highlight-1"
            right_sidebar_html += (
                f'<div class="prompt-section {highlight_class}">{section}</div>'
            )
        else:  # Question and answer
            right_sidebar_html += f'<div class="prompt-section">{section}</div>'
else:
    right_sidebar_html += '<div class="prompt-section">No prompt available yet. Ask a question to see the prompt here.</div>'

# Close the rightmost sidebar HTML
right_sidebar_html += """
    </div>
</div>
"""

# Inject the rightmost sidebar HTML directly
st.markdown(right_sidebar_html, unsafe_allow_html=True)

# Empty the original right_sidebar column to prevent duplicate content
with right_sidebar:
    pass
