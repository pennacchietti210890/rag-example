# RAG Financial Reports

A Retrieval-Augmented Generation (RAG) system for analyzing financial reports. This system allows users to upload financial documents and ask questions about their contents using either a local Llama model or the Groq API.

## Features

- PDF document upload and processing
- Document chunking and embedding generation
- Semantic search using FAISS
- Question answering using either:
  - Local Llama 3B model
  - Groq API (Llama 70B)
- Session management for multiple documents
- Thread-safe operations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-financial-reports.git
cd rag-financial-reports
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

4. Set up environment variables:
Create a `.env` file in the backend directory with:
```
GROQ_API_URL=your_groq_api_url
GROQ_API_KEY=your_groq_api_key
MODEL_NAME=your_model_name
```

## Usage

1. Start the backend server:
```bash
cd backend
python main.py
```

2. Start the frontend:
```bash
cd frontend
streamlit run app.py
```

3. Open your browser and navigate to `http://localhost:8501`

## Project Structure

```
rag-financial-reports/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── rag/
│   │   ├── __init__.py
│   │   └── rag.py
│   └── session_manager.py
├── frontend/
│   └── app.py
├── setup.py
└── README.md
```

## License

MIT License
