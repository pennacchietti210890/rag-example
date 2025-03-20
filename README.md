# RAG Example

A Retrieval-Augmented Generation (RAG) system for analyzing pdf documents. This system allows users to upload a pdf document and ask questions about their contents via open source LLMs available via Groq (the user needs a free API from Groq), showing you how setting up different LLM or RAG parameters influence what kind of context is retrieved for generation.

## UI Preview

![RAG Application UI Preview](images/UI_preview.png)

## Features

- PDF document upload and processing
- Document chunking and embedding generation
- Semantic search using FAISS
- Question answering using Groq API (Llama 3, Mixtral, etc.)
- Session management for multiple documents
- Thread-safe operations
- Docker support for easy deployment
- Secure API key handling with encryption

## Installation

### Using Poetry (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/pennacchietti210890/rag-example
cd rag-exapmle
```

2. Install Poetry if you don't have it already:
```bash
pip install poetry
```

3. Install dependencies using Poetry:
```bash
poetry install
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```
GROQ_API_URL=https://api.groq.com/openai/v1/chat/completions
GROQ_API_KEY=your_groq_api_key
```

### Using Docker

1. Clone the repository:
```bash
git clone https://github.com/pennacchietti210890/rag-example
cd rag-example
```

2. Build and run with Docker Compose:
```bash
docker-compose up --build
```

## Usage

### Running with Poetry

1. Start the backend server:
```bash
poetry run start-backend
```

2. Start the frontend:
```bash
poetry run start-frontend
```

3. Open your browser and navigate to `http://localhost:8501`

### Running with Docker

After running `docker-compose up --build`, open your browser and navigate to `http://localhost:8501`

## Project Structure

```
rag-example/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── session_manager.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── groq.py
│   │   └── prompts/
│   └── rag/
│       ├── __init__.py
│       └── rag.py
├── frontend/
│   └── app.py
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── pyproject.toml
├── poetry.lock
├── requirements.txt
├── .env
└── README.md
```

## How It Works

1. **Document Processing**: Upload a PDF document through the Streamlit interface.
2. **Chunking and Embedding**: The document is split into chunks and embedded using sentence-transformers.
3. **Query Processing**: When you ask a question, the system:
   - Finds the most relevant chunks using semantic search
   - Constructs a prompt with the relevant context
   - Sends the prompt to Groq API
   - Returns the answer

## Security

- User API keys are encrypted during transmission between frontend and backend
- API keys are never stored persistently
- Password field masks API key input

## License

MIT License
