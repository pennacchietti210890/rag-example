import logging
import os
import textwrap
import uuid
from enum import Enum
from logging.handlers import RotatingFileHandler
from threading import Lock
from typing import Dict, List, Optional
from builtins import Exception
import faiss
import fitz  # PyMuPDF
import numpy as np
import requests
import torch
import uvicorn
from ctransformers import AutoModelForCausalLM
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from llama_cpp import Llama
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline

from backend.rag.rag import DocumentManager, DocumentProcessingError
from backend.session_manager import SessionManager

from backend.llm.hf import load_local_model, ModelError


# Configure logging
def setup_logging():
    """Configure logging with both file and console handlers"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Configure logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler (rotating file handler to manage log size)
    file_handler = RotatingFileHandler(
        "logs/app.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger


# Initialize logging
logger = setup_logging()


# Model type enum
class ModelType(str, Enum):
    GROQ = "groq"
    LOCAL = "local"

    @classmethod
    def values(cls) -> List[str]:
        return [member.value for member in cls]


class QueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, description="The question to ask about the document"
    )
    model_type: ModelType = Field(
        ..., description="The type of model to use for generating the response"
    )
    session_id: str = Field(..., description="The session ID from the upload response")


# Custom exceptions
class APIError(Exception):
    """Raised when there's an error with external API calls"""

    pass


# Initialize session manager
session_manager = SessionManager()


async def get_document_manager(session_id: str) -> DocumentManager:
    """Dependency function to get or create a document manager for the current session"""
    logger.info(f"Received request with session_id: {session_id}")

    if not session_id:
        session_id = session_manager.create_session()
        logger.info(f"Created new session with ID: {session_id}")

    document_manager = session_manager.get_session(session_id)
    if not document_manager:
        logger.info(
            f"No document manager found for session {session_id}, creating new one"
        )
        document_manager = DocumentManager()
        session_manager._sessions[session_id] = document_manager

    logger.info(f"Document manager initialized: {document_manager.is_initialized}")
    return document_manager


async def get_document_manager_dep(session_id: Optional[str] = None) -> DocumentManager:
    """Dependency wrapper for get_document_manager"""
    return await get_document_manager(session_id)


async def get_document_manager_for_query(
    query_request: QueryRequest,
) -> DocumentManager:
    """Dependency function to get document manager for query endpoint"""
    return await get_document_manager(query_request.session_id)


# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(env_path)
app = FastAPI()

# Load the embedding model
try:
    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {str(e)}")
    raise ModelError(f"Failed to load embedding model: {str(e)}")

# Larger models via Groq API
GROQ_API_URL = os.getenv("GROQ_API_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Validate required environment variables
if not all([GROQ_API_URL, GROQ_API_KEY, MODEL_NAME]):
    logger.error("Missing required environment variables")
    raise ValueError(
        "Missing required environment variables. Please check your .env file."
    )


@app.post("/upload", include_in_schema=False)
@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    document_manager: DocumentManager = Depends(get_document_manager_dep),
):
    logger.info(f"Processing upload request for file: {file.filename}")

    if not file.filename.endswith(".pdf"):
        logger.warning(f"Invalid file type attempted: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        text = ""
        file_content = await file.read()

        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()

        if not text.strip():
            logger.warning("Empty PDF content detected")
            raise DocumentProcessingError("No text content found in the PDF")

        # Process document using document manager
        result = document_manager.process_document(text, embedding_model)
        logger.info(f"Successfully processed file with {result['num_chunks']} chunks")

        # Get the current session ID
        session_id = next(
            (
                sid
                for sid, dm in session_manager._sessions.items()
                if dm == document_manager
            ),
            None,
        )
        if not session_id:
            logger.error("No session ID found for document manager")
            raise HTTPException(status_code=500, detail="Session management error")

        logger.info(f"Returning session ID: {session_id}")
        return {
            "message": f"File processed with {result['num_chunks']} chunks indexed.",
            "session_id": session_id,
        }

    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            f"Unexpected error during file processing: {str(e)}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/query/")
async def query_doc(
    query_request: QueryRequest,
    document_manager: DocumentManager = Depends(get_document_manager_for_query),
):
    logger.info(f"Processing query request: {query_request.query[:50]}...")
    logger.info(f"Using session ID: {query_request.session_id}")

    if not document_manager.is_initialized:
        logger.warning("Query attempted before document upload")
        raise HTTPException(
            status_code=400,
            detail="No file uploaded or indexed. Please upload a document first.",
        )

    try:
        # Retrieve relevant chunks using document manager
        retrieved_chunks = document_manager.search_chunks(
            query_request.query, embedding_model
        )

        if not retrieved_chunks:
            logger.warning("No relevant chunks found for query")
            raise DocumentProcessingError("No relevant chunks found for the query")

        # Combine context with user query
        small_chunks = [chunk[:5000] for chunk in retrieved_chunks]
        context = "\n".join(small_chunks)

        # Check if all chunks are identical
        if len(set(small_chunks)) == 1:
            logger.warning(
                "FAISS retrieved identical chunks - potential issue with chunking"
            )

        prompt = f"You are a financial analyst. You are given a document and a question. You need to answer the question based on the document. Only provide the answer in your response and nothing else. Below is the data you need. Document Context:\n{context}\n\nUser Question: {query_request.query}\n\n Answer:"

        if query_request.model_type == ModelType.GROQ:
            try:
                logger.info("Using Groq API for response generation")
                # Query Groq API
                response = requests.post(
                    GROQ_API_URL,
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL_NAME,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an AI assistant specialized in analyzing financial reports.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 200,
                    },
                    timeout=30,  # Add timeout
                )

                if response.status_code == 200:
                    answer = (
                        response.json()
                        .get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "No answer found.")
                    )
                    logger.info("Successfully received response from Groq API")
                    return {"answer": answer.strip()}
                else:
                    logger.error(
                        f"Groq API error: {response.status_code} - {response.text}"
                    )
                    raise APIError(
                        f"Groq API returned status code {response.status_code}: {response.text}"
                    )

            except requests.Timeout:
                logger.error("Groq API request timed out")
                raise APIError("Request to Groq API timed out")
            except requests.RequestException as e:
                logger.error(f"Groq API request failed: {str(e)}")
                raise APIError(f"Error calling Groq API: {str(e)}")

        elif query_request.model_type == ModelType.LOCAL:
            try:
                logger.info("Using local model for response generation")
                llama_pipe, tokenizer = load_local_model()
                response = llama_pipe(
                    prompt,
                    max_new_tokens=200,
                    return_full_text=False,
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                )[0]["generated_text"]
                logger.info("Successfully generated response with local model")
                return {"answer": response}
            except Exception as e:
                logger.error(f"Local model error: {str(e)}", exc_info=True)
                raise ModelError(
                    f"Error generating response with local model: {str(e)}"
                )

        else:
            logger.warning(f"Invalid model type requested: {query_request.model_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model type. Must be one of: {', '.join(ModelType.values())}",
            )

    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except APIError as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except ModelError as e:
        logger.error(f"Model error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


def main():
    """Entry point for running the FastAPI application"""
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
