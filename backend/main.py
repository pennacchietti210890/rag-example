import logging
import os
import textwrap
import uuid
import base64
import secrets
from builtins import Exception
from enum import Enum
from logging.handlers import RotatingFileHandler
from threading import Lock
from typing import Dict, List, Optional

import faiss
import fitz  # PyMuPDF
import numpy as np
import requests
import torch
import uvicorn
from ctransformers import AutoModelForCausalLM
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline

from backend.rag.rag import DocumentManager, DocumentProcessingError
from backend.session_manager import SessionManager
from backend.llm.groq import generate_response, APIError, get_available_models


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


# Decrypt function specifically for API keys
def decrypt_api_key(encrypted_key: str, key: str) -> str:
    """Decrypt an API key that was encrypted with XOR cipher"""
    try:
        # Decode base64
        encrypted_bytes = base64.b64decode(encrypted_key)
        # Convert to string for XOR
        encrypted_str = encrypted_bytes.decode("utf-8", errors="ignore")
        # Apply XOR with key
        key_bytes = (key * (len(encrypted_str) // len(key) + 1))[
            : len(encrypted_str)
        ].encode()
        result_bytes = bytes(a ^ b for a, b in zip(encrypted_str.encode(), key_bytes))
        # Return decrypted result
        return result_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Error decrypting API key: {str(e)}")
        # Return a placeholder to avoid breaking the app during debugging
        return "invalid_key"


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

# Generate a secure encryption key
ENCRYPTION_KEY = secrets.token_hex(16)  # 32 character hex string


class QueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, description="The question to ask about the document"
    )
    model_name: str = Field(..., description="The specific Groq model to use")
    encrypted_api_key: str = Field(..., description="User's encrypted Groq API key")
    session_id: str = Field(..., description="The session ID from the upload response")
    # LLM parameters
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=200, ge=1, le=2000)
    # RAG parameters
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=200)
    num_chunks: int = Field(default=3, ge=1, le=10)
    # RAG mode parameters
    rag_enabled: bool = Field(default=True, description="Whether to use RAG or not")
    rag_mode: str = Field(default="rag", description="RAG mode: 'rag' or 'self-rag'")

    class Config:
        schema_extra = {
            "example": {
                "query": "What is the revenue for Q1?",
                "model_name": "llama3-70b-8192",
                "encrypted_api_key": "gsk_xxx",
                "session_id": "abc123",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 200,
                "chunk_size": 500,
                "chunk_overlap": 50,
                "num_chunks": 3,
                "rag_enabled": True,
                "rag_mode": "rag",
            }
        }


# Initialize session manager
session_manager = SessionManager()


async def get_document_manager(
    session_id: str, chunk_size: int = 500, chunk_overlap: int = 50, num_chunks: int = 3
) -> DocumentManager:
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
        document_manager = DocumentManager(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_chunks=num_chunks,
        )
        session_manager._sessions[session_id] = document_manager
    else:
        # Update existing document manager parameters
        document_manager.chunk_size = chunk_size
        document_manager.chunk_overlap = chunk_overlap
        document_manager.num_chunks = num_chunks

    logger.info(f"Document manager initialized: {document_manager.is_initialized}")
    return document_manager


async def get_document_manager_dep(
    session_id: Optional[str] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    num_chunks: int = 3,
) -> DocumentManager:
    """Dependency wrapper for get_document_manager"""
    return await get_document_manager(
        session_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        num_chunks=num_chunks,
    )


async def get_document_manager_for_query(
    query_request: QueryRequest,
) -> DocumentManager:
    """Dependency function to get document manager for query endpoint"""
    return await get_document_manager(
        query_request.session_id,
        chunk_size=query_request.chunk_size,
        chunk_overlap=query_request.chunk_overlap,
        num_chunks=query_request.num_chunks,
    )


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

# Remove global API key variables since we'll use per-request keys
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


@app.post("/upload", include_in_schema=False)
@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    num_chunks: int = Form(3),
    document_manager: DocumentManager = Depends(get_document_manager_dep),
):
    logger.info(f"Processing upload request for file: {file.filename}")
    logger.info(
        f"RAG parameters - chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}, num_chunks: {num_chunks}"
    )

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

        # Update document manager parameters
        document_manager.chunk_size = chunk_size
        document_manager.chunk_overlap = chunk_overlap
        document_manager.num_chunks = num_chunks

        # Process or reprocess document using document manager
        if document_manager.is_initialized:
            logger.info("Reprocessing document with new parameters")
            result = document_manager.reprocess_document(embedding_model)
        else:
            logger.info("Processing new document")
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
    logger.info(
        f"RAG parameters - chunk_size: {query_request.chunk_size}, chunk_overlap: {query_request.chunk_overlap}, num_chunks: {query_request.num_chunks}"
    )
    logger.info(
        f"RAG mode: {query_request.rag_mode}, RAG enabled: {query_request.rag_enabled}"
    )

    if not document_manager.is_initialized:
        logger.warning("Query attempted before document upload")
        raise HTTPException(
            status_code=400,
            detail="No file uploaded or indexed. Please upload a document first.",
        )

    try:
        # Decrypt the API key using the new function
        decrypted_api_key = decrypt_api_key(
            query_request.encrypted_api_key, ENCRYPTION_KEY
        )
        logger.info(f"API key decryption successful")

        # Update document manager parameters
        document_manager.chunk_size = query_request.chunk_size
        document_manager.chunk_overlap = query_request.chunk_overlap
        document_manager.num_chunks = query_request.num_chunks

        # Reprocess document with new parameters
        logger.info("Reprocessing document with new parameters")
        document_manager.reprocess_document(embedding_model)

        # Choose retrieval method based on RAG mode
        if query_request.rag_mode.lower() == "self-rag":
            logger.info("Using Self-RAG retrieval mode")
            # Self-RAG returns both the final answer and the retrieved chunks
            llm_response, retrieved_chunks = document_manager.search_chunks_self_rag(
                query_request.query,
                embedding_model,
                num_chunks=query_request.num_chunks,
                model_name=query_request.model_name,
                api_key=decrypted_api_key,
            )

            if not retrieved_chunks:
                logger.warning("No relevant chunks found for Self-RAG query")
                raise DocumentProcessingError("No relevant chunks found for the query")

            prompt_sections = [
                "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise",
                *[
                    f"Passage {i+1}:\n{chunk}\n"
                    for i, chunk in enumerate(retrieved_chunks)
                ],
                f"\n\nUser Question: {query_request.query}\n\nAnswer:",
            ]
            # For Self-RAG, we'll use the LLM response directly
            return {
                "answer": llm_response,
                "prompt_sections": prompt_sections,
                "retrieved_passages": retrieved_chunks,
            }
        else:
            logger.info("Using standard RAG retrieval mode")
            # Standard RAG mode
            retrieved_chunks = document_manager.search_chunks(
                query_request.query,
                embedding_model,
                num_chunks=query_request.num_chunks,
            )

            if not retrieved_chunks:
                logger.warning("No relevant chunks found for query")
                raise DocumentProcessingError("No relevant chunks found for the query")

            # Combine context with user query
            small_chunks = [chunk[:] for chunk in retrieved_chunks]
            context = "\n".join(small_chunks)

            # Check if all chunks are identical
            if len(set(small_chunks)) == 1:
                logger.warning(
                    "FAISS retrieved identical chunks - potential issue with chunking"
                )

            # Split the prompt into sections for highlighting
            prompt_sections = [
                "You are given a document and a question. You need to answer the question based on the document. Only provide the answer in your response and nothing else. Below is the data you need.\n\nDocument Context:\n",
                *[f"Passage {i+1}:\n{chunk}\n" for i, chunk in enumerate(small_chunks)],
                f"\n\nUser Question: {query_request.query}\n\nAnswer:",
            ]

            prompt = f"You are given a document and a question. You need to answer the question based on the document. Only provide the answer in your response and nothing else. Below is the data you need.\n\nDocument Context:\n{context}\n\nUser Question: {query_request.query}\n\nAnswer:"

        # For standard RAG, generate response using decrypted API key
        if (
            query_request.rag_mode.lower() != "self-rag"
            or not query_request.rag_enabled
        ):
            response = generate_response(
                prompt=prompt,
                groq_api_key=decrypted_api_key,
                groq_api_url=GROQ_API_URL,
                model_name=query_request.model_name,
                sys_prompt="You are given a document and a question. You need to answer the question based on the document. Only provide the answer in your response and nothing else.",
                temperature=query_request.temperature,
                top_p=query_request.top_p,
                max_tokens=query_request.max_tokens,
            )

            return {
                "answer": response.get("answer", ""),
                "prompt_sections": prompt_sections,
                "retrieved_passages": retrieved_chunks,
            }

    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except APIError as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/models/")
async def get_models(encrypted_api_key: str):
    """Get available models from Groq using user's encrypted API key"""
    try:
        logger.info(
            f"Received encrypted API key (first 10 chars): {encrypted_api_key[:10]}..."
        )

        # Decrypt the API key using the new function
        decrypted_api_key = decrypt_api_key(encrypted_api_key, ENCRYPTION_KEY)
        logger.info(f"API key decryption successful for models endpoint")

        # Validate API key format
        if not decrypted_api_key.startswith("gsk_"):
            logger.warning(
                f"Decrypted API key doesn't have expected format (should start with 'gsk_')"
            )
            # Try a direct approach as fallback
            if encrypted_api_key.startswith("gsk_"):
                logger.info("Using original API key as fallback")
                decrypted_api_key = encrypted_api_key

        logger.info(
            f"Fetching models with API key (first 5 chars): {decrypted_api_key[:5]}..."
        )
        models = get_available_models(decrypted_api_key, "https://api.groq.com/openai")
        return {"models": models}
    except APIError as e:
        logger.error(f"Failed to fetch Groq models: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error fetching models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/encryption-key")
async def get_encryption_key():
    """Get the encryption key for secure API key transmission"""
    return {"encryption_key": ENCRYPTION_KEY}


def main():
    """Entry point for running the FastAPI application"""
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
