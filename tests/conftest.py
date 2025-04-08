import os
import pytest
from unittest.mock import MagicMock, patch
import tempfile
import numpy as np
import requests_mock
import json

from backend.session_manager import SessionManager
from backend.rag.rag import DocumentManager


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model that returns fixed embeddings"""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
    return mock_model


@pytest.fixture
def sample_text():
    """Sample text for testing document processing"""
    return """
    This is a sample financial report for testing purposes.
    The company reported revenue of $10 million in Q1 2023.
    Expenses were $7 million, resulting in a profit of $3 million.
    The board expects growth of 15% in the next quarter.
    """


@pytest.fixture
def document_manager():
    """Create a document manager for testing"""
    return DocumentManager(
        chunk_size=100, chunk_overlap=20, num_chunks=2, distance_metric="l2"
    )


@pytest.fixture
def session_manager():
    """Create a session manager for testing"""
    return SessionManager()


@pytest.fixture
def mock_groq_response():
    """Mock response from Groq API"""
    return {
        "id": "mock-response-id",
        "object": "chat.completion",
        "created": 1679825466,
        "model": "llama3-70b-8192",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The company reported revenue of $10 million in Q1 2023.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 150, "completion_tokens": 20, "total_tokens": 170},
    }


@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
        # This is just a placeholder - in a real test you'd create an actual PDF
        temp.write(b"%PDF-1.5\nSome dummy PDF content")
        temp_name = temp.name

    yield temp_name

    # Cleanup after test
    if os.path.exists(temp_name):
        os.unlink(temp_name)


@pytest.fixture
def mock_api_responses():
    """Mock API responses for testing"""
    with requests_mock.Mocker() as m:
        # Mock encryption key endpoint
        m.get("http://testserver/encryption-key", json={"encryption_key": "test_key"})

        # Mock models endpoint
        m.get(
            "http://testserver/models/",
            json={"models": ["llama3-70b-8192", "mixtral-8x7b-32768"]},
        )

        # Mock upload endpoint
        m.post(
            "http://testserver/upload/",
            json={"session_id": "test_session_id", "message": "Document processed"},
        )

        # Mock query endpoint
        m.post(
            "http://testserver/query/",
            json={
                "answer": "The revenue was $10 million in Q1 2023.",
                "prompt_sections": ["section1", "section2"],
                "retrieved_passages": ["passage1", "passage2"],
            },
        )

        yield m
