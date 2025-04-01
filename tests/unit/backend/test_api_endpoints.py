import pytest
from unittest.mock import patch, MagicMock
import io
import json
import requests
import requests_mock

from backend.main import app


@pytest.mark.unit
class TestAPIEndpoints:
    def test_encryption_key_endpoint(self, mock_api_responses):
        """Test the encryption key endpoint"""
        # When
        response = requests.get("http://testserver/encryption-key")

        # Then
        assert response.status_code == 200
        assert "encryption_key" in response.json()
        assert isinstance(response.json()["encryption_key"], str)

    @patch("backend.main.get_available_models")
    def test_models_endpoint(self, mock_get_models, mock_api_responses):
        """Test the models endpoint"""
        # Setup mock
        mock_get_models.return_value = ["llama3-70b-8192", "mixtral-8x7b-32768"]

        # When
        response = requests.get("http://testserver/models/?encrypted_api_key=test_key")

        # Then
        assert response.status_code == 200
        assert "models" in response.json()
        assert len(response.json()["models"]) == 2
        assert "llama3-70b-8192" in response.json()["models"]

    @patch("backend.main.get_available_models")
    def test_models_endpoint_error(self, mock_get_models, mock_api_responses):
        """Test error handling in the models endpoint"""
        # Setup mock
        from backend.llm.groq import APIError

        mock_get_models.side_effect = APIError("API Error")

        # Mock error response
        mock_api_responses.get(
            "http://testserver/models/?encrypted_api_key=test_key",
            status_code=503,
            json={"detail": "API Error"},
        )

        # When
        response = requests.get("http://testserver/models/?encrypted_api_key=test_key")

        # Then
        assert response.status_code == 503
        assert "detail" in response.json()

    @patch("backend.main.DocumentManager.process_document")
    @patch("backend.main.fitz.open")
    def test_upload_endpoint(
        self, mock_fitz_open, mock_process_document, mock_api_responses
    ):
        """Test the upload endpoint"""
        # Setup mocks
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Sample text for testing"
        mock_doc.__enter__.return_value = [mock_page]
        mock_fitz_open.return_value = mock_doc

        mock_process_document.return_value = {"num_chunks": 3}

        # Create a test PDF file
        test_file = io.BytesIO(b"%PDF-1.5\nTest PDF content")
        test_file.name = "test.pdf"

        # When
        files = {"file": ("test.pdf", test_file, "application/pdf")}
        data = {
            "chunk_size": "500", 
            "chunk_overlap": "50", 
            "num_chunks": "3",
            "distance_metric": "l2"
        }
        response = requests.post("http://testserver/upload/", files=files, data=data)

        # Then
        assert response.status_code == 200
        assert "session_id" in response.json()
        assert "message" in response.json()

    @patch("backend.main.generate_response")
    def test_query_endpoint(self, mock_generate_response, mock_api_responses):
        """Test the query endpoint"""
        # Setup mock
        mock_generate_response.return_value = {"answer": "Test answer"}

        # Create a test query
        query_data = {
            "query": "What was the revenue?",
            "model_name": "llama3-70b-8192",
            "encrypted_api_key": "test_encrypted_key",
            "session_id": "test_session_id",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "chunk_size": 500,
            "chunk_overlap": 50,
            "num_chunks": 3,
            "rag_enabled": True,
            "rag_mode": "rag",
            "distance_metric": "l2"
        }

        # Mock the document manager dependency
        with patch("backend.main.get_document_manager_for_query") as mock_get_dm:
            # Setup the mock document manager
            mock_dm = MagicMock()
            mock_dm.is_initialized = True
            mock_dm.search_chunks.return_value = ["Chunk 1", "Chunk 2"]
            mock_get_dm.return_value = mock_dm

            # When
            response = requests.post("http://testserver/query/", json=query_data)

            # Then
            assert response.status_code == 200
            assert "answer" in response.json()
            assert "prompt_sections" in response.json()
            assert "retrieved_passages" in response.json()

    def test_query_endpoint_no_document(self, mock_api_responses):
        """Test query endpoint when no document has been uploaded"""
        # Create a test query
        query_data = {
            "query": "What was the revenue?",
            "model_name": "llama3-70b-8192",
            "encrypted_api_key": "test_encrypted_key",
            "session_id": "nonexistent_session",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "chunk_size": 500,
            "chunk_overlap": 50,
            "num_chunks": 3,
        }

        # Mock error response
        mock_api_responses.post(
            "http://testserver/query/",
            status_code=400,
            json={"detail": "No document has been processed for this session"},
        )

        # Mock the document manager dependency
        with patch("backend.main.get_document_manager_for_query") as mock_get_dm:
            # Setup the mock document manager
            mock_dm = MagicMock()
            mock_dm.is_initialized = False
            mock_get_dm.return_value = mock_dm

            # When
            response = requests.post("http://testserver/query/", json=query_data)

            # Then
            assert response.status_code == 400
            assert "detail" in response.json()
