import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
import requests

from backend.main import app


@pytest.fixture
async def test_client():
    """Create a test client for the FastAPI app"""
    client = AsyncClient(app=app, base_url="http://test")
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def sample_pdf():
    """Create a sample PDF file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
        # This is just a placeholder - in a real test you'd create an actual PDF
        temp.write(b"%PDF-1.5\nSample financial report content for testing.")
        temp_name = temp.name

    yield temp_name

    # Cleanup after test
    if os.path.exists(temp_name):
        os.unlink(temp_name)


@pytest.mark.asyncio
@pytest.mark.functional
class TestRAGFlow:
    @patch("backend.main.generate_response")
    @patch("backend.main.get_available_models")
    def test_end_to_end_flow(
        self, mock_get_models, mock_generate_response, mock_api_responses, sample_pdf
    ):
        """Test the complete RAG flow from upload to query"""
        # Create a test PDF file
        with open(sample_pdf, "rb") as f:
            pdf_content = f.read()

        # Create a test file for upload
        files = {"file": ("test.pdf", pdf_content, "application/pdf")}
        data = {
            "chunk_size": "500",
            "chunk_overlap": "50",
            "num_chunks": "3",
            "distance_metric": "l2",
        }

        # Step 1: Get encryption key
        encryption_response = requests.get("http://testserver/encryption-key")
        assert encryption_response.status_code == 200
        assert "encryption_key" in encryption_response.json()

        # Step 2: Get available models
        mock_get_models.return_value = ["llama3-70b-8192", "mixtral-8x7b-32768"]
        models_response = requests.get("http://testserver/models/")
        assert models_response.status_code == 200
        assert "models" in models_response.json()
        assert "llama3-70b-8192" in models_response.json()["models"]

        # Step 3: Upload document
        upload_response = requests.post(
            "http://testserver/upload/", files=files, data=data
        )
        assert upload_response.status_code == 200
        assert "session_id" in upload_response.json()
        session_id = upload_response.json()["session_id"]

        # Step 4: Query the document
        mock_generate_response.return_value = {
            "answer": "The revenue was $10 million in Q1 2023."
        }
        query_data = {
            "query": "What was the revenue in Q1?",
            "model_name": "llama3-70b-8192",
            "encrypted_api_key": "test_encrypted_key",
            "session_id": session_id,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "chunk_size": 500,
            "chunk_overlap": 50,
            "num_chunks": 3,
            "rag_enabled": True,
            "rag_mode": "rag",
            "distance_metric": "l2",
        }

        query_response = requests.post("http://testserver/query/", json=query_data)
        assert query_response.status_code == 200
        assert "answer" in query_response.json()
        assert (
            query_response.json()["answer"] == "The revenue was $10 million in Q1 2023."
        )

        # Verify that the response includes prompt sections and retrieved passages
        assert "prompt_sections" in query_response.json()
        assert "retrieved_passages" in query_response.json()
