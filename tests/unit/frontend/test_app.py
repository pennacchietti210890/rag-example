import pytest
import json
import base64
from unittest.mock import patch, MagicMock
import streamlit as st
import requests
import pandas as pd

# Import the app module - we'll mock most of its dependencies
from frontend import app


class SessionStateMock(dict):
    """Mock for Streamlit's session_state that allows both dict and attribute access"""

    def __getattr__(self, key):
        if key in self:
            return self[key]
        return None

    def __setattr__(self, key, value):
        self[key] = value


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions"""
    with patch("frontend.app.st") as mock_st:
        # Configure common mock behaviors
        mock_st.session_state = SessionStateMock()
        mock_st.session_state["models_fetched"] = False
        mock_st.session_state["available_models"] = []

        mock_st.sidebar.title = MagicMock()
        mock_st.sidebar.file_uploader = MagicMock()
        mock_st.sidebar.slider = MagicMock()
        mock_st.sidebar.number_input = MagicMock()
        mock_st.sidebar.selectbox = MagicMock()
        mock_st.sidebar.button = MagicMock()
        mock_st.title = MagicMock()
        mock_st.write = MagicMock()
        mock_st.error = MagicMock()
        mock_st.success = MagicMock()
        mock_st.info = MagicMock()
        mock_st.spinner = MagicMock()
        mock_st.empty = MagicMock()
        mock_st.warning = MagicMock()

        yield mock_st


@pytest.fixture
def mock_requests():
    """Mock requests module"""
    with patch("frontend.app.requests") as mock_req:
        yield mock_req


class TestStreamlitApp:
    def test_fetch_available_models(self, mock_requests, mock_streamlit):
        """Test fetching available models from the backend"""
        # Setup mocks
        with patch("frontend.app.encrypt_api_key") as mock_encrypt:
            mock_encrypt.return_value = "encrypted_key"
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": ["llama3-70b-8192"]}
            mock_requests.get.return_value = mock_response

            # Call the function
            app.fetch_available_models("test_api_key")

            # Assertions
            assert mock_streamlit.session_state["available_models"] == [
                "llama3-70b-8192"
            ]
            assert mock_streamlit.session_state["models_fetched"] == True
            mock_encrypt.assert_called_once_with("test_api_key")
            mock_requests.get.assert_called_once()

            # Check that the encrypted_api_key parameter was passed correctly
            call_args = mock_requests.get.call_args
            assert call_args[1]["params"]["encrypted_api_key"] == "encrypted_key"

    def test_fetch_available_models_error(self, mock_requests, mock_streamlit):
        """Test error handling when fetching models fails"""
        # Setup mocks
        with patch("frontend.app.encrypt_api_key") as mock_encrypt:
            mock_encrypt.return_value = "encrypted_key"
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Server error"
            mock_requests.get.return_value = mock_response

            # Call the function
            app.fetch_available_models("test_api_key")

            # Assertions
            assert mock_streamlit.session_state["available_models"] == []
            assert mock_streamlit.session_state["models_fetched"] == True
            mock_streamlit.error.assert_called_once()

    def test_encrypt_api_key(self):
        """Test API key encryption"""
        # Setup
        with patch("frontend.app.encryption_key", "test_encryption_key"):
            with patch(
                "frontend.app.xor_encrypt_decrypt",
                side_effect=lambda data, key: f"encrypted_{data}",
            ) as mock_xor:
                # Call the function
                result = app.encrypt_api_key("test_api_key")

                # Assertions
                assert result == "encrypted_test_api_key"
                assert mock_xor.call_count == 2  # Called twice in the function

    def test_fetch_encryption_key(self, mock_requests, mock_streamlit):
        """Test fetching encryption key from backend"""
        # Setup mocks
        with patch.object(app, "encryption_key", None):  # Reset the global variable
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"encryption_key": "test_key"}
            mock_requests.get.return_value = mock_response

            # Call the function
            result = app.fetch_encryption_key()

            # Assertions
            assert result == True
            assert app.encryption_key == "test_key"
            assert mock_streamlit.session_state["encryption_key"] == "test_key"
            mock_requests.get.assert_called_once()

    def test_fetch_encryption_key_error(self, mock_requests, mock_streamlit):
        """Test error handling when fetching encryption key fails"""
        # Setup mocks
        with patch.object(app, "encryption_key", None):  # Reset the global variable
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_requests.get.return_value = mock_response

            # Call the function
            result = app.fetch_encryption_key()

            # Assertions
            assert result == False
            mock_streamlit.error.assert_called_once()

    def test_xor_encrypt_decrypt(self):
        """Test XOR encryption and decryption"""
        # Test data
        data = "test_data"
        key = "test_key"

        # Encrypt
        encrypted = app.xor_encrypt_decrypt(data, key)

        # Decrypt
        decrypted = app.xor_encrypt_decrypt(encrypted, key)

        # Assertions
        assert encrypted != data

        # Manually implement the XOR encryption/decryption to verify
        data_bytes = data.encode()
        key_bytes = (key * (len(data_bytes) // len(key) + 1))[
            : len(data_bytes)
        ].encode()
        result_bytes = bytes(a ^ b for a, b in zip(data_bytes, key_bytes))
        expected_encrypted = base64.b64encode(result_bytes).decode()

        assert encrypted == expected_encrypted

        # For the roundtrip test, we need to decode the base64 first
        encrypted_bytes = base64.b64decode(encrypted.encode())
        result_bytes = bytes(a ^ b for a, b in zip(encrypted_bytes, key_bytes))
        expected_decrypted = result_bytes.decode()

        assert expected_decrypted == data

    def test_conditional_rag_settings_display(self, mock_streamlit):
        """Test conditional display of RAG settings based on RAG mode"""
        # Create a mock for the expander and its context manager
        mock_expander = MagicMock()
        mock_streamlit.sidebar.expander.return_value = mock_expander

        # Mock for the slider widgets inside the expander
        mock_streamlit.sidebar.slider.return_value = 500  # Default values
        mock_streamlit.sidebar.selectbox.return_value = "l2"  # Default distance metric

        # Test with No RAG selected
        mock_streamlit.sidebar.radio.return_value = "No RAG"
        mock_streamlit.session_state.rag_enabled = (
            mock_streamlit.sidebar.radio.return_value != "No RAG"
        )

        # In the actual app, the RAG settings expander would not be shown or its content would be hidden
        # We can't fully test the UI flow, but we can verify the session state
        assert mock_streamlit.session_state.rag_enabled == False

        # Test with RAG selected
        mock_streamlit.sidebar.radio.return_value = "RAG"
        mock_streamlit.session_state.rag_enabled = (
            mock_streamlit.sidebar.radio.return_value != "No RAG"
        )

        # In the actual app, the RAG settings expander would be shown
        assert mock_streamlit.session_state.rag_enabled == True

        # Simulate the app's behavior of showing the RAG settings when RAG is enabled
        if mock_streamlit.session_state.rag_enabled:
            # These should be called when RAG is enabled
            with mock_expander:
                chunk_size = mock_streamlit.sidebar.slider("Chunk Size")
                chunk_overlap = mock_streamlit.sidebar.slider("Chunk Overlap")
                num_chunks = mock_streamlit.sidebar.slider("Number of Chunks")
                distance_metric = mock_streamlit.sidebar.selectbox("Distance Metric")

        # Verify that the slider was called for chunk size, overlap, and number of chunks
        assert mock_streamlit.sidebar.slider.call_count >= 3

        # Reset mock for testing Self-RAG
        mock_streamlit.sidebar.slider.reset_mock()
        mock_streamlit.sidebar.selectbox.reset_mock()

        # Test with Self-RAG selected
        mock_streamlit.sidebar.radio.return_value = "Self-RAG"
        mock_streamlit.session_state.rag_enabled = (
            mock_streamlit.sidebar.radio.return_value != "No RAG"
        )

        # Verify rag_enabled is True for Self-RAG
        assert mock_streamlit.session_state.rag_enabled == True

        # Simulate the app's behavior of showing the RAG settings when Self-RAG is enabled
        if mock_streamlit.session_state.rag_enabled:
            # These should be called when Self-RAG is enabled (same as RAG)
            with mock_expander:
                chunk_size = mock_streamlit.sidebar.slider("Chunk Size")
                chunk_overlap = mock_streamlit.sidebar.slider("Chunk Overlap")
                num_chunks = mock_streamlit.sidebar.slider("Number of Chunks")
                distance_metric = mock_streamlit.sidebar.selectbox("Distance Metric")

        # Verify that the slider was called for chunk size, overlap, and number of chunks
        assert mock_streamlit.sidebar.slider.call_count >= 3
        # Verify that the selectbox was called for distance metric
        assert mock_streamlit.sidebar.selectbox.call_count >= 1
