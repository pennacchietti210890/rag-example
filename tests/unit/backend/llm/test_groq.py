import pytest
from unittest.mock import patch, MagicMock
import json

from backend.llm.groq import generate_response, get_available_models, APIError


@pytest.mark.unit
class TestGroqIntegration:
    
    @patch('requests.post')
    def test_generate_response_success(self, mock_post, mock_groq_response):
        """Test successful response generation from Groq API"""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_groq_response
        mock_post.return_value = mock_response
        
        # When
        result = generate_response(
            prompt="What was the revenue in Q1?",
            groq_api_key="test_api_key",
            groq_api_url="https://api.test.com",
            model_name="llama3-70b-8192",
            sys_prompt="You are a helpful assistant",
            temperature=0.7,
            top_p=0.9,
            max_tokens=100
        )
        
        # Then
        assert result["answer"] == "The company reported revenue of $10 million in Q1 2023."
        mock_post.assert_called_once()
        
        # Verify the correct parameters were sent
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://api.test.com"
        
        # Check headers
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test_api_key"
        assert headers["Content-Type"] == "application/json"
        
        # Check payload
        payload = call_args[1]["json"]
        assert payload["model"] == "llama3-70b-8192"
        assert payload["max_tokens"] == 100
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
    
    @patch('requests.post')
    def test_generate_response_api_error(self, mock_post):
        """Test handling of API errors"""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response
        
        # When/Then
        with pytest.raises(APIError):
            generate_response(
                prompt="Test prompt",
                groq_api_key="invalid_key",
                groq_api_url="https://api.test.com",
                model_name="llama3-70b-8192",
                sys_prompt="You are a helpful assistant",
            )
    
    @patch('requests.post')
    def test_generate_response_connection_error(self, mock_post):
        """Test handling of connection errors"""
        # Setup mock
        mock_post.side_effect = Exception("Connection error")
        
        # When/Then
        with pytest.raises(Exception):
            generate_response(
                prompt="Test prompt",
                groq_api_key="test_key",
                groq_api_url="https://api.test.com",
                model_name="llama3-70b-8192",
                sys_prompt="You are a helpful assistant",
            )
    
    @patch('requests.get')
    def test_get_available_models_success(self, mock_get):
        """Test successful retrieval of available models"""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "llama3-70b-8192", "object": "model"},
                {"id": "mixtral-8x7b-32768", "object": "model"}
            ]
        }
        mock_get.return_value = mock_response
        
        # When
        models = get_available_models("test_api_key", "https://api.test.com")
        
        # Then
        assert len(models) == 2
        assert "llama3-70b-8192" in models
        assert "mixtral-8x7b-32768" in models
        
        # Verify the correct parameters were sent
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://api.test.com/v1/models"
        
        # Check headers
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test_api_key"
    
    @patch('requests.get')
    def test_get_available_models_error(self, mock_get):
        """Test handling of errors when retrieving models"""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_get.return_value = mock_response
        
        # When/Then
        with pytest.raises(APIError):
            get_available_models("invalid_key", "https://api.test.com") 