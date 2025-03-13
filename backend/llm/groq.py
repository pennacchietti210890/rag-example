import json
import logging
import os
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Raised when there's an error with external API calls"""

    pass


def get_available_models(api_key: str, api_url: str) -> List[str]:
    """Fetch available models from Groq API"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Remove any trailing slash from the API URL
        base_url = api_url.rstrip('/')
        response = requests.get(f"{base_url}/v1/models", headers=headers)
        
        if response.status_code == 200:
            models_data = response.json()
            # Extract model IDs from the response
            return [model["id"] for model in models_data.get("data", [])]
        else:
            logger.error(f"Failed to fetch models: {response.text}")
            raise APIError(f"Failed to fetch models: {response.text}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error while fetching models: {str(e)}")
        raise APIError(f"Request error while fetching models: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while fetching models: {str(e)}")
        raise APIError(f"Unexpected error while fetching models: {str(e)}")


def generate_response(
    prompt: str,
    groq_api_key: str,
    groq_api_url: str,
    model_name: str,
    sys_prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 200,
) -> Dict[str, str]:
    try:
        response = requests.post(
            groq_api_url,
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": sys_prompt,
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
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
            logger.error(f"Groq API error: {response.status_code} - {response.text}")
            raise APIError(
                f"Groq API returned status code {response.status_code}: {response.text}"
            )

    except requests.Timeout:
        logger.error("Groq API request timed out")
        raise APIError("Request to Groq API timed out")
    except requests.RequestException as e:
        logger.error(f"Groq API request failed: {str(e)}")
        raise APIError(f"Error calling Groq API: {str(e)}")
