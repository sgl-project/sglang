"""
Pytest configuration and fixtures for Anthropic API tests.
"""

import os

import anthropic
import httpx
import pytest


def get_base_url() -> str:
    """Get base URL from environment, normalizing /v1 suffix."""
    url = os.environ.get("ANTHROPIC_BASE_URL", "http://localhost:8000")
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


@pytest.fixture(scope="session")
def client():
    """Create Anthropic client configured for SGLang server.

    Uses custom httpx client to send Authorization header (OpenAI-style)
    since SGLang's auth middleware expects Bearer tokens.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy-key")
    http_client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"})
    return anthropic.Anthropic(
        base_url=get_base_url(),
        api_key=api_key,
        http_client=http_client,
    )


@pytest.fixture(scope="session")
def model_name():
    """Model name for tests. SGLang uses whatever model is loaded."""
    return os.environ.get("TEST_MODEL_NAME", "default")


@pytest.fixture
def weather_tool():
    """Weather tool definition."""
    return {
        "name": "get_weather",
        "description": "Get the current weather for a specific location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g. 'Paris, France'",
                }
            },
            "required": ["location"],
        },
    }


@pytest.fixture
def calculator_tool():
    """Calculator tool definition."""
    return {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    }
