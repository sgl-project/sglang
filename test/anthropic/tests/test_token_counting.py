"""
Tests for Anthropic token counting endpoint.

Tests the /v1/messages/count_tokens endpoint.
"""

import os

import httpx
import pytest


def get_base_url() -> str:
    """Get base URL from environment, normalizing /v1 suffix."""
    url = os.environ.get("ANTHROPIC_BASE_URL", "http://localhost:8000")
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


@pytest.fixture
def api_key():
    """Get API key from environment."""
    return os.environ.get("ANTHROPIC_API_KEY", "dummy-key")


@pytest.fixture
def model_name():
    """Model name for tests."""
    return os.environ.get("TEST_MODEL_NAME", "default")


class TestTokenCounting:
    """Test token counting endpoint."""

    def test_count_tokens__returns_positive_count_for_simple_message(
        self, api_key, model_name
    ):
        """Test basic token counting."""
        response = httpx.post(
            f"{get_base_url()}/v1/messages/count_tokens",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Hello, world!"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "input_tokens" in data
        assert isinstance(data["input_tokens"], int)
        assert data["input_tokens"] > 0

    def test_count_tokens__includes_system_message_tokens(self, api_key, model_name):
        """Test token counting with system message."""
        response = httpx.post(
            f"{get_base_url()}/v1/messages/count_tokens",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "input_tokens" in data
        assert data["input_tokens"] > 0

    def test_count_tokens__includes_tool_definition_tokens(self, api_key, model_name):
        """Test token counting includes tool definitions."""
        response = httpx.post(
            f"{get_base_url()}/v1/messages/count_tokens",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "What's the weather?"}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                            "required": ["location"],
                        },
                    }
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "input_tokens" in data
        assert data["input_tokens"] > 0

    def test_count_tokens__counts_multi_turn_conversation(self, api_key, model_name):
        """Test token counting for multi-turn conversation."""
        response = httpx.post(
            f"{get_base_url()}/v1/messages/count_tokens",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there! How can I help?"},
                    {"role": "user", "content": "What is 2+2?"},
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "input_tokens" in data
        assert data["input_tokens"] > 0

    def test_count_tokens__longer_content_returns_more_tokens(
        self, api_key, model_name
    ):
        """Test that longer content results in more tokens."""
        short_response = httpx.post(
            f"{get_base_url()}/v1/messages/count_tokens",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        long_response = httpx.post(
            f"{get_base_url()}/v1/messages/count_tokens",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": "This is a much longer message that should result in significantly more tokens being counted by the tokenizer.",
                    }
                ],
            },
        )

        assert short_response.status_code == 200
        assert long_response.status_code == 200

        short_tokens = short_response.json()["input_tokens"]
        long_tokens = long_response.json()["input_tokens"]

        assert long_tokens > short_tokens

    def test_count_tokens__succeeds_with_x_api_key_header(self, api_key, model_name):
        """Test token counting with x-api-key auth header."""
        response = httpx.post(
            f"{get_base_url()}/v1/messages/count_tokens",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "input_tokens" in data
