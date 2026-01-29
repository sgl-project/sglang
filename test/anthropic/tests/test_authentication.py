"""
Tests for Anthropic API authentication.

Verifies both OpenAI-style (Authorization: Bearer) and
Anthropic-style (x-api-key) authentication headers work.
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


class TestAuthentication:
    """Test authentication header handling."""

    def test_messages_create__succeeds_with_x_api_key_header(self):
        """Test native Anthropic SDK auth using x-api-key header."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy-key")
        client = anthropic.Anthropic(
            base_url=get_base_url(),
            api_key=api_key,
        )
        model_name = os.environ.get("TEST_MODEL_NAME", "default")

        response = client.messages.create(
            model=model_name,
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'auth test' exactly."}],
        )

        assert response.id is not None
        assert response.role == "assistant"
        assert len(response.content) > 0

    def test_messages_create__succeeds_with_bearer_token_header(self):
        """Test OpenAI-style Authorization: Bearer header."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy-key")
        http_client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"})
        client = anthropic.Anthropic(
            base_url=get_base_url(),
            api_key=api_key,
            http_client=http_client,
        )
        model_name = os.environ.get("TEST_MODEL_NAME", "default")

        response = client.messages.create(
            model=model_name,
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'bearer test' exactly."}],
        )

        assert response.id is not None
        assert response.role == "assistant"
        assert len(response.content) > 0

    def test_messages_create__returns_401_when_api_key_invalid(self):
        """Test that invalid API key is rejected with 401."""
        client = anthropic.Anthropic(
            base_url=get_base_url(),
            api_key="wrong-api-key-that-should-fail",
        )
        model_name = os.environ.get("TEST_MODEL_NAME", "default")

        with pytest.raises(anthropic.AuthenticationError):
            client.messages.create(
                model=model_name,
                max_tokens=50,
                messages=[{"role": "user", "content": "This should fail."}],
            )
