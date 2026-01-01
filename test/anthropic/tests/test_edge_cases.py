"""
Tests for edge cases in Anthropic API.

Tests boundary conditions and special inputs.
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


@pytest.fixture
def client():
    """Create Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy-key")
    http_client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"})
    return anthropic.Anthropic(
        base_url=get_base_url(),
        api_key=api_key,
        http_client=http_client,
    )


@pytest.fixture
def model_name():
    """Model name for tests."""
    return os.environ.get("TEST_MODEL_NAME", "default")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_messages_create__handles_unicode_content(self, client, model_name):
        """Test handling of unicode characters."""
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "Repeat exactly: 你好世界 🌍 Ωmega café",
                }
            ],
        )

        assert response.id is not None
        assert len(response.content) > 0

    def test_messages_create__handles_emoji_content(self, client, model_name):
        """Test handling of emoji-heavy content."""
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "What emojis are these? 🎉🚀💻🔥✨🎨🌈",
                }
            ],
        )

        assert response.id is not None
        assert len(response.content) > 0

    def test_messages_create__handles_whitespace_only_content(self, client, model_name):
        """Test handling of whitespace-only content."""
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "   \n\t   ",
                }
            ],
        )

        assert response.id is not None

    def test_messages_create__truncates_when_max_tokens_1(self, client, model_name):
        """Test with max_tokens=1."""
        response = client.messages.create(
            model=model_name,
            max_tokens=1,
            messages=[{"role": "user", "content": "Say a very long sentence."}],
        )

        assert response.id is not None
        assert response.stop_reason in ["max_tokens", "end_turn"]

    def test_messages_create__handles_newlines_in_content(self, client, model_name):
        """Test handling of newlines in content."""
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "Line 1\nLine 2\n\nLine 4\n\n\nLine 7",
                }
            ],
        )

        assert response.id is not None
        assert len(response.content) > 0

    def test_messages_create__handles_special_characters(self, client, model_name):
        """Test handling of special characters."""
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "Handle these: <script>alert('xss')</script> & \"quotes\" 'apostrophe'",
                }
            ],
        )

        assert response.id is not None
        assert len(response.content) > 0

    def test_messages_create__handles_json_in_content(self, client, model_name):
        """Test handling of JSON-like content."""
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": 'Parse this JSON: {"name": "test", "values": [1, 2, 3]}',
                }
            ],
        )

        assert response.id is not None
        assert len(response.content) > 0

    def test_messages_create__handles_code_in_content(self, client, model_name):
        """Test handling of code in content."""
        response = client.messages.create(
            model=model_name,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": "What does this do?\n```python\ndef foo(x):\n    return x * 2\n```",
                }
            ],
        )

        assert response.id is not None
        assert len(response.content) > 0

    def test_messages_create__handles_multiple_text_blocks(self, client, model_name):
        """Test mixing text content blocks in single message."""
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First part."},
                        {"type": "text", "text": "Second part."},
                        {"type": "text", "text": "Third part."},
                    ],
                }
            ],
        )

        assert response.id is not None
        assert len(response.content) > 0

    def test_messages_create__handles_long_system_prompt(self, client, model_name):
        """Test with a long system prompt."""
        long_system = "You are a helpful assistant. " * 100
        response = client.messages.create(
            model=model_name,
            max_tokens=50,
            system=long_system,
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert response.id is not None
        assert len(response.content) > 0

    def test_messages_create__handles_many_turn_conversation(self, client, model_name):
        """Test conversation with many turns."""
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"Message {i * 2 + 1}"})
            messages.append({"role": "assistant", "content": f"Response {i * 2 + 2}"})
        messages.append({"role": "user", "content": "Final question"})

        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            messages=messages,
        )

        assert response.id is not None
        assert len(response.content) > 0

    def test_messages_create__handles_repeated_identical_messages(
        self, client, model_name
    ):
        """Test handling of repeated identical messages."""
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Hello"},
            ],
        )

        assert response.id is not None
        assert len(response.content) > 0

    def test_messages_create__handles_system_as_content_blocks(
        self, client, model_name
    ):
        """Test system prompt as content blocks."""
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            system=[
                {"type": "text", "text": "You are helpful."},
                {"type": "text", "text": "Be concise."},
            ],
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert response.id is not None
        assert len(response.content) > 0
