"""
Test basic chat completions via Anthropic Messages API.

These tests verify that SGLang's Anthropic API compatibility layer
correctly handles basic message requests without tools.
"""

import pytest


def get_text_content(response):
    """Extract text content from response, skipping thinking blocks."""
    for block in response.content:
        if block.type == "text":
            return block.text
    return None


class TestBasicChat:
    """Test basic chat functionality."""

    def test_messages_create__returns_valid_response_for_simple_message(
        self, client, model_name
    ):
        """Test a simple single-turn message."""
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        )

        assert response.id is not None
        assert response.id.startswith("msg_")
        assert response.model is not None
        assert response.stop_reason in ["end_turn", "stop_sequence", "max_tokens"]
        assert len(response.content) >= 1
        text = get_text_content(response)
        assert text is not None, "Response should contain a text block"
        assert len(text) > 0
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    def test_messages_create__uses_system_message_string(self, client, model_name):
        """Test with a system message as string."""
        response = client.messages.create(
            model=model_name,
            max_tokens=500,
            system="You are a helpful assistant that responds in exactly one word.",
            messages=[{"role": "user", "content": "What color is the sky?"}],
        )

        assert response.id is not None
        assert len(response.content) >= 1
        text = get_text_content(response)
        assert text is not None, "Response should contain a text block"

    def test_messages_create__uses_system_message_blocks(self, client, model_name):
        """Test with system message as array of blocks."""
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            system=[
                {"type": "text", "text": "You are a pirate."},
                {"type": "text", "text": "Always respond with 'Arrr!'"},
            ],
            messages=[{"role": "user", "content": "Hello!"}],
        )

        assert response.id is not None
        assert len(response.content) >= 1

    def test_messages_create__remembers_context_in_multi_turn(self, client, model_name):
        """Test multi-turn conversation."""
        response = client.messages.create(
            model=model_name,
            max_tokens=500,
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
        )

        assert response.id is not None
        assert len(response.content) >= 1
        text = get_text_content(response)
        assert text is not None, "Response should contain a text block"
        assert "alice" in text.lower()

    def test_messages_create__respects_max_tokens_limit(self, client, model_name):
        """Test that max_tokens is respected."""
        response = client.messages.create(
            model=model_name,
            max_tokens=5,
            messages=[
                {"role": "user", "content": "Write a very long essay about space."}
            ],
        )

        assert response.id is not None
        assert response.usage.output_tokens <= 10

    def test_messages_create__produces_deterministic_output_when_temperature_zero(
        self, client, model_name
    ):
        """Test with temperature=0 for deterministic output."""
        messages = [
            {"role": "user", "content": "What is 2+2? Reply with just the number."}
        ]

        response1 = client.messages.create(
            model=model_name, max_tokens=10, temperature=0, messages=messages
        )

        response2 = client.messages.create(
            model=model_name, max_tokens=10, temperature=0, messages=messages
        )

        text1 = get_text_content(response1)
        text2 = get_text_content(response2)
        assert text1 == text2

    def test_messages_create__stops_at_custom_stop_sequence(self, client, model_name):
        """Test custom stop sequences."""
        response = client.messages.create(
            model=model_name,
            max_tokens=100,
            stop_sequences=["STOP"],
            messages=[
                {
                    "role": "user",
                    "content": "Count from 1 to 10, then say STOP, then continue.",
                }
            ],
        )

        assert response.id is not None
        text = get_text_content(response)
        assert (
            text is None or "11" not in text or response.stop_reason == "stop_sequence"
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
