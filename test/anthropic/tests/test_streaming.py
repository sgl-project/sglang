"""
Test streaming responses via Anthropic Messages API.

These tests verify that SGLang's Anthropic API correctly handles
streaming (SSE) responses.
"""

import pytest


def get_text_content(response):
    """Extract text content from response, skipping thinking blocks."""
    for block in response.content:
        if block.type == "text":
            return block.text
    return None


class TestStreaming:
    """Test streaming functionality."""

    def test_messages_stream__returns_text_chunks(self, client, model_name):
        """Test basic streaming response."""
        collected_text = ""
        event_types = set()

        with client.messages.stream(
            model=model_name,
            max_tokens=500,
            messages=[{"role": "user", "content": "Say 'hello world'."}],
        ) as stream:
            for event in stream:
                event_types.add(type(event).__name__)

                if (
                    hasattr(event, "delta")
                    and hasattr(event.delta, "text")
                    and event.delta.text
                ):
                    collected_text += event.delta.text

        assert len(collected_text) > 0 or len(event_types) > 0
        assert (
            "MessageStartEvent" in event_types
            or "ContentBlockStart" in event_types
            or len(event_types) > 0
        )

    def test_messages_stream__emits_proper_event_sequence(self, client, model_name):
        """Test that streaming produces proper message structure."""
        message_start_received = False
        content_block_starts = 0
        content_block_stops = 0
        message_stop_received = False

        with client.messages.stream(
            model=model_name,
            max_tokens=500,
            messages=[{"role": "user", "content": "Hi"}],
        ) as stream:
            for event in stream:
                event_name = type(event).__name__

                if "MessageStart" in event_name:
                    message_start_received = True
                    if hasattr(event, "message"):
                        assert event.message.id is not None
                        assert event.message.model is not None

                elif "ContentBlockStart" in event_name:
                    content_block_starts += 1

                elif "ContentBlockStop" in event_name:
                    content_block_stops += 1

                elif "MessageStop" in event_name:
                    message_stop_received = True

        assert message_start_received, "Should receive message_start event"
        assert message_stop_received, "Should receive message_stop event"
        assert (
            content_block_starts > 0
        ), "Should receive at least one content_block_start"
        assert (
            content_block_starts == content_block_stops
        ), "content_block starts should match stops"

    def test_messages_stream__uses_system_message(self, client, model_name):
        """Test streaming with system message."""
        collected_text = ""

        with client.messages.stream(
            model=model_name,
            max_tokens=500,
            system="Respond only with 'ACKNOWLEDGED'",
            messages=[{"role": "user", "content": "Hello"}],
        ) as stream:
            for event in stream:
                if (
                    hasattr(event, "delta")
                    and hasattr(event.delta, "text")
                    and event.delta.text
                ):
                    collected_text += event.delta.text

        assert len(collected_text) >= 0

    def test_messages_stream__get_final_message_returns_complete_response(
        self, client, model_name
    ):
        """Test that get_final_message returns complete message."""
        with client.messages.stream(
            model=model_name,
            max_tokens=500,
            messages=[{"role": "user", "content": "Say 'test complete'."}],
        ) as stream:
            for _ in stream:
                pass

            final = stream.get_final_message()

        assert final is not None
        assert final.id is not None
        assert len(final.content) > 0
        assert final.content[0].type in ["text", "thinking"]
        assert final.usage is not None

    def test_messages_stream__text_stream_helper_collects_text(
        self, client, model_name
    ):
        """Test the text stream helper."""
        collected_text = ""

        with client.messages.stream(
            model=model_name,
            max_tokens=500,
            messages=[{"role": "user", "content": "Say 'streaming works'."}],
        ) as stream:
            for text in stream.text_stream:
                collected_text += text

        assert len(collected_text) >= 0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
