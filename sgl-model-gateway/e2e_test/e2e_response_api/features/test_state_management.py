"""
State management tests for Response API.

Tests both previous_response_id and conversation-based state management.
These tests should work across all backends (OpenAI, XAI, gRPC).
"""

import openai
import pytest


@pytest.mark.parametrize(
    "setup_backend", ["openai", "xai", "grpc", "grpc_harmony"], indirect=True
)
class TestStateManagement:
    """Tests for state management using previous_response_id and conversation."""

    def test_basic_response_creation(self, setup_backend):
        """Test basic response creation without state."""
        _, model, client = setup_backend

        resp = client.responses.create(model=model, input="What is 2+2?")

        assert resp.id is not None
        assert resp.error is None
        assert resp.status == "completed"
        assert len(resp.output_text) > 0
        assert resp.usage is not None

    def test_streaming_response(self, setup_backend):
        """Test streaming response."""
        _, model, client = setup_backend

        resp = client.responses.create(
            model=model, input="Count to 5", stream=True, max_output_tokens=50
        )

        # Check for response.created event
        events = [event for event in resp]
        created_events = [event for event in events if event.type == "response.created"]
        assert len(created_events) > 0

        # Check for final completed event or in_progress events
        assert any(
            event.type in ["response.completed", "response.in_progress"]
            for event in events
        )

    def test_previous_response_id_chaining(self, setup_backend):
        """Test chaining responses using previous_response_id."""
        _, model, client = setup_backend
        # First response
        resp1 = client.responses.create(
            model=model, input="My name is Alice and my friend is Bob. Remember it."
        )
        assert resp1.error is None
        assert resp1.status == "completed"
        response1_id = resp1.id

        # Second response referencing first
        resp2 = client.responses.create(
            model=model, input="What is my name", previous_response_id=response1_id
        )
        assert resp2.error is None
        assert resp2.status == "completed"

        # The model should remember the name from previous response
        assert "Alice" in resp2.output_text

        # Third response referencing second
        resp3 = client.responses.create(
            model=model,
            input="What is my friend name?",
            previous_response_id=resp2.id,
        )
        assert resp3.error is None
        assert resp3.status == "completed"
        assert "Bob" in resp3.output_text

    @pytest.mark.skip(reason="TODO: Add the invalid previous_response_id check")
    def test_previous_response_id_invalid(self, setup_backend):
        """Test using invalid previous_response_id."""
        _, model, client = setup_backend
        with pytest.raises(openai.BadRequestError):
            client.responses.create(
                model=model,
                input="Test",
                previous_response_id="resp_invalid123",
                max_output_tokens=50,
            )

    def test_conversation_with_multiple_turns(self, setup_backend):
        """Test state management using conversation ID."""
        backend, model, client = setup_backend

        if backend in ["grpc", "grpc_harmony"]:
            pytest.skip("TODO: 501 Not Implemented")

        # Create conversation
        conv_resp = client.conversations.create(metadata={"topic": "math"})
        assert conv_resp.id is not None
        assert conv_resp.created_at is not None

        conversation_id = conv_resp.id

        # First response in conversation
        resp1 = client.responses.create(
            model=model, input="I have 5 apples.", conversation=conversation_id
        )
        assert resp1.error is None
        assert resp1.status == "completed"

        # Second response in same conversation
        resp2 = client.responses.create(
            model=model,
            input="How many apples do I have?",
            conversation=conversation_id,
        )
        assert resp2.error is None
        assert resp2.status == "completed"
        output_text = resp2.output_text

        # Should remember "5 apples"
        assert "5" in output_text or "five" in output_text.lower()

        # Third response in same conversation
        resp3 = client.responses.create(
            model=model,
            input="If I get 3 more, how many total?",
            conversation=conversation_id,
        )
        assert resp3.error is None
        assert resp3.status == "completed"
        output_text = resp3.output_text

        # Should calculate 5 + 3 = 8
        assert "8" in output_text or "eight" in output_text.lower()
        list_resp = client.conversations.items.list(conversation_id)
        assert list_resp.data is not None
        items = list_resp.data
        # Should have at least 6 items (3 inputs + 3 outputs)
        assert len(items) >= 6

    def test_mutually_exclusive_parameters(self, setup_backend):
        """Test that previous_response_id and conversation are mutually exclusive."""
        _, model, client = setup_backend

        # TODO: Remove this once the conversation API is implemented for GRPC backend
        conversation_id = "conv_123"

        resp1 = client.responses.create(model=model, input="Test")
        response1_id = resp1.id

        # Try to use both parameters
        with pytest.raises(openai.BadRequestError):
            client.responses.create(
                model=model,
                input="This should fail",
                previous_response_id=response1_id,
                conversation=conversation_id,
            )
