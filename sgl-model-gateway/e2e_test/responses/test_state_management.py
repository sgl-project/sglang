"""State management tests for Response API.

Tests both previous_response_id and conversation-based state management.
These tests work across local (gRPC) and cloud (OpenAI, xAI) backends.

Source: Migrated from e2e_response_api/features/test_state_management.py
"""

from __future__ import annotations

import logging

import openai
import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Cloud Backend Tests (OpenAI, xAI)
# =============================================================================


@pytest.mark.parametrize("setup_backend", ["openai", "xai"], indirect=True)
class TestStateManagementCloud:
    """State management tests against cloud APIs."""

    def test_basic_response_creation(self, setup_backend):
        """Test basic response creation without state."""
        _, model, client, gateway = setup_backend

        resp = client.responses.create(model=model, input="What is 2+2?")

        assert resp.id is not None
        assert resp.error is None
        assert resp.status == "completed"
        assert len(resp.output_text) > 0
        assert resp.usage is not None

    def test_streaming_response(self, setup_backend):
        """Test streaming response."""
        _, model, client, gateway = setup_backend

        resp = client.responses.create(
            model=model, input="Count to 5", stream=True, max_output_tokens=50
        )

        events = list(resp)
        created_events = [e for e in events if e.type == "response.created"]
        assert len(created_events) > 0

        assert any(
            e.type in ["response.completed", "response.in_progress"] for e in events
        )

    def test_previous_response_id_chaining(self, setup_backend):
        """Test chaining responses using previous_response_id."""
        _, model, client, gateway = setup_backend

        # First response
        resp1 = client.responses.create(
            model=model, input="My name is Alice and my friend is Bob. Remember it."
        )
        assert resp1.error is None
        assert resp1.status == "completed"

        # Second response referencing first
        resp2 = client.responses.create(
            model=model, input="What is my name", previous_response_id=resp1.id
        )
        assert resp2.error is None
        assert resp2.status == "completed"
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

    def test_conversation_with_multiple_turns(self, setup_backend):
        """Test state management using conversation ID."""
        _, model, client, gateway = setup_backend

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
        assert "5" in resp2.output_text or "five" in resp2.output_text.lower()

        # Third response in same conversation
        resp3 = client.responses.create(
            model=model,
            input="If I get 3 more, how many total?",
            conversation=conversation_id,
        )
        assert resp3.error is None
        assert resp3.status == "completed"
        assert "8" in resp3.output_text or "eight" in resp3.output_text.lower()

        items = client.conversations.items.list(conversation_id)
        assert items.data is not None
        assert len(items.data) >= 6  # 3 inputs + 3 outputs

    @pytest.mark.skip(reason="TODO: Add the invalid previous_response_id check")
    def test_previous_response_id_invalid(self, setup_backend):
        """Test using invalid previous_response_id."""
        _, model, client, gateway = setup_backend
        with pytest.raises(openai.BadRequestError):
            client.responses.create(
                model=model,
                input="Test",
                previous_response_id="resp_invalid123",
                max_output_tokens=50,
            )

    def test_mutually_exclusive_parameters(self, setup_backend):
        """Test that previous_response_id and conversation are mutually exclusive."""
        _, model, client, gateway = setup_backend

        conversation_id = "conv_123"
        resp1 = client.responses.create(model=model, input="Test")

        with pytest.raises(openai.BadRequestError):
            client.responses.create(
                model=model,
                input="This should fail",
                previous_response_id=resp1.id,
                conversation=conversation_id,
            )


# =============================================================================
# Local Backend Tests (gRPC with Qwen model)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.model("qwen-14b")
@pytest.mark.gateway(
    extra_args=["--tool-call-parser", "qwen", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestStateManagementLocal:
    """State management tests against local gRPC backend."""

    @pytest.mark.skip(reason="TODO: Add the invalid previous_response_id check")
    def test_previous_response_id_invalid(self, setup_backend):
        """Test using invalid previous_response_id."""
        _, model, client, gateway = setup_backend
        with pytest.raises(openai.BadRequestError):
            client.responses.create(
                model=model,
                input="Test",
                previous_response_id="resp_invalid123",
                max_output_tokens=50,
            )

    def test_basic_response_creation(self, setup_backend):
        """Test basic response creation without state."""
        _, model, client, gateway = setup_backend

        resp = client.responses.create(model=model, input="What is 2+2?")

        assert resp.id is not None
        assert resp.error is None
        assert resp.status == "completed"
        assert len(resp.output_text) > 0
        assert resp.usage is not None

    def test_streaming_response(self, setup_backend):
        """Test streaming response."""
        _, model, client, gateway = setup_backend

        resp = client.responses.create(
            model=model, input="Count to 5", stream=True, max_output_tokens=50
        )

        events = list(resp)
        created_events = [e for e in events if e.type == "response.created"]
        assert len(created_events) > 0

        assert any(
            e.type in ["response.completed", "response.in_progress"] for e in events
        )

    def test_previous_response_id_chaining(self, setup_backend):
        """Test chaining responses using previous_response_id."""
        _, model, client, gateway = setup_backend

        # First response
        resp1 = client.responses.create(
            model=model, input="My name is Alice and my friend is Bob. Remember it."
        )
        assert resp1.error is None
        assert resp1.status == "completed"

        # Second response referencing first
        resp2 = client.responses.create(
            model=model, input="What is my name", previous_response_id=resp1.id
        )
        assert resp2.error is None
        assert resp2.status == "completed"
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

    def test_mutually_exclusive_parameters(self, setup_backend):
        """Test that previous_response_id and conversation are mutually exclusive."""
        _, model, client, gateway = setup_backend

        conversation_id = "conv_123"
        resp1 = client.responses.create(model=model, input="Test")

        with pytest.raises(openai.BadRequestError):
            client.responses.create(
                model=model,
                input="This should fail",
                previous_response_id=resp1.id,
                conversation=conversation_id,
            )


# =============================================================================
# Local Backend Tests (gRPC with Harmony/Reasoning model)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.model("gpt-oss")
@pytest.mark.gateway(
    extra_args=["--reasoning-parser=gpt-oss", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestStateManagementHarmony:
    """State management tests against local gRPC backend with Harmony model."""

    @pytest.mark.skip(reason="TODO: Add the invalid previous_response_id check")
    def test_previous_response_id_invalid(self, setup_backend):
        """Test using invalid previous_response_id."""
        _, model, client, gateway = setup_backend
        with pytest.raises(openai.BadRequestError):
            client.responses.create(
                model=model,
                input="Test",
                previous_response_id="resp_invalid123",
                max_output_tokens=50,
            )

    def test_basic_response_creation(self, setup_backend):
        """Test basic response creation without state."""
        _, model, client, gateway = setup_backend

        resp = client.responses.create(model=model, input="What is 2+2?")

        assert resp.id is not None
        assert resp.error is None
        assert resp.status == "completed"
        assert len(resp.output_text) > 0
        assert resp.usage is not None

    def test_streaming_response(self, setup_backend):
        """Test streaming response."""
        _, model, client, gateway = setup_backend

        resp = client.responses.create(
            model=model, input="Count to 5", stream=True, max_output_tokens=50
        )

        events = list(resp)
        created_events = [e for e in events if e.type == "response.created"]
        assert len(created_events) > 0

        assert any(
            e.type in ["response.completed", "response.in_progress"] for e in events
        )

    def test_previous_response_id_chaining(self, setup_backend):
        """Test chaining responses using previous_response_id."""
        _, model, client, gateway = setup_backend

        # First response
        resp1 = client.responses.create(
            model=model, input="My name is Alice and my friend is Bob. Remember it."
        )
        assert resp1.error is None
        assert resp1.status == "completed"

        # Second response referencing first
        resp2 = client.responses.create(
            model=model, input="What is my name", previous_response_id=resp1.id
        )
        assert resp2.error is None
        assert resp2.status == "completed"
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

    def test_mutually_exclusive_parameters(self, setup_backend):
        """Test that previous_response_id and conversation are mutually exclusive."""
        _, model, client, gateway = setup_backend

        conversation_id = "conv_123"
        resp1 = client.responses.create(model=model, input="Test")

        with pytest.raises(openai.BadRequestError):
            client.responses.create(
                model=model,
                input="This should fail",
                previous_response_id=resp1.id,
                conversation=conversation_id,
            )
