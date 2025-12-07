"""
Base test class for Response API e2e tests.

This module provides base test classes that can be reused across different backends
(OpenAI, XAI, gRPC) with common test logic.
"""

import sys
import time
from pathlib import Path

import openai
import pytest
from openai import OpenAI
from openai.types import responses

# Add current directory for local imports
_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR))


@pytest.mark.parametrize("setup_backend", ["openai", "oracle_store"], indirect=True)
class TestResponseCRUD:
    """Base class for Response API CRUD tests."""

    def test_create_and_get_response(self, setup_backend):
        """Test creating response and retrieving it."""
        _, model, client = setup_backend

        # Create response
        create_resp = client.responses.create(model=model, input="Hello, world!")
        assert create_resp.id is not None
        assert create_resp.error is None
        assert create_resp.status == "completed"
        assert len(create_resp.output_text) > 0
        response_id = create_resp.id

        # Get response
        get_resp = client.responses.retrieve(response_id=response_id)
        assert get_resp.error is None
        assert get_resp.id == response_id
        assert get_resp.status == "completed"

        input_resp = client.responses.input_items.list(response_id=get_resp.id)
        assert input_resp.data is not None
        assert len(input_resp.data) > 0

    @pytest.mark.skip(reason="TODO: Add delete response feature")
    def test_delete_response(self, setup_backend):
        """Test deleting response."""
        _, model, client = setup_backend

        # Create response
        create_resp = client.responses.create(model=model, input="Test deletion")
        assert create_resp.id is not None
        assert create_resp.error is None
        assert create_resp.status == "completed"
        assert len(create_resp.output_text) > 0

        response_id = create_resp.id

        # Delete response
        client.responses.delete(response_id=response_id)

        # Verify it's deleted (should return 404)
        with pytest.raises(openai.NotFoundError):
            client.responses.retrieve(response_id=response_id)

    @pytest.mark.skip(reason="TODO: Add background response feature")
    def test_background_response(self, setup_backend):
        """Test background response execution."""
        _, model, client = setup_backend

        # Create background response
        create_resp = client.responses.create(
            model=model,
            input="Write a short story",
            background=True,
            max_output_tokens=100,
        )
        assert create_resp.id is not None
        assert create_resp.error is None
        assert create_resp.status in ["in_progress", "queued"]

        response_id = create_resp.id

        # Wait for completion
        final_data = wait_for_background_task(client, response_id, timeout=60)
        assert final_data.status == "completed"


@pytest.mark.parametrize("setup_backend", ["openai", "oracle_store"], indirect=True)
class TestConversationCRUD:
    """Base class for Conversation API CRUD tests."""

    def test_create_and_get_conversation(self, setup_backend):
        """Test creating and retrieving conversation."""
        _, model, client = setup_backend

        # Create conversation
        create_resp = client.conversations.create(metadata={"user": "test_user"})
        assert create_resp.id is not None
        assert create_resp.created_at is not None

        create_data = create_resp.metadata
        assert create_data["user"] == "test_user"
        conversation_id = create_resp.id

        # Get conversation
        get_resp = client.conversations.retrieve(conversation_id=conversation_id)
        assert get_resp.id is not None
        assert get_resp.created_at is not None

        get_data = get_resp.metadata
        assert get_resp.id == conversation_id
        assert get_data["user"] == "test_user"

    def test_update_conversation(self, setup_backend):
        """Test updating conversation metadata."""
        _, model, client = setup_backend

        # Create conversation
        create_resp = client.conversations.create(metadata={"key1": "value1"})
        assert create_resp.id is not None
        assert create_resp.created_at is not None

        create_data = create_resp.metadata
        assert create_data["key1"] == "value1"
        assert "key2" not in create_data
        conversation_id = create_resp.id

        # Update conversation
        update_resp = client.conversations.update(
            conversation_id=conversation_id,
            metadata={"key1": "value1", "key2": "value2"},
        )
        assert update_resp.id == conversation_id
        update_data = update_resp.metadata
        assert update_data["key1"] == "value1"
        assert update_data["key2"] == "value2"

        # Verify update
        get_resp = client.conversations.retrieve(conversation_id=conversation_id)
        get_data = get_resp.metadata
        assert update_data["key1"] == "value1"
        assert update_data["key2"] == "value2"

    def test_delete_conversation(self, setup_backend):
        """Test deleting conversation."""
        _, model, client = setup_backend

        # Create conversation
        create_resp = client.conversations.create()
        assert create_resp.id is not None
        assert create_resp.created_at is not None
        conversation_id = create_resp.id

        # Delete conversation
        delete_resp = client.conversations.delete(conversation_id=conversation_id)
        assert delete_resp.id is not None
        assert delete_resp.deleted

        # Verify deletion
        with pytest.raises(openai.NotFoundError):
            client.conversations.retrieve(conversation_id=conversation_id)

    def test_list_conversation_items(self, setup_backend):
        """Test listing conversation items."""
        _, model, client = setup_backend

        # Create conversation
        conv_resp = client.conversations.create()
        assert conv_resp.id is not None
        conversation_id = conv_resp.id

        # Create response with conversation
        resp1 = client.responses.create(
            model=model,
            input="First message",
            conversation=conversation_id,
            max_output_tokens=50,
        )
        assert resp1.error is None
        resp2 = client.responses.create(
            model=model,
            input="Second message",
            conversation=conversation_id,
            max_output_tokens=50,
        )
        assert resp2.error is None

        # List items
        list_resp = client.conversations.items.list(conversation_id=conversation_id)
        assert list_resp is not None
        assert list_resp.data is not None

        list_data = list_resp.data
        # Should have at least 4 items (2 inputs + 2 outputs)
        assert len(list_data) >= 4


def wait_for_background_task(
    client: OpenAI, response_id: str, timeout: int = 30, poll_interval: float = 0.5
) -> responses.Response:
    """
    Wait for background task to complete.

    Args:
        client: openai client
        response_id: Response ID to poll
        timeout: Max seconds to wait
        poll_interval: Seconds between polls

    Returns:
        Final response data

    Raises:
        TimeoutError: If task doesn't complete in time
        AssertionError: If task fails
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        resp = client.responses.retrieve(response_id=response_id)
        assert resp.error is None
        assert resp.id == response_id

        status = resp.status

        if status == "completed":
            return resp
        elif status == "failed":
            raise AssertionError(f"Background task failed: {resp.error}")
        elif status == "cancelled":
            raise AssertionError("Background task was cancelled")

        time.sleep(poll_interval)

    raise TimeoutError(
        f"Background task {response_id} did not complete within {timeout}s"
    )
