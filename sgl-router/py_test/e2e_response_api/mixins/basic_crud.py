"""
Base test class for Response API e2e tests.

This module provides base test classes that can be reused across different backends
(OpenAI, XAI, gRPC) with common test logic.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from typing import Optional, Union

import openai
from openai.types import conversations, responses

# Add current directory for local imports
_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR))

from util import CustomTestCase


class ResponseAPIBaseTest(CustomTestCase):
    """Base class for Response API tests with common utilities."""

    # To be set by subclasses
    base_url: str = None
    api_key: str = None
    model: str = None
    client: openai.OpenAI = None

    def create_response(
        self,
        input: Union[str, responses.ResponseInputParam],
        instructions: Optional[str] = None,
        stream: bool = False,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        previous_response_id: Optional[str] = None,
        conversation: Optional[str] = None,
        tools: Optional[list] = None,
        background: bool = False,
        **kwargs,
    ) -> responses.Response | openai.Stream[responses.ResponseStreamEvent]:
        """
        Create a response via POST /v1/responses.

        Args:
            input: User input
            instructions: Optional system instructions
            stream: Whether to stream response
            max_output_tokens: Optional max tokens to generate
            temperature: Sampling temperature
            previous_response_id: Optional previous response ID for state management
            conversation: Optional conversation ID for state management
            tools: Optional list of MCP tools
            background: Whether to run in background mode
            **kwargs: Additional request parameters

        Returns:
            Response object for non-stream request
            ResponseStreamEvent for stream request
        """
        params = {
            "model": self.model,
            "input": input,
            "stream": stream,
            **kwargs,
        }

        if instructions:
            params["instructions"] = instructions

        if max_output_tokens is not None:
            params["max_output_tokens"] = max_output_tokens

        if temperature is not None:
            params["temperature"] = temperature

        if previous_response_id:
            params["previous_response_id"] = previous_response_id

        if conversation:
            params["conversation"] = conversation

        if tools:
            params["tools"] = tools

        if background:
            params["background"] = background

        return self.client.responses.create(**params)

    def get_response(
        self, response_id: str
    ) -> responses.Response | openai.Stream[responses.ResponseStreamEvent]:
        """Get response by ID via GET /v1/responses/{response_id}."""
        return self.client.responses.retrieve(response_id=response_id)

    def delete_response(self, response_id: str) -> None:
        """Delete response by ID via DELETE /v1/responses/{response_id}."""
        return self.client.responses.delete(response_id=response_id)

    def cancel_response(self, response_id: str) -> responses.Response:
        """Cancel response by ID via POST /v1/responses/{response_id}/cancel."""
        return self.client.responses.cancel(response_id=response_id)

    def get_response_input_items(
        self, response_id: str
    ) -> openai.pagination.SyncCursorPage[responses.ResponseItem]:
        """Get response input items via GET /v1/responses/{response_id}/input_items."""
        return self.client.responses.input_items.list(response_id=response_id)

    def create_conversation(
        self, metadata: Optional[dict] = None
    ) -> conversations.Conversation:
        """Create conversation via POST /v1/conversations."""
        params = {}
        if metadata:
            params["metadata"] = metadata
        return self.client.conversations.create(**params)

    def get_conversation(self, conversation_id: str) -> conversations.Conversation:
        """Get conversation by ID via GET /v1/conversations/{conversation_id}."""
        return self.client.conversations.retrieve(conversation_id=conversation_id)

    def update_conversation(
        self, conversation_id: str, metadata: dict
    ) -> conversations.Conversation:
        """Update conversation via POST /v1/conversations/{conversation_id}."""
        return self.client.conversations.update(
            conversation_id=conversation_id, metadata=metadata
        )

    def delete_conversation(
        self, conversation_id: str
    ) -> conversations.ConversationDeletedResource:
        """Delete conversation via DELETE /v1/conversations/{conversation_id}."""
        return self.client.conversations.delete(conversation_id=conversation_id)

    def list_conversation_items(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        after: Optional[str] = None,
        order: str = "asc",
    ) -> openai.pagination.SyncConversationCursorPage[conversations.ConversationItem]:
        """List conversation items via GET /v1/conversations/{conversation_id}/items."""
        params = {"conversation_id": conversation_id, "order": order}
        if limit:
            params["limit"] = limit
        if after:
            params["after"] = after
        return self.client.conversations.items.list(**params)

    def create_conversation_items(
        self, conversation_id: str, items: list
    ) -> conversations.ConversationItemList:
        """Create conversation items via POST /v1/conversations/{conversation_id}/items."""
        return self.client.conversations.items.create(
            conversation_id=conversation_id, items=items
        )

    def get_conversation_item(
        self, conversation_id: str, item_id: str
    ) -> conversations.ConversationItem:
        """Get conversation item via GET /v1/conversations/{conversation_id}/items/{item_id}."""
        return self.client.conversations.items.retrieve(
            conversation_id=conversation_id, item_id=item_id
        )

    def delete_conversation_item(
        self, conversation_id: str, item_id: str
    ) -> conversations.Conversation:
        """Delete conversation item via DELETE /v1/conversations/{conversation_id}/items/{item_id}."""
        return self.client.conversations.items.delete(
            conversation_id=conversation_id, item_id=item_id
        )

    def wait_for_background_task(
        self, response_id: str, timeout: int = 30, poll_interval: float = 0.5
    ) -> responses.Response:
        """
        Wait for background task to complete.

        Args:
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
            resp = self.get_response(response_id)
            self.assertIsNone(resp.error)
            self.assertEqual(resp.id, response_id)

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


class StateManagementBaseTest(ResponseAPIBaseTest):
    """Base class for state management tests (previous_response_id and conversation)."""

    def test_basic_response_creation(self):
        """Test basic response creation without state."""
        resp = self.create_response("What is 2+2?", max_output_tokens=50)

        self.assertIsNotNone(resp.id)
        self.assertIsNone(resp.error)
        self.assertEqual(resp.status, "completed")
        self.assertGreater(len(resp.output_text), 0)
        self.assertGreater(resp.usage.input_tokens, 0)
        self.assertGreater(resp.usage.output_tokens, 0)
        self.assertGreater(resp.usage.total_tokens, 0)

    def test_streaming_response(self):
        """Test streaming response."""
        resp = self.create_response("Count to 5", stream=True, max_output_tokens=50)

        # Check for response.created event
        events = [event for event in resp]
        created_events = [event for event in events if event.type == "response.created"]
        self.assertGreater(len(created_events), 0)

        # Check for final completed event or in_progress events
        self.assertTrue(
            any(
                event.type in ["response.completed", "response.in_progress"]
                for event in events
            )
        )


class ResponseCRUDBaseTest(ResponseAPIBaseTest):
    """Base class for Response API CRUD tests."""

    def test_create_and_get_response(self):
        """Test creating response and retrieving it."""
        # Create response
        create_resp = self.create_response("Hello, world!")
        self.assertIsNotNone(create_resp.id)
        self.assertIsNone(create_resp.error)
        self.assertEqual(create_resp.status, "completed")
        self.assertGreater(len(create_resp.output_text), 0)
        response_id = create_resp.id

        # Get response
        get_resp = self.get_response(response_id)
        self.assertIsNone(get_resp.error)
        self.assertEqual(get_resp.id, response_id)
        self.assertEqual(get_resp.status, "completed")

        input_resp = self.get_response_input_items(get_resp.id)
        self.assertIsNotNone(input_resp.data)
        self.assertGreater(len(input_resp.data), 0)

    @unittest.skip("TODO: Add delete response feature")
    def test_delete_response(self):
        """Test deleting response."""
        # Create response
        create_resp = self.create_response("Test deletion")
        self.assertIsNotNone(create_resp.id)
        self.assertIsNone(create_resp.error)
        self.assertEqual(create_resp.status, "completed")
        self.assertGreater(len(create_resp.output_text), 0)

        response_id = create_resp.id

        # Delete response
        self.delete_response(response_id)

        # Verify it's deleted (should return 404)
        with self.assertRaises(openai.NotFoundError):
            self.get_response(response_id)

    @unittest.skip("TODO: Add background response feature")
    def test_background_response(self):
        """Test background response execution."""
        # Create background response
        create_resp = self.create_response(
            "Write a short story", background=True, max_output_tokens=100
        )
        self.assertIsNotNone(create_resp.id)
        self.assertIsNone(create_resp.error)
        self.assertIn(create_resp.status, ["in_progress", "queued"])

        response_id = create_resp.id

        # Wait for completion
        final_data = self.wait_for_background_task(response_id, timeout=60)
        self.assertEqual(final_data.status, "completed")


class ConversationCRUDBaseTest(ResponseAPIBaseTest):
    """Base class for Conversation API CRUD tests."""

    def test_create_and_get_conversation(self):
        """Test creating and retrieving conversation."""
        # Create conversation
        create_resp = self.create_conversation(metadata={"user": "test_user"})
        self.assertIsNotNone(create_resp.id)
        self.assertIsNotNone(create_resp.created_at)

        create_data = create_resp.metadata
        self.assertEqual(create_data["user"], "test_user")
        conversation_id = create_resp.id

        # Get conversation
        get_resp = self.get_conversation(conversation_id)
        self.assertIsNotNone(get_resp.id)
        self.assertIsNotNone(get_resp.created_at)

        get_data = get_resp.metadata
        self.assertEqual(get_resp.id, conversation_id)
        self.assertEqual(get_data["user"], "test_user")

    def test_update_conversation(self):
        """Test updating conversation metadata."""
        # Create conversation
        create_resp = self.create_conversation(metadata={"key1": "value1"})
        self.assertIsNotNone(create_resp.id)
        self.assertIsNotNone(create_resp.created_at)

        create_data = create_resp.metadata
        self.assertEqual(create_data["key1"], "value1")
        self.assertNotIn("key2", create_data)
        conversation_id = create_resp.id

        # Update conversation
        update_resp = self.update_conversation(
            conversation_id, metadata={"key1": "value1", "key2": "value2"}
        )
        self.assertEqual(update_resp.id, conversation_id)
        update_data = update_resp.metadata
        self.assertEqual(update_data["key1"], "value1")
        self.assertEqual(update_data["key2"], "value2")

        # Verify update
        get_resp = self.get_conversation(conversation_id)
        get_data = get_resp.metadata
        self.assertEqual(get_data["key1"], "value1")
        self.assertEqual(get_data["key2"], "value2")

    def test_delete_conversation(self):
        """Test deleting conversation."""
        # Create conversation
        create_resp = self.create_conversation()
        self.assertIsNotNone(create_resp.id)
        self.assertIsNotNone(create_resp.created_at)
        conversation_id = create_resp.id

        # Delete conversation
        delete_resp = self.delete_conversation(conversation_id)
        self.assertIsNotNone(delete_resp.id)
        self.assertTrue(delete_resp.deleted)

        # Verify deletion
        with self.assertRaises(openai.NotFoundError):
            self.get_conversation(conversation_id)

    def test_list_conversation_items(self):
        """Test listing conversation items."""
        # Create conversation
        conv_resp = self.create_conversation()
        self.assertIsNotNone(conv_resp.id)
        conversation_id = conv_resp.id

        # Create response with conversation
        resp1 = self.create_response(
            "First message", conversation=conversation_id, max_output_tokens=50
        )
        self.assertIsNone(resp1.error)
        resp2 = self.create_response(
            "Second message", conversation=conversation_id, max_output_tokens=50
        )
        self.assertIsNone(resp2.error)

        # List items
        list_resp = self.list_conversation_items(conversation_id)
        self.assertIsNotNone(list_resp)
        self.assertIsNotNone(list_resp.data)

        list_data = list_resp.data
        # Should have at least 4 items (2 inputs + 2 outputs)
        self.assertGreaterEqual(len(list_data), 4)
