"""
Base test class for Response API e2e tests.

This module provides base test classes that can be reused across different backends
(OpenAI, XAI, gRPC) with common test logic.
"""

import json
import sys
import time
import unittest
from pathlib import Path
from typing import Optional

import requests

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

    def make_request(
        self,
        endpoint: str,
        method: str = "POST",
        json_data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> requests.Response:
        """
        Make HTTP request to router.

        Args:
            endpoint: Endpoint path (e.g., "/v1/responses")
            method: HTTP method (GET, POST, DELETE)
            json_data: JSON body for POST requests
            params: Query parameters

        Returns:
            requests.Response object
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if method == "POST":
            resp = requests.post(url, json=json_data, headers=headers, params=params)
        elif method == "GET":
            resp = requests.get(url, headers=headers, params=params)
        elif method == "DELETE":
            resp = requests.delete(url, headers=headers, params=params)
        else:
            raise ValueError(f"Unsupported method: {method}")
        return resp

    def create_response(
        self,
        input_text: str,
        instructions: Optional[str] = None,
        stream: bool = False,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        previous_response_id: Optional[str] = None,
        conversation: Optional[str] = None,
        tools: Optional[list] = None,
        background: bool = False,
        **kwargs,
    ) -> requests.Response:
        """
        Create a response via POST /v1/responses.

        Args:
            input_text: User input
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
            requests.Response object
        """
        data = {
            "model": self.model,
            "input": input_text,
            "stream": stream,
            **kwargs,
        }

        if instructions:
            data["instructions"] = instructions

        if max_output_tokens is not None:
            data["max_output_tokens"] = max_output_tokens

        if temperature is not None:
            data["temperature"] = temperature

        if previous_response_id:
            data["previous_response_id"] = previous_response_id

        if conversation:
            data["conversation"] = conversation

        if tools:
            data["tools"] = tools

        if background:
            data["background"] = background

        if stream:
            # For streaming, we need to handle SSE
            return self._create_streaming_response(data)
        else:
            return self.make_request("/v1/responses", "POST", data)

    def _create_streaming_response(self, data: dict) -> requests.Response:
        """Handle streaming response creation."""
        url = f"{self.base_url}/v1/responses"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Return response object with stream=True
        return requests.post(url, json=data, headers=headers, stream=True)

    def get_response(self, response_id: str) -> requests.Response:
        """Get response by ID via GET /v1/responses/{response_id}."""
        return self.make_request(f"/v1/responses/{response_id}", "GET")

    def delete_response(self, response_id: str) -> requests.Response:
        """Delete response by ID via DELETE /v1/responses/{response_id}."""
        return self.make_request(f"/v1/responses/{response_id}", "DELETE")

    def cancel_response(self, response_id: str) -> requests.Response:
        """Cancel response by ID via POST /v1/responses/{response_id}/cancel."""
        return self.make_request(f"/v1/responses/{response_id}/cancel", "POST", {})

    def get_response_input_items(self, response_id: str) -> requests.Response:
        """Get response input items via GET /v1/responses/{response_id}/input_items."""
        return self.make_request(f"/v1/responses/{response_id}/input_items", "GET")

    def create_conversation(self, metadata: Optional[dict] = None) -> requests.Response:
        """Create conversation via POST /v1/conversations."""
        data = {}
        if metadata:
            data["metadata"] = metadata
        return self.make_request("/v1/conversations", "POST", data)

    def get_conversation(self, conversation_id: str) -> requests.Response:
        """Get conversation by ID via GET /v1/conversations/{conversation_id}."""
        return self.make_request(f"/v1/conversations/{conversation_id}", "GET")

    def update_conversation(
        self, conversation_id: str, metadata: dict
    ) -> requests.Response:
        """Update conversation via POST /v1/conversations/{conversation_id}."""
        return self.make_request(
            f"/v1/conversations/{conversation_id}", "POST", {"metadata": metadata}
        )

    def delete_conversation(self, conversation_id: str) -> requests.Response:
        """Delete conversation via DELETE /v1/conversations/{conversation_id}."""
        return self.make_request(f"/v1/conversations/{conversation_id}", "DELETE")

    def list_conversation_items(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        order: str = "asc",
    ) -> requests.Response:
        """List conversation items via GET /v1/conversations/{conversation_id}/items."""
        params = {"order": order}
        if limit:
            params["limit"] = limit
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        return self.make_request(
            f"/v1/conversations/{conversation_id}/items", "GET", params=params
        )

    def create_conversation_items(
        self, conversation_id: str, items: list
    ) -> requests.Response:
        """Create conversation items via POST /v1/conversations/{conversation_id}/items."""
        return self.make_request(
            f"/v1/conversations/{conversation_id}/items", "POST", {"items": items}
        )

    def get_conversation_item(
        self, conversation_id: str, item_id: str
    ) -> requests.Response:
        """Get conversation item via GET /v1/conversations/{conversation_id}/items/{item_id}."""
        return self.make_request(
            f"/v1/conversations/{conversation_id}/items/{item_id}", "GET"
        )

    def delete_conversation_item(
        self, conversation_id: str, item_id: str
    ) -> requests.Response:
        """Delete conversation item via DELETE /v1/conversations/{conversation_id}/items/{item_id}."""
        return self.make_request(
            f"/v1/conversations/{conversation_id}/items/{item_id}", "DELETE"
        )

    def parse_sse_events(self, response: requests.Response) -> list:
        """
        Parse Server-Sent Events from streaming response.

        Args:
            response: requests.Response with stream=True

        Returns:
            List of event dictionaries with 'event' and 'data' keys
        """
        events = []
        current_event = None

        for line in response.iter_lines():
            if not line:
                # Empty line signals end of event
                if current_event and current_event.get("data"):
                    events.append(current_event)
                current_event = None
                continue

            line = line.decode("utf-8")

            if line.startswith("event:"):
                current_event = {"event": line[6:].strip()}
            elif line.startswith("data:"):
                if current_event is None:
                    current_event = {}
                data_str = line[5:].strip()
                try:
                    current_event["data"] = json.loads(data_str)
                except json.JSONDecodeError:
                    current_event["data"] = data_str

        # Don't forget the last event if stream ends without empty line
        if current_event and current_event.get("data"):
            events.append(current_event)

        return events

    def wait_for_background_task(
        self, response_id: str, timeout: int = 30, poll_interval: float = 0.5
    ) -> dict:
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
            self.assertEqual(resp.status_code, 200)

            data = resp.json()
            status = data.get("status")

            if status == "completed":
                return data
            elif status == "failed":
                raise AssertionError(
                    f"Background task failed: {data.get('error', 'Unknown error')}"
                )
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
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertIn("id", data)
        self.assertIn("output", data)
        self.assertEqual(data["status"], "completed")
        self.assertIn("usage", data)

    def test_streaming_response(self):
        """Test streaming response."""
        resp = self.create_response("Count to 5", stream=True, max_output_tokens=50)
        self.assertEqual(resp.status_code, 200)

        events = self.parse_sse_events(resp)
        self.assertGreater(len(events), 0)

        # Check for response.created event
        created_events = [e for e in events if e.get("event") == "response.created"]
        self.assertGreater(len(created_events), 0)

        # Check for final completed event or in_progress events
        self.assertTrue(
            any(
                e.get("event") in ["response.completed", "response.in_progress"]
                for e in events
            )
        )


class ResponseCRUDBaseTest(ResponseAPIBaseTest):
    """Base class for Response API CRUD tests."""

    def test_create_and_get_response(self):
        """Test creating response and retrieving it."""
        # Create response
        create_resp = self.create_response("Hello, world!")
        self.assertEqual(create_resp.status_code, 200)

        create_data = create_resp.json()
        response_id = create_data["id"]

        # Get response
        get_resp = self.get_response(response_id)
        self.assertEqual(get_resp.status_code, 200)

        get_data = get_resp.json()
        self.assertEqual(get_data["id"], response_id)
        self.assertEqual(get_data["status"], "completed")

        input_resp = self.get_response_input_items(get_data["id"])
        self.assertEqual(input_resp.status_code, 200)
        input_data = input_resp.json()
        self.assertIn("data", input_data)
        self.assertGreater(len(input_data["data"]), 0)

    @unittest.skip("TODO: Add delete response feature")
    def test_delete_response(self):
        """Test deleting response."""
        # Create response
        create_resp = self.create_response("Test deletion", max_output_tokens=50)
        self.assertEqual(create_resp.status_code, 200)

        response_id = create_resp.json()["id"]

        # Delete response
        delete_resp = self.delete_response(response_id)
        self.assertEqual(delete_resp.status_code, 200)

        # Verify it's deleted (should return 404)
        get_resp = self.get_response(response_id)
        self.assertEqual(get_resp.status_code, 404)

    @unittest.skip("TODO: Add background response feature")
    def test_background_response(self):
        """Test background response execution."""
        # Create background response
        create_resp = self.create_response(
            "Write a short story", background=True, max_output_tokens=100
        )
        self.assertEqual(create_resp.status_code, 200)

        create_data = create_resp.json()
        response_id = create_data["id"]
        self.assertEqual(create_data["status"], "in_progress")

        # Wait for completion
        final_data = self.wait_for_background_task(response_id, timeout=60)
        self.assertEqual(final_data["status"], "completed")


class ConversationCRUDBaseTest(ResponseAPIBaseTest):
    """Base class for Conversation API CRUD tests."""

    def test_create_and_get_conversation(self):
        """Test creating and retrieving conversation."""
        # Create conversation
        create_resp = self.create_conversation(metadata={"user": "test_user"})
        self.assertEqual(create_resp.status_code, 200)

        create_data = create_resp.json()
        conversation_id = create_data["id"]
        self.assertEqual(create_data["metadata"]["user"], "test_user")

        # Get conversation
        get_resp = self.get_conversation(conversation_id)
        self.assertEqual(get_resp.status_code, 200)

        get_data = get_resp.json()
        self.assertEqual(get_data["id"], conversation_id)
        self.assertEqual(get_data["metadata"]["user"], "test_user")

    def test_update_conversation(self):
        """Test updating conversation metadata."""
        # Create conversation
        create_resp = self.create_conversation(metadata={"key1": "value1"})
        self.assertEqual(create_resp.status_code, 200)
        conversation_id = create_resp.json()["id"]

        # Update conversation
        update_resp = self.update_conversation(
            conversation_id, metadata={"key1": "value1", "key2": "value2"}
        )
        self.assertEqual(update_resp.status_code, 200)

        # Verify update
        get_resp = self.get_conversation(conversation_id)
        get_data = get_resp.json()
        self.assertEqual(get_data["metadata"]["key2"], "value2")

    def test_delete_conversation(self):
        """Test deleting conversation."""
        # Create conversation
        create_resp = self.create_conversation()
        self.assertEqual(create_resp.status_code, 200)
        conversation_id = create_resp.json()["id"]

        # Delete conversation
        delete_resp = self.delete_conversation(conversation_id)
        self.assertEqual(delete_resp.status_code, 200)

        # Verify deletion
        get_resp = self.get_conversation(conversation_id)
        self.assertEqual(get_resp.status_code, 404)

    def test_list_conversation_items(self):
        """Test listing conversation items."""
        # Create conversation
        conv_resp = self.create_conversation()
        conversation_id = conv_resp.json()["id"]

        # Create response with conversation
        self.create_response(
            "First message", conversation=conversation_id, max_output_tokens=50
        )
        self.create_response(
            "Second message", conversation=conversation_id, max_output_tokens=50
        )

        # List items
        list_resp = self.list_conversation_items(conversation_id)
        self.assertEqual(list_resp.status_code, 200)

        list_data = list_resp.json()
        self.assertIn("data", list_data)
        # Should have at least 4 items (2 inputs + 2 outputs)
        self.assertGreaterEqual(len(list_data["data"]), 4)


class FunctionCallingBaseTest(ResponseAPIBaseTest):
    """Base class for function calling tests."""

    def test_basic_function_call(self):
        """
        Test basic function calling workflow.

        This test follows the pattern from function_call_test.py:
        1. Define a function tool (get_horoscope)
        2. Send user message asking for horoscope
        3. Model should return function_call
        4. Execute function locally and provide output
        5. Model should generate final response using the function output
        """
        # 1. Define a list of callable tools for the model
        tools = [
            {
                "type": "function",
                "name": "get_horoscope",
                "description": "Get today's horoscope for an astrological sign.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sign": {
                            "type": "string",
                            "description": "An astrological sign like Taurus or Aquarius",
                        },
                    },
                    "required": ["sign"],
                },
            },
        ]

        # Create a running input list we will add to over time
        input_list = [
            {"role": "user", "content": "What is my horoscope? I am an Aquarius."}
        ]

        # 2. Prompt the model with tools defined
        resp = self.make_request(
            "/v1/responses",
            "POST",
            {
                "model": self.model,
                "tools": tools,
                "input": input_list,
            },
        )

        # Should successfully make the request
        self.assertEqual(resp.status_code, 200)

        data = resp.json()

        # Basic response structure
        self.assertIn("id", data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "completed")
        self.assertIn("output", data)

        # Verify output array is not empty
        output = data["output"]
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)

        # Check for function_call in output
        function_calls = [item for item in output if item.get("type") == "function_call"]
        self.assertGreater(
            len(function_calls), 0, "Response should contain at least one function_call"
        )

        # Verify function_call structure
        function_call = function_calls[0]
        self.assertIn("call_id", function_call)
        self.assertIn("name", function_call)
        self.assertEqual(function_call["name"], "get_horoscope")
        self.assertIn("arguments", function_call)

        # Parse arguments
        args = json.loads(function_call["arguments"])
        self.assertIn("sign", args)
        self.assertEqual(args["sign"].lower(), "aquarius")

        # 3. Save function call outputs for subsequent requests
        input_list += output

        # 4. Execute the function logic for get_horoscope
        horoscope = f"{args['sign']}: Next Tuesday you will befriend a baby otter."

        # 5. Provide function call results to the model
        input_list.append(
            {
                "type": "function_call_output",
                "call_id": function_call["call_id"],
                "output": json.dumps({"horoscope": horoscope}),
            }
        )

        # 6. Make second request with function output
        resp2 = self.make_request(
            "/v1/responses",
            "POST",
            {
                "model": self.model,
                "instructions": "Respond only with a horoscope generated by a tool.",
                "tools": tools,
                "input": input_list,
            },
        )

        self.assertEqual(resp2.status_code, 200)

        data2 = resp2.json()
        self.assertEqual(data2["status"], "completed")

        # The model should be able to give a response using the function output
        output2 = data2["output"]
        self.assertGreater(len(output2), 0)

        # Find message output
        messages = [item for item in output2 if item.get("type") == "message"]
        self.assertGreater(
            len(messages), 0, "Response should contain at least one message"
        )

        # Verify message contains the horoscope
        message = messages[0]
        self.assertIn("content", message)
        content_parts = message["content"]
        self.assertGreater(len(content_parts), 0)

        # Get text from content
        text_parts = [
            part.get("text", "")
            for part in content_parts
            if part.get("type") == "output_text"
        ]
        full_text = " ".join(text_parts).lower()

        # Should mention the horoscope or baby otter
        self.assertTrue(
            "baby otter" in full_text or "aquarius" in full_text,
            "Response should reference the horoscope content",
        )
