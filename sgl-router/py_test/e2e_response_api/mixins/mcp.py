"""
MCP (Model Context Protocol) tests for Response API.

Tests MCP tool calling in both streaming and non-streaming modes.
These tests should work across all backends that support MCP (OpenAI, XAI).
"""

import json

from basic_crud import ResponseAPIBaseTest


class MCPTests(ResponseAPIBaseTest):
    """Tests for MCP tool calling in both streaming and non-streaming modes."""

    # Class attribute to control validation strictness
    # Subclasses can override this to enable strict validation
    mcp_validation_mode = "relaxed"

    # Shared constants for MCP tests
    BRAVE_MCP_TOOL = {
        "type": "mcp",
        "server_label": "brave",
        "server_description": "A Tool to do web search",
        "server_url": "http://localhost:8001/sse",
        "require_approval": "never",
    }

    MCP_TEST_PROMPT = (
        "show me some news about sglang router, use the tool to just search "
        "one result and return one sentence response"
    )

    GET_WEATHER_FUNCTION = {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    }

    def test_mcp_basic_tool_call(self):
        """Test basic MCP tool call (non-streaming).

        Validation strictness is controlled by the class attribute `mcp_validation_mode`.
        Set to "strict" in subclasses for additional HTTP-specific validation.
        """
        resp = self.create_response(
            self.MCP_TEST_PROMPT,
            tools=[self.BRAVE_MCP_TOOL],
            stream=False,
            reasoning={"effort": "low"},
        )

        # Should successfully make the request
        self.assertIsNone(resp.error)

        # Basic response structure
        self.assertIsNotNone(resp.id)
        self.assertEqual(resp.status, "completed")
        self.assertIsNotNone(resp.model)
        self.assertIsNotNone(resp.output)

        # Verify output array is not empty
        self.assertGreater(len(resp.output_text), 0)

        # Check for MCP-specific output types
        output_types = [item.type for item in resp.output]

        # Should have mcp_list_tools - tools are listed before calling
        self.assertIn(
            "mcp_list_tools", output_types, "Response should contain mcp_list_tools"
        )

        # Should have at least one mcp_call
        mcp_calls = [item for item in resp.output if item.type == "mcp_call"]
        self.assertGreater(
            len(mcp_calls), 0, "Response should contain at least one mcp_call"
        )

        # Verify mcp_call structure
        for mcp_call in mcp_calls:
            self.assertIsNotNone(mcp_call.id)
            self.assertEqual(mcp_call.status, "completed")
            self.assertEqual(mcp_call.server_label, "brave")
            self.assertIsNotNone(mcp_call.name)
            self.assertIsNotNone(mcp_call.arguments)
            self.assertIsNotNone(mcp_call.output)

        # Strict mode: additional validation for HTTP backends
        if self.mcp_validation_mode == "strict":
            # Should have final message output
            messages = [item for item in resp.output if item.type == "message"]
            self.assertGreater(
                len(messages), 0, "Response should contain at least one message"
            )
            # Verify message structure
            for msg in messages:
                self.assertIsNotNone(msg.content)
                self.assertIsInstance(msg.content, list)

                # Check content has text
                for content_item in msg.content:
                    if content_item.type == "output_text":
                        self.assertIsNotNone(content_item.text)
                        self.assertIsInstance(content_item.text, str)
                        self.assertGreater(len(content_item.text), 0)

    def test_mcp_basic_tool_call_streaming(self):
        """Test basic MCP tool call (streaming).

        Validation strictness is controlled by the class attribute `mcp_validation_mode`.
        Set to "strict" in subclasses for additional HTTP-specific validation.
        """
        resp = self.create_response(
            self.MCP_TEST_PROMPT,
            tools=[self.BRAVE_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )

        # Should successfully make the request
        events = [event for event in resp]
        self.assertGreater(len(events), 0)

        event_types = [event.type for event in events]
        # Check for lifecycle events
        self.assertIn(
            "response.created", event_types, "Should have response.created event"
        )
        self.assertIn(
            "response.completed", event_types, "Should have response.completed event"
        )

        # Check for MCP list tools events
        self.assertIn(
            "response.output_item.added",
            event_types,
            "Should have output_item.added events",
        )
        self.assertIn(
            "response.mcp_list_tools.in_progress",
            event_types,
            "Should have mcp_list_tools.in_progress event",
        )
        self.assertIn(
            "response.mcp_list_tools.completed",
            event_types,
            "Should have mcp_list_tools.completed event",
        )

        # Check for MCP call events
        self.assertIn(
            "response.mcp_call.in_progress",
            event_types,
            "Should have mcp_call.in_progress event",
        )
        self.assertIn(
            "response.mcp_call_arguments.delta",
            event_types,
            "Should have mcp_call_arguments.delta event",
        )
        self.assertIn(
            "response.mcp_call_arguments.done",
            event_types,
            "Should have mcp_call_arguments.done event",
        )
        self.assertIn(
            "response.mcp_call.completed",
            event_types,
            "Should have mcp_call.completed event",
        )

        # Verify final completed event has full response
        completed_events = [e for e in events if e.type == "response.completed"]
        self.assertEqual(len(completed_events), 1)

        final_response = completed_events[0].response
        self.assertIsNotNone(final_response.id)
        self.assertEqual(final_response.status, "completed")
        self.assertIsNotNone(final_response.output)

        # Verify final output contains expected items
        final_output = final_response.output
        final_output_types = [item.type for item in final_output]

        self.assertIn("mcp_list_tools", final_output_types)
        self.assertIn("mcp_call", final_output_types)

        # Verify mcp_call items in final output
        mcp_calls = [item for item in final_output if item.type == "mcp_call"]
        self.assertGreater(len(mcp_calls), 0)

        for mcp_call in mcp_calls:
            self.assertEqual(mcp_call.status, "completed")
            self.assertEqual(mcp_call.server_label, "brave")
            self.assertIsNotNone(mcp_call.name)
            self.assertIsNotNone(mcp_call.arguments)
            self.assertIsNotNone(mcp_call.output)

        # Strict mode: additional validation for HTTP backends
        if self.mcp_validation_mode == "strict":
            # Check for text output events
            self.assertIn(
                "response.content_part.added",
                event_types,
                "Should have content_part.added event",
            )
            self.assertIn(
                "response.output_text.delta",
                event_types,
                "Should have output_text.delta events",
            )
            self.assertIn(
                "response.output_text.done",
                event_types,
                "Should have output_text.done event",
            )
            self.assertIn(
                "response.content_part.done",
                event_types,
                "Should have content_part.done event",
            )

            self.assertIn("message", final_output_types)

            # Verify text deltas combine to final message
            text_deltas = [
                e.delta for e in events if e.type == "response.output_text.delta"
            ]
            self.assertGreater(len(text_deltas), 0, "Should have text deltas")

            # Get final text from output_text.done event
            text_done_events = [
                e for e in events if e.type == "response.output_text.done"
            ]
            self.assertGreater(len(text_done_events), 0)

            final_text = text_done_events[0].text
            self.assertGreater(len(final_text), 0, "Final text should not be empty")

    def test_mixed_mcp_and_function_tools(self):
        """Test mixed MCP and function tools (non-streaming)."""
        resp = self.create_response(
            "What is the weather in seattle now?",
            tools=[self.BRAVE_MCP_TOOL, self.GET_WEATHER_FUNCTION],
            stream=False,
            tool_choice="auto",
        )

        # Should successfully make the request
        self.assertIsNone(resp.error)

        # Basic response structure
        self.assertIsNotNone(resp.id)
        self.assertIsNotNone(resp.status)
        self.assertIsNotNone(resp.output)

        # Verify output array is not empty
        output = resp.output
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)

        # Check for function_call (not mcp_call for get_weather)
        function_calls = [item for item in output if item.type == "function_call"]
        self.assertGreater(
            len(function_calls), 0, "Response should contain at least one function_call"
        )

        # Verify function_call structure for get_weather
        weather_call = function_calls[0]
        self.assertEqual(weather_call.name, "get_weather")
        self.assertIsNotNone(weather_call.call_id)
        self.assertIsNotNone(weather_call.arguments)
        self.assertIsNotNone(weather_call.status)

        # Parse and verify arguments
        args = json.loads(weather_call.arguments)
        self.assertIn("location", args)
        self.assertIn("seattle", args["location"].lower())

    def test_mixed_mcp_and_function_tools_streaming(self):
        """Test mixed MCP and function tools (streaming)."""
        resp = self.create_response(
            "What is the weather in seattle now?",
            tools=[self.BRAVE_MCP_TOOL, self.GET_WEATHER_FUNCTION],
            stream=True,
            tool_choice="auto",  # Encourage tool usage
        )

        # Should successfully make the request
        events = [event for event in resp]
        self.assertGreater(len(events), 0)

        event_types = [e.type for e in events]

        # Check for lifecycle events
        self.assertIn(
            "response.created", event_types, "Should have response.created event"
        )

        # Should have mcp_list_tools events
        self.assertIn(
            "response.mcp_list_tools.completed",
            event_types,
            "Should have mcp_list_tools.completed event",
        )

        # Should have function_call_arguments events (not mcp_call_arguments)
        self.assertIn(
            "response.function_call_arguments.delta",
            event_types,
            "Should have function_call_arguments.delta event for function tools",
        )
        self.assertIn(
            "response.function_call_arguments.done",
            event_types,
            "Should have function_call_arguments.done event for function tools",
        )

        # Should NOT have mcp_call_arguments events for function tools
        # (get_weather should use function_call_arguments, not mcp_call_arguments)
        mcp_call_arg_events = [
            e
            for e in events
            if e.type == "response.mcp_call_arguments.delta"
            and "get_weather" in str(e.delta)
        ]
        self.assertEqual(
            len(mcp_call_arg_events),
            0,
            "Should NOT emit mcp_call_arguments.delta for function tools (get_weather)",
        )

        # Verify function_call_arguments.delta event structure
        func_arg_deltas = [
            e for e in events if e.type == "response.function_call_arguments.delta"
        ]
        self.assertGreater(
            len(func_arg_deltas), 0, "Should have function_call_arguments.delta events"
        )

        # Check that at least one delta event contains location arguments
        has_location = False
        for event in func_arg_deltas:
            delta = event.delta
            if "location" in delta.lower() or "seattle" in delta.lower():
                has_location = True
                break

        self.assertTrue(
            has_location,
            "function_call_arguments.delta should contain location/seattle",
        )
