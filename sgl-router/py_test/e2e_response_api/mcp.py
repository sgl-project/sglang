"""
MCP (Model Context Protocol) tests for Response API.

Tests MCP tool calling in both streaming and non-streaming modes.
These tests should work across all backends that support MCP (OpenAI, XAI).
"""

from basic_crud import ResponseAPIBaseTest


class MCPTests(ResponseAPIBaseTest):
    """Tests for MCP tool calling in both streaming and non-streaming modes."""

    def test_mcp_basic_tool_call(self):
        """Test basic MCP tool call (non-streaming)."""
        tools = [
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "require_approval": "never",
            }
        ]

        resp = self.create_response(
            "What transport protocols does the 2025-03-26 version of the MCP spec (modelcontextprotocol/modelcontextprotocol) support?",
            tools=tools,
            stream=False,
        )

        # Should successfully make the request
        self.assertEqual(resp.status_code, 200)

        data = resp.json()

        # Basic response structure
        self.assertIn("id", data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "completed")
        self.assertIn("output", data)
        self.assertIn("model", data)

        # Verify output array is not empty
        output = data["output"]
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)

        # Check for MCP-specific output types
        output_types = [item.get("type") for item in output]

        # Should have mcp_list_tools - tools are listed before calling
        self.assertIn(
            "mcp_list_tools", output_types, "Response should contain mcp_list_tools"
        )

        # Should have at least one mcp_call
        mcp_calls = [item for item in output if item.get("type") == "mcp_call"]
        self.assertGreater(
            len(mcp_calls), 0, "Response should contain at least one mcp_call"
        )

        # Verify mcp_call structure
        for mcp_call in mcp_calls:
            self.assertIn("id", mcp_call)
            self.assertIn("status", mcp_call)
            self.assertEqual(mcp_call["status"], "completed")
            self.assertIn("server_label", mcp_call)
            self.assertEqual(mcp_call["server_label"], "deepwiki")
            self.assertIn("name", mcp_call)
            self.assertIn("arguments", mcp_call)
            self.assertIn("output", mcp_call)

        # Should have final message output
        messages = [item for item in output if item.get("type") == "message"]
        self.assertGreater(
            len(messages), 0, "Response should contain at least one message"
        )

        # Verify message structure
        for msg in messages:
            self.assertIn("content", msg)
            self.assertIsInstance(msg["content"], list)

            # Check content has text
            for content_item in msg["content"]:
                if content_item.get("type") == "output_text":
                    self.assertIn("text", content_item)
                    self.assertIsInstance(content_item["text"], str)
                    self.assertGreater(len(content_item["text"]), 0)

    def test_mcp_basic_tool_call_streaming(self):
        """Test basic MCP tool call (streaming)."""
        tools = [
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "require_approval": "never",
            }
        ]

        resp = self.create_response(
            "What transport protocols does the 2025-03-26 version of the MCP spec (modelcontextprotocol/modelcontextprotocol) support?",
            tools=tools,
            stream=True,
        )

        # Should successfully make the request
        self.assertEqual(resp.status_code, 200)

        events = self.parse_sse_events(resp)
        self.assertGreater(len(events), 0)

        event_types = [e.get("event") for e in events]

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

        # Verify final completed event has full response
        completed_events = [e for e in events if e.get("event") == "response.completed"]
        self.assertEqual(len(completed_events), 1)

        final_response = completed_events[0].get("data", {}).get("response", {})
        self.assertIn("id", final_response)
        self.assertEqual(final_response.get("status"), "completed")
        self.assertIn("output", final_response)

        # Verify final output contains expected items
        final_output = final_response.get("output", [])
        final_output_types = [item.get("type") for item in final_output]

        self.assertIn("mcp_list_tools", final_output_types)
        self.assertIn("mcp_call", final_output_types)
        self.assertIn("message", final_output_types)

        # Verify mcp_call items in final output
        mcp_calls = [item for item in final_output if item.get("type") == "mcp_call"]
        self.assertGreater(len(mcp_calls), 0)

        for mcp_call in mcp_calls:
            self.assertEqual(mcp_call.get("status"), "completed")
            self.assertEqual(mcp_call.get("server_label"), "deepwiki")
            self.assertIn("name", mcp_call)
            self.assertIn("arguments", mcp_call)
            self.assertIn("output", mcp_call)

        # Verify text deltas combine to final message
        text_deltas = [
            e.get("data", {}).get("delta", "")
            for e in events
            if e.get("event") == "response.output_text.delta"
        ]
        self.assertGreater(len(text_deltas), 0, "Should have text deltas")

        # Get final text from output_text.done event
        text_done_events = [
            e for e in events if e.get("event") == "response.output_text.done"
        ]
        self.assertGreater(len(text_done_events), 0)

        final_text = text_done_events[0].get("data", {}).get("text", "")
        self.assertGreater(len(final_text), 0, "Final text should not be empty")
