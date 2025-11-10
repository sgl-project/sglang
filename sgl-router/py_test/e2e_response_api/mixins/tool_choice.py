"""
Tool choice tests for Response API.

Tests for tool_choice parameter support.

Note: These tests require --reasoning-parser gpt-oss in the worker.
"""

from basic_crud import ResponseAPIBaseTest


class ToolChoiceTests(ResponseAPIBaseTest):
    """Tests for tool_choice parameter in Responses API."""

    def test_tool_choice_auto(self):
        """
        Test tool_choice="auto" allows model to decide whether to use tools.

        The model should be able to choose to call a tool or not.
        """
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g., San Francisco",
                        }
                    },
                    "required": ["location"],
                },
            }
        ]

        # Query that should trigger tool use
        resp = self.create_response(
            "What is the weather in Seattle?",
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["status"], "completed")

        output = data.get("output", [])
        self.assertGreater(len(output), 0)

        # With auto, model should choose to call get_weather for this query
        function_calls = [
            item for item in output if item.get("type") == "function_call"
        ]
        self.assertGreater(
            len(function_calls),
            0,
            "Model should choose to call function with tool_choice='auto'",
        )

    def test_tool_choice_required(self):
        """
        Test tool_choice="required" forces the model to call at least one tool.

        The model must make at least one function call.
        """
        tools = [
            {
                "type": "function",
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            }
        ]

        resp = self.create_response(
            "What is 15 * 23?",
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        output = data.get("output", [])

        # Must have at least one function call
        function_calls = [
            item for item in output if item.get("type") == "function_call"
        ]
        self.assertGreater(
            len(function_calls),
            0,
            "tool_choice='required' must force at least one function call",
        )

    def test_tool_choice_specific_function(self):
        """
        Test tool_choice with specific function name forces that function to be called.

        The model must call the specified function.
        """
        tools = [
            {
                "type": "function",
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        ]

        # Force specific function call
        resp = self.create_response(
            "What's happening in the news today?",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "search_web"}},
            stream=False,
        )

        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        output = data.get("output", [])

        # Must have function call
        function_calls = [
            item for item in output if item.get("type") == "function_call"
        ]
        self.assertGreater(len(function_calls), 0, "Must call the specified function")

        # Must be the specified function
        called_function = function_calls[0]
        self.assertEqual(
            called_function.get("name"),
            "search_web",
            "Must call the function specified in tool_choice",
        )

    def test_tool_choice_streaming(self):
        """
        Test tool_choice parameter works correctly with streaming.

        Verifies that tool_choice constraints are applied in streaming mode.
        """
        tools = [
            {
                "type": "function",
                "name": "calculate",
                "description": "Perform calculations",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            }
        ]

        resp = self.create_response(
            "Calculate 42 * 17",
            tools=tools,
            tool_choice="required",
            stream=True,
        )

        self.assertEqual(resp.status_code, 200)

        events = self.parse_sse_events(resp)
        self.assertGreater(len(events), 0)

        event_types = [e.get("event") for e in events]

        # Should have function call events
        self.assertIn(
            "response.function_call_arguments.delta",
            event_types,
            "Should have function_call_arguments.delta events",
        )

        # Verify completed event has function call
        completed_events = [e for e in events if e.get("event") == "response.completed"]
        self.assertEqual(len(completed_events), 1)

        response_data = completed_events[0].get("data", {}).get("response", {})
        output = response_data.get("output", [])

        function_calls = [
            item for item in output if item.get("type") == "function_call"
        ]
        self.assertGreater(
            len(function_calls),
            0,
            "Streaming with tool_choice='required' must produce function call",
        )

    def test_tool_choice_with_mcp_tools(self):
        """
        Test tool_choice parameter works with MCP tools.

        Verifies that tool_choice can control MCP tool usage.
        """
        tools = [
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "require_approval": "never",
            }
        ]

        # With tool_choice="auto", should allow MCP tool calls
        resp = self.create_response(
            "What transport protocols does the 2025-03-26 version of the MCP spec (modelcontextprotocol/modelcontextprotocol) support?",
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        output = data.get("output", [])

        # Should have mcp_call with auto
        mcp_calls = [item for item in output if item.get("type") == "mcp_call"]
        self.assertGreater(
            len(mcp_calls), 0, "tool_choice='auto' should allow MCP tool calls"
        )

    def test_tool_choice_mixed_function_and_mcp(self):
        """
        Test tool_choice with mixed function and MCP tools.

        Verifies tool_choice can select specific tools when both function and MCP tools are available.
        """
        tools = [
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "require_approval": "never",
            },
            {
                "type": "function",
                "name": "local_search",
                "description": "Search local database",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        ]

        # Force specific function call
        resp = self.create_response(
            "Search for information about Python",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "local_search"}},
            stream=False,
        )

        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        output = data.get("output", [])

        # Must call local_search, not MCP
        function_calls = [
            item for item in output if item.get("type") == "function_call"
        ]
        self.assertGreater(len(function_calls), 0)
        self.assertEqual(function_calls[0].get("name"), "local_search")

        # Should not have mcp_call
        mcp_calls = [item for item in output if item.get("type") == "mcp_call"]
        self.assertEqual(
            len(mcp_calls),
            0,
            "Should only call specified function, not MCP tools",
        )
