"""
Test class for tool calling tests.

This module provides test cases for function calling functionality, tool choices
and mcp calling functionality across different backends.
"""

import json
import sys
import time
from pathlib import Path

import pytest

# Add current directory for local imports
_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR))


@pytest.mark.parametrize(
    "setup_backend", ["openai", "grpc", "grpc_harmony"], indirect=True
)
class TestToolCalling:

    # Shared function tool definitions
    SYSTEM_DIAGNOSTICS_FUNCTION = {
        "type": "function",
        "name": "get_system_diagnostics",
        "description": "Retrieve real-time diagnostics for a spacecraft system.",
        "parameters": {
            "type": "object",
            "properties": {
                "system_name": {
                    "type": "string",
                    "description": "Name of the spacecraft system to query. "
                    "Example: 'Astra-7 Core Reactor'.",
                }
            },
            "required": ["system_name"],
        },
    }

    GET_WEATHER_FUNCTION = {
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

    CALCULATE_FUNCTION = {
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

    SEARCH_WEB_FUNCTION = {
        "type": "function",
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }

    LOCAL_SEARCH_FUNCTION = {
        "type": "function",
        "name": "local_search",
        "description": "Search local database",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }

    # Shared constants for MCP tests
    BRAVE_MCP_TOOL = {
        "type": "mcp",
        "server_label": "brave",
        "server_description": "A Tool to do web search",
        "server_url": "http://localhost:8001/sse",
        "require_approval": "never",
    }

    DEEPWIKI_MCP_TOOL = {
        "type": "mcp",
        "server_label": "deepwiki",
        "server_url": "https://mcp.deepwiki.com/mcp",
        "require_approval": "never",
    }

    MCP_TEST_PROMPT = (
        "show me some news about sglang router, use the tool to just search "
        "one result and return one sentence response"
    )

    # Test cases for basic function calling functionality

    def test_basic_function_call(self, setup_backend):
        """
        Test basic function calling workflow.

        This test follows the pattern from function_call_test.py:
        1. Define a function tool (get_horoscope)
        2. Send user message asking for horoscope
        3. Model should return function_call
        4. Execute function locally and provide output
        5. Model should generate final response using the function output
        """
        backend, model, client = setup_backend

        if backend in ["grpc"]:
            pytest.skip("skip for grpc")

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
        system_prompt = (
            "You are a helpful assistant that can call functions. "
            "When a user asks for horoscope information, call the function. "
            "IMPORTANT: Don't reply directly to the user, only call the function. "
        )

        # Create a running input list we will add to over time
        input_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is my horoscope? I am an Aquarius."},
        ]

        # 2. Prompt the model with tools defined
        resp = client.responses.create(model=model, input=input_list, tools=tools)

        # Should successfully make the request
        assert resp.error is None

        # Basic response structure
        assert resp.id is not None
        assert resp.status == "completed"
        assert resp.output is not None

        # Verify output array is not empty
        output = resp.output
        assert isinstance(output, list)
        assert len(output) > 0

        # Check for function_call in output
        function_calls = [item for item in output if item.type == "function_call"]
        assert (
            len(function_calls) > 0
        ), "Response should contain at least one function_call"

        # Verify function_call structure
        function_call = function_calls[0]
        assert function_call.call_id is not None
        assert function_call.name is not None
        assert function_call.name == "get_horoscope"
        assert function_call.arguments is not None

        # Parse arguments
        args = json.loads(function_call.arguments)
        assert "sign" in args
        assert args["sign"].lower() == "aquarius"

        # 3. Save function call outputs for subsequent requests
        input_list.append(function_call)

        # 4. Execute the function logic for get_horoscope
        horoscope = f"{args['sign']}: Next Tuesday you will befriend a baby otter."

        # 5. Provide function call results to the model
        input_list.append(
            {
                "type": "function_call_output",
                "call_id": function_call.call_id,
                "output": json.dumps({"horoscope": horoscope}),
            }
        )

        # 6. Make second request with function output
        resp2 = client.responses.create(
            model=model,
            input=input_list,
            instructions="Respond only with a horoscope generated by a tool.",
            tools=tools,
        )
        assert resp2.error is None
        assert resp2.status == "completed"

        # The model should be able to give a response using the function output
        output2 = resp2.output
        assert len(output2) > 0

        # Find message output
        messages = [item for item in output2 if item.type == "message"]
        assert len(messages) > 0, "Response should contain at least one message"

        # Verify message contains the horoscope
        message = messages[0]
        assert message.content is not None
        content_parts = message.content
        assert len(content_parts) > 0

        # Get text from content
        text_parts = [part.text for part in content_parts if part.type == "output_text"]
        full_text = " ".join(text_parts).lower()

        # Should mention the horoscope or baby otter
        assert (
            "baby otter" in full_text or "aquarius" in full_text
        ), "Response should reference the horoscope content"

    # Test cases for tool_choice parameter support, these tests require --reasoning-parser

    def test_tool_choice_auto(self, setup_backend):
        """
        Test tool_choice="auto" allows model to decide whether to use tools.

        The model should be able to choose to call a tool or not.
        """
        backend, model, client = setup_backend

        if backend in ["openai"]:
            pytest.skip("skip for openai")

        tools = [self.GET_WEATHER_FUNCTION]

        # Query that should trigger tool use
        resp = client.responses.create(
            model=model,
            input="What is the weather in Seattle?",
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output
        assert len(output) > 0

        # With auto, model should choose to call get_weather for this query
        function_calls = [item for item in output if item.type == "function_call"]
        assert (
            len(function_calls) > 0
        ), "Model should choose to call function with tool_choice='auto'"

    def test_tool_choice_required(self, setup_backend):
        """
        Test tool_choice="required" forces the model to call at least one tool.

        The model must make at least one function call.
        """
        backend, model, client = setup_backend

        if backend in ["openai"]:
            pytest.skip("skip for openai")

        tools = [self.CALCULATE_FUNCTION]

        resp = client.responses.create(
            model=model,
            input="What is 15 * 23?",
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output

        # Must have at least one function call
        function_calls = [item for item in output if item.type == "function_call"]
        assert (
            len(function_calls) > 0
        ), "tool_choice='required' must force at least one function call"

    def test_tool_choice_specific_function(self, setup_backend):
        """
        Test tool_choice with specific function name forces that function to be called.

        The model must call the specified function.
        """
        backend, model, client = setup_backend

        if backend in ["openai"]:
            pytest.skip("skip for openai")

        tools = [self.SEARCH_WEB_FUNCTION, self.GET_WEATHER_FUNCTION]

        # Force specific function call
        resp = client.responses.create(
            model=model,
            input="What's happening in the news today?",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "search_web"}},
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output

        # Must have function call
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0, "Must call the specified function"

        # Must be the specified function
        called_function = function_calls[0]
        assert (
            called_function.name == "search_web"
        ), "Must call the function specified in tool_choice"

    def test_tool_choice_streaming(self, setup_backend):
        """
        Test tool_choice parameter works correctly with streaming.

        Verifies that tool_choice constraints are applied in streaming mode.
        """
        backend, model, client = setup_backend

        if backend in ["openai", "grpc"]:
            pytest.skip("skip for openai")

        tools = [self.CALCULATE_FUNCTION]

        resp = client.responses.create(
            model=model,
            input="Calculate 42 * 17",
            tools=tools,
            tool_choice="required",
            stream=True,
        )

        events = [event for event in resp]
        assert len(events) > 0

        event_types = [e.type for e in events]

        # Should have function call events
        assert (
            "response.function_call_arguments.delta" in event_types
        ), "Should have function_call_arguments.delta events"

        # Verify completed event has function call
        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1

        output = completed_events[0].response.output

        function_calls = [item for item in output if item.type == "function_call"]
        assert (
            len(function_calls) > 0
        ), "Streaming with tool_choice='required' must produce function call"

    def test_tool_choice_with_mcp_tools(self, setup_backend):
        """
        Test tool_choice parameter works with MCP tools.

        Verifies that tool_choice can control MCP tool usage.
        """
        backend, model, client = setup_backend

        if backend in ["openai"]:
            pytest.skip("skip for openai")

        tools = [self.DEEPWIKI_MCP_TOOL]

        # With tool_choice="auto", should allow MCP tool calls
        resp = client.responses.create(
            model=model,
            input="What transport protocols does the 2025-03-26 version of the MCP spec (modelcontextprotocol/modelcontextprotocol) support?",
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output

        # Should have mcp_call with auto
        mcp_calls = [item for item in output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0, "tool_choice='auto' should allow MCP tool calls"

    def test_tool_choice_mixed_function_and_mcp(self, setup_backend):
        """
        Test tool_choice with mixed function and MCP tools.

        Verifies tool_choice can select specific tools when both function and MCP tools are available.
        """
        backend, model, client = setup_backend

        if backend in ["openai"]:
            pytest.skip("skip for openai")

        tools = [self.DEEPWIKI_MCP_TOOL, self.LOCAL_SEARCH_FUNCTION]

        # Force specific function call
        resp = client.responses.create(
            model=model,
            input="Search for information about Python",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "local_search"}},
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output

        # Must call local_search, not MCP
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0
        assert function_calls[0].name == "local_search"

        # Should not have mcp_call
        mcp_calls = [item for item in output if item.type == "mcp_call"]
        assert len(mcp_calls) == 0, "Should only call specified function, not MCP tools"

    # Tests for MCP tool calling in both streaming and non-streaming modes.

    def test_mcp_basic_tool_call(self, setup_backend):
        """
        Test basic MCP tool call (non-streaming).
        Validation strictness is controlled by parameter `backend` from setup_backend fixture.
        Set to "strict" if backend is http.
        """
        backend, model, client = setup_backend

        # To avoid being rate-limited by brave search server
        time.sleep(2)

        resp = client.responses.create(
            model=model,
            input=self.MCP_TEST_PROMPT,
            tools=[self.BRAVE_MCP_TOOL],
            stream=False,
            reasoning={"effort": "low"},
        )

        # Should successfully make the request
        assert resp.error is None

        # Basic response structure
        assert resp.id is not None
        assert resp.status == "completed"
        assert resp.model is not None
        assert resp.output is not None

        # Verify output array is not empty
        assert len(resp.output_text) > 0

        # Check for MCP-specific output types
        output_types = [item.type for item in resp.output]

        # Should have mcp_list_tools - tools are listed before calling
        assert (
            "mcp_list_tools" in output_types
        ), "Response should contain mcp_list_tools"

        # Should have at least one mcp_call
        mcp_calls = [item for item in resp.output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0, "Response should contain at least one mcp_call"

        # Verify mcp_call structure
        for mcp_call in mcp_calls:
            assert mcp_call.id is not None
            assert mcp_call.error is None
            assert mcp_call.status == "completed"
            assert mcp_call.server_label == "brave"
            assert mcp_call.name is not None
            assert mcp_call.arguments is not None
            assert mcp_call.output is not None

        # Strict mode: additional validation for HTTP backends
        if backend == "openai":
            # Should have final message output
            messages = [item for item in resp.output if item.type == "message"]
            assert len(messages) > 0, "Response should contain at least one message"
            # Verify message structure
            for msg in messages:
                assert msg.content is not None
                assert isinstance(msg.content, list)

                # Check content has text
                for content_item in msg.content:
                    if content_item.type == "output_text":
                        assert content_item.text is not None
                        assert isinstance(content_item.text, str)
                        assert len(content_item.text) > 0

    def test_mcp_basic_tool_call_streaming(self, setup_backend):
        """Test basic MCP tool call (streaming).

        Validation strictness is controlled by the class attribute `mcp_validation_mode`.
        Set to "strict" in subclasses for additional HTTP-specific validation.
        """
        backend, model, client = setup_backend

        # To avoid being rate-limited by brave search server
        time.sleep(2)

        resp = client.responses.create(
            model=model,
            input=self.MCP_TEST_PROMPT,
            tools=[self.BRAVE_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )

        # Should successfully make the request
        events = [event for event in resp]
        assert len(events) > 0

        event_types = [event.type for event in events]
        # Check for lifecycle events
        assert "response.created" in event_types, "Should have response.created event"
        assert (
            "response.completed" in event_types
        ), "Should have response.completed event"

        # Check for MCP list tools events
        assert (
            "response.output_item.added" in event_types
        ), "Should have output_item.added events"
        assert (
            "response.mcp_list_tools.in_progress" in event_types
        ), "Should have mcp_list_tools.in_progress event"
        assert (
            "response.mcp_list_tools.completed" in event_types
        ), "Should have mcp_list_tools.completed event"

        # Check for MCP call events
        assert (
            "response.mcp_call.in_progress" in event_types
        ), "Should have mcp_call.in_progress event"
        assert (
            "response.mcp_call_arguments.delta" in event_types
        ), "Should have mcp_call_arguments.delta event"
        assert (
            "response.mcp_call_arguments.done" in event_types
        ), "Should have mcp_call_arguments.done event"
        assert (
            "response.mcp_call.completed" in event_types
        ), "Should have mcp_call.completed event"

        # Verify final completed event has full response
        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1

        final_response = completed_events[0].response
        assert final_response.id is not None
        assert final_response.status == "completed"
        assert final_response.output is not None

        # Verify final output contains expected items
        final_output = final_response.output
        final_output_types = [item.type for item in final_output]

        assert "mcp_list_tools" in final_output_types
        assert "mcp_call" in final_output_types

        # Verify mcp_call items in final output
        mcp_calls = [item for item in final_output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0

        for mcp_call in mcp_calls:
            assert mcp_call.error is None
            assert mcp_call.status == "completed"
            assert mcp_call.server_label == "brave"
            assert mcp_call.name is not None
            assert mcp_call.arguments is not None
            assert mcp_call.output is not None

        # Strict mode: additional validation for HTTP backends
        if backend == "openai":
            # Check for text output events
            assert (
                "response.content_part.added" in event_types
            ), "Should have content_part.added event"
            assert (
                "response.output_text.delta" in event_types
            ), "Should have output_text.delta events"
            assert (
                "response.output_text.done" in event_types
            ), "Should have output_text.done event"
            assert (
                "response.content_part.done" in event_types
            ), "Should have content_part.done event"

            assert "message" in final_output_types

            # Verify text deltas combine to final message
            text_deltas = [
                e.delta for e in events if e.type == "response.output_text.delta"
            ]
            assert len(text_deltas) > 0, "Should have text deltas"

            # Get final text from output_text.done event
            text_done_events = [
                e for e in events if e.type == "response.output_text.done"
            ]
            assert len(text_done_events) > 0

            final_text = text_done_events[0].text
            assert len(final_text) > 0, "Final text should not be empty"

    def test_mixed_mcp_and_function_tools(self, setup_backend):
        """Test mixed MCP and function tools (non-streaming)."""
        backend, model, client = setup_backend

        if backend in ["openai"]:
            pytest.skip(
                "Requires external MCP server (deepwiki) - may not be accessible in CI"
            )

        resp = client.responses.create(
            model=model,
            input="Give me diagnostics for the Astra-7 Core Reactor.",
            tools=[self.BRAVE_MCP_TOOL, self.SYSTEM_DIAGNOSTICS_FUNCTION],
            stream=False,
            tool_choice="auto",
        )

        # Should successfully make the request
        assert resp.error is None

        # Basic response structure
        assert resp.id is not None
        assert resp.status is not None
        assert resp.output is not None

        # Verify output array is not empty
        output = resp.output
        assert isinstance(output, list)
        assert len(output) > 0

        # Check for function_call (not mcp_call for get_system_diagnostics)
        function_calls = [item for item in output if item.type == "function_call"]
        assert (
            len(function_calls) > 0
        ), "Response should contain at least one function_call"

        # Verify function_call structure for get_system_diagnostics
        system_diagnostics_call = function_calls[0]
        assert system_diagnostics_call.name == "get_system_diagnostics"
        assert system_diagnostics_call.call_id is not None
        assert system_diagnostics_call.arguments is not None
        assert system_diagnostics_call.status is not None

        # Parse and verify arguments
        args = json.loads(system_diagnostics_call.arguments)
        assert "system_name" in args
        assert "astra-7" in args["system_name"].lower()

    def test_mixed_mcp_and_function_tools_streaming(self, setup_backend):
        """Test mixed MCP and function tools (streaming)."""
        backend, model, client = setup_backend

        if backend in ["openai"]:
            pytest.skip(
                "Requires external MCP server (deepwiki) - may not be accessible in CI"
            )

        resp = client.responses.create(
            model=model,
            input="Give me diagnostics for the Astra-7 Core Reactor.",
            tools=[self.BRAVE_MCP_TOOL, self.SYSTEM_DIAGNOSTICS_FUNCTION],
            stream=True,
            tool_choice="auto",  # Encourage tool usage
        )

        # Should successfully make the request
        events = [event for event in resp]
        assert len(events) > 0

        event_types = [e.type for e in events]

        # Check for lifecycle events
        assert "response.created" in event_types, "Should have response.created event"

        # Should have mcp_list_tools events
        assert (
            "response.mcp_list_tools.completed" in event_types
        ), "Should have mcp_list_tools.completed event"

        # Should have function_call_arguments events (not mcp_call_arguments)
        assert (
            "response.function_call_arguments.delta" in event_types
        ), "Should have function_call_arguments.delta event for function tools"
        assert (
            "response.function_call_arguments.done" in event_types
        ), "Should have function_call_arguments.done event for function tools"

        # Should NOT have mcp_call_arguments events for function tools
        # (get_system_diagnostics should use function_call_arguments, not mcp_call_arguments)
        mcp_call_arg_events = [
            e
            for e in events
            if e.type == "response.mcp_call_arguments.delta"
            and "get_system_diagnostics" in str(e.delta)
        ]
        assert (
            len(mcp_call_arg_events) == 0
        ), "Should NOT emit mcp_call_arguments.delta for function tools (get_system_diagnostics)"

        # Verify function_call_arguments.delta event structure
        func_arg_deltas = [
            e for e in events if e.type == "response.function_call_arguments.delta"
        ]
        assert (
            len(func_arg_deltas) > 0
        ), "Should have function_call_arguments.delta events"

        # Check that delta event contains system_name arguments
        full_delta_event = ""
        for event in func_arg_deltas:
            full_delta_event += event.delta

        assert (
            "system_name" in full_delta_event.lower()
            and "astra-7" in full_delta_event.lower()
        ), "function_call_arguments.delta should contain system_name and astra-7"
