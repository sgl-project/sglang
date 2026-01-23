"""Tool calling tests for Response API.

Tests for function calling functionality, tool choices and MCP calling
functionality across different backends.

Source: Migrated from e2e_response_api/features/test_tools_call.py
"""

from __future__ import annotations

import json
import logging
import time

import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Tool Definitions
# =============================================================================


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

GET_HOROSCOPE_FUNCTION = {
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
}

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


# =============================================================================
# Cloud Backend Tests (OpenAI) - Basic Function Calling
# =============================================================================


@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestToolCallingCloud:
    """Tool calling tests against cloud APIs."""

    def test_basic_function_call(self, setup_backend):
        """Test basic function calling workflow."""
        _, model, client, gateway = setup_backend

        tools = [GET_HOROSCOPE_FUNCTION]
        system_prompt = (
            "You are a helpful assistant that can call functions. "
            "When a user asks for horoscope information, call the function. "
            "IMPORTANT: Don't reply directly to the user, only call the function. "
        )

        input_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is my horoscope? I am an Aquarius."},
        ]

        resp = client.responses.create(model=model, input=input_list, tools=tools)

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"
        assert resp.output is not None

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
        assert function_call.name == "get_horoscope"
        assert function_call.arguments is not None

        # Parse arguments
        args = json.loads(function_call.arguments)
        assert "sign" in args
        assert args["sign"].lower() == "aquarius"

        # Provide function call output
        input_list.append(function_call)
        horoscope = f"{args['sign']}: Next Tuesday you will befriend a baby otter."
        input_list.append(
            {
                "type": "function_call_output",
                "call_id": function_call.call_id,
                "output": json.dumps({"horoscope": horoscope}),
            }
        )

        # Second request with function output
        resp2 = client.responses.create(
            model=model,
            input=input_list,
            instructions="Respond only with a horoscope generated by a tool.",
            tools=tools,
        )
        assert resp2.error is None
        assert resp2.status == "completed"

        output2 = resp2.output
        assert len(output2) > 0

        messages = [item for item in output2 if item.type == "message"]
        assert len(messages) > 0

        message = messages[0]
        assert message.content is not None
        text_parts = [
            part.text for part in message.content if part.type == "output_text"
        ]
        full_text = " ".join(text_parts).lower()
        assert "baby otter" in full_text or "aquarius" in full_text

    def test_mcp_basic_tool_call(self, setup_backend):
        """Test basic MCP tool call (non-streaming)."""
        _, model, client, gateway = setup_backend

        time.sleep(2)  # Avoid rate limiting

        resp = client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL],
            stream=False,
            reasoning={"effort": "low"},
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"
        assert resp.model is not None
        assert resp.output is not None
        assert len(resp.output_text) > 0

        output_types = [item.type for item in resp.output]
        assert "mcp_list_tools" in output_types

        mcp_calls = [item for item in resp.output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0

        for mcp_call in mcp_calls:
            assert mcp_call.id is not None
            assert mcp_call.error is None
            assert mcp_call.status == "completed"
            assert mcp_call.server_label == "brave"
            assert mcp_call.name is not None
            assert mcp_call.arguments is not None
            assert mcp_call.output is not None

        # Strict validation for cloud backends
        messages = [item for item in resp.output if item.type == "message"]
        assert len(messages) > 0, "Response should contain at least one message"
        for msg in messages:
            assert msg.content is not None
            assert isinstance(msg.content, list)
            for content_item in msg.content:
                if content_item.type == "output_text":
                    assert content_item.text is not None
                    assert isinstance(content_item.text, str)
                    assert len(content_item.text) > 0

    def test_mcp_basic_tool_call_streaming(self, setup_backend):
        """Test basic MCP tool call (streaming)."""
        _, model, client, gateway = setup_backend

        time.sleep(2)  # Avoid rate limiting

        resp = client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )

        events = list(resp)
        assert len(events) > 0

        event_types = [event.type for event in events]
        assert "response.created" in event_types, "Should have response.created event"
        assert (
            "response.completed" in event_types
        ), "Should have response.completed event"
        assert (
            "response.output_item.added" in event_types
        ), "Should have output_item.added events"
        assert (
            "response.mcp_list_tools.in_progress" in event_types
        ), "Should have mcp_list_tools.in_progress event"
        assert (
            "response.mcp_list_tools.completed" in event_types
        ), "Should have mcp_list_tools.completed event"
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

        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1

        final_response = completed_events[0].response
        assert final_response.id is not None
        assert final_response.status == "completed"
        assert final_response.output is not None

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

        # Strict validation for cloud backends - check for text output events
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
        text_done_events = [e for e in events if e.type == "response.output_text.done"]
        assert len(text_done_events) > 0

        final_text = text_done_events[0].text
        assert len(final_text) > 0, "Final text should not be empty"

    def test_mcp_multi_server_tool_call(self, setup_backend):
        """Test MCP tool call with multiple servers (non-streaming)."""
        _, model, client, gateway = setup_backend

        time.sleep(2)  # Avoid rate limiting

        resp = client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL, DEEPWIKI_MCP_TOOL],
            stream=False,
            reasoning={"effort": "low"},
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"
        assert resp.output is not None

        list_tools_items = [
            item for item in resp.output if item.type == "mcp_list_tools"
        ]
        assert len(list_tools_items) == 2
        labels = {item.server_label for item in list_tools_items}
        assert labels == {"brave", "deepwiki"}

        mcp_calls = [item for item in resp.output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0

    def test_mcp_multi_server_tool_call_streaming(self, setup_backend):
        """Test MCP tool call with multiple servers (streaming)."""
        _, model, client, gateway = setup_backend

        time.sleep(2)  # Avoid rate limiting

        resp = client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL, DEEPWIKI_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )

        events = list(resp)
        assert len(events) > 0

        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1

        final_output = completed_events[0].response.output
        list_tools_items = [
            item for item in final_output if item.type == "mcp_list_tools"
        ]
        assert len(list_tools_items) == 2
        labels = {item.server_label for item in list_tools_items}
        assert labels == {"brave", "deepwiki"}

        mcp_calls = [item for item in final_output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0


# =============================================================================
# Local Backend Tests (gRPC with Harmony model) - Tool Choice
# =============================================================================


@pytest.mark.e2e
@pytest.mark.model("gpt-oss")
@pytest.mark.gateway(
    extra_args=["--reasoning-parser=gpt-oss", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestToolChoiceHarmony:
    """Tool choice tests against local gRPC backend with Harmony model."""

    def test_tool_choice_auto(self, setup_backend):
        """Test tool_choice="auto" allows model to decide whether to use tools."""
        _, model, client, gateway = setup_backend

        tools = [GET_WEATHER_FUNCTION]

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

        function_calls = [item for item in output if item.type == "function_call"]
        assert (
            len(function_calls) > 0
        ), "Model should choose to call function with tool_choice='auto'"

    def test_tool_choice_required(self, setup_backend):
        """Test tool_choice="required" forces the model to call at least one tool."""
        _, model, client, gateway = setup_backend

        tools = [CALCULATE_FUNCTION]

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
        function_calls = [item for item in output if item.type == "function_call"]
        assert (
            len(function_calls) > 0
        ), "tool_choice='required' must force at least one function call"

    def test_tool_choice_specific_function(self, setup_backend):
        """Test tool_choice with specific function name forces that function to be called."""
        _, model, client, gateway = setup_backend

        tools = [SEARCH_WEB_FUNCTION, GET_WEATHER_FUNCTION]

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
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0, "Must call the specified function"
        assert (
            function_calls[0].name == "search_web"
        ), "Must call the function specified in tool_choice"

    def test_tool_choice_streaming(self, setup_backend):
        """Test tool_choice parameter works correctly with streaming."""
        _, model, client, gateway = setup_backend

        tools = [CALCULATE_FUNCTION]

        resp = client.responses.create(
            model=model,
            input="Calculate 42 * 17",
            tools=tools,
            tool_choice="required",
            stream=True,
        )

        events = list(resp)
        assert len(events) > 0

        event_types = [e.type for e in events]
        assert "response.function_call_arguments.delta" in event_types

        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1

        output = completed_events[0].response.output
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0

    def test_tool_choice_with_mcp_tools(self, setup_backend):
        """Test tool_choice parameter works with MCP tools."""
        _, model, client, gateway = setup_backend

        tools = [DEEPWIKI_MCP_TOOL]

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
        mcp_calls = [item for item in output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0, "tool_choice='auto' should allow MCP tool calls"

    def test_mcp_multi_server_tool_call(self, setup_backend):
        """Test MCP tool call with multiple servers (non-streaming)."""
        _, model, client, gateway = setup_backend

        time.sleep(2)

        resp = client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL, DEEPWIKI_MCP_TOOL],
            stream=False,
            reasoning={"effort": "low"},
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"
        assert resp.output is not None

        list_tools_items = [
            item for item in resp.output if item.type == "mcp_list_tools"
        ]
        assert len(list_tools_items) == 2
        labels = {item.server_label for item in list_tools_items}
        assert labels == {"brave", "deepwiki"}

        mcp_calls = [item for item in resp.output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0

    def test_mcp_multi_server_tool_call_streaming(self, setup_backend):
        """Test MCP tool call with multiple servers (streaming)."""
        _, model, client, gateway = setup_backend

        time.sleep(2)

        resp = client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL, DEEPWIKI_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )

        events = list(resp)
        assert len(events) > 0

        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1

        final_output = completed_events[0].response.output
        list_tools_items = [
            item for item in final_output if item.type == "mcp_list_tools"
        ]
        assert len(list_tools_items) == 2
        labels = {item.server_label for item in list_tools_items}
        assert labels == {"brave", "deepwiki"}

        mcp_calls = [item for item in final_output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0

    def test_tool_choice_mixed_function_and_mcp(self, setup_backend):
        """Test tool_choice with mixed function and MCP tools."""
        _, model, client, gateway = setup_backend

        tools = [DEEPWIKI_MCP_TOOL, LOCAL_SEARCH_FUNCTION]

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
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0
        assert function_calls[0].name == "local_search"

        mcp_calls = [item for item in output if item.type == "mcp_call"]
        assert len(mcp_calls) == 0, "Should only call specified function, not MCP tools"

    def test_basic_function_call(self, setup_backend):
        """Test basic function calling workflow."""
        _, model, client, gateway = setup_backend

        tools = [GET_HOROSCOPE_FUNCTION]
        system_prompt = (
            "You are a helpful assistant that can call functions. "
            "When a user asks for horoscope information, call the function. "
            "IMPORTANT: Don't reply directly to the user, only call the function. "
        )

        input_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is my horoscope? I am an Aquarius."},
        ]

        resp = client.responses.create(model=model, input=input_list, tools=tools)

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"

        output = resp.output
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0

        function_call = function_calls[0]
        assert function_call.name == "get_horoscope"

        args = json.loads(function_call.arguments)
        assert "sign" in args
        assert args["sign"].lower() == "aquarius"

    def test_mcp_basic_tool_call(self, setup_backend):
        """Test basic MCP tool call (non-streaming)."""
        _, model, client, gateway = setup_backend

        time.sleep(2)

        resp = client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL],
            stream=False,
            reasoning={"effort": "low"},
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"
        assert len(resp.output_text) > 0

        output_types = [item.type for item in resp.output]
        assert "mcp_list_tools" in output_types

        mcp_calls = [item for item in resp.output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0

        for mcp_call in mcp_calls:
            assert mcp_call.id is not None
            assert mcp_call.error is None
            assert mcp_call.status == "completed"
            assert mcp_call.server_label == "brave"

    def test_mcp_basic_tool_call_streaming(self, setup_backend):
        """Test basic MCP tool call (streaming)."""
        _, model, client, gateway = setup_backend

        time.sleep(2)

        resp = client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )

        events = list(resp)
        assert len(events) > 0

        event_types = [event.type for event in events]
        assert "response.created" in event_types
        assert "response.completed" in event_types
        assert "response.mcp_list_tools.completed" in event_types
        assert "response.mcp_call.completed" in event_types

    def test_mixed_mcp_and_function_tools(self, setup_backend):
        """Test mixed MCP and function tools (non-streaming)."""
        _, model, client, gateway = setup_backend

        resp = client.responses.create(
            model=model,
            input="Give me diagnostics for the Astra-7 Core Reactor.",
            tools=[BRAVE_MCP_TOOL, SYSTEM_DIAGNOSTICS_FUNCTION],
            stream=False,
            tool_choice="auto",
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.output is not None

        output = resp.output
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0

        system_diagnostics_call = function_calls[0]
        assert system_diagnostics_call.name == "get_system_diagnostics"
        assert system_diagnostics_call.call_id is not None

        args = json.loads(system_diagnostics_call.arguments)
        assert "system_name" in args
        assert "astra-7" in args["system_name"].lower()

    def test_mixed_mcp_and_function_tools_streaming(self, setup_backend):
        """Test mixed MCP and function tools (streaming)."""
        _, model, client, gateway = setup_backend

        resp = client.responses.create(
            model=model,
            input="Give me diagnostics for the Astra-7 Core Reactor.",
            tools=[BRAVE_MCP_TOOL, SYSTEM_DIAGNOSTICS_FUNCTION],
            stream=True,
            tool_choice="auto",
        )

        events = list(resp)
        assert len(events) > 0

        event_types = [e.type for e in events]
        assert "response.created" in event_types
        assert "response.mcp_list_tools.completed" in event_types
        assert "response.function_call_arguments.delta" in event_types
        assert "response.function_call_arguments.done" in event_types

        func_arg_deltas = [
            e for e in events if e.type == "response.function_call_arguments.delta"
        ]
        assert len(func_arg_deltas) > 0

        full_delta_event = "".join(e.delta for e in func_arg_deltas)
        assert (
            "system_name" in full_delta_event.lower()
            and "astra-7" in full_delta_event.lower()
        )


# =============================================================================
# Local Backend Tests (gRPC with Qwen model) - Tool Choice
# =============================================================================


@pytest.mark.e2e
@pytest.mark.model("qwen-14b")
@pytest.mark.gateway(
    extra_args=["--tool-call-parser", "qwen", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestToolChoiceLocal:
    """Tool choice tests against local gRPC backend with Qwen model."""

    def test_tool_choice_auto(self, setup_backend):
        """Test tool_choice="auto" allows model to decide whether to use tools."""
        _, model, client, gateway = setup_backend

        tools = [GET_WEATHER_FUNCTION]

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

        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0

    def test_tool_choice_required(self, setup_backend):
        """Test tool_choice="required" forces the model to call at least one tool."""
        _, model, client, gateway = setup_backend

        tools = [CALCULATE_FUNCTION]

        resp = client.responses.create(
            model=model,
            input="What is 15 * 23?",
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        function_calls = [item for item in resp.output if item.type == "function_call"]
        assert len(function_calls) > 0

    def test_tool_choice_specific_function(self, setup_backend):
        """Test tool_choice with specific function name forces that function to be called."""
        _, model, client, gateway = setup_backend

        tools = [SEARCH_WEB_FUNCTION, GET_WEATHER_FUNCTION]

        resp = client.responses.create(
            model=model,
            input="What's happening in the news today?",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "search_web"}},
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        function_calls = [item for item in resp.output if item.type == "function_call"]
        assert len(function_calls) > 0
        assert function_calls[0].name == "search_web"

    def test_mcp_basic_tool_call(self, setup_backend):
        """Test basic MCP tool call (non-streaming)."""
        _, model, client, gateway = setup_backend

        time.sleep(2)

        resp = client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL],
            stream=False,
            reasoning={"effort": "low"},
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"

        output_types = [item.type for item in resp.output]
        assert "mcp_list_tools" in output_types

        mcp_calls = [item for item in resp.output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0

    def test_mcp_basic_tool_call_streaming(self, setup_backend):
        """Test basic MCP tool call (streaming)."""
        _, model, client, gateway = setup_backend

        time.sleep(2)

        resp = client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )

        events = list(resp)
        assert len(events) > 0

        event_types = [event.type for event in events]
        assert "response.created" in event_types
        assert "response.completed" in event_types

    def test_mcp_multi_server_tool_call(self, setup_backend):
        """Test MCP tool call with multiple servers (non-streaming)."""
        _, model, client, gateway = setup_backend

        time.sleep(2)

        resp = client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL, DEEPWIKI_MCP_TOOL],
            stream=False,
            reasoning={"effort": "low"},
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"
        assert resp.output is not None

        list_tools_items = [
            item for item in resp.output if item.type == "mcp_list_tools"
        ]
        assert len(list_tools_items) == 2
        labels = {item.server_label for item in list_tools_items}
        assert labels == {"brave", "deepwiki"}

        mcp_calls = [item for item in resp.output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0

    def test_mcp_multi_server_tool_call_streaming(self, setup_backend):
        """Test MCP tool call with multiple servers (streaming)."""
        _, model, client, gateway = setup_backend

        time.sleep(2)

        resp = client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL, DEEPWIKI_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )

        events = list(resp)
        assert len(events) > 0

        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1

        final_output = completed_events[0].response.output
        list_tools_items = [
            item for item in final_output if item.type == "mcp_list_tools"
        ]
        assert len(list_tools_items) == 2
        labels = {item.server_label for item in list_tools_items}
        assert labels == {"brave", "deepwiki"}

        mcp_calls = [item for item in final_output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0

    def test_tool_choice_with_mcp_tools(self, setup_backend):
        """Test tool_choice parameter works with MCP tools."""
        _, model, client, gateway = setup_backend

        tools = [DEEPWIKI_MCP_TOOL]

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
        mcp_calls = [item for item in output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0, "tool_choice='auto' should allow MCP tool calls"

    def test_tool_choice_mixed_function_and_mcp(self, setup_backend):
        """Test tool_choice with mixed function and MCP tools."""
        _, model, client, gateway = setup_backend

        tools = [DEEPWIKI_MCP_TOOL, LOCAL_SEARCH_FUNCTION]

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
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0
        assert function_calls[0].name == "local_search"

        mcp_calls = [item for item in output if item.type == "mcp_call"]
        assert len(mcp_calls) == 0, "Should only call specified function, not MCP tools"

    def test_mixed_mcp_and_function_tools(self, setup_backend):
        """Test mixed MCP and function tools (non-streaming)."""
        _, model, client, gateway = setup_backend

        resp = client.responses.create(
            model=model,
            input="Give me diagnostics for the Astra-7 Core Reactor.",
            tools=[BRAVE_MCP_TOOL, SYSTEM_DIAGNOSTICS_FUNCTION],
            stream=False,
            tool_choice="auto",
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.output is not None

        output = resp.output
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0

        system_diagnostics_call = function_calls[0]
        assert system_diagnostics_call.name == "get_system_diagnostics"
        assert system_diagnostics_call.call_id is not None

        args = json.loads(system_diagnostics_call.arguments)
        assert "system_name" in args
        assert "astra-7" in args["system_name"].lower()

    def test_mixed_mcp_and_function_tools_streaming(self, setup_backend):
        """Test mixed MCP and function tools (streaming)."""
        _, model, client, gateway = setup_backend

        resp = client.responses.create(
            model=model,
            input="Give me diagnostics for the Astra-7 Core Reactor.",
            tools=[BRAVE_MCP_TOOL, SYSTEM_DIAGNOSTICS_FUNCTION],
            stream=True,
            tool_choice="auto",
        )

        events = list(resp)
        assert len(events) > 0

        event_types = [e.type for e in events]
        assert "response.created" in event_types
        assert "response.mcp_list_tools.completed" in event_types
        assert "response.function_call_arguments.delta" in event_types
        assert "response.function_call_arguments.done" in event_types

        func_arg_deltas = [
            e for e in events if e.type == "response.function_call_arguments.delta"
        ]
        assert len(func_arg_deltas) > 0

        full_delta_event = "".join(e.delta for e in func_arg_deltas)
        assert (
            "system_name" in full_delta_event.lower()
            and "astra-7" in full_delta_event.lower()
        )
