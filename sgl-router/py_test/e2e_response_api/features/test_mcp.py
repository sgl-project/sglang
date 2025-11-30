"""
MCP (Model Context Protocol) tests for Response API.

Tests MCP tool calling in both streaming and non-streaming modes.
These tests should work across all backends that support MCP (OpenAI, XAI).
"""

import json
import time

import pytest


@pytest.mark.parametrize(
    "setup_backend", ["openai", "grpc", "grpc_harmony"], indirect=True
)
class TestMcp:
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

    def test_mcp_basic_tool_call(self, setup_backend):
        """Test basic MCP tool call (non-streaming).

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
