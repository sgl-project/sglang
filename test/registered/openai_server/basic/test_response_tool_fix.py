"""Test cases for ResponseTool bug (function type constraint issue)."""

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="default")


def test_response_tool_function_type_rejected():
    """Test that function type is now accepted after fix."""
    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

    # This should now work without ValidationError
    request = ResponsesRequest(
        input="test",
        tools=[
            {
                "type": "function",
                "name": "calculate",
                "description": "Calculate something",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )

    # Verify the tool was correctly parsed
    assert len(request.tools) == 1
    assert request.tools[0].type == "function"
    assert request.tools[0].name == "calculate"
    assert request.tools[0].description == "Calculate something"
    assert request.tools[0].parameters == {"type": "object", "properties": {}}


def test_builtin_tools_work():
    """Test that built-in tools work."""
    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

    request = ResponsesRequest(
        input="test",
        tools=[{"type": "code_interpreter"}, {"type": "web_search_preview"}],
    )

    assert len(request.tools) == 2
    assert request.tools[0].type == "code_interpreter"
    assert request.tools[1].type == "web_search_preview"


def test_mixed_tools():
    """Test mixing built-in and function tools - should work after fix."""
    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

    # This should now work without ValidationError
    request = ResponsesRequest(
        input="test",
        tools=[
            {"type": "code_interpreter"},
            {
                "type": "function",
                "name": "my_custom_tool",
                "description": "Custom function",
                "parameters": {"type": "object", "properties": {}},
            },
        ],
    )

    # Verify both tools were correctly parsed
    assert len(request.tools) == 2
    assert request.tools[0].type == "code_interpreter"
    assert request.tools[1].type == "function"
    assert request.tools[1].name == "my_custom_tool"
    assert request.tools[1].description == "Custom function"


def test_empty_tools():
    """Test that empty tools list works."""
    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

    request = ResponsesRequest(input="test")
    assert request.tools == []

    request_with_empty = ResponsesRequest(input="test", tools=[])
    assert request_with_empty.tools == []


def test_function_tool_with_full_schema():
    """Test function tool with complete schema - should work after fix."""
    from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

    function_def = {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use",
                },
            },
            "required": ["location"],
        },
    }

    # This should now work without ValidationError
    request = ResponsesRequest(input="What's the weather like?", tools=[function_def])

    # Verify the tool was correctly parsed
    assert len(request.tools) == 1
    assert request.tools[0].type == "function"
    assert request.tools[0].name == "get_weather"
    assert request.tools[0].parameters["type"] == "object"
    assert "location" in request.tools[0].parameters["properties"]
    assert request.tools[0].parameters["required"] == ["location"]


if __name__ == "__main__":
    # Run tests manually
    pytest.main([__file__, "-v"])
