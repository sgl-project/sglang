"""
Test tool calling via Anthropic Messages API.

These tests verify that SGLang's Anthropic API correctly handles
tool definitions and tool use responses.
"""


class TestToolCalling:
    """Test tool calling functionality."""

    def test_messages_create__generates_tool_call_when_appropriate(
        self, client, model_name, weather_tool
    ):
        """Test that model generates a tool call when appropriate."""
        response = client.messages.create(
            model=model_name,
            max_tokens=500,
            tools=[weather_tool],
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        )

        assert response.id is not None

        tool_uses = [c for c in response.content if c.type == "tool_use"]

        if response.stop_reason == "tool_use":
            assert (
                len(tool_uses) > 0
            ), "Should have tool_use block when stop_reason is tool_use"
            assert tool_uses[0].name == "get_weather"
            assert "location" in tool_uses[0].input
        else:
            text_blocks = [c for c in response.content if c.type == "text"]
            assert len(text_blocks) > 0, "Should have text response if no tool call"

    def test_messages_create__incorporates_tool_result_in_response(
        self, client, model_name, weather_tool
    ):
        """Test multi-turn with tool result."""
        response1 = client.messages.create(
            model=model_name,
            max_tokens=500,
            tools=[weather_tool],
            messages=[
                {"role": "user", "content": "What's the weather in Paris, France?"}
            ],
        )

        tool_uses = [c for c in response1.content if c.type == "tool_use"]
        assert (
            len(tool_uses) > 0
        ), "Model must generate tool call for multi-turn tool test"

        tool_use = tool_uses[0]
        tool_use_id = tool_use.id

        assistant_content = [
            block.model_dump() if hasattr(block, "model_dump") else block
            for block in response1.content
        ]

        response2 = client.messages.create(
            model=model_name,
            max_tokens=500,
            tools=[weather_tool],
            messages=[
                {"role": "user", "content": "What's the weather in Paris, France?"},
                {"role": "assistant", "content": assistant_content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": "22°C, partly cloudy",
                        }
                    ],
                },
            ],
        )

        assert response2.id is not None
        text_blocks = [c for c in response2.content if c.type == "text"]
        assert len(text_blocks) > 0, "Should have text response after tool result"
        response_text = text_blocks[0].text.lower()
        assert (
            "22" in response_text
            or "paris" in response_text
            or "cloud" in response_text
        )

    def test_messages_create__selects_correct_tool_from_multiple(
        self, client, model_name, weather_tool, calculator_tool
    ):
        """Test with multiple tool definitions."""
        response = client.messages.create(
            model=model_name,
            max_tokens=500,
            tools=[weather_tool, calculator_tool],
            messages=[{"role": "user", "content": "What is 15 times 7?"}],
        )

        assert response.id is not None

        tool_uses = [c for c in response.content if c.type == "tool_use"]

        if response.stop_reason == "tool_use":
            assert len(tool_uses) > 0
            assert tool_uses[0].name == "calculate"

    def test_messages_create__allows_text_or_tool_with_tool_choice_auto(
        self, client, model_name, weather_tool
    ):
        """Test tool_choice='auto' (default behavior)."""
        response = client.messages.create(
            model=model_name,
            max_tokens=500,
            tools=[weather_tool],
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

        assert response.id is not None

    def test_messages_create__generates_tool_use_id_with_correct_format(
        self, client, model_name, weather_tool
    ):
        """Test that tool_use IDs have correct format."""
        response = client.messages.create(
            model=model_name,
            max_tokens=500,
            tools=[weather_tool],
            messages=[{"role": "user", "content": "What's the weather in London?"}],
        )

        tool_uses = [c for c in response.content if c.type == "tool_use"]

        for tool_use in tool_uses:
            assert tool_use.id is not None
            assert tool_use.id.startswith(
                "toolu_"
            ), f"Tool use ID should start with 'toolu_', got: {tool_use.id}"

    def test_messages_create__generates_valid_tool_input_for_complex_schema(
        self, client, model_name
    ):
        """Test that tool input matches schema."""
        complex_tool = {
            "name": "search_products",
            "description": "Search for products in the catalog",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10,
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "food"],
                        "description": "Product category",
                    },
                },
                "required": ["query"],
            },
        }

        response = client.messages.create(
            model=model_name,
            max_tokens=500,
            tools=[complex_tool],
            messages=[{"role": "user", "content": "Find me some electronics laptops"}],
        )

        tool_uses = [c for c in response.content if c.type == "tool_use"]

        if tool_uses:
            tool_input = tool_uses[0].input
            assert (
                "query" in tool_input
            ), "Required 'query' field should be in tool input"


class TestToolCallingStreaming:
    """Test tool calling with streaming."""

    def test_messages_stream__includes_tool_call_in_final_message(
        self, client, model_name, weather_tool
    ):
        """Test streaming response with tool call."""
        tool_name = None
        tool_input_parts = []

        with client.messages.stream(
            model=model_name,
            max_tokens=500,
            tools=[weather_tool],
            messages=[{"role": "user", "content": "What's the weather in Sydney?"}],
        ) as stream:
            for event in stream:
                event_name = type(event).__name__

                if "ContentBlockStart" in event_name:
                    if hasattr(event, "content_block"):
                        if event.content_block.type == "tool_use":
                            tool_name = event.content_block.name

                if hasattr(event, "delta"):
                    if hasattr(event.delta, "partial_json"):
                        tool_input_parts.append(event.delta.partial_json)

            final = stream.get_final_message()

        tool_uses = [c for c in final.content if c.type == "tool_use"]

        if final.stop_reason == "tool_use":
            assert len(tool_uses) > 0
            assert tool_uses[0].name == "get_weather"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
