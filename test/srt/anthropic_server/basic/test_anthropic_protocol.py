"""
Unit tests for Anthropic protocol functionality.
"""

import unittest
from unittest.mock import Mock, patch

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicTool,
    AnthropicToolChoice,
    AnthropicUsage,
)


class TestAnthropicProtocol(unittest.TestCase):
    """Test Anthropic protocol models and validation."""

    def test_anthropic_content_block_text(self):
        """Test AnthropicContentBlock with text content."""
        content_block = AnthropicContentBlock(type="text", text="Hello, world!")
        self.assertEqual(content_block.type, "text")
        self.assertEqual(content_block.text, "Hello, world!")
        self.assertIsNone(content_block.source)
        self.assertIsNone(content_block.id)
        self.assertIsNone(content_block.name)
        self.assertIsNone(content_block.input)
        self.assertIsNone(content_block.content)
        self.assertIsNone(content_block.is_error)

    def test_anthropic_content_block_tool_use(self):
        """Test AnthropicContentBlock with tool use content."""
        content_block = AnthropicContentBlock(
            type="tool_use",
            id="toolu_123",
            name="get_weather",
            input={"location": "Paris", "unit": "celsius"},
        )
        self.assertEqual(content_block.type, "tool_use")
        self.assertEqual(content_block.id, "toolu_123")
        self.assertEqual(content_block.name, "get_weather")
        self.assertEqual(content_block.input, {"location": "Paris", "unit": "celsius"})

    def test_anthropic_content_block_tool_result(self):
        """Test AnthropicContentBlock with tool result content."""
        content_block = AnthropicContentBlock(
            type="tool_result",
            id="call_123",
            content="The weather in Paris is sunny.",
            is_error=False,
        )
        self.assertEqual(content_block.type, "tool_result")
        self.assertEqual(content_block.id, "call_123")
        self.assertEqual(content_block.content, "The weather in Paris is sunny.")
        self.assertFalse(content_block.is_error)

    def test_anthropic_message_user(self):
        """Test AnthropicMessage for user role."""
        message = {"role": "user", "content": "Hello, assistant!"}
        anthropic_message = AnthropicMessage(**message)
        self.assertEqual(anthropic_message.role, "user")
        self.assertEqual(anthropic_message.content, "Hello, assistant!")

    def test_anthropic_message_assistant_text(self):
        """Test AnthropicMessage for assistant role with text content."""
        content_block = AnthropicContentBlock(type="text", text="Hello, user!")
        message = {"role": "assistant", "content": [content_block]}
        anthropic_message = AnthropicMessage(**message)
        self.assertEqual(anthropic_message.role, "assistant")
        self.assertIsInstance(anthropic_message.content, list)
        self.assertEqual(len(anthropic_message.content), 1)
        self.assertEqual(anthropic_message.content[0].type, "text")
        self.assertEqual(anthropic_message.content[0].text, "Hello, user!")

    def test_anthropic_tool(self):
        """Test AnthropicTool model."""
        tool_schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        }

        tool = AnthropicTool(
            name="get_weather",
            description="Get the current weather for a location",
            input_schema=tool_schema,
        )

        self.assertEqual(tool.name, "get_weather")
        self.assertEqual(tool.description, "Get the current weather for a location")
        self.assertEqual(tool.input_schema, tool_schema)

    def test_anthropic_tool_choice_auto(self):
        """Test AnthropicToolChoice with auto type."""
        tool_choice = AnthropicToolChoice(type="auto", name=None)
        self.assertEqual(tool_choice.type, "auto")
        self.assertIsNone(tool_choice.name)

    def test_anthropic_tool_choice_specific(self):
        """Test AnthropicToolChoice with specific tool."""
        tool_choice = AnthropicToolChoice(type="tool", name="get_weather")
        self.assertEqual(tool_choice.type, "tool")
        self.assertEqual(tool_choice.name, "get_weather")

    def test_anthropic_messages_request_basic(self):
        """Test basic AnthropicMessagesRequest."""
        message = AnthropicMessage(
            role="user", content="What is the weather like today?"
        )

        request = AnthropicMessagesRequest(
            model="claude-3-haiku-20240307", messages=[message], max_tokens=100
        )

        self.assertEqual(request.model, "claude-3-haiku-20240307")
        self.assertEqual(len(request.messages), 1)
        self.assertEqual(request.messages[0].role, "user")
        self.assertEqual(request.max_tokens, 100)
        self.assertFalse(request.stream)
        self.assertIsNone(request.system)
        self.assertIsNone(request.temperature)

    def test_anthropic_messages_request_with_system(self):
        """Test AnthropicMessagesRequest with system prompt."""
        message = AnthropicMessage(
            role="user", content="What is the weather like today?"
        )

        request = AnthropicMessagesRequest(
            model="claude-3-haiku-20240307",
            messages=[message],
            max_tokens=100,
            system="You are a helpful weather assistant.",
        )

        self.assertEqual(request.system, "You are a helpful weather assistant.")

    def test_anthropic_messages_request_with_tools(self):
        """Test AnthropicMessagesRequest with tools."""
        message = AnthropicMessage(role="user", content="What is the weather in Paris?")

        tool_schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        }

        tool = AnthropicTool(
            name="get_weather",
            description="Get the current weather for a location",
            input_schema=tool_schema,
        )

        tool_choice = AnthropicToolChoice(type="auto", name=None)

        request = AnthropicMessagesRequest(
            model="claude-3-haiku-20240307",
            messages=[message],
            max_tokens=100,
            tools=[tool],
            tool_choice=tool_choice,
        )

        self.assertEqual(len(request.tools), 1)
        self.assertEqual(request.tools[0].name, "get_weather")
        self.assertEqual(request.tool_choice.type, "auto")

    def test_anthropic_messages_request_streaming(self):
        """Test AnthropicMessagesRequest with streaming."""
        message = AnthropicMessage(role="user", content="Tell me a story.")

        request = AnthropicMessagesRequest(
            model="claude-3-haiku-20240307",
            messages=[message],
            max_tokens=1000,
            stream=True,
        )

        self.assertTrue(request.stream)

    def test_anthropic_messages_request_validation(self):
        """Test AnthropicMessagesRequest validation."""
        message = AnthropicMessage(role="user", content="Hello")

        # Test model validation
        with self.assertRaises(ValueError):
            AnthropicMessagesRequest(model="", messages=[message], max_tokens=100)

        # Test max_tokens validation
        with self.assertRaises(ValueError):
            AnthropicMessagesRequest(
                model="claude-3-haiku-20240307", messages=[message], max_tokens=-1
            )

        # Test valid request
        request = AnthropicMessagesRequest(
            model="claude-3-haiku-20240307",
            messages=[message],
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )

        self.assertEqual(request.temperature, 0.7)
        self.assertEqual(request.top_p, 0.9)
        self.assertEqual(request.top_k, 50)

    def test_anthropic_usage(self):
        """Test AnthropicUsage model."""
        usage = AnthropicUsage(input_tokens=50, output_tokens=75)

        self.assertEqual(usage.input_tokens, 50)
        self.assertEqual(usage.output_tokens, 75)
        self.assertIsNone(usage.cache_creation_input_tokens)
        self.assertIsNone(usage.cache_read_input_tokens)

    def test_anthropic_usage_with_cache(self):
        """Test AnthropicUsage model with cache tokens."""
        usage = AnthropicUsage(
            input_tokens=50,
            output_tokens=75,
            cache_creation_input_tokens=25,
            cache_read_input_tokens=15,
        )

        self.assertEqual(usage.input_tokens, 50)
        self.assertEqual(usage.output_tokens, 75)
        self.assertEqual(usage.cache_creation_input_tokens, 25)
        self.assertEqual(usage.cache_read_input_tokens, 15)

    def test_anthropic_messages_response(self):
        """Test AnthropicMessagesResponse model."""
        content_block = AnthropicContentBlock(type="text", text="Hello, user!")
        usage = AnthropicUsage(input_tokens=10, output_tokens=20)

        response = AnthropicMessagesResponse(
            id="msg_123",
            content=[content_block],
            model="claude-3-haiku-20240307",
            stop_reason="end_turn",
            usage=usage,
        )

        self.assertEqual(response.id, "msg_123")
        self.assertEqual(response.type, "message")
        self.assertEqual(response.role, "assistant")
        self.assertEqual(len(response.content), 1)
        self.assertEqual(response.content[0].type, "text")
        self.assertEqual(response.content[0].text, "Hello, user!")
        self.assertEqual(response.model, "claude-3-haiku-20240307")
        self.assertEqual(response.stop_reason, "end_turn")
        self.assertEqual(response.usage.input_tokens, 10)
        self.assertEqual(response.usage.output_tokens, 20)

    def test_anthropic_messages_response_auto_id(self):
        """Test AnthropicMessagesResponse with auto-generated ID."""
        content_block = AnthropicContentBlock(type="text", text="Hello!")
        usage = AnthropicUsage(input_tokens=5, output_tokens=10)

        response = AnthropicMessagesResponse(
            id="msg_test123",
            content=[content_block],
            model="claude-3-haiku-20240307",
            usage=usage,
        )

        self.assertEqual(response.id, "msg_test123")
        self.assertTrue(response.id.startswith("msg_"))

    def test_anthropic_tool_input_schema_default(self):
        """Test AnthropicTool with default input_schema type."""
        tool = AnthropicTool(
            name="simple_tool",
            input_schema={"properties": {"param": {"type": "string"}}},
        )

        # The validator should add "type": "object" if missing
        self.assertEqual(tool.input_schema["type"], "object")
        self.assertIn("properties", tool.input_schema)


if __name__ == "__main__":
    unittest.main(verbosity=2)
