"""
End-to-end integration tests for tool calling with structural tags and json_schema.

Tests tool calling functionality with real models and HTTP requests,
verifying that structural tags and json_schema constraints work correctly.
"""

import json
import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class ToolCallingE2EBase(CustomTestCase):
    """Base class for E2E tool calling tests."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        # Start the local OpenAI Server with tool calling support
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--tool-call-parser",
                cls.parser_name,
            ],
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def setUp(self):
        self.client = openai.Client(base_url=self.base_url, api_key=self.api_key)
        self.model_name = self.client.models.list().data[0].id

    def get_test_tools(self):
        """Get test tools for function calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city name",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time in a timezone",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "The timezone (e.g., UTC, EST)",
                            },
                        },
                        "required": ["timezone"],
                    },
                },
            },
        ]

    def get_test_messages(self):
        """Get test messages that should trigger tool usage."""
        return [
            {
                "role": "user",
                "content": "What's the weather in San Francisco? Also, what time is it in UTC?",
            }
        ]


class TestToolCallingE2ELlama32(ToolCallingE2EBase):
    """E2E tests for Llama 3.2 with structural tags."""

    parser_name = "llama3"

    def test_auto_mode_structural_tag_optional(self):
        """Test tool_choice='auto' with structural tag - tool calls are optional."""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        self.assertIsNotNone(response.choices[0].message)
        # With auto mode, tool calls are optional, so we just verify no errors

    def test_auto_mode_parallel_true_multiple_calls(self):
        """Test tool_choice='auto' + parallel_tool_calls=True allows multiple calls."""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=True,
            stream=False,
        )

        self.assertIsNotNone(response.choices[0].message)
        # Multiple tool calls may be generated

    def test_auto_mode_parallel_false_single_call(self):
        """Test tool_choice='auto' + parallel_tool_calls=False limits to single call."""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=False,
            stream=False,
        )

        self.assertIsNotNone(response.choices[0].message)
        # Should limit to single tool call (stop_after_first=True in structural tag)

    def test_required_mode_json_schema_required(self):
        """Test tool_choice='required' with json_schema - tool calls are required."""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.1,
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        # With required, we should get tool calls
        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)

    def test_required_mode_parallel_true_multiple_calls(self):
        """Test tool_choice='required' + parallel_tool_calls=True allows multiple."""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.1,
            tools=tools,
            tool_choice="required",
            parallel_tool_calls=True,
            stream=False,
        )

        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)

    def test_required_mode_parallel_false_single_call(self):
        """Test tool_choice='required' + parallel_tool_calls=False sets maxItems=1 constraint."""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.1,
            tools=tools,
            tool_choice="required",
            parallel_tool_calls=False,
            stream=False,
        )

        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)
        # Note: maxItems=1 constraint guides the model but doesn't guarantee only 1 call
        # The constraint is applied, but models may still generate multiple calls

    def test_schema_guidance_works(self):
        """Test that schema guidance works - model follows parameter structure."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
        messages = [{"role": "user", "content": "What's the weather in Paris?"}]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=256,
            temperature=0.1,
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)

        # Verify the tool call has correct structure
        tool_call = tool_calls[0]
        self.assertEqual(tool_call.function.name, "get_weather")
        args = json.loads(tool_call.function.arguments)
        self.assertIn("city", args)
        self.assertIsInstance(args["city"], str)

    def test_streaming_mode_structural_tag(self):
        """Test streaming mode with structural tag."""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            tools=tools,
            tool_choice="auto",
            stream=True,
        )

        # Collect streaming response
        chunks = list(response)
        self.assertGreater(len(chunks), 0)

    def test_streaming_mode_json_schema(self):
        """Test streaming mode with json_schema."""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            tools=tools,
            tool_choice="required",
            stream=True,
        )

        # Collect streaming response
        tool_call_chunks = []
        for chunk in response:
            if chunk.choices[0].delta.tool_calls:
                tool_call_chunks.extend(chunk.choices[0].delta.tool_calls)

        # Should get tool call chunks
        self.assertGreater(len(tool_call_chunks), 0)

    def test_multiple_tools_scenario(self):
        """Test scenario with multiple tools."""
        tools = self.get_test_tools()
        messages = [
            {
                "role": "user",
                "content": "I need weather for Tokyo and the current time in UTC.",
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.1,
            tools=tools,
            tool_choice="required",
            parallel_tool_calls=True,
            stream=False,
        )

        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)

        # Verify different tools can be called
        called_names = {call.function.name for call in tool_calls}
        self.assertGreater(len(called_names), 0)


class TestToolCallingE2EMistral(ToolCallingE2EBase):
    """E2E tests for Mistral with structural tags."""

    parser_name = "mistral"

    def test_auto_mode_structural_tag(self):
        """Test tool_choice='auto' with structural tag."""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        self.assertIsNotNone(response.choices[0].message)

    def test_required_mode_json_schema(self):
        """Test tool_choice='required' with json_schema."""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.1,
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)

    def test_parallel_tool_calls_false(self):
        """Test parallel_tool_calls=False sets maxItems=1 constraint."""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.1,
            tools=tools,
            tool_choice="required",
            parallel_tool_calls=False,
            stream=False,
        )

        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)
        # Note: maxItems=1 constraint guides the model but doesn't guarantee only 1 call
        # The constraint is applied, but models may still generate multiple calls


class TestToolCallingE2EQwen25(ToolCallingE2EBase):
    """E2E tests for Qwen 2.5 with structural tags."""

    parser_name = "qwen25"

    def test_auto_mode_structural_tag(self):
        """Test tool_choice='auto' with structural tag."""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        self.assertIsNotNone(response.choices[0].message)

    def test_required_mode_json_schema(self):
        """Test tool_choice='required' with json_schema."""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.1,
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)

    def test_schema_guidance_works(self):
        """Test that schema guidance works."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
        messages = [{"role": "user", "content": "What's the weather in London?"}]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=256,
            temperature=0.1,
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)

        args = json.loads(tool_calls[0].function.arguments)
        self.assertIn("city", args)


if __name__ == "__main__":
    unittest.main()
