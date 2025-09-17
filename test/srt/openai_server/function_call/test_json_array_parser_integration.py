"""
Integration tests for JsonArrayParser with tool_choice functionality
Tests the integration of JsonArrayParser with OpenAI API tool_choice scenarios
"""

import json
import unittest

import openai

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestJsonArrayParserIntegration(CustomTestCase):
    """Integration tests for JsonArrayParser with tool_choice functionality"""

    @classmethod
    def setUpClass(cls):
        # Use a model that supports function calling
        cls.model = "meta-llama/Llama-3.2-1B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Start the local OpenAI Server with tool calling support
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--tool-call-parser",
                "llama3",  # Default parser for the test model
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
        """Get test tools for function calling"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit",
                            },
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def get_test_messages(self):
        """Get test messages that should trigger tool usage"""
        return [
            {
                "role": "user",
                "content": "What's the weather in Tokyo and search for 'weather forecast'?",
            }
        ]

    def test_tool_choice_required_with_json_array_parser(self):
        """Test that tool_choice='required' uses JsonArrayParser"""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        # Should get tool calls
        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)

        # Verify tool call structure
        for tool_call in tool_calls:
            self.assertIsNotNone(tool_call.function.name)
            self.assertIsNotNone(tool_call.function.arguments)
            
            # Verify arguments are valid JSON
            try:
                args = json.loads(tool_call.function.arguments)
                self.assertIsInstance(args, dict)
            except json.JSONDecodeError:
                self.fail(f"Invalid JSON in tool call arguments: {tool_call.function.arguments}")

    def test_tool_choice_specific_function_with_json_array_parser(self):
        """Test that specific tool choice uses JsonArrayParser"""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        tool_choice = {"type": "function", "function": {"name": "get_weather"}}

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
        )

        # Should get tool calls for the specific function
        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)

        # All tool calls should be for the specified function
        for tool_call in tool_calls:
            self.assertEqual(tool_call.function.name, "get_weather")
            
            # Verify arguments are valid JSON
            try:
                args = json.loads(tool_call.function.arguments)
                self.assertIsInstance(args, dict)
            except json.JSONDecodeError:
                self.fail(f"Invalid JSON in tool call arguments: {tool_call.function.arguments}")

    def test_tool_choice_required_streaming_with_json_array_parser(self):
        """Test that tool_choice='required' uses JsonArrayParser in streaming mode"""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            tools=tools,
            tool_choice="required",
            stream=True,
        )

        # Collect streaming response
        tool_call_chunks = []
        content_chunks = []

        for chunk in response:
            if chunk.choices[0].delta.tool_calls:
                tool_call_chunks.extend(chunk.choices[0].delta.tool_calls)
            if chunk.choices[0].delta.content:
                content_chunks.append(chunk.choices[0].delta.content)

        # Should get tool call chunks
        self.assertGreater(len(tool_call_chunks), 0)

        # Verify tool call structure in chunks
        for chunk in tool_call_chunks:
            if chunk.function and chunk.function.name:
                self.assertIsNotNone(chunk.function.name)
                if chunk.function.arguments:
                    try:
                        args = json.loads(chunk.function.arguments)
                        self.assertIsInstance(args, dict)
                    except json.JSONDecodeError:
                        # Partial arguments are expected in streaming
                        pass

    def test_tool_choice_specific_function_streaming_with_json_array_parser(self):
        """Test that specific tool choice uses JsonArrayParser in streaming mode"""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        tool_choice = {"type": "function", "function": {"name": "search"}}

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )

        # Collect streaming response
        tool_call_chunks = []

        for chunk in response:
            if chunk.choices[0].delta.tool_calls:
                tool_call_chunks.extend(chunk.choices[0].delta.tool_calls)

        # Should get tool call chunks
        self.assertGreater(len(tool_call_chunks), 0)

        # Find function name in chunks
        found_name = None
        for chunk in tool_call_chunks:
            if chunk.function and chunk.function.name:
                found_name = chunk.function.name
                break

        self.assertEqual(found_name, "search")

    def test_json_array_parser_handles_multiple_tool_calls(self):
        """Test that JsonArrayParser correctly handles multiple tool calls in one response"""
        tools = self.get_test_tools()
        messages = [
            {
                "role": "user",
                "content": "Get weather for Tokyo and search for 'weather forecast'",
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        # Should get multiple tool calls
        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreaterEqual(len(tool_calls), 1)

        # Verify each tool call has valid structure
        for tool_call in tool_calls:
            self.assertIsNotNone(tool_call.function.name)
            self.assertIsNotNone(tool_call.function.arguments)
            
            # Verify arguments are valid JSON
            try:
                args = json.loads(tool_call.function.arguments)
                self.assertIsInstance(args, dict)
            except json.JSONDecodeError:
                self.fail(f"Invalid JSON in tool call arguments: {tool_call.function.arguments}")

    def test_json_array_parser_error_handling(self):
        """Test that JsonArrayParser handles malformed JSON gracefully"""
        tools = self.get_test_tools()
        messages = [
            {
                "role": "user",
                "content": "Get weather for Tokyo",
            }
        ]

        # This should not crash even if the model generates malformed JSON
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=100,  # Short response to potentially trigger edge cases
            temperature=0.1,
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        # Should complete without crashing
        self.assertIsNotNone(response.choices[0].message)

    def test_json_array_parser_with_complex_parameters(self):
        """Test JsonArrayParser with complex nested parameters"""
        complex_tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_data",
                    "description": "Analyze complex data structures",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object",
                                "properties": {
                                    "metrics": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "config": {
                                        "type": "object",
                                        "properties": {
                                            "threshold": {"type": "number"},
                                            "enabled": {"type": "boolean"},
                                        },
                                    },
                                },
                                "required": ["metrics"],
                            },
                            "options": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "value": {"type": "string"},
                                    },
                                },
                            },
                        },
                        "required": ["data"],
                    },
                },
            }
        ]

        messages = [
            {
                "role": "user",
                "content": "Analyze some data with metrics and configuration",
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            tools=complex_tools,
            tool_choice="required",
            stream=False,
        )

        # Should get tool calls with complex parameters
        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)

        # Verify complex parameters are valid JSON
        for tool_call in tool_calls:
            self.assertEqual(tool_call.function.name, "analyze_data")
            try:
                args = json.loads(tool_call.function.arguments)
                self.assertIsInstance(args, dict)
                # Verify required fields exist
                self.assertIn("data", args)
                self.assertIsInstance(args["data"], dict)
            except json.JSONDecodeError:
                self.fail(f"Invalid JSON in complex tool call arguments: {tool_call.function.arguments}")


if __name__ == "__main__":
    unittest.main()
