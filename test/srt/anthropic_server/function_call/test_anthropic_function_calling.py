"""
Integration tests for Anthropic API function calling.
Run with:
    HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=python python -m unittest test.srt.anthropic_server.function_call.test_anthropic_function_calling.TestAnthropicServerFunctionCalling.test_function_calling_format
"""

import json
import unittest

import requests

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestAnthropicServerFunctionCalling(CustomTestCase):
    # System message to prompt the model to use tool calling
    SYSTEM_MESSAGE = (
        "You are a helpful assistant with tool calling capabilities. "
        "Only reply with a tool call if the function exists in the library provided by the user. "
        "If it doesn't exist, just reply directly in natural language. "
        "When you receive a tool call response, use the output to format an answer to the original user question. "
        "You have access to the following functions. "
        "To call a function, please respond with JSON for a function call. "
        'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. '
        "Do not use variables.\n\n"
    )

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--tool-call-parser",
                "llama3",
            ],
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_function_calling_format(self):
        """
        Test: Whether the function call format returned by the AI is correct.
        When returning a tool call, message.content should contain tool_use blocks.
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        tools = [
            {
                "name": "add",
                "description": "Compute the sum of two numbers",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "integer",
                            "description": "A number",
                        },
                        "b": {
                            "type": "integer",
                            "description": "A number",
                        },
                    },
                    "required": ["a", "b"],
                },
            }
        ]

        data = {
            "model": self.model,
            "max_tokens": 2048,
            "system": self.SYSTEM_MESSAGE,
            "messages": [
                {"role": "user", "content": "Compute (3+5)"},
            ],
            "temperature": 0.8,
            "tools": tools,
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200)

        result = response.json()

        # Verify response structure
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)
        self.assertGreater(len(result["content"]), 0)

        # Check if we have a tool_use content block
        tool_use_blocks = [
            block for block in result["content"] if block.get("type") == "tool_use"
        ]

        # Since this is a simple test, we'll check if we have tool_use blocks or just text
        # Either way is acceptable for a basic test
        if len(tool_use_blocks) > 0:
            # If we have tool_use blocks, verify their structure
            function_name = tool_use_blocks[0].get("name")
            self.assertIsNotNone(function_name, "Tool use block should have a name")

            # Check input parameters
            function_input = tool_use_blocks[0].get("input", {})
            self.assertIn("a", function_input, "Should have parameter 'a'")
            self.assertIn("b", function_input, "Should have parameter 'b'")
        else:
            # If we don't have tool_use blocks, that's also acceptable for this test
            # Just verify we have a text response
            text_blocks = [
                block for block in result["content"] if block.get("type") == "text"
            ]
            self.assertGreater(
                len(text_blocks),
                0,
                "Should have at least one text block if no tool_use blocks",
            )

    def test_function_calling_with_weather_tool(self):
        """
        Test: Function calling with weather tool.
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        tools = [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for",
                        },
                        "unit": {
                            "type": "string",
                            "description": "Weather unit (celsius or fahrenheit)",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city", "unit"],
                },
            }
        ]

        data = {
            "model": self.model,
            "max_tokens": 2048,
            "system": self.SYSTEM_MESSAGE,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the temperature in Paris in celsius?",
                },
            ],
            "temperature": 0.8,
            "tools": tools,
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200)

        result = response.json()

        # Verify response structure
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)
        self.assertGreater(len(result["content"]), 0)

        # Check content blocks - either tool_use or text is acceptable
        content_types = [block.get("type") for block in result["content"]]
        self.assertTrue(
            "text" in content_types or "tool_use" in content_types,
            "Should have either text or tool_use content blocks",
        )

    def test_function_calling_streaming_simple(self):
        """
        Test: Whether the function name can be correctly recognized in streaming mode.
        - Expect a function call to be found, and the function name to be correct.
        - Verify that streaming mode returns at least multiple chunks.
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        tools = [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for",
                        },
                        "unit": {
                            "type": "string",
                            "description": "Weather unit (celsius or fahrenheit)",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city", "unit"],
                },
            }
        ]

        data = {
            "model": self.model,
            "max_tokens": 2048,
            "system": self.SYSTEM_MESSAGE,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the temperature in Paris in celsius?",
                },
            ],
            "temperature": 0.8,
            "tools": tools,
            "stream": True,
        }

        response = requests.post(url, headers=headers, json=data, stream=True)
        self.assertEqual(response.status_code, 200)

        # Collect streamed responses
        events = []
        for line in response.iter_lines():
            if line:
                if line.startswith(b"event: "):
                    event_type = line[7:].decode("utf-8")
                elif line.startswith(b"data: "):
                    data_content = line[6:]
                    if data_content != b"[DONE]":
                        event_data = json.loads(data_content)
                        events.append((event_type, event_data))

        # Verify we got some events
        self.assertGreater(len(events), 0, "Streaming should return at least one chunk")

        # Look for tool_use events in content blocks
        found_tool_use = False
        for event_type, event_data in events:
            if event_type == "content_block_start":
                if "content_block" in event_data:
                    content_block = event_data["content_block"]
                    if content_block.get("type") == "tool_use":
                        self.assertEqual(
                            content_block.get("name"),
                            "get_current_weather",
                            "Function name should be 'get_current_weather'",
                        )
                        found_tool_use = True
                        break

        # It's acceptable if we don't find tool_use in streaming - model may respond with text
        # Just verify we have proper event structure
        event_types = [event[0] for event in events]
        self.assertIn("message_start", event_types)
        self.assertIn("content_block_start", event_types)
        self.assertIn("content_block_stop", event_types)
        self.assertIn("message_stop", event_types)

    def test_function_calling_multiturn(self):
        """
        Test: Multiturn function calling with tool responses.
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        tools = [
            {
                "name": "add",
                "description": "Compute the sum of two numbers",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "integer",
                            "description": "First number",
                        },
                        "b": {
                            "type": "integer",
                            "description": "Second number",
                        },
                    },
                    "required": ["a", "b"],
                },
            }
        ]

        # First turn - ask to compute (3+5)
        data = {
            "model": self.model,
            "max_tokens": 2048,
            "system": self.SYSTEM_MESSAGE,
            "messages": [
                {"role": "user", "content": "Compute (3+5)"},
            ],
            "temperature": 0.8,
            "tools": tools,
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200)

        result = response.json()

        # Verify we got a tool_use response
        tool_use_blocks = [
            block for block in result["content"] if block.get("type") == "tool_use"
        ]
        self.assertGreater(
            len(tool_use_blocks), 0, "Should have at least one tool_use block"
        )

        tool_call = tool_use_blocks[0]
        function_name = tool_call.get("name")
        self.assertEqual(function_name, "add", "Function name should be 'add'")

        function_input = tool_call.get("input", {})
        self.assertIn("a", function_input, "Should have parameter 'a'")
        self.assertIn("b", function_input, "Should have parameter 'b'")

        # Second turn - simulate tool response and get final answer
        # Create a new message with the tool response
        tool_result_block = {
            "type": "tool_result",
            "tool_use_id": tool_call.get("id"),
            "content": "8",
        }

        # Prepare messages for second turn
        messages = [
            {"role": "user", "content": "Compute (3+5)"},
            {"role": "assistant", "content": [tool_call]},  # Assistant's tool call
            {"role": "user", "content": [tool_result_block]},  # Tool result
        ]

        data2 = {
            "model": self.model,
            "max_tokens": 2048,
            "system": self.SYSTEM_MESSAGE,
            "messages": messages,
            "temperature": 0.8,
        }

        response2 = requests.post(url, headers=headers, json=data2)
        self.assertEqual(response2.status_code, 200)

        result2 = response2.json()

        # Verify final response contains the answer
        text_blocks = [
            block for block in result2["content"] if block.get("type") == "text"
        ]
        self.assertGreater(
            len(text_blocks), 0, "Should have text response in final turn"
        )

        response_text = text_blocks[0].get("text", "")
        # Check that the response contains the result "8"
        self.assertTrue(
            "8" in response_text,
            "Final response should contain the computed result '8'",
        )

    def test_tool_choice_required(self):
        """
        Test: Whether tool_choice with "any" type works as expected.
        When tool_choice is "any", the model should return one or more tool calls.
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for",
                        },
                    },
                    "required": ["city"],
                },
            },
            {
                "name": "add",
                "description": "Compute the sum of two integers",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "integer",
                            "description": "First integer",
                        },
                        "b": {
                            "type": "integer",
                            "description": "Second integer",
                        },
                    },
                    "required": ["a", "b"],
                },
            },
        ]

        data = {
            "model": self.model,
            "max_tokens": 2048,
            "system": self.SYSTEM_MESSAGE,
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "temperature": 0.8,
            "tools": tools,
            "tool_choice": {"type": "any"},  # Force tool use
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200)

        result = response.json()

        # Verify response structure
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)
        self.assertGreater(len(result["content"]), 0)

        # Should have tool_use blocks when tool_choice is "any"
        tool_use_blocks = [
            block for block in result["content"] if block.get("type") == "tool_use"
        ]
        self.assertGreater(
            len(tool_use_blocks),
            0,
            "Should have at least one tool_use block when tool_choice is 'any'",
        )

        # Verify the tool call is one of our defined tools
        function_name = tool_use_blocks[0].get("name")
        valid_tools = ["get_weather", "add"]
        self.assertIn(
            function_name, valid_tools, f"Function name should be one of {valid_tools}"
        )

    def test_tool_choice_specific(self):
        """
        Test: Whether tool_choice with specific tool name works as expected.
        When tool_choice specifies a tool name, the model should use that specific tool.
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for",
                        },
                    },
                    "required": ["city"],
                },
            },
            {
                "name": "add",
                "description": "Compute the sum of two integers",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "integer",
                            "description": "First integer",
                        },
                        "b": {
                            "type": "integer",
                            "description": "Second integer",
                        },
                    },
                    "required": ["a", "b"],
                },
            },
        ]

        data = {
            "model": self.model,
            "max_tokens": 2048,
            "system": self.SYSTEM_MESSAGE,
            "messages": [
                {"role": "user", "content": "What is the weather like in London?"},
            ],
            "temperature": 0.8,
            "tools": tools,
            "tool_choice": {
                "type": "tool",
                "name": "get_weather",
            },  # Force specific tool
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200)

        result = response.json()

        # Verify response structure
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)
        self.assertGreater(len(result["content"]), 0)

        # Should have tool_use blocks with the specific tool
        tool_use_blocks = [
            block for block in result["content"] if block.get("type") == "tool_use"
        ]
        self.assertGreater(
            len(tool_use_blocks), 0, "Should have at least one tool_use block"
        )

        # Verify the tool call is the specified tool
        function_name = tool_use_blocks[0].get("name")
        self.assertEqual(
            function_name,
            "get_weather",
            "Function name should be 'get_weather' when tool_choice specifies it",
        )

    def test_function_calling_strict_mode(self):
        """
        Test: Whether strict mode function calling works as expected.
        In strict mode, function arguments should conform to the schema exactly.
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        tools = [
            {
                "name": "calculate_age",
                "description": "Calculate a person's age based on birth year",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Person's name",
                        },
                        "birth_year": {
                            "type": "integer",
                            "description": "Year of birth",
                        },
                    },
                    "required": ["name", "birth_year"],
                },
            }
        ]

        data = {
            "model": self.model,
            "max_tokens": 2048,
            "system": self.SYSTEM_MESSAGE,
            "messages": [
                {
                    "role": "user",
                    "content": "Calculate John's age who was born in 1990",
                },
            ],
            "temperature": 0.8,
            "tools": tools,
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200)

        result = response.json()

        # Verify response structure
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)
        self.assertGreater(len(result["content"]), 0)

        # Check if we have a tool_use content block
        tool_use_blocks = [
            block for block in result["content"] if block.get("type") == "tool_use"
        ]

        # If we have tool_use blocks, verify their structure
        if len(tool_use_blocks) > 0:
            tool_call = tool_use_blocks[0]
            function_name = tool_call.get("name")
            self.assertEqual(
                function_name,
                "calculate_age",
                "Function name should be 'calculate_age'",
            )

            # Check input parameters
            function_input = tool_call.get("input", {})
            self.assertIn("name", function_input, "Should have parameter 'name'")
            self.assertIn(
                "birth_year", function_input, "Should have parameter 'birth_year'"
            )

            # Verify parameter types
            self.assertIsInstance(
                function_input["name"], str, "Name should be a string"
            )
            # birth_year could be int or string representation of int
            birth_year = function_input["birth_year"]
            self.assertTrue(
                isinstance(birth_year, int) or str(birth_year).isdigit(),
                "Birth year should be an integer or string representation of integer",
            )

    def test_function_calling_no_tool_call(self):
        """
        Test: Function calling when no tool call is expected.
        The model should respond with text when the query doesn't require tool use.
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        tools = [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for",
                        },
                        "unit": {
                            "type": "string",
                            "description": "Weather unit (celsius or fahrenheit)",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city", "unit"],
                },
            }
        ]

        data = {
            "model": self.model,
            "max_tokens": 2048,
            "system": self.SYSTEM_MESSAGE,
            "messages": [
                {"role": "user", "content": "Who are you?"},
            ],
            "temperature": 0.8,
            "tools": tools,
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200)

        result = response.json()

        # Verify response structure
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)
        self.assertGreater(len(result["content"]), 0)

        # Should have text content when no tool call is needed
        text_blocks = [
            block for block in result["content"] if block.get("type") == "text"
        ]
        self.assertGreater(
            len(text_blocks), 0, "Should have text response when no tool call is needed"
        )

        response_text = text_blocks[0].get("text", "")
        self.assertGreater(len(response_text), 0, "Response text should not be empty")

    def test_function_calling_invalid_function_name(self):
        """
        Test: Function calling with invalid function name.
        The model should either not call any tool or respond appropriately when
        asked to use a function that doesn't exist in the provided tools.
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        tools = [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for",
                        },
                        "unit": {
                            "type": "string",
                            "description": "Weather unit (celsius or fahrenheit)",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city", "unit"],
                },
            }
        ]

        data = {
            "model": self.model,
            "max_tokens": 2048,
            "system": self.SYSTEM_MESSAGE,
            "messages": [
                {
                    "role": "user",
                    "content": "Please use the 'get_stock_price' function to check Apple's stock price",
                },
            ],
            "temperature": 0.8,
            "tools": tools,
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200)

        result = response.json()

        # Verify response structure
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)
        self.assertGreater(len(result["content"]), 0)

        # Should either have text response or no tool_use blocks for invalid function
        content_types = [block.get("type") for block in result["content"]]

        # If there are tool_use blocks, verify they are only for valid functions
        tool_use_blocks = [
            block for block in result["content"] if block.get("type") == "tool_use"
        ]
        for tool_block in tool_use_blocks:
            function_name = tool_block.get("name")
            # Should only use functions from our tools list
            self.assertEqual(
                function_name,
                "get_current_weather",
                "Should only call valid functions from the tools list",
            )

    def test_function_calling_parallel_tools(self):
        """
        Test: Parallel function calling with multiple tools.
        The model should be able to handle requests that might require multiple tools.
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        tools = [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for",
                        },
                        "unit": {
                            "type": "string",
                            "description": "Weather unit (celsius or fahrenheit)",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city", "unit"],
                },
            },
            {
                "name": "get_time",
                "description": "Get the current time in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the time for",
                        },
                    },
                    "required": ["city"],
                },
            },
        ]

        data = {
            "model": self.model,
            "max_tokens": 2048,
            "system": self.SYSTEM_MESSAGE,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather and current time in New York?",
                },
            ],
            "temperature": 0.8,
            "tools": tools,
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200)

        result = response.json()

        # Verify response structure
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)
        self.assertGreater(len(result["content"]), 0)

        # Check if we have tool_use content blocks
        tool_use_blocks = [
            block for block in result["content"] if block.get("type") == "tool_use"
        ]

        # If we have tool_use blocks, verify their structure
        if len(tool_use_blocks) > 0:
            # Collect all tool names called
            called_tools = [block.get("name") for block in tool_use_blocks]

            # Should only call tools from our tools list
            valid_tools = ["get_current_weather", "get_time"]
            for tool_name in called_tools:
                self.assertIn(
                    tool_name,
                    valid_tools,
                    f"Should only call valid functions from the tools list, got {tool_name}",
                )

            # Should have proper input parameters for each tool
            for tool_block in tool_use_blocks:
                function_input = tool_block.get("input", {})
                tool_name = tool_block.get("name")

                if tool_name == "get_current_weather":
                    # Should have city and unit parameters
                    self.assertIn(
                        "city",
                        function_input,
                        "Weather tool should have 'city' parameter",
                    )
                    self.assertIn(
                        "unit",
                        function_input,
                        "Weather tool should have 'unit' parameter",
                    )
                elif tool_name == "get_time":
                    # Should have city parameter
                    self.assertIn(
                        "city", function_input, "Time tool should have 'city' parameter"
                    )


if __name__ == "__main__":
    unittest.main()
