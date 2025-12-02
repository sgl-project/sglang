"""
Test script for tool_choice functionality in SGLang
Tests: required, auto, and specific function choices in both streaming and non-streaming modes

# To run the tests, use the following command:
#
# python3 -m unittest openai_server.function_call.test_tool_choice
"""

import json
import sys
import unittest
from pathlib import Path

import openai

_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR.parent))
from fixtures import popen_launch_workers_and_router
from util import (
    DEFAULT_MISTRAL_FUNCTION_CALLING_MODEL_PATH,
    DEFAULT_QWEN_FUNCTION_CALLING_MODEL_PATH,
    DEFAULT_SMALL_MODEL_PATH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_tokenizer,
    kill_process_tree,
)


class TestToolChoiceLlama32(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        # CHANGE: Launch gRPC router with integrated workers (single command)
        # Mark flaky tests for this model
        cls.flaky_tests = {
            "test_multi_tool_scenario_auto",
            "test_multi_tool_scenario_required",
        }

        # Use a model that supports function calling
        cls.model = DEFAULT_SMALL_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Start the local OpenAI Server with tool calling support
        cls.cluster = popen_launch_workers_and_router(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            router_args=[
                "--tool-call-parser",
                "llama",  # Default parser for the test model
            ],
            num_workers=1,
            tp_size=2,
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        # Cleanup router and workers
        kill_process_tree(cls.cluster["router"].pid)
        for worker in cls.cluster.get("workers", []):
            kill_process_tree(worker.pid)

    def setUp(self):
        self.client = openai.Client(base_url=self.base_url, api_key=self.api_key)
        self.model_name = self.client.models.list().data[0].id

    def _is_flaky_test(self):
        """Check if the current test is marked as flaky for this class"""
        return (
            hasattr(self.__class__, "flaky_tests")
            and self._testMethodName in self.__class__.flaky_tests
        )

    def get_test_tools(self):
        """Get the test tools for function calling"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "use this to get latest weather information for a city given its name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "name of the city to get weather for",
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
                    "name": "get_pokemon_info",
                    "description": "get detailed information about a pokemon given its name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "name of the pokemon to get info for",
                            }
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "make_next_step_decision",
                    "description": "You will be given a trace of thinking process in the following format.\n\nQuestion: the input question you must answer\nTOOL: think about what to do, and choose a tool to use ONLY IF there are defined tools. \n  You should never call the same tool with the same input twice in a row.\n  If the previous conversation history already contains the information that can be retrieved from the tool, you should not call the tool again.\nOBSERVATION: the result of the tool call, NEVER include this in your response, this information will be provided\n... (this TOOL/OBSERVATION can repeat N times)\nANSWER: If you know the answer to the original question, require for more information,\n  or you don't know the answer and there are no defined tools or all available tools are not helpful, respond with the answer without mentioning anything else.\n  If the previous conversation history already contains the answer, respond with the answer right away.\n\n  If no tools are configured, naturally mention this limitation while still being helpful. Briefly note that adding tools in the agent configuration would expand capabilities.\n\nYour task is to respond with the next step to take, based on the traces, \nor answer the question if you have enough information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "decision": {
                                "type": "string",
                                "description": 'The next step to take, it must be either "TOOL" or "ANSWER". If the previous conversation history already contains the information that can be retrieved from the tool, you should not call the tool again. If there are no defined tools, you should not return "TOOL" in your response.',
                            },
                            "content": {
                                "type": "string",
                                "description": 'The content of the next step. If the decision is "TOOL", this should be a short and concise reasoning of why you chose the tool, MUST include the tool name. If the decision is "ANSWER", this should be the answer to the question. If no tools are available, integrate this limitation conversationally without sounding scripted.',
                            },
                        },
                        "required": ["decision", "content"],
                    },
                },
            },
        ]

    def get_test_messages(self):
        """Get test messages that should trigger tool usage"""
        return [
            {
                "role": "user",
                "content": "Answer the following questions as best you can:\n\nYou will be given a trace of thinking process in the following format.\n\nQuestion: the input question you must answer\nTOOL: think about what to do, and choose a tool to use ONLY IF there are defined tools\nOBSERVATION: the result of the tool call or the observation of the current task, NEVER include this in your response, this information will be provided\n... (this TOOL/OBSERVATION can repeat N times)\nANSWER: If you know the answer to the original question, require for more information, \nif the previous conversation history already contains the answer, \nor you don't know the answer and there are no defined tools or all available tools are not helpful, respond with the answer without mentioning anything else.\nYou may use light Markdown formatting to improve clarity (e.g. lists, **bold**, *italics*), but keep it minimal and unobtrusive.\n\nYour task is to respond with the next step to take, based on the traces, \nor answer the question if you have enough information.\n\nQuestion: what is the weather in top 5 populated cities in the US in celsius?\n\nTraces:\n\n\nThese are some additional instructions that you should follow:",
            }
        ]

    def get_travel_tools(self):
        """Get tools for travel assistant scenario that should trigger multiple tool calls"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The name of the city or location.",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_tourist_attractions",
                    "description": "Get a list of top tourist attractions for a given city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The name of the city to find attractions for.",
                            }
                        },
                        "required": ["city"],
                    },
                },
            },
        ]

    def get_travel_messages(self):
        """Get travel assistant messages that should trigger multiple tool calls"""
        return [
            {
                "content": "You are a travel assistant providing real-time weather updates and top tourist attractions.",
                "role": "system",
            },
            {
                "content": "I'm planning a trip to Tokyo next week. What's the weather like? What are the most amazing sights?",
                "role": "user",
            },
        ]

    def test_tool_choice_auto_non_streaming(self):
        """Test tool_choice='auto' in non-streaming mode"""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2048,
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        self.assertIsNotNone(response.choices[0].message)
        # With auto, tool calls are optional

    def test_tool_choice_auto_streaming(self):
        """Test tool_choice='auto' in streaming mode"""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2048,
            tools=tools,
            tool_choice="auto",
            stream=True,
        )

        # Collect streaming response
        content_chunks = []
        tool_call_chunks = []

        for chunk in response:
            if chunk.choices[0].delta.content:
                content_chunks.append(chunk.choices[0].delta.content)
            elif chunk.choices[0].delta.tool_calls:
                tool_call_chunks.extend(chunk.choices[0].delta.tool_calls)

        # Should complete without errors
        self.assertIsInstance(content_chunks, list)
        self.assertIsInstance(tool_call_chunks, list)

    def test_tool_choice_required_non_streaming(self):
        """Test tool_choice='required' in non-streaming mode"""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2048,
            temperature=0.2,
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        # With required, we should get tool calls
        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)

    def test_tool_choice_required_streaming(self):
        """Test tool_choice='required' in streaming mode"""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2048,
            tools=tools,
            tool_choice="required",
            stream=True,
        )

        # Collect streaming response
        tool_call_chunks = []

        for chunk in response:
            if chunk.choices[0].delta.tool_calls:
                tool_call_chunks.extend(chunk.choices[0].delta.tool_calls)

        # With required, we should get tool call chunks
        self.assertGreater(len(tool_call_chunks), 0)

    def test_tool_choice_specific_function_non_streaming(self):
        """Test tool_choice with specific function in non-streaming mode"""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        tool_choice = {"type": "function", "function": {"name": "get_weather"}}

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2048,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
        )

        # Should call the specific function
        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        # Our messages ask the top 5 populated cities in the US, so the model could get 5 tool calls
        self.assertGreaterEqual(len(tool_calls), 1)
        for tool_call in tool_calls:
            self.assertEqual(tool_call.function.name, "get_weather")

    def test_tool_choice_specific_function_streaming(self):
        """Test tool_choice with specific function in streaming mode"""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        tool_choice = {"type": "function", "function": {"name": "get_weather"}}

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2048,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )

        # Collect streaming response
        tool_call_chunks = []

        for chunk in response:
            if chunk.choices[0].delta.tool_calls:
                tool_call_chunks.extend(chunk.choices[0].delta.tool_calls)

        # Should get tool call chunks for the specific function
        self.assertGreater(len(tool_call_chunks), 0)

        # Find function name in chunks
        found_name = None
        for chunk in tool_call_chunks:
            if chunk.function and chunk.function.name:
                found_name = chunk.function.name
                break

        self.assertEqual(found_name, "get_weather")

    def test_required_streaming_arguments_chunks_json(self):
        """In streaming required mode, complete tool call arguments should be valid JSON when all chunks are combined"""
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

        # Collect all tool call chunks and reconstruct complete tool calls
        tool_calls_by_index = {}
        for chunk in response:
            if chunk.choices[0].delta.tool_calls:
                for tool_call_delta in chunk.choices[0].delta.tool_calls:
                    tool_index = tool_call_delta.index

                    # Initialize tool call if not seen before
                    if tool_index not in tool_calls_by_index:
                        tool_calls_by_index[tool_index] = {
                            "id": tool_call_delta.id,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }

                    # Update function name if present (first chunk)
                    if tool_call_delta.function and tool_call_delta.function.name:
                        tool_calls_by_index[tool_index]["function"][
                            "name"
                        ] = tool_call_delta.function.name

                    # Accumulate arguments (all chunks)
                    if tool_call_delta.function and tool_call_delta.function.arguments:
                        tool_calls_by_index[tool_index]["function"][
                            "arguments"
                        ] += tool_call_delta.function.arguments

        self.assertGreater(len(tool_calls_by_index), 0)

        # Validate that complete tool calls have valid JSON arguments
        for tool_call in tool_calls_by_index.values():
            self.assertIsNotNone(tool_call["function"]["name"])
            self.assertIsNotNone(tool_call["function"]["arguments"])

            # The complete arguments should be valid JSON
            try:
                args = json.loads(tool_call["function"]["arguments"])
                self.assertIsInstance(args, dict)
            except json.JSONDecodeError:
                self.fail(
                    f"Invalid JSON in complete tool call arguments: {tool_call['function']['arguments']}"
                )

    def test_complex_parameters_required_non_streaming(self):
        """Validate complex nested parameter schemas in non-streaming required mode"""
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

        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)

        for tool_call in tool_calls:
            self.assertEqual(tool_call.function.name, "analyze_data")
            try:
                args = json.loads(tool_call.function.arguments)
                self.assertIsInstance(args, dict)
                self.assertIn("data", args)
                self.assertIsInstance(args["data"], dict)
            except json.JSONDecodeError:
                self.fail(
                    f"Invalid JSON in complex tool call arguments: {tool_call.function.arguments}"
                )

    def test_multi_tool_scenario_auto(self):
        """Test multi-tool scenario with tool_choice='auto'"""
        tools = self.get_travel_tools()
        messages = self.get_travel_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2048,
            temperature=0.2,
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        # Should complete without errors
        self.assertIsNotNone(response.choices[0].message)

        tool_calls = response.choices[0].message.tool_calls
        expected_functions = {"get_weather", "get_tourist_attractions"}

        if self._is_flaky_test():
            # For flaky tests, just verify all called functions are available tools
            if tool_calls:
                available_names = [tool["function"]["name"] for tool in tools]
                for call in tool_calls:
                    self.assertIn(call.function.name, available_names)
        else:
            # For non-flaky tests, enforce strict requirements
            self.assertIsNotNone(tool_calls, "Expected tool calls but got none")
            self.assertEqual(
                len(tool_calls), 2, f"Expected 2 tool calls, got {len(tool_calls)}"
            )

            called_functions = {call.function.name for call in tool_calls}
            self.assertEqual(
                called_functions,
                expected_functions,
                f"Expected functions {expected_functions}, got {called_functions}",
            )

    def test_multi_tool_scenario_required(self):
        """Test multi-tool scenario with tool_choice='required'"""
        tools = self.get_travel_tools()
        messages = self.get_travel_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2048,
            temperature=0.2,
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        # With required, we should get at least one tool call
        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)

        # Verify all called functions are available tools
        available_names = [tool["function"]["name"] for tool in tools]
        expected_functions = {"get_weather", "get_tourist_attractions"}

        for tool_call in tool_calls:
            self.assertIsNotNone(tool_call.function.name)
            self.assertIsNotNone(tool_call.function.arguments)

        if self._is_flaky_test():
            # For flaky tests, just ensure basic functionality works
            self.assertGreater(
                len(tool_calls),
                0,
                f"Expected at least 1 tool call, got {len(tool_calls)}",
            )
            for call in tool_calls:
                self.assertIn(call.function.name, available_names)
        else:
            # For non-flaky tests, enforce strict requirements
            self.assertEqual(
                len(tool_calls), 2, f"Expected 2 tool calls, got {len(tool_calls)}"
            )

            called_functions = {call.function.name for call in tool_calls}
            self.assertEqual(
                called_functions,
                expected_functions,
                f"Expected functions {expected_functions}, got {called_functions}",
            )

    def test_error_handling_invalid_tool_choice(self):
        """Test error handling for invalid tool_choice"""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        # Test with invalid function name
        tool_choice = {"type": "function", "function": {"name": "nonexistent_function"}}

        # Expect a 400 BadRequestError to be raised for invalid tool_choice
        with self.assertRaises(openai.BadRequestError) as context:
            self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=2048,
                tools=tools,
                tool_choice=tool_choice,
                stream=False,
            )

        # Verify the error message contains the expected text
        self.assertIn(
            "function 'nonexistent_function' not found in",
            str(context.exception),
        )

    def test_invalid_tool_missing_name(self):
        """Test what happens when user doesn't provide a tool name in request"""
        # Test with malformed JSON in tool parameters - missing required "name" field
        invalid_tools = [
            {
                "type": "function",
                "function": {
                    # Missing required "name" field
                    "description": "Test function with invalid schema",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_field": {
                                "type": "string",
                                "description": "Test field",
                            }
                        },
                        "required": ["test_field"],
                    },
                },
            }
        ]

        messages = [
            {
                "role": "user",
                "content": "Test the function",
            }
        ]

        # Should raise BadRequestError due to missing required 'name' field
        with self.assertRaises(openai.BadRequestError) as context:
            self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=100,
                temperature=0.1,
                tools=invalid_tools,
                tool_choice="required",
                stream=False,
            )

        # Verify the error message indicates missing name field
        error_msg = str(context.exception).lower()
        self.assertIn("name", error_msg)

    def test_conflicting_defs_required_tool_choice(self):
        """Test that conflicting $defs with required tool_choice returns 400 error"""
        conflicting_tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool1",
                    "description": "Tool 1 with conflicting $defs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {"$ref": "#/$defs/DataType"},
                        },
                        "required": ["data"],
                        "$defs": {
                            "DataType": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                                "required": ["value"],
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool2",
                    "description": "Tool 2 with conflicting $defs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {"$ref": "#/$defs/DataType"},
                        },
                        "required": ["data"],
                        "$defs": {
                            "DataType": {  # Different definition for DataType
                                "type": "object",
                                "properties": {"value": {"type": "number"}},
                                "required": ["value"],
                            },
                        },
                    },
                },
            },
        ]

        messages = [
            {
                "role": "user",
                "content": "Test the conflicting tools",
            }
        ]

        # Should raise BadRequestError due to conflicting $defs
        with self.assertRaises(openai.BadRequestError) as context:
            self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=100,
                temperature=0.1,
                tools=conflicting_tools,
                tool_choice="required",
                stream=False,
            )

        # Verify the error message indicates conflicting tool definitions
        error_msg = str(context.exception).lower()
        self.assertIn("invalid tool configuration", error_msg)
        self.assertIn("not supported", error_msg)


class TestToolChoiceQwen25(TestToolChoiceLlama32):
    """Test tool_choice functionality with Qwen2.5 model"""

    @classmethod
    def setUpClass(cls):
        # CHANGE: Launch gRPC router with integrated workers (single command)
        cls.flaky_tests = {}

        cls.model = DEFAULT_QWEN_FUNCTION_CALLING_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        cls.cluster = popen_launch_workers_and_router(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            router_args=[
                "--tool-call-parser",
                "qwen",
            ],
            num_workers=1,
            tp_size=2,
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)


class TestToolChoiceMistral(TestToolChoiceLlama32):
    """Test tool_choice functionality with Mistral model"""

    @classmethod
    def setUpClass(cls):
        # CHANGE: Launch gRPC router with integrated workers (single command)
        # Mark flaky tests for this model
        cls.flaky_tests = {
            "test_multi_tool_scenario_auto",
            "test_multi_tool_scenario_required",
        }

        cls.model = DEFAULT_MISTRAL_FUNCTION_CALLING_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        cls.cluster = popen_launch_workers_and_router(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            router_args=[
                "--tool-call-parser",
                "mistral",
            ],
            num_workers=1,
            tp_size=2,
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @unittest.skip("Fails due to whitespace issue with Mistral - skipping")
    def test_complex_parameters_required_non_streaming(self):
        """Validate complex nested parameter schemas in non-streaming required mode"""
        super().test_complex_parameters_required_non_streaming()


if __name__ == "__main__":
    unittest.main()
