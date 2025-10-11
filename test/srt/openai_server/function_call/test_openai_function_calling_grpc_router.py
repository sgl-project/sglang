"""
Test script for OpenAI function calling through router with gRPC worker.

This test suite validates function calling through the router → gRPC worker pathway
to catch potential issues at streaming boundaries and transport conversions
(e.g., partial JSON frames, interleaving, retries, backpressure).

Focuses on critical streaming and tool choice tests.
"""

import json
import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_router_with_grpc_worker,
)


class TestOpenAIServerFunctionCallingGRPCRouter(CustomTestCase):
    """Test function calling through router with gRPC worker.

    This class tests critical function calling scenarios through the router → gRPC
    worker pathway to validate streaming boundaries and transport conversions.
    """

    # System message for Llama3.2
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

        # Launch router with single gRPC worker
        # Note: This can take 5+ minutes due to router connection time
        print("Launching router with gRPC worker (may take 5+ minutes)...")
        cls.router_process, cls.worker_process, cls.worker_port = (
            popen_launch_router_with_grpc_worker(
                cls.model,
                cls.base_url,
                timeout=600,  # 10 minutes for router + worker startup
                api_key=cls.api_key,
                worker_other_args=[
                    "--tool-call-parser",
                    "llama3",
                ],
                router_other_args=[],
            )
        )

        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        # Clean up both router and worker processes
        kill_process_tree(cls.router_process.pid)
        kill_process_tree(cls.worker_process.pid)

    def test_function_calling_streaming_simple(self):
        """
        Test: Whether the function name can be correctly recognized in streaming mode through gRPC.
        This is critical for catching transport layer issues with partial chunks.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
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
            }
        ]

        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": "What is the temperature in Paris in celsius??",
            },
        ]

        response_stream = client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=True,
            tools=tools,
        )

        chunks = list(response_stream)
        self.assertTrue(len(chunks) > 0, "Streaming should return at least one chunk")

        found_function_name = False
        for chunk in chunks:
            choice = chunk.choices[0]
            if choice.delta.tool_calls:
                tool_call = choice.delta.tool_calls[0]
                if tool_call.function.name:
                    self.assertEqual(
                        tool_call.function.name,
                        "get_current_weather",
                        "Function name should be 'get_current_weather'",
                    )
                    found_function_name = True
                    break

        self.assertTrue(
            found_function_name,
            "Target function name 'get_current_weather' was not found in the streaming chunks",
        )

        finish_reason = chunks[-1].choices[0].finish_reason
        self.assertEqual(
            finish_reason,
            "tool_calls",
            "Final response of function calling should have finish_reason 'tool_calls'",
        )

    def test_function_calling_streaming_args_parsing(self):
        """
        Test: Whether function call arguments can be correctly parsed through gRPC streaming.
        This validates that JSON fragments are correctly assembled across transport boundaries.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Compute the sum of two integers",
                    "parameters": {
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
                    "strict": True,
                },
            }
        ]

        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": "Please sum 5 and 7, just call the function."},
        ]

        response_stream = client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=messages,
            temperature=0.9,
            top_p=0.9,
            stream=True,
            tools=tools,
        )

        argument_fragments = []
        chunks = list(response_stream)
        function_name = None
        for chunk in chunks:
            choice = chunk.choices[0]
            if choice.delta.tool_calls:
                tool_call = choice.delta.tool_calls[0]
                function_name = tool_call.function.name or function_name
                if tool_call.function.arguments is not None:
                    argument_fragments.append(tool_call.function.arguments)

        self.assertEqual(function_name, "add", "Function name should be 'add'")
        joined_args = "".join(argument_fragments)
        self.assertTrue(
            len(joined_args) > 0,
            "No parameter fragments were returned in the function call",
        )

        finish_reason = chunks[-1].choices[0].finish_reason
        self.assertEqual(
            finish_reason,
            "tool_calls",
            "Final response of function calling should have finish_reason 'tool_calls'",
        )

        # Check whether the concatenated JSON is valid
        try:
            args_obj = json.loads(joined_args)
        except json.JSONDecodeError:
            self.fail(
                "The concatenated tool call arguments are not valid JSON, parsing failed"
            )

        self.assertIn("a", args_obj, "Missing parameter 'a'")
        self.assertIn("b", args_obj, "Missing parameter 'b'")
        self.assertEqual(str(args_obj["a"]), "5", "Parameter a should be 5")
        self.assertEqual(str(args_obj["b"]), "7", "Parameter b should be 7")

    def test_function_call_required(self):
        """
        Test: Whether tool_choice: "required" works through gRPC router
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "sub",
                    "description": "Compute the difference of two integers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "int_a": {
                                "type": "integer",
                                "description": "First integer",
                            },
                            "int_b": {
                                "type": "integer",
                                "description": "Second integer",
                            },
                        },
                        "required": ["int_a", "int_b"],
                    },
                    "strict": True,
                },
            },
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
                            }
                        },
                        "required": ["city"],
                    },
                },
            },
        ]

        messages = [{"role": "user", "content": "What is the capital of France?"}]
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            tools=tools,
            tool_choice="required",
        )

        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls, "No tool_calls in the response")
        function_name = tool_calls[0].function.name
        arguments = tool_calls[0].function.arguments
        args_obj = json.loads(arguments)

        self.assertEqual(
            function_name,
            "get_weather",
            f"Function name should be 'get_weather', got: {function_name}",
        )
        self.assertIn(
            "city", args_obj, f"Function arguments should have 'city', got: {args_obj}"
        )

        # Make the test more robust by checking type and accepting valid responses
        city_value = args_obj["city"]
        self.assertIsInstance(
            city_value,
            str,
            f"Parameter city should be a string, got: {type(city_value)}",
        )
        self.assertTrue(
            "Paris" in city_value or "France" in city_value,
            f"Parameter city should contain either 'Paris' or 'France', got: {city_value}",
        )


if __name__ == "__main__":
    unittest.main()
