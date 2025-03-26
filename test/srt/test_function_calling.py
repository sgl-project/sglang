import json
import time
import unittest

import openai

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestOpenAIServerFunctionCalling(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Replace with the model name needed for testing; if not required, reuse DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Start the local OpenAI Server. If necessary, you can add other parameters such as --enable-tools.
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                # If your server needs extra parameters to test function calling, please add them here.
                "--tool-call-parser",
                "llama3",
            ],
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_function_calling_format(self):
        """
        Test: Whether the function call format returned by the AI is correct.
        When returning a tool call, message.content should be None, and tool_calls should be a list.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Compute the sum of two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "int",
                                "description": "A number",
                            },
                            "b": {
                                "type": "int",
                                "description": "A number",
                            },
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]

        messages = [{"role": "user", "content": "Compute (3+5)"}]
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            tools=tools,
        )

        content = response.choices[0].message.content
        tool_calls = response.choices[0].message.tool_calls

        assert content is None, (
            "When function call is successful, message.content should be None, "
            f"but got: {content}"
        )
        assert (
            isinstance(tool_calls, list) and len(tool_calls) > 0
        ), "tool_calls should be a non-empty list"

        function_name = tool_calls[0].function.name
        assert function_name == "add", "Function name should be 'add'"

    def test_function_calling_streaming_simple(self):
        """
        Test: Whether the function name can be correctly recognized in streaming mode.
        - Expect a function call to be found, and the function name to be correct.
        - Verify that streaming mode returns at least multiple chunks.
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

        messages = [{"role": "user", "content": "What is the temperature in Paris?"}]

        response_stream = client.chat.completions.create(
            model=self.model,
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
            # Check whether the current chunk contains tool_calls
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

    def test_function_calling_streaming_args_parsing(self):
        """
        Test: Whether the function call arguments returned in streaming mode can be correctly concatenated into valid JSON.
        - The user request requires multiple parameters.
        - AI may return the arguments in chunks that need to be concatenated.
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
                                "type": "int",
                                "description": "First integer",
                            },
                            "b": {
                                "type": "int",
                                "description": "Second integer",
                            },
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]

        messages = [
            {"role": "user", "content": "Please sum 5 and 7, just call the function."}
        ]

        response_stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.9,
            top_p=0.9,
            stream=True,
            tools=tools,
        )

        argument_fragments = []
        function_name = None
        for chunk in response_stream:
            choice = chunk.choices[0]
            if choice.delta.tool_calls:
                tool_call = choice.delta.tool_calls[0]
                # Record the function name on first occurrence
                function_name = tool_call.function.name or function_name
                # In case of multiple chunks, JSON fragments may need to be concatenated
                if tool_call.function.arguments:
                    argument_fragments.append(tool_call.function.arguments)

        self.assertEqual(function_name, "add", "Function name should be 'add'")
        joined_args = "".join(argument_fragments)
        self.assertTrue(
            len(joined_args) > 0,
            "No parameter fragments were returned in the function call",
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

    def test_function_call_strict(self):
        """
        Test: Whether the strict mode of function calling works as expected.
        - When strict mode is enabled, the AI should not return a function call if the function name is not recognized.
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
                                "type": "int",
                                "description": "First integer",
                            },
                            "int_b": {
                                "type": "int",
                                "description": "Second integer",
                            },
                        },
                        "required": ["int_a", "int_b"],
                    },
                    "strict": True,
                },
            }
        ]

        messages = [
            {"role": "user", "content": "Please compute 5 - 7, using your tool."}
        ]
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            tools=tools,
        )

        tool_calls = response.choices[0].message.tool_calls
        function_name = tool_calls[0].function.name
        arguments = tool_calls[0].function.arguments
        args_obj = json.loads(arguments)

        self.assertEqual(function_name, "sub", "Function name should be 'sub'")
        self.assertEqual(str(args_obj["int_a"]), "5", "Parameter int_a should be 5")
        self.assertEqual(str(args_obj["int_b"]), "7", "Parameter int_b should be 7")


if __name__ == "__main__":
    unittest.main()
