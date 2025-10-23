import json
import time
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


class TestOpenAIServerFunctionCalling(CustomTestCase):
    # NOTE: this system_message is for Llama3.2 system prompt. Without this,
    # sometimes Llama3.2 gives a different tool call format such as:
    # '<|python_tag|>{"type": "function", "function": "add", "parameters": {"a": "3", "b": "5"}}'
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
                },
            }
        ]

        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": "Compute (3+5)"},
        ]
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            tools=tools,
        )

        tool_calls = response.choices[0].message.tool_calls

        assert (
            isinstance(tool_calls, list) and len(tool_calls) > 0
        ), "tool_calls should be a non-empty list"

        function_name = tool_calls[0].function.name
        assert function_name == "add", "Function name should be 'add'"

    # This unit test is too difficult for default model. Mark it as optional unit tests so it won't trigger unless specified.
    def _test_function_calling_multiturn(self):
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
                },
            }
        ]

        messages = [{"role": "user", "content": "Compute (3+5)"}]

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            tools=tools,
        )

        tool_call = response.choices[0].message.tool_calls[0]
        function_name = tool_call.function.name
        assert function_name == "add", "Function name should be 'add'"
        function_arguments = tool_call.function.arguments
        function_arguments = json.loads(tool_call.function.arguments)
        assert function_arguments in [
            {"a": 3, "b": 5},
            {"a": "3", "b": "5"},
        ], f"Unexpected function arguments: {function_arguments}"

        messages.append(response.choices[0].message)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "8",
                "name": function_name,
            }
        )

        final_response = client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            tools=tools,
        )

        assert (
            "8" in final_response.choices[0].message.content
        ), "tool_call response should have the sum 8 in the content"

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

        finish_reason = chunks[-1].choices[0].finish_reason
        self.assertEqual(
            finish_reason,
            "tool_calls",
            "Final response of function calling should have finish_reason 'tool_calls'",
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
                    "strict": True,  # Llama-3.2-1B is flaky in tool call. It won't always respond with parameters unless we set strict.
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
                # Record the function name on first occurrence
                function_name = tool_call.function.name or function_name
                # In case of multiple chunks, JSON fragments may need to be concatenated
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
            }
        ]

        messages = [
            {"role": "user", "content": "Please compute 5 - 7, using your tool."}
        ]
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
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

    def test_function_call_required(self):
        """
        Test: Whether tool_choice: "required" works as expected
        - When tool_choice == "required", the model should return one or more tool_calls.
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

    def test_function_call_specific(self):
        """
        Test: Whether tool_choice: ToolChoice works as expected
        - When tool_choice is a specific ToolChoice, the model should return one or more tool_calls.
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
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls, "No tool_calls in the response")
        function_name = tool_calls[0].function.name
        arguments = tool_calls[0].function.arguments
        args_obj = json.loads(arguments)

        self.assertEqual(
            function_name, "get_weather", "Function name should be 'get_weather'"
        )
        self.assertIn("city", args_obj, "Function arguments should have 'city'")

    def test_streaming_multiple_choices_finish_reason(self):
        """
        Test: Verify that each choice gets its own finish_reason chunk in streaming mode with n > 1.
        This tests the fix for the bug where only the last index got a finish_reason chunk.
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
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        messages = [
            {"role": "user", "content": "What is the weather like in Los Angeles?"}
        ]

        # Request with n=2 to get multiple choices
        response_stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.8,
            stream=True,
            tools=tools,
            tool_choice="required",  # Force tool calls
            n=2,  # Multiple choices
        )

        chunks = list(response_stream)

        # Track finish_reason chunks for each index
        finish_reason_chunks = {}
        for chunk in chunks:
            if chunk.choices:
                for choice in chunk.choices:
                    if choice.finish_reason is not None:
                        index = choice.index
                        if index not in finish_reason_chunks:
                            finish_reason_chunks[index] = []
                        finish_reason_chunks[index].append(choice.finish_reason)

        # Verify we got finish_reason chunks for both indices
        self.assertEqual(
            len(finish_reason_chunks),
            2,
            f"Expected finish_reason chunks for 2 indices, got {len(finish_reason_chunks)}",
        )

        # Verify both index 0 and 1 have finish_reason
        self.assertIn(
            0, finish_reason_chunks, "Missing finish_reason chunk for index 0"
        )
        self.assertIn(
            1, finish_reason_chunks, "Missing finish_reason chunk for index 1"
        )

        # Verify the finish_reason is "tool_calls" since we forced tool calls
        for index, reasons in finish_reason_chunks.items():
            self.assertEqual(
                reasons[-1],  # Last finish_reason for this index
                "tool_calls",
                f"Expected finish_reason 'tool_calls' for index {index}, got {reasons[-1]}",
            )

    def test_function_calling_streaming_no_tool_call(self):
        """
        Test: Whether the finish_reason is stop in streaming mode when no tool call is given.
        - Expect no function call to be found.
        - Verify that finish_reason is stop
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

        messages = [{"role": "user", "content": "Who are you?"}]

        response_stream = client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=True,
            tools=tools,
            tool_choice="none",
        )

        chunks = list(response_stream)
        self.assertTrue(len(chunks) > 0, "Streaming should return at least one chunk")

        found_tool_call = False
        for chunk in chunks:
            choice = chunk.choices[0]
            # Check whether the current chunk contains tool_calls
            found_tool_call = choice.delta.tool_calls is not None

        self.assertFalse(
            found_tool_call,
            "Shouldn't have any tool_call in the streaming chunks",
        )

        finish_reason = chunks[-1].choices[0].finish_reason
        self.assertEqual(
            finish_reason,
            "stop",
            "Final response of no function calling should have finish_reason 'stop'",
        )

    def test_streaming_multiple_choices_without_tools(self):
        """
        Test: Verify that each choice gets its own finish_reason chunk without tool calls.
        This tests the fix for regular content streaming with multiple choices.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        messages = [{"role": "user", "content": "Say hello in one word."}]

        # Request with n=2 to get multiple choices, no tools
        response_stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            stream=True,
            max_tokens=10,  # Keep it short
            n=2,  # Multiple choices
        )

        chunks = list(response_stream)

        # Track finish_reason chunks for each index
        finish_reason_chunks = {}
        for chunk in chunks:
            if chunk.choices:
                for choice in chunk.choices:
                    if choice.finish_reason is not None:
                        index = choice.index
                        if index not in finish_reason_chunks:
                            finish_reason_chunks[index] = []
                        finish_reason_chunks[index].append(choice.finish_reason)

        # Verify we got finish_reason chunks for both indices
        self.assertEqual(
            len(finish_reason_chunks),
            2,
            f"Expected finish_reason chunks for 2 indices, got {len(finish_reason_chunks)}",
        )

        # Verify both index 0 and 1 have finish_reason
        self.assertIn(
            0, finish_reason_chunks, "Missing finish_reason chunk for index 0"
        )
        self.assertIn(
            1, finish_reason_chunks, "Missing finish_reason chunk for index 1"
        )

        # Verify the finish_reason is "stop" (regular completion)
        for index, reasons in finish_reason_chunks.items():
            self.assertIn(
                reasons[-1],
                ["stop", "length"],  # Could be either depending on how model responds
                f"Expected finish_reason 'stop' or 'length' for index {index}, got {reasons[-1]}",
            )


class TestOpenAIPythonicFunctionCalling(CustomTestCase):
    PYTHONIC_TOOLS = [
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
                        }
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

    PYTHONIC_MESSAGES = [
        {
            "role": "system",
            "content": (
                "You are a travel assistant. "
                "When asked to call functions, ALWAYS respond ONLY with a python list of function calls, "
                "using this format: [func_name1(param1=value1, param2=value2), func_name2(param=value)]. "
                "Do NOT use JSON, do NOT use variables, do NOT use any other format. "
                "Here is an example:\n"
                '[get_weather(location="Paris"), get_tourist_attractions(city="Paris")]'
            ),
        },
        {
            "role": "user",
            "content": (
                "I'm planning a trip to Tokyo next week. What's the weather like and what are some top tourist attractions? "
                "Propose parallel tool calls at once, using the python list of function calls format as shown above."
            ),
        },
    ]

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
                "pythonic",
            ],
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_pythonic_tool_call_prompt(self):
        """
        Test: Explicit prompt for pythonic tool call format without chat template.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=self.PYTHONIC_MESSAGES,
            tools=self.PYTHONIC_TOOLS,
            temperature=0.1,
            stream=False,
        )
        tool_calls = response.choices[0].message.tool_calls
        self.assertIsInstance(tool_calls, list, "No tool_calls found")
        self.assertGreaterEqual(len(tool_calls), 1)
        names = [tc.function.name for tc in tool_calls]
        self.assertTrue(
            "get_weather" in names or "get_tourist_attractions" in names,
            f"Function name '{names}' should container either 'get_weather' or 'get_tourist_attractions'",
        )

    def test_pythonic_tool_call_streaming(self):
        """
        Test: Streaming pythonic tool call format; assert tool_call index is present.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response_stream = client.chat.completions.create(
            model=self.model,
            messages=self.PYTHONIC_MESSAGES,
            tools=self.PYTHONIC_TOOLS,
            temperature=0.1,
            stream=True,
        )
        found_tool_calls = False
        found_index = False
        found_names = set()
        for chunk in response_stream:
            choice = chunk.choices[0]
            if getattr(choice.delta, "tool_calls", None):
                found_tool_calls = True
                tool_call = choice.delta.tool_calls[0]
                if hasattr(tool_call, "index") or (
                    isinstance(tool_call, dict) and "index" in tool_call
                ):
                    found_index = True
                found_names.add(str(tool_call.function.name))

        self.assertTrue(found_tool_calls, "No tool_calls found in streaming response")
        self.assertTrue(found_index, "No index field found in any streamed tool_call")
        self.assertTrue(
            "get_weather" in found_names or "get_tourist_attractions" in found_names,
            f"Function name '{found_names}' should container either 'get_weather' or 'get_tourist_attractions'",
        )


# Skip for ci test
# class TestGLM45ServerFunctionCalling(TestOpenAIServerFunctionCalling):
#     @classmethod
#     def setUpClass(cls):
#         # Replace with the model name needed for testing; if not required, reuse DEFAULT_SMALL_MODEL_NAME_FOR_TEST
#         cls.model = "THUDM/GLM-4.5"
#         cls.base_url = DEFAULT_URL_FOR_TEST
#         cls.api_key = "sk-123456"

#         # Start the local OpenAI Server. If necessary, you can add other parameters such as --enable-tools.
#         cls.process = popen_launch_server(
#             cls.model,
#             cls.base_url,
#             timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
#             api_key=cls.api_key,
#             other_args=[
#                 # If your server needs extra parameters to test function calling, please add them here.
#                 "--tool-call-parser",
#                 "glm45",
#                 "--reasoning-parser",
#                 "glm45",
#                 "--tp-size",
#                 "8"
#             ],
#         )
#         cls.base_url += "/v1"
#         cls.tokenizer = get_tokenizer(cls.model)

#     # This test is too difficult for GLM4-moe. Skip it from the UT
#     def test_function_call_required(self):
#         pass

#     def test_function_calling_multiturn(self):
#         self._test_function_calling_multiturn()


if __name__ == "__main__":
    unittest.main()
