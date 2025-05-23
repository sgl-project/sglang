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
                if tool_call.function.arguments is not None:
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

    def test_double_dumping_detection(self):
        """
        Test: Detect potential double dumping issues in function call arguments.
        - Verify that arguments are valid JSON and not double-encoded strings.
        - Check for common double dumping patterns like escaped quotes.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_location",
                    "description": "Search for information about a location with complex nested data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location name",
                            },
                            "search_options": {
                                "type": "object",
                                "properties": {
                                    "include_weather": {"type": "boolean"},
                                    "include_attractions": {"type": "boolean"},
                                    "radius_km": {"type": "integer"},
                                },
                                "required": ["include_weather"],
                            },
                            "metadata": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string"},
                                    "timestamp": {"type": "string"},
                                },
                            },
                        },
                        "required": ["location", "search_options"],
                    },
                },
            }
        ]

        messages = [
            {
                "role": "user",
                "content": "Search for information about Tokyo with weather data, attractions within 5km, from source 'travel_api' at timestamp '2024-01-01T00:00:00Z'",
            }
        ]

        # Test non-streaming response
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            top_p=0.9,
            stream=False,
            tools=tools,
        )

        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            arguments_str = tool_call.function.arguments

            # Check 1: Arguments should be valid JSON
            try:
                args_obj = json.loads(arguments_str)
            except json.JSONDecodeError as e:
                self.fail(
                    f"Function arguments are not valid JSON: {e}. Arguments: {arguments_str}"
                )

            # Check 2: Arguments should not be a string representation of JSON (double dumping)
            self.assertIsInstance(
                args_obj,
                dict,
                f"Arguments should be parsed as dict, not string. Got: {type(args_obj)}. Value: {args_obj}",
            )

            # Check 3: Nested objects should also be properly parsed
            if "search_options" in args_obj:
                search_options = args_obj["search_options"]
                self.assertIsInstance(
                    search_options,
                    dict,
                    f"Nested search_options should be dict, not string. Got: {type(search_options)}. Value: {search_options}",
                )

            # Check 4: Look for double dumping patterns in the raw string
            # Double dumping typically results in escaped quotes like \"
            if arguments_str.count('\\"') > 0:
                # Count the ratio of escaped quotes to total quotes
                escaped_quotes = arguments_str.count('\\"')
                total_quotes = arguments_str.count('"')
                escaped_ratio = escaped_quotes / total_quotes if total_quotes > 0 else 0

                # If more than 30% of quotes are escaped, it might indicate double dumping
                if escaped_ratio > 0.3:
                    self.fail(
                        f"Potential double dumping detected: {escaped_ratio:.2%} of quotes are escaped. "
                        f"Arguments: {arguments_str}"
                    )

            # Check 5: Arguments string should not start and end with quotes (indicating it's a JSON string)
            arguments_stripped = arguments_str.strip()
            if (
                arguments_stripped.startswith('"')
                and arguments_stripped.endswith('"')
                and len(arguments_stripped) > 2
            ):
                # Try to parse it as a string to see if it contains JSON
                try:
                    inner_content = json.loads(arguments_stripped)
                    if isinstance(inner_content, str):
                        # This might be double dumped JSON
                        try:
                            json.loads(
                                inner_content
                            )  # Try to parse the inner string as JSON
                            self.fail(
                                f"Double dumping detected: arguments appear to be a JSON string containing JSON. "
                                f"Arguments: {arguments_str}"
                            )
                        except json.JSONDecodeError:
                            pass  # Inner content is not JSON, so it's fine
                except json.JSONDecodeError:
                    pass  # Not a JSON string, so it's fine

        # Test streaming response for the same issue
        response_stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            top_p=0.9,
            stream=True,
            tools=tools,
        )

        argument_fragments = []
        for chunk in response_stream:
            choice = chunk.choices[0]
            if choice.delta.tool_calls:
                tool_call = choice.delta.tool_calls[0]
                if tool_call.function.arguments is not None:
                    argument_fragments.append(tool_call.function.arguments)

        if argument_fragments:
            joined_args = "".join(argument_fragments)

            # Same checks for streaming response
            try:
                args_obj = json.loads(joined_args)
            except json.JSONDecodeError as e:
                self.fail(
                    f"Streaming function arguments are not valid JSON: {e}. Arguments: {joined_args}"
                )

            self.assertIsInstance(
                args_obj,
                dict,
                f"Streaming arguments should be parsed as dict, not string. Got: {type(args_obj)}. Value: {args_obj}",
            )

            # Check for double dumping patterns in streaming
            if joined_args.count('\\"') > 0:
                escaped_quotes = joined_args.count('\\"')
                total_quotes = joined_args.count('"')
                escaped_ratio = escaped_quotes / total_quotes if total_quotes > 0 else 0

                if escaped_ratio > 0.3:
                    self.fail(
                        f"Potential double dumping detected in streaming response: {escaped_ratio:.2%} of quotes are escaped. "
                        f"Arguments: {joined_args}"
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
        self.assertIsInstance(tool_calls, list)
        self.assertGreaterEqual(len(tool_calls), 1)
        names = [tc.function.name for tc in tool_calls]
        self.assertIn("get_weather", names)
        self.assertIn("get_tourist_attractions", names)

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
        self.assertIn("get_weather", found_names)
        self.assertIn("get_tourist_attractions", found_names)


if __name__ == "__main__":
    unittest.main()
