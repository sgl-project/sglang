"""Function Calling E2E Tests.

Tests for OpenAI-compatible function calling and tool choice functionality.

Source: Migrated from e2e_grpc/function_call/test_openai_function_calling.py
        and e2e_grpc/function_call/test_tool_choice.py
"""

from __future__ import annotations

import json
import logging

import openai
import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Tool Definitions
# =============================================================================

# System message for Llama3.2 function calling
LLAMA_SYSTEM_MESSAGE = (
    "You are a helpful assistant with tool calling capabilities. "
    "Only reply with a tool call if the function exists in the library provided by the user. "
    "If it doesn't exist, just reply directly in natural language. "
    "When you receive a tool call response, use the output to format an answer to the original user question. "
    "You have access to the following functions. "
    "To call a function, please respond with JSON for a function call. "
    'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. '
    "Do not use variables.\n\n"
)

# Tools for pythonic function calling tests
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


# =============================================================================
# OpenAI Function Calling Tests (Llama 1B with llama parser)
# =============================================================================


@pytest.mark.model("llama-1b")
@pytest.mark.gateway(
    extra_args=["--tool-call-parser", "llama", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestOpenAIServerFunctionCalling:
    """Tests for OpenAI-compatible function calling with Llama tool parser."""

    def test_function_calling_format(self, setup_backend):
        """Test: Whether the function call format returned by the AI is correct.

        When returning a tool call, message.content should be None, and tool_calls should be a list.
        """
        _, model, client, _ = setup_backend

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
            {"role": "system", "content": LLAMA_SYSTEM_MESSAGE},
            {"role": "user", "content": "Compute (3+5)"},
        ]
        response = client.chat.completions.create(
            model=model,
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

    def test_function_calling_streaming_simple(self, setup_backend):
        """Test: Whether the function name can be correctly recognized in streaming mode.

        - Expect a function call to be found, and the function name to be correct.
        - Verify that streaming mode returns at least multiple chunks.
        """
        _, model, client, _ = setup_backend

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
            {"role": "system", "content": LLAMA_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": "What is the temperature in Paris in celsius??",
            },
        ]

        response_stream = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=True,
            tools=tools,
        )

        chunks = list(response_stream)
        assert len(chunks) > 0, "Streaming should return at least one chunk"

        found_function_name = False
        for chunk in chunks:
            choice = chunk.choices[0]
            # Check whether the current chunk contains tool_calls
            if choice.delta.tool_calls:
                tool_call = choice.delta.tool_calls[0]
                if tool_call.function.name:
                    assert (
                        tool_call.function.name == "get_current_weather"
                    ), "Function name should be 'get_current_weather'"
                    found_function_name = True
                    break

        assert (
            found_function_name
        ), "Target function name 'get_current_weather' was not found in the streaming chunks"

        finish_reason = chunks[-1].choices[0].finish_reason
        assert (
            finish_reason == "tool_calls"
        ), "Final response of function calling should have finish_reason 'tool_calls'"

    def test_function_calling_streaming_args_parsing(self, setup_backend):
        """Test: Whether the function call arguments returned in streaming mode can be correctly concatenated into valid JSON.

        - The user request requires multiple parameters.
        - AI may return the arguments in chunks that need to be concatenated.
        """
        _, model, client, _ = setup_backend

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
            {"role": "system", "content": LLAMA_SYSTEM_MESSAGE},
            {"role": "user", "content": "Please sum 5 and 7, just call the function."},
        ]

        response_stream = client.chat.completions.create(
            model=model,
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

        assert function_name == "add", "Function name should be 'add'"
        joined_args = "".join(argument_fragments)
        assert (
            len(joined_args) > 0
        ), "No parameter fragments were returned in the function call"

        finish_reason = chunks[-1].choices[0].finish_reason
        assert (
            finish_reason == "tool_calls"
        ), "Final response of function calling should have finish_reason 'tool_calls'"

        # Check whether the concatenated JSON is valid
        try:
            args_obj = json.loads(joined_args)
        except json.JSONDecodeError:
            pytest.fail(
                "The concatenated tool call arguments are not valid JSON, parsing failed"
            )

        assert "a" in args_obj, "Missing parameter 'a'"
        assert "b" in args_obj, "Missing parameter 'b'"
        assert str(args_obj["a"]) == "5", "Parameter a should be 5"
        assert str(args_obj["b"]) == "7", "Parameter b should be 7"

    @pytest.mark.skip(
        reason="Skipping function call strict test as it is not supported by the router"
    )
    def test_function_call_strict(self, setup_backend):
        """Test: Whether the strict mode of function calling works as expected.

        - When strict mode is enabled, the AI should not return a function call if the function name is not recognized.
        """
        _, model, client, _ = setup_backend

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
            model=model,
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

        assert function_name == "sub", "Function name should be 'sub'"
        assert str(args_obj["int_a"]) == "5", "Parameter int_a should be 5"
        assert str(args_obj["int_b"]) == "7", "Parameter int_b should be 7"

    def test_function_call_required(self, setup_backend):
        """Test: Whether tool_choice: "required" works as expected.

        - When tool_choice == "required", the model should return one or more tool_calls.
        """
        _, model, client, _ = setup_backend

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
            model=model,
            max_tokens=2048,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            tools=tools,
            tool_choice="required",
        )

        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls is not None, "No tool_calls in the response"
        function_name = tool_calls[0].function.name
        arguments = tool_calls[0].function.arguments
        args_obj = json.loads(arguments)

        assert (
            function_name == "get_weather"
        ), f"Function name should be 'get_weather', got: {function_name}"
        assert (
            "city" in args_obj
        ), f"Function arguments should have 'city', got: {args_obj}"

        # Make the test more robust by checking type and accepting valid responses
        city_value = args_obj["city"]
        assert isinstance(
            city_value, str
        ), f"Parameter city should be a string, got: {type(city_value)}"
        assert (
            "Paris" in city_value or "France" in city_value
        ), f"Parameter city should contain either 'Paris' or 'France', got: {city_value}"

    def test_function_call_specific(self, setup_backend):
        """Test: Whether tool_choice: ToolChoice works as expected.

        - When tool_choice is a specific ToolChoice, the model should return one or more tool_calls.
        """
        _, model, client, _ = setup_backend

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
            model=model,
            max_tokens=2048,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls is not None, "No tool_calls in the response"
        function_name = tool_calls[0].function.name
        arguments = tool_calls[0].function.arguments
        args_obj = json.loads(arguments)

        assert function_name == "get_weather", "Function name should be 'get_weather'"
        assert "city" in args_obj, "Function arguments should have 'city'"

    def test_streaming_multiple_choices_finish_reason(self, setup_backend):
        """Test: Verify that each choice gets its own finish_reason chunk in streaming mode with n > 1.

        This tests the fix for the bug where only the last index got a finish_reason chunk.
        """
        _, model, client, _ = setup_backend

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
            model=model,
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
        assert (
            len(finish_reason_chunks) == 2
        ), f"Expected finish_reason chunks for 2 indices, got {len(finish_reason_chunks)}"

        # Verify both index 0 and 1 have finish_reason
        assert 0 in finish_reason_chunks, "Missing finish_reason chunk for index 0"
        assert 1 in finish_reason_chunks, "Missing finish_reason chunk for index 1"

        # Verify the finish_reason is "tool_calls" since we forced tool calls
        for index, reasons in finish_reason_chunks.items():
            assert (
                reasons[-1] == "tool_calls"
            ), f"Expected finish_reason 'tool_calls' for index {index}, got {reasons[-1]}"

    def test_function_calling_streaming_no_tool_call(self, setup_backend):
        """Test: Whether the finish_reason is stop in streaming mode when no tool call is given.

        - Expect no function call to be found.
        - Verify that finish_reason is stop
        """
        _, model, client, _ = setup_backend

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
            model=model,
            max_tokens=2048,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=True,
            tools=tools,
            tool_choice="none",
        )

        chunks = list(response_stream)
        assert len(chunks) > 0, "Streaming should return at least one chunk"

        found_tool_call = False
        for chunk in chunks:
            choice = chunk.choices[0]
            # Check whether the current chunk contains tool_calls
            if choice.delta.tool_calls is not None:
                found_tool_call = True
                break

        assert (
            not found_tool_call
        ), "Shouldn't have any tool_call in the streaming chunks"

        finish_reason = chunks[-1].choices[0].finish_reason
        assert (
            finish_reason == "stop"
        ), "Final response of no function calling should have finish_reason 'stop'"

    def test_streaming_multiple_choices_without_tools(self, setup_backend):
        """Test: Verify that each choice gets its own finish_reason chunk without tool calls.

        This tests the fix for regular content streaming with multiple choices.
        """
        _, model, client, _ = setup_backend

        messages = [{"role": "user", "content": "Say hello in one word."}]

        # Request with n=2 to get multiple choices, no tools
        response_stream = client.chat.completions.create(
            model=model,
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
        assert (
            len(finish_reason_chunks) == 2
        ), f"Expected finish_reason chunks for 2 indices, got {len(finish_reason_chunks)}"

        # Verify both index 0 and 1 have finish_reason
        assert 0 in finish_reason_chunks, "Missing finish_reason chunk for index 0"
        assert 1 in finish_reason_chunks, "Missing finish_reason chunk for index 1"

        # Verify the finish_reason is "stop" (regular completion)
        for index, reasons in finish_reason_chunks.items():
            assert reasons[-1] in [
                "stop",
                "length",
            ], f"Expected finish_reason 'stop' or 'length' for index {index}, got {reasons[-1]}"


# =============================================================================
# Pythonic Function Calling Tests (Llama 8B with pythonic parser)
# =============================================================================


@pytest.mark.model("llama-8b")
@pytest.mark.gateway(
    extra_args=["--tool-call-parser", "pythonic", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestOpenAIPythonicFunctionCalling:
    """Tests for pythonic function calling format."""

    def test_pythonic_tool_call_prompt(self, setup_backend):
        """Test: Explicit prompt for pythonic tool call format without chat template."""
        _, model, client, _ = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=PYTHONIC_MESSAGES,
            tools=PYTHONIC_TOOLS,
            temperature=0.1,
            stream=False,
        )
        tool_calls = response.choices[0].message.tool_calls
        assert isinstance(tool_calls, list), "No tool_calls found"
        assert len(tool_calls) >= 1
        names = [tc.function.name for tc in tool_calls]
        assert (
            "get_weather" in names or "get_tourist_attractions" in names
        ), f"Function name '{names}' should contain either 'get_weather' or 'get_tourist_attractions'"

    def test_pythonic_tool_call_streaming(self, setup_backend):
        """Test: Streaming pythonic tool call format; assert tool_call index is present."""
        _, model, client, _ = setup_backend

        response_stream = client.chat.completions.create(
            model=model,
            messages=PYTHONIC_MESSAGES,
            tools=PYTHONIC_TOOLS,
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

        assert found_tool_calls, "No tool_calls found in streaming response"
        assert found_index, "No index field found in any streamed tool_call"
        assert (
            "get_weather" in found_names or "get_tourist_attractions" in found_names
        ), f"Function name '{found_names}' should contain either 'get_weather' or 'get_tourist_attractions'"


# =============================================================================
# Tool Choice Tests - Base Class (Llama 1B with llama parser)
# =============================================================================


def get_test_tools():
    """Get the test tools for function calling."""
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


def get_test_messages():
    """Get test messages that should trigger tool usage."""
    return [
        {
            "role": "user",
            "content": "Answer the following questions as best you can:\n\nYou will be given a trace of thinking process in the following format.\n\nQuestion: the input question you must answer\nTOOL: think about what to do, and choose a tool to use ONLY IF there are defined tools\nOBSERVATION: the result of the tool call or the observation of the current task, NEVER include this in your response, this information will be provided\n... (this TOOL/OBSERVATION can repeat N times)\nANSWER: If you know the answer to the original question, require for more information, \nif the previous conversation history already contains the answer, \nor you don't know the answer and there are no defined tools or all available tools are not helpful, respond with the answer without mentioning anything else.\nYou may use light Markdown formatting to improve clarity (e.g. lists, **bold**, *italics*), but keep it minimal and unobtrusive.\n\nYour task is to respond with the next step to take, based on the traces, \nor answer the question if you have enough information.\n\nQuestion: what is the weather in top 5 populated cities in the US in celsius?\n\nTraces:\n\n\nThese are some additional instructions that you should follow:",
        }
    ]


def get_travel_tools():
    """Get tools for travel assistant scenario that should trigger multiple tool calls."""
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


def get_travel_messages():
    """Get travel assistant messages that should trigger multiple tool calls."""
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


class _TestToolChoiceBase:
    """Base class for tool_choice tests. Not collected by pytest (underscore prefix)."""

    # Subclasses should override this
    FLAKY_TESTS = set()

    def _is_flaky_test(self, test_name):
        """Check if the current test is marked as flaky for this class."""
        return test_name in self.FLAKY_TESTS

    def test_tool_choice_auto_non_streaming(self, setup_backend):
        """Test tool_choice='auto' in non-streaming mode."""
        _, model, client, _ = setup_backend

        tools = get_test_tools()
        messages = get_test_messages()

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        assert response.choices[0].message is not None
        # With auto, tool calls are optional

    def test_tool_choice_auto_streaming(self, setup_backend):
        """Test tool_choice='auto' in streaming mode."""
        _, model, client, _ = setup_backend

        tools = get_test_tools()
        messages = get_test_messages()

        response = client.chat.completions.create(
            model=model,
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
        assert isinstance(content_chunks, list)
        assert isinstance(tool_call_chunks, list)

    def test_tool_choice_required_non_streaming(self, setup_backend):
        """Test tool_choice='required' in non-streaming mode."""
        _, model, client, _ = setup_backend

        tools = get_test_tools()
        messages = get_test_messages()

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
            temperature=0.2,
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        # With required, we should get tool calls
        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls is not None
        assert len(tool_calls) > 0

    def test_tool_choice_required_streaming(self, setup_backend):
        """Test tool_choice='required' in streaming mode."""
        _, model, client, _ = setup_backend

        tools = get_test_tools()
        messages = get_test_messages()

        response = client.chat.completions.create(
            model=model,
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
        assert len(tool_call_chunks) > 0

    def test_tool_choice_specific_function_non_streaming(self, setup_backend):
        """Test tool_choice with specific function in non-streaming mode."""
        _, model, client, _ = setup_backend

        tools = get_test_tools()
        messages = get_test_messages()

        tool_choice = {"type": "function", "function": {"name": "get_weather"}}

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
        )

        # Should call the specific function
        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls is not None
        # Our messages ask the top 5 populated cities in the US, so the model could get 5 tool calls
        assert len(tool_calls) >= 1
        for tool_call in tool_calls:
            assert tool_call.function.name == "get_weather"

    def test_tool_choice_specific_function_streaming(self, setup_backend):
        """Test tool_choice with specific function in streaming mode."""
        _, model, client, _ = setup_backend

        tools = get_test_tools()
        messages = get_test_messages()

        tool_choice = {"type": "function", "function": {"name": "get_weather"}}

        response = client.chat.completions.create(
            model=model,
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
        assert len(tool_call_chunks) > 0

        # Find function name in chunks
        found_name = None
        for chunk in tool_call_chunks:
            if chunk.function and chunk.function.name:
                found_name = chunk.function.name
                break

        assert found_name == "get_weather"

    def test_required_streaming_arguments_chunks_json(self, setup_backend):
        """In streaming required mode, complete tool call arguments should be valid JSON when all chunks are combined."""
        _, model, client, _ = setup_backend

        tools = get_test_tools()
        messages = get_test_messages()

        response = client.chat.completions.create(
            model=model,
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
                    tool_call = tool_calls_by_index.setdefault(
                        tool_index,
                        {
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )

                    # Update fields from the delta
                    if tool_call_delta.id is not None:
                        tool_call["id"] = tool_call_delta.id
                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            tool_call["function"][
                                "name"
                            ] = tool_call_delta.function.name
                        if tool_call_delta.function.arguments:
                            tool_call["function"][
                                "arguments"
                            ] += tool_call_delta.function.arguments

        assert len(tool_calls_by_index) > 0

        # Validate that complete tool calls have valid JSON arguments
        for tool_call in tool_calls_by_index.values():
            assert tool_call["function"]["name"] is not None
            assert tool_call["function"]["arguments"] is not None

            # The complete arguments should be valid JSON
            try:
                args = json.loads(tool_call["function"]["arguments"])
                assert isinstance(args, dict)
            except json.JSONDecodeError:
                pytest.fail(
                    f"Invalid JSON in complete tool call arguments: {tool_call['function']['arguments']}"
                )

    def test_complex_parameters_required_non_streaming(self, setup_backend):
        """Validate complex nested parameter schemas in non-streaming required mode."""
        _, model, client, _ = setup_backend

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

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            tools=complex_tools,
            tool_choice="required",
            stream=False,
        )

        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls is not None
        assert len(tool_calls) > 0

        for tool_call in tool_calls:
            assert tool_call.function.name == "analyze_data"
            try:
                args = json.loads(tool_call.function.arguments)
                assert isinstance(args, dict)
                assert "data" in args
                assert isinstance(args["data"], dict)
            except json.JSONDecodeError:
                pytest.fail(
                    f"Invalid JSON in complex tool call arguments: {tool_call.function.arguments}"
                )

    def test_multi_tool_scenario_auto(self, setup_backend):
        """Test multi-tool scenario with tool_choice='auto'."""
        _, model, client, _ = setup_backend

        tools = get_travel_tools()
        messages = get_travel_messages()

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
            temperature=0.2,
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        # Should complete without errors
        assert response.choices[0].message is not None

        tool_calls = response.choices[0].message.tool_calls
        expected_functions = {"get_weather", "get_tourist_attractions"}

        if self._is_flaky_test("test_multi_tool_scenario_auto"):
            # For flaky tests, just verify all called functions are available tools
            if tool_calls:
                available_names = [tool["function"]["name"] for tool in tools]
                for call in tool_calls:
                    assert call.function.name in available_names
        else:
            # For non-flaky tests, enforce strict requirements
            assert tool_calls is not None, "Expected tool calls but got none"
            assert len(tool_calls) == 2, f"Expected 2 tool calls, got {len(tool_calls)}"

            called_functions = {call.function.name for call in tool_calls}
            assert (
                called_functions == expected_functions
            ), f"Expected functions {expected_functions}, got {called_functions}"

    def test_multi_tool_scenario_required(self, setup_backend):
        """Test multi-tool scenario with tool_choice='required'."""
        _, model, client, _ = setup_backend

        tools = get_travel_tools()
        messages = get_travel_messages()

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
            temperature=0.2,
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        # With required, we should get at least one tool call
        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls is not None
        assert len(tool_calls) > 0

        # Verify all called functions are available tools
        available_names = [tool["function"]["name"] for tool in tools]
        expected_functions = {"get_weather", "get_tourist_attractions"}

        for tool_call in tool_calls:
            assert tool_call.function.name is not None
            assert tool_call.function.arguments is not None

        if self._is_flaky_test("test_multi_tool_scenario_required"):
            # For flaky tests, just ensure basic functionality works
            assert (
                len(tool_calls) > 0
            ), f"Expected at least 1 tool call, got {len(tool_calls)}"
            for call in tool_calls:
                assert call.function.name in available_names
        else:
            # For non-flaky tests, enforce strict requirements
            assert len(tool_calls) == 2, f"Expected 2 tool calls, got {len(tool_calls)}"

            called_functions = {call.function.name for call in tool_calls}
            assert (
                called_functions == expected_functions
            ), f"Expected functions {expected_functions}, got {called_functions}"

    def test_error_handling_invalid_tool_choice(self, setup_backend):
        """Test error handling for invalid tool_choice."""
        _, model, client, _ = setup_backend

        tools = get_test_tools()
        messages = get_test_messages()

        # Test with invalid function name
        tool_choice = {"type": "function", "function": {"name": "nonexistent_function"}}

        # Expect a 400 BadRequestError to be raised for invalid tool_choice
        with pytest.raises(openai.BadRequestError) as exc_info:
            client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2048,
                tools=tools,
                tool_choice=tool_choice,
                stream=False,
            )

        # Verify the error message contains the expected text
        assert "function 'nonexistent_function' not found in" in str(exc_info.value)

    def test_invalid_tool_missing_name(self, setup_backend):
        """Test what happens when user doesn't provide a tool name in request."""
        _, model, client, _ = setup_backend

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
        with pytest.raises(openai.BadRequestError) as exc_info:
            client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=100,
                temperature=0.1,
                tools=invalid_tools,
                tool_choice="required",
                stream=False,
            )

        # Verify the error message indicates missing name field
        error_msg = str(exc_info.value).lower()
        assert "name" in error_msg

    def test_conflicting_defs_required_tool_choice(self, setup_backend):
        """Test that conflicting $defs with required tool_choice returns 400 error."""
        _, model, client, _ = setup_backend

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
        with pytest.raises(openai.BadRequestError) as exc_info:
            client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=100,
                temperature=0.1,
                tools=conflicting_tools,
                tool_choice="required",
                stream=False,
            )

        # Verify the error message indicates conflicting tool definitions
        error_msg = str(exc_info.value).lower()
        assert "invalid tool configuration" in error_msg
        assert "not supported" in error_msg


# =============================================================================
# Tool Choice Tests - Llama Model (llama parser)
# =============================================================================


@pytest.mark.model("llama-1b")
@pytest.mark.gateway(
    extra_args=["--tool-call-parser", "llama", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestToolChoiceLlama(_TestToolChoiceBase):
    """Tests for tool_choice functionality with Llama model."""

    # Mark flaky tests for this model
    FLAKY_TESTS = {
        "test_multi_tool_scenario_auto",
        "test_multi_tool_scenario_required",
    }


# =============================================================================
# Tool Choice Tests - Qwen Model (qwen parser)
# =============================================================================


@pytest.mark.model("qwen-7b")
@pytest.mark.gateway(
    extra_args=["--tool-call-parser", "qwen", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestToolChoiceQwen(_TestToolChoiceBase):
    """Tests for tool_choice functionality with Qwen model."""

    # No flaky tests for Qwen
    FLAKY_TESTS = set()


# =============================================================================
# Tool Choice Tests - Mistral Model (mistral parser)
# =============================================================================


@pytest.mark.model("mistral-7b")
@pytest.mark.gateway(
    extra_args=["--tool-call-parser", "mistral", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestToolChoiceMistral(_TestToolChoiceBase):
    """Tests for tool_choice functionality with Mistral model."""

    # Mark flaky tests for this model
    FLAKY_TESTS = {
        "test_multi_tool_scenario_auto",
        "test_multi_tool_scenario_required",
    }

    @pytest.mark.skip(reason="Fails due to whitespace issue with Mistral - skipping")
    def test_complex_parameters_required_non_streaming(self, setup_backend):
        """Validate complex nested parameter schemas in non-streaming required mode."""
        super().test_complex_parameters_required_non_streaming(setup_backend)
