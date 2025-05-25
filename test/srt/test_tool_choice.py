"""
Test script for tool_choice functionality in SGLang
Tests: required, auto, and specific function choices in both streaming and non-streaming modes

python3 -m unittest test_tool_choice.TestToolChoice
"""

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


class TestToolChoice(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        # Clear any previous warnings
        cls._test_warnings = []

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
        # Display summary of warnings before killing the process
        if cls._test_warnings:
            print(f"\n{'='*80}")
            print(
                f"🚨 {cls.__name__} TEST SUMMARY: {len(cls._test_warnings)} WARNINGS DETECTED"
            )
            print(f"{'='*80}")

            # Group warnings by test method
            warnings_by_test = {}
            for full_test_name, warning_msg in cls._test_warnings:
                test_method = full_test_name.split(".")[-1]  # Get just the method name
                if test_method not in warnings_by_test:
                    warnings_by_test[test_method] = []
                warnings_by_test[test_method].append(warning_msg)

            # Display grouped warnings
            for test_method, warnings in warnings_by_test.items():
                print(f"\n{test_method}:")
                for i, warning in enumerate(warnings, 1):
                    print(f"   {i}. {warning}")

            print(f"\n{'='*80}")

        kill_process_tree(cls.process.pid)

    def setUp(self):
        self.client = openai.Client(base_url=self.base_url, api_key=self.api_key)
        self.model_name = self.client.models.list().data[0].id

    def _add_test_warning(self, message):
        """Add a warning to the class collection for end-of-test summary"""
        test_name = self._testMethodName
        class_name = self.__class__.__name__
        full_test_name = f"{class_name}.{test_name}"
        formatted_message = f"{message}"
        self._test_warnings.append((full_test_name, formatted_message))

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
                            }
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
                "content": "Answer the following questions as best you can:\n\nYou will be given a trace of thinking process in the following format.\n\nQuestion: the input question you must answer\nTOOL: think about what to do, and choose a tool to use ONLY IF there are defined tools\nOBSERVATION: the result of the tool call or the observation of the current task, NEVER include this in your response, this information will be provided\n... (this TOOL/OBSERVATION can repeat N times)\nANSWER: If you know the answer to the original question, require for more information, \nif the previous conversation history already contains the answer, \nor you don't know the answer and there are no defined tools or all available tools are not helpful, respond with the answer without mentioning anything else.\nYou may use light Markdown formatting to improve clarity (e.g. lists, **bold**, *italics*), but keep it minimal and unobtrusive.\n\nYour task is to respond with the next step to take, based on the traces, \nor answer the question if you have enough information.\n\nQuestion: what is the weather in top 5 populated cities in the US?\n\nTraces:\n\n\nThese are some additional instructions that you should follow:",
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
                        "required": ["location", "unit"],
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
            max_tokens=200,
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        self.assertIsNotNone(response.choices[0].message)
        # With auto, tool calls are optional
        print(f"Tool calls: {response.choices[0].message.tool_calls}")

    def test_tool_choice_auto_streaming(self):
        """Test tool_choice='auto' in streaming mode"""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=200,
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
            max_tokens=200,
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
            max_tokens=200,
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
            max_tokens=200,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
        )

        # Should call the specific function
        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "get_weather")

    def test_tool_choice_specific_function_streaming(self):
        """Test tool_choice with specific function in streaming mode"""
        tools = self.get_test_tools()
        messages = self.get_test_messages()

        tool_choice = {"type": "function", "function": {"name": "get_weather"}}

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=200,
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

    def test_multi_tool_scenario_auto(self):
        """Test multi-tool scenario with tool_choice='auto'"""
        tools = self.get_travel_tools()
        messages = self.get_travel_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=400,
            temperature=0.2,
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        # Should complete without errors
        self.assertIsNotNone(response.choices[0].message)

        # Log the tool calls for debugging
        tool_calls = response.choices[0].message.tool_calls
        expected_functions = {"get_weather", "get_tourist_attractions"}

        # Check if we got both expected functions
        if not tool_calls:
            self._add_test_warning(
                f"⚠️  AUTO MODE LIMITATION: No tool calls made! "
                f"Model chose not to use available tools. This is technically correct "
                f"but not ideal for a travel assistant scenario."
            )
            return

        # With auto, the model may or may not call multipletools due to its capabilities
        called_functions = set()

        for call in tool_calls:
            # Verify the called function is one of the available tools
            available_names = [tool["function"]["name"] for tool in tools]
            self.assertIn(call.function.name, available_names)
            called_functions.add(call.function.name)

        if len(called_functions) == 1:
            missing = expected_functions - called_functions
            self._add_test_warning(
                f"⚠️  AUTO MODE LIMITATION: Only called {called_functions}, "
                f"missing {missing}. This is acceptable but not ideal for travel scenario."
            )
        elif not expected_functions.issubset(called_functions):
            self._add_test_warning(
                f"⚠️  UNEXPECTED: Called {called_functions}, expected {expected_functions}"
            )

    def test_multi_tool_scenario_required(self):
        """Test multi-tool scenario with tool_choice='required'"""
        tools = self.get_travel_tools()
        messages = self.get_travel_messages()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=400,
            temperature=0.2,
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        # With required, we should get at least one tool call
        tool_calls = response.choices[0].message.tool_calls
        self.assertIsNotNone(tool_calls)
        self.assertGreater(len(tool_calls), 0)

        # Verify all called functions are available tools and analyze coverage
        available_names = [tool["function"]["name"] for tool in tools]
        called_functions = set()
        expected_functions = {"get_weather", "get_tourist_attractions"}

        for call in tool_calls:
            self.assertIn(call.function.name, available_names)
            called_functions.add(call.function.name)

        # Analyze the results
        if len(called_functions) == 1:
            missing = expected_functions - called_functions
            self._add_test_warning(
                f"⚠️  REQUIRED MODE LIMITATION: Only called {called_functions}, "
                f"missing {missing}. Required mode should ideally call both functions "
                f"for a comprehensive travel assistant response."
            )
        elif not expected_functions.issubset(called_functions):
            self._add_test_warning(
                f"⚠️  UNEXPECTED BEHAVIOR: Called {called_functions}, "
                f"expected {expected_functions} for travel scenario"
            )

    def test_error_handling_invalid_tool_choice(self):
        """Test error handling for invalid tool_choice"""
        import logging
        from unittest.mock import patch

        tools = self.get_test_tools()
        messages = self.get_test_messages()

        # Test with invalid function name
        tool_choice = {"type": "function", "function": {"name": "nonexistent_function"}}

        # The behavior could be either:
        # 1. Log a warning and continue (if fallback is implemented)
        # 2. Raise an exception (if strict validation is implemented)

        # First try to capture any logging that might happen
        with patch("logging.warning") as mock_warning:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=200,
                tools=tools,
                tool_choice=tool_choice,
                stream=False,
            )

            self.assertIsNotNone(response.choices[0].message)

            if mock_warning.called:
                warning_message = mock_warning.call_args[0][0]
                self.assertIn("nonexistent_function", warning_message)
            else:
                print(
                    "No warnings logged - this might be handled elsewhere in the stack"
                )


class TestToolChoiceQwen25(TestToolChoice):
    """Test tool_choice functionality with Qwen2.5 model"""

    @classmethod
    def setUpClass(cls):
        # Each test class gets its own warning collection
        cls._test_warnings = []

        cls.model = "Qwen/Qwen2.5-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--tool-call-parser",
                "qwen25",
            ],
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)


class TestToolChoiceMistral(TestToolChoice):
    """Test tool_choice functionality with Mistral model"""

    @classmethod
    def setUpClass(cls):
        # Each test class gets its own warning collection
        cls._test_warnings = []

        cls.model = "mistralai/Mistral-7B-Instruct-v0.3"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--tool-call-parser",
                "mistral",
            ],
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)


if __name__ == "__main__":
    unittest.main()
