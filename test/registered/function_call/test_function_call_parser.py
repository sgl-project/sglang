import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.srt.function_call.mistral_detector import MistralDetector
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "default")


class TestPythonicDetector(unittest.TestCase):
    def setUp(self):
        # Create sample tools for testing
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location to get weather for",
                            },
                            "unit": {
                                "type": "string",
                                "description": "Temperature unit",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search for information",
                    parameters={
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ),
        ]
        self.detector = PythonicDetector()

    def test_parse_streaming_no_brackets(self):
        """Test parsing text with no brackets (no tool calls)."""
        text = "This is just normal text without any tool calls."
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])
        self.assertEqual(self.detector._buffer, "")  # Buffer should be cleared

    def test_parse_streaming_complete_tool_call(self):
        """Test parsing a complete tool call."""
        text = "Here's a tool call: [get_weather(location='New York', unit='celsius')]"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "Here's a tool call: ")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            self.detector._buffer, ""
        )  # Buffer should be cleared after processing

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "New York")
        self.assertEqual(params["unit"], "celsius")

    def test_parse_streaming_text_before_tool_call(self):
        """Test parsing text that appears before a tool call."""
        text = "This is some text before [get_weather(location='London')]"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "This is some text before ")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "London")

    def test_parse_streaming_partial_tool_call(self):
        """Test parsing a partial tool call that spans multiple chunks."""
        # First chunk with opening bracket but no closing bracket
        text1 = "Let me check the weather: [get_weather(location="
        result1 = self.detector.parse_streaming_increment(text1, self.tools)

        self.assertEqual(result1.normal_text, "Let me check the weather: ")
        self.assertEqual(result1.calls, [])
        self.assertEqual(
            self.detector._buffer, "[get_weather(location="
        )  # Partial tool call remains in buffer

        # Second chunk completing the tool call
        text2 = "'Paris')]"
        result2 = self.detector.parse_streaming_increment(text2, self.tools)

        self.assertEqual(result2.normal_text, "")
        self.assertEqual(len(result2.calls), 1)
        self.assertEqual(result2.calls[0].name, "get_weather")

        # Check the parameters
        params = json.loads(result2.calls[0].parameters)
        self.assertEqual(params["location"], "Paris")
        self.assertEqual(
            self.detector._buffer, ""
        )  # Buffer should be cleared after processing

    def test_parse_streaming_bracket_without_text_before(self):
        """Test parsing a tool call that starts at the beginning of the text."""
        text = "[search(query='python programming')]"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "python programming")

    def test_parse_streaming_text_after_tool_call(self):
        """Test parsing text that appears after a tool call."""
        # First chunk with complete tool call and some text after
        text = "[get_weather(location='Tokyo')] Here's the forecast:"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            self.detector._buffer, " Here's the forecast:"
        )  # Text after tool call remains in buffer

        # Process the remaining text in buffer
        result2 = self.detector.parse_streaming_increment("", self.tools)
        self.assertEqual(result2.normal_text, " Here's the forecast:")
        self.assertEqual(result2.calls, [])
        self.assertEqual(self.detector._buffer, "")  # Buffer should be cleared

    def test_parse_streaming_multiple_tool_calls(self):
        """Test parsing multiple tool calls in sequence."""
        text = "[get_weather(location='Berlin')] and [search(query='restaurants')]"

        # First tool call
        result1 = self.detector.parse_streaming_increment(text, self.tools)
        self.assertEqual(len(result1.calls), 1)
        self.assertEqual(result1.calls[0].name, "get_weather")
        self.assertEqual(self.detector._buffer, " and [search(query='restaurants')]")

        # Second tool call
        result2 = self.detector.parse_streaming_increment("", self.tools)
        self.assertEqual(result2.normal_text, " and ")
        self.assertEqual(len(result2.calls), 1)
        self.assertEqual(result2.calls[0].name, "search")
        self.assertEqual(self.detector._buffer, "")

    def test_parse_streaming_opening_bracket_only(self):
        """Test parsing text with only an opening bracket but no closing bracket."""
        text = "Let's try this: ["
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "Let's try this: ")
        self.assertEqual(result.calls, [])
        self.assertEqual(
            self.detector._buffer, "["
        )  # Opening bracket remains in buffer

    def test_parse_streaming_nested_brackets(self):
        """Test parsing tool calls with nested brackets in arguments."""
        # Test with list argument containing nested brackets
        text = "[get_weather(location='New York', unit='celsius', data=[1, 2, 3])]"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(self.detector._buffer, "")

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "New York")
        self.assertEqual(params["unit"], "celsius")
        self.assertEqual(params["data"], [1, 2, 3])

    def test_parse_streaming_nested_brackets_dict(self):
        """Test parsing tool calls with nested dictionaries and lists."""
        # Test with nested dict and list arguments
        text = "[search(query='test', config={'options': [1, 2], 'nested': {'key': 'value'}})]"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        self.assertEqual(self.detector._buffer, "")

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "test")
        self.assertEqual(params["config"]["options"], [1, 2])
        self.assertEqual(params["config"]["nested"]["key"], "value")

    def test_parse_streaming_multiple_tools_with_nested_brackets(self):
        """Test parsing multiple tool calls with nested brackets."""
        text = "[get_weather(location='Paris', data=[10, 20]), search(query='test', filters=['a', 'b'])]"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(self.detector._buffer, "")

        # Check first tool call
        params1 = json.loads(result.calls[0].parameters)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(params1["location"], "Paris")
        self.assertEqual(params1["data"], [10, 20])

        # Check second tool call
        params2 = json.loads(result.calls[1].parameters)
        self.assertEqual(result.calls[1].name, "search")
        self.assertEqual(params2["query"], "test")
        self.assertEqual(params2["filters"], ["a", "b"])

    def test_parse_streaming_partial_nested_brackets(self):
        """Test parsing partial tool calls with nested brackets across chunks."""
        # First chunk with nested brackets but incomplete
        text1 = "Here's a call: [get_weather(location='Tokyo', data=[1, 2"
        result1 = self.detector.parse_streaming_increment(text1, self.tools)

        self.assertEqual(result1.normal_text, "Here's a call: ")
        self.assertEqual(result1.calls, [])
        self.assertEqual(
            self.detector._buffer, "[get_weather(location='Tokyo', data=[1, 2"
        )

        # Second chunk completing the nested brackets
        text2 = ", 3])]"
        result2 = self.detector.parse_streaming_increment(text2, self.tools)

        self.assertEqual(result2.normal_text, "")
        self.assertEqual(len(result2.calls), 1)
        self.assertEqual(result2.calls[0].name, "get_weather")
        self.assertEqual(self.detector._buffer, "")

        # Check the parameters
        params = json.loads(result2.calls[0].parameters)
        self.assertEqual(params["location"], "Tokyo")
        self.assertEqual(params["data"], [1, 2, 3])

    def test_parse_streaming_with_python_start_and_end_token(self):
        """Test parsing a message that starts with <|python_start|> and <|python_end|> across chunks."""
        chunks = [
            "Here's a call: ",
            "<|python_",
            "start|>[get_weather(location=",
            "'Tokyo', data=[1, 2",
            ", 3])]<|python_end|>",
        ]

        normal_text = ""
        call_name = ""
        parameters = ""
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.normal_text:
                normal_text += result.normal_text
            if result.calls:
                call_name += result.calls[0].name
                parameters += result.calls[0].parameters

        self.assertEqual(normal_text, "Here's a call: ")
        self.assertEqual(call_name, "get_weather")
        self.assertEqual(self.detector._buffer, "")
        self.assertEqual(
            result.normal_text, "", "Final result should have no normal text"
        )

        # Check the parameters
        params = json.loads(parameters)
        self.assertEqual(params["location"], "Tokyo")
        self.assertEqual(params["data"], [1, 2, 3])

        chunks = [
            "Here's a call: <|python_start|>[get_weather(location='Tokyo', data=[1, 2, 3])]<|python_end|>"
        ]

        normal_text = ""
        call_name = ""
        parameters = ""
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.normal_text:
                normal_text += result.normal_text
            if result.calls:
                call_name += result.calls[0].name
                parameters += result.calls[0].parameters

        self.assertEqual(normal_text, "Here's a call: ")
        self.assertEqual(call_name, "get_weather")
        self.assertEqual(self.detector._buffer, "")

        # Check the parameters
        params = json.loads(parameters)
        self.assertEqual(params["location"], "Tokyo")
        self.assertEqual(params["data"], [1, 2, 3])

    def test_detect_and_parse_with_python_start_and_end_token(self):
        """Test parsing a message that starts with <|python_start|> and contains a valid tool call."""
        text = "User wants to get the weather in Mars. <|python_start|>[get_weather(location='Mars', unit='celsius')]<|python_end|> In this way we will get the weather in Mars."
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(
            result.normal_text,
            "User wants to get the weather in Mars.  In this way we will get the weather in Mars.",
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(self.detector._buffer, "")

        # Check the parameters
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "Mars")
        self.assertEqual(params["unit"], "celsius")


class TestMistralDetector(unittest.TestCase):
    def setUp(self):
        """Set up test tools and detector for Mistral format testing."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="make_next_step_decision",
                    description="Test function for decision making",
                    parameters={
                        "type": "object",
                        "properties": {
                            "decision": {
                                "type": "string",
                                "description": "The next step to take",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content of the next step",
                            },
                        },
                        "required": ["decision", "content"],
                    },
                ),
            ),
        ]
        self.detector = MistralDetector()

    def test_detect_and_parse_with_nested_brackets_in_content(self):
        """Test parsing Mistral format with nested brackets in JSON content.

        This test case specifically addresses the issue where the regex pattern
        was incorrectly truncating JSON when it contained nested brackets like [City Name].
        """
        # This is the exact problematic text from the original test failure
        test_text = '[TOOL_CALLS] [{"name":"make_next_step_decision", "arguments":{"decision":"","content":"```\\nTOOL: Access a weather API or service\\nOBSERVATION: Retrieve the current weather data for the top 5 populated cities in the US\\nANSWER: The weather in the top 5 populated cities in the US is as follows: [City Name] - [Weather Conditions] - [Temperature]\\n```"}}]'

        result = self.detector.detect_and_parse(test_text, self.tools)

        # Verify that the parsing was successful
        self.assertEqual(len(result.calls), 1, "Should detect exactly one tool call")

        call = result.calls[0]
        self.assertEqual(
            call.name,
            "make_next_step_decision",
            "Should detect the correct function name",
        )

        # Verify that the parameters are valid JSON and contain the expected content
        params = json.loads(call.parameters)
        self.assertEqual(
            params["decision"], "", "Decision parameter should be empty string"
        )

        # The content should contain the full text including the nested brackets [City Name]
        expected_content = "```\nTOOL: Access a weather API or service\nOBSERVATION: Retrieve the current weather data for the top 5 populated cities in the US\nANSWER: The weather in the top 5 populated cities in the US is as follows: [City Name] - [Weather Conditions] - [Temperature]\n```"
        self.assertEqual(
            params["content"],
            expected_content,
            "Content should include nested brackets without truncation",
        )

        # Verify that normal text is empty (since the entire input is a tool call)
        self.assertEqual(
            result.normal_text, "", "Normal text should be empty for pure tool call"
        )

    def test_detect_and_parse_simple_case(self):
        """Test parsing a simple Mistral format tool call without nested brackets."""
        test_text = '[TOOL_CALLS] [{"name":"make_next_step_decision", "arguments":{"decision":"TOOL", "content":"Use weather API"}}]'

        result = self.detector.detect_and_parse(test_text, self.tools)

        self.assertEqual(len(result.calls), 1)
        call = result.calls[0]
        self.assertEqual(call.name, "make_next_step_decision")

        params = json.loads(call.parameters)
        self.assertEqual(params["decision"], "TOOL")
        self.assertEqual(params["content"], "Use weather API")

    def test_detect_and_parse_no_tool_calls(self):
        """Test parsing text without any tool calls."""
        test_text = "This is just normal text without any tool calls."

        result = self.detector.detect_and_parse(test_text, self.tools)

        self.assertEqual(len(result.calls), 0, "Should detect no tool calls")
        self.assertEqual(
            result.normal_text,
            test_text,
            "Should return the original text as normal text",
        )

    def test_detect_and_parse_with_text_before_tool_call(self):
        """Test parsing text that has content before the tool call."""
        test_text = 'Here is some text before the tool call: [TOOL_CALLS] [{"name":"make_next_step_decision", "arguments":{"decision":"ANSWER", "content":"The answer is 42"}}]'

        result = self.detector.detect_and_parse(test_text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.normal_text, "Here is some text before the tool call:")

        call = result.calls[0]
        self.assertEqual(call.name, "make_next_step_decision")

        params = json.loads(call.parameters)
        self.assertEqual(params["decision"], "ANSWER")
        self.assertEqual(params["content"], "The answer is 42")

    def test_detect_and_parse_compact_args_format(self):
        """Test parsing compact format: [TOOL_CALLS]name[ARGS]{...}."""
        test_text = '[TOOL_CALLS]make_next_step_decision[ARGS]{"decision":"TOOL", "content":"Use weather API"}'

        result = self.detector.detect_and_parse(test_text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "make_next_step_decision")
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["decision"], "TOOL")
        self.assertEqual(params["content"], "Use weather API")

    def test_streaming_compact_args_format_emits_tool_calls(self):
        """Test streaming chunks for compact format produce tool_calls items."""
        chunks = [
            "[TOOL_CALLS]make_next_step_decision[ARGS]",
            '{"decision":"TOOL", ',
            '"content":"Use weather API"}',
        ]

        emitted = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                emitted.extend(result.calls)

        # Expect two items: name chunk + full args chunk
        self.assertEqual(len(emitted), 2)
        self.assertEqual(emitted[0].name, "make_next_step_decision")
        self.assertEqual(emitted[0].parameters, "")
        self.assertIsNone(emitted[1].name)
        params = json.loads(emitted[1].parameters)
        self.assertEqual(params["decision"], "TOOL")
        self.assertEqual(params["content"], "Use weather API")


class TestBaseFormatDetector(unittest.TestCase):
    """Test buffer management and sequential tool index assignment in BaseFormatDetector."""

    def setUp(self):
        """Set up test detector and tools."""

        # Create a concrete implementation of BaseFormatDetector for testing
        class TestFormatDetector(BaseFormatDetector):
            def __init__(self):
                super().__init__()
                self.bot_token = "<tool_call>"
                self.eot_token = "</tool_call>"

            def detect_and_parse(self, text, tools):
                # Not used in streaming tests
                pass

            def has_tool_call(self, text):
                return "<tool_call>" in text

            def structure_info(self):
                # Not used in streaming tests
                pass

        self.detector = TestFormatDetector()
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_tourist_attractions",
                    description="Get tourist attractions",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]

    def test_sequential_tool_index_assignment(self):
        """Test that multiple tool calls get sequential tool_index values (0, 1, 2, ...)."""
        # Simulate streaming chunks for two consecutive tool calls
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", ',
            '"arguments": {"city": "Paris"}}',
            ", ",
            '{"name": "get_tourist_attractions", ',
            '"arguments": {"city": "London"}}',
            "</tool_call>",
        ]

        tool_indices_seen = []

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

            if result.calls:
                for call in result.calls:
                    if call.tool_index is not None:
                        tool_indices_seen.append(call.tool_index)

        # Verify we got sequential tool indices
        unique_indices = sorted(set(tool_indices_seen))
        self.assertEqual(
            unique_indices,
            [0, 1],
            f"Expected sequential tool indices [0, 1], got {unique_indices}",
        )

    def test_buffer_content_preservation(self):
        """Test that buffer correctly preserves unprocessed content when tool completes."""
        # Test simpler scenario: tool completion followed by new tool start
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", ',
            '"arguments": {"city": "Paris"}}',
            ", ",
            '{"name": "get_tourist_attractions", ',
            '"arguments": {"city": "London"}} </tool_call>',
        ]

        tool_calls_seen = []

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                for call in result.calls:
                    if (
                        call.name
                    ):  # Only count calls with names (not just parameter updates)
                        tool_calls_seen.append(call.name)

        # Should see both tool names
        self.assertIn("get_weather", tool_calls_seen, "Should process first tool")
        self.assertIn(
            "get_tourist_attractions", tool_calls_seen, "Should process second tool"
        )

    def test_current_tool_id_increment_on_completion(self):
        """Test that current_tool_id increments when a tool completes."""
        # Initial state
        self.assertEqual(
            self.detector.current_tool_id, -1, "Should start with current_tool_id=-1"
        )

        # Process first tool completely
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", ',
        ]

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            self.detector.current_tool_id, 0, "current_tool_id should be 0"
        )
        self.assertEqual(
            result.calls[0].name, "get_weather", "The first tool should be get_weather"
        )
        self.assertEqual(
            result.calls[0].tool_index, 0, "The first tool index should be 0"
        )

        # Complete second tool name - this should show that current_tool_id is now 1
        result = self.detector.parse_streaming_increment(
            '"arguments": {"city": "Paris"}}, {"name": "get_', self.tools
        )
        self.assertEqual(result.calls[0].parameters, '{"city": "Paris"}')

        self.assertEqual(
            self.detector.current_tool_id,
            1,
            "current_tool_id should be 1 after first tool completes and second tool starts",
        )

        result = self.detector.parse_streaming_increment(
            'tourist_attractions", ', self.tools
        )

        # Second tool should have tool_index=1
        tourist_calls = [
            call for call in result.calls if call.name == "get_tourist_attractions"
        ]
        self.assertEqual(
            tourist_calls[0].tool_index, 1, "Second tool should have tool_index=1"
        )

    def test_tool_name_streaming_with_correct_index(self):
        """Test that tool names are streamed with correct tool_index values."""
        # Process first tool
        self.detector.parse_streaming_increment("<tool_call>", self.tools)
        result1 = self.detector.parse_streaming_increment(
            '{"name": "get_weather", ', self.tools
        )

        # First tool name should have tool_index=0
        weather_calls = [call for call in result1.calls if call.name == "get_weather"]
        self.assertEqual(len(weather_calls), 1, "Should have one weather call")
        self.assertEqual(
            weather_calls[0].tool_index, 0, "First tool should have tool_index=0"
        )

        # Complete first tool
        self.detector.parse_streaming_increment(
            '"arguments": {"city": "Paris"}}', self.tools
        )

        # Start second tool
        self.detector.parse_streaming_increment(", ", self.tools)
        result2 = self.detector.parse_streaming_increment(
            '{"name": "get_tourist_attractions", ', self.tools
        )

        # Second tool name should have tool_index=1
        tourist_calls = [
            call for call in result2.calls if call.name == "get_tourist_attractions"
        ]
        self.assertEqual(
            len(tourist_calls), 1, "Should have one tourist attractions call"
        )
        self.assertEqual(
            tourist_calls[0].tool_index, 1, "Second tool should have tool_index=1"
        )

    def test_buffer_reset_on_invalid_tool(self):
        """Test that buffer and state are reset when an invalid tool name is encountered."""
        # Start fresh with an invalid tool name from the beginning
        result = self.detector.parse_streaming_increment(
            '<tool_call>{"name": "invalid_tool", ', self.tools
        )

        # Should return empty result and reset state
        self.assertEqual(result.calls, [], "Should return no calls for invalid tool")
        self.assertEqual(
            self.detector.current_tool_id,
            -1,
            "current_tool_id should remain -1 for invalid tool",
        )
        self.assertEqual(
            self.detector._buffer, "", "Buffer should be cleared for invalid tool"
        )

    def test_chinese_characters_not_double_escaped(self):
        """Test that Chinese characters in tool call parameters are not double-escaped."""
        # Test with Chinese city name "杭州" (Hangzhou)
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", ',
            '"arguments": {"city": "杭州"}}',
            "</tool_call>",
        ]

        accumulated_parameters = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                for call in result.calls:
                    if call.parameters:
                        tool_idx = call.tool_index if call.tool_index is not None else 0
                        if tool_idx not in accumulated_parameters:
                            accumulated_parameters[tool_idx] = ""
                        accumulated_parameters[tool_idx] += call.parameters

        # Verify that Chinese characters are preserved (not escaped as \uXXXX)
        self.assertGreater(
            len(accumulated_parameters), 0, "Should have parsed parameters"
        )
        final_params_str = accumulated_parameters[0]

        # The parameters string should contain the actual Chinese characters, not escaped Unicode
        self.assertIn(
            "杭州", final_params_str, "Should contain actual Chinese characters"
        )
        self.assertNotIn(
            "\\u676d", final_params_str, "Should not contain escaped Unicode sequences"
        )
        self.assertNotIn(
            "\\u5dde", final_params_str, "Should not contain escaped Unicode sequences"
        )

        # Verify the JSON can be parsed and contains the correct value
        params = json.loads(final_params_str)
        self.assertEqual(
            params["city"], "杭州", "Should correctly parse Chinese city name"
        )

    def test_chinese_characters_incremental_streaming(self):
        """Test that Chinese characters work correctly with incremental streaming."""
        # Test incremental streaming with Chinese characters
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", ',
            '"arguments": {"city": "',
            "杭州",
            '"}}',
            "</tool_call>",
        ]

        accumulated_parameters = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                for call in result.calls:
                    if call.parameters:
                        tool_idx = call.tool_index if call.tool_index is not None else 0
                        if tool_idx not in accumulated_parameters:
                            accumulated_parameters[tool_idx] = ""
                        accumulated_parameters[tool_idx] += call.parameters

        # Verify Chinese characters are preserved throughout streaming
        self.assertGreater(
            len(accumulated_parameters), 0, "Should have parsed parameters"
        )
        final_params_str = accumulated_parameters[0]

        # Should contain actual Chinese characters, not escaped
        self.assertIn(
            "杭州", final_params_str, "Should contain actual Chinese characters"
        )

        # Parse and verify
        params = json.loads(final_params_str)
        self.assertEqual(
            params["city"], "杭州", "Should correctly parse Chinese city name"
        )

    def test_multiple_chinese_parameters(self):
        """Test multiple tool calls with Chinese parameters."""
        # Test with multiple tool calls containing Chinese characters
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", "arguments": {"city": "北京"}}, ',
            '{"name": "get_tourist_attractions", "arguments": {"city": "上海"}}',
            "</tool_call>",
        ]

        accumulated_parameters = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                for call in result.calls:
                    if call.parameters:
                        tool_idx = call.tool_index if call.tool_index is not None else 0
                        if tool_idx not in accumulated_parameters:
                            accumulated_parameters[tool_idx] = ""
                        accumulated_parameters[tool_idx] += call.parameters

        # Verify both tool calls have correct Chinese characters
        self.assertGreaterEqual(
            len(accumulated_parameters), 1, "Should have parsed parameters"
        )

        # Check first tool call (北京 - Beijing)
        if 0 in accumulated_parameters:
            params0 = json.loads(accumulated_parameters[0])
            self.assertIn(
                "北京",
                accumulated_parameters[0],
                "Should contain actual Chinese characters",
            )
            self.assertEqual(
                params0["city"], "北京", "Should correctly parse first Chinese city"
            )

        # Check second tool call (上海 - Shanghai) if present
        if 1 in accumulated_parameters:
            params1 = json.loads(accumulated_parameters[1])
            self.assertIn(
                "上海",
                accumulated_parameters[1],
                "Should contain actual Chinese characters",
            )
            self.assertEqual(
                params1["city"], "上海", "Should correctly parse second Chinese city"
            )


class TestLlama32Detector(unittest.TestCase):
    def setUp(self):
        """Set up test tools and detector for Mistral format testing."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_tourist_attractions",
                    description="Get tourist attractions",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = Llama32Detector()

    def test_single_json(self):
        text = '{"name": "get_weather", "parameters": {"city": "Paris"}}'
        result = self.detector.detect_and_parse(text, self.tools)
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert result.normal_text == ""

    def test_multiple_json_with_separator(self):
        text = (
            '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}};'
            '{"name": "get_tourist_attractions", "parameters": {"city": "Paris"}}'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[1].name, "get_tourist_attractions")
        self.assertEqual(result.normal_text, "")

    def test_multiple_json_with_separator_customized(self):
        text = (
            '<|python_tag|>{"name": "get_weather", "parameters": {}}'
            '<|python_tag|>{"name": "get_tourist_attractions", "parameters": {}}'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[1].name, "get_tourist_attractions")
        self.assertEqual(result.normal_text, "")

    def test_json_with_trailing_text(self):
        text = '{"name": "get_weather", "parameters": {}} Some follow-up text'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertIn("follow-up", result.normal_text)

    def test_invalid_then_valid_json(self):
        text = (
            '{"name": "get_weather", "parameters": {'  # malformed
            '{"name": "get_weather", "parameters": {}}'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_plain_text_only(self):
        text = "This is just plain explanation text."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, text)

    def test_with_python_tag_prefix(self):
        text = 'Some intro. <|python_tag|>{"name": "get_weather", "parameters": {}}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertTrue(result.normal_text.strip().startswith("Some intro."))


class TestKimiK2Detector(unittest.TestCase):

    def setUp(self):
        """Set up test tools and detector."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_tourist_attractions",
                    description="Get tourist attractions",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = KimiK2Detector()

    def test_single_tool_call(self):
        """Test parsing a single tool call in a complete text."""
        text = '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|><|tool_calls_section_end|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[0].parameters, '{"city": "Paris"}')
        self.assertEqual(result.normal_text, "")

    def test_multiple_tool_calls(self):
        """Test parsing multiple tool calls in a complete text."""
        text = '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|><|tool_call_begin|>functions.get_tourist_attractions:1<|tool_call_argument_begin|>{"city": "London"}<|tool_call_end|><|tool_calls_section_end|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[0].parameters, '{"city": "Paris"}')
        self.assertEqual(result.calls[1].name, "get_tourist_attractions")
        self.assertEqual(result.calls[1].parameters, '{"city": "London"}')
        self.assertEqual(result.normal_text, "")

    def test_streaming_tool_call(self):
        """Test streaming incremental parsing of a tool call."""
        chunks = [
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{",
            '"city": "Paris"',
            "}",
            "<|tool_call_end|><|tool_calls_section_end|>",
        ]

        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if tool_call_chunk.tool_index is not None:

                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})

                    tc = tool_calls[tool_call_chunk.tool_index]

                    if tool_call_chunk.name:
                        tc["name"] += tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(tool_calls[0]["parameters"], '{"city": "Paris"}')

    def test_streaming_multiple_tool_calls(self):
        """Test streaming incremental parsing of multiple tool calls."""
        chunks = [
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{",
            '"city": "Paris"',
            "}<|tool_call_end|>",
            "<|tool_call_begin|>functions.get_tourist_attractions:1<|tool_call_argument_begin|>{",
            '"city": "London"',
            "}<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]

        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if tool_call_chunk.tool_index is not None:

                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})

                    tc = tool_calls[tool_call_chunk.tool_index]

                    if tool_call_chunk.name:
                        tc["name"] += tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters

        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(tool_calls[0]["parameters"], '{"city": "Paris"}')
        self.assertEqual(tool_calls[1]["name"], "get_tourist_attractions")
        self.assertEqual(tool_calls[1]["parameters"], '{"city": "London"}')

    def test_tool_call_completion(self):
        """Test that the buffer and state are reset after a tool call is completed."""
        chunks = [
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{",
            '"city": "Paris"',
            "}",
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

        # After processing all chunks, the buffer should be empty and current_tool_id should be reset
        self.assertEqual(self.detector._buffer, "")
        self.assertEqual(self.detector.current_tool_id, 1)

    def test_tool_name_streaming(self):
        """Test that tool names are streamed correctly with the right index."""
        chunks = [
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{",
            '"city": "Paris"',
            "}",
            "<|tool_call_end|>",
            "<|tool_call_begin|>functions.get_tourist_attractions:1<|tool_call_argument_begin|>{",
        ]

        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if tool_call_chunk.tool_index is not None:

                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})

                    tc = tool_calls[tool_call_chunk.tool_index]

                    if tool_call_chunk.name:
                        tc["name"] += tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters

        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(tool_calls[0]["parameters"], '{"city": "Paris"}')
        self.assertEqual(tool_calls[1]["name"], "get_tourist_attractions")

    def test_invalid_tool_call(self):
        """Test that invalid tool calls are handled correctly."""
        text = 'invalid_tool:0<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|><|tool_calls_section_end|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_partial_tool_call(self):
        """Test that partial tool calls are handled correctly in streaming mode."""
        chunks = [
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{",
            '"city": "Paris"',
        ]

        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if tool_call_chunk.tool_index is not None:

                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})

                    tc = tool_calls[tool_call_chunk.tool_index]

                    if tool_call_chunk.name:
                        tc["name"] += tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(tool_calls[0]["parameters"], '{"city": "Paris"')


class TestDeepSeekV3Detector(unittest.TestCase):
    def setUp(self):
        """Set up test tools and detector for DeepSeekV3 format testing."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_tourist_attractions",
                    description="Get tourist attractions",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = DeepSeekV3Detector()

    def test_parse_streaming_multiple_tool_calls_with_multi_token_chunk(self):
        """Test parsing multiple tool calls when streaming chunks contains multi-tokens (e.g. DeepSeekV3 enable MTP)"""
        # Simulate streaming chunks with multi-tokens for two consecutive tool calls
        chunks = [
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁call▁begin｜>function",
            "<｜tool▁sep｜>get",
            "_weather\n",
            "```json\n",
            '{"city":',
            '"Shanghai',
            '"}\n```<｜tool▁call▁end｜>',
            "\n<｜tool▁call▁begin｜>",
            "function<｜tool▁sep｜>",
            "get_tour",
            "ist_att",
            "ractions\n```" 'json\n{"',
            'city": "',
            'Beijing"}\n',
            "```<｜tool▁call▁end｜>",
            "<｜tool▁calls▁end｜>",
        ]

        tool_calls_seen = []
        tool_calls_parameters = []

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                for call in result.calls:
                    if call.name:
                        tool_calls_seen.append(call.name)
                    if call.parameters:
                        tool_calls_parameters.append(call.parameters)

        # Should see both tool names
        self.assertIn("get_weather", tool_calls_seen, "Should process first tool")
        self.assertIn(
            "get_tourist_attractions", tool_calls_seen, "Should process second tool"
        )

        # Verify that the parameters are valid JSON and contain the expected content
        params1 = json.loads(tool_calls_parameters[0])
        params2 = json.loads(tool_calls_parameters[1])
        self.assertEqual(params1["city"], "Shanghai")
        self.assertEqual(params2["city"], "Beijing")


class TestDeepSeekV32Detector(unittest.TestCase):
    def setUp(self):
        """Set up test tools and detector for DeepSeekV32 format testing."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Searches for information related to query and displays topn results.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string",
                            },
                            "topn": {
                                "type": "integer",
                                "description": "Number of top results to display",
                                "default": 10,
                            },
                            "source": {
                                "type": "string",
                                "description": "Source to search within",
                                "enum": ["web", "news"],
                                "default": "web",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_favorite_tourist_spot",
                    description="Return the favorite tourist spot for a given city.",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = DeepSeekV32Detector()
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3.2")
        self.interval = 1

    def test_detect_and_parse_xml_format(self):
        """Test parsing standard XML format (DSML)"""
        text = """I'll help you with information about San Francisco and get its favorite tourist spot for you.\n\n
        <｜DSML｜function_calls>\n
            <｜DSML｜invoke name="get_favorite_tourist_spot">\n
                <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>\n
            </｜DSML｜invoke>\n
            <｜DSML｜invoke name="search">
                <｜DSML｜parameter name="query" string="true">WebNav benchmark</｜DSML｜parameter>
                <｜DSML｜parameter name="topn" string="false">10</｜DSML｜parameter>
                <｜DSML｜parameter name="source" string="true">web</｜DSML｜parameter>
            </｜DSML｜invoke>
        </｜DSML｜function_calls>
        """
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertIn("I'll help you with information", result.normal_text)
        self.assertEqual(len(result.calls), 2)

        # Check first call
        call1 = result.calls[0]
        self.assertEqual(call1.name, "get_favorite_tourist_spot")
        params1 = json.loads(call1.parameters)
        self.assertEqual(params1["city"], "San Francisco")

        # Check second call
        call2 = result.calls[1]
        self.assertEqual(call2.name, "search")
        params2 = json.loads(call2.parameters)
        self.assertEqual(params2["query"], "WebNav benchmark")
        self.assertEqual(params2["topn"], 10)
        self.assertEqual(params2["source"], "web")

    def test_detect_and_parse_json_format(self):
        """Test parsing JSON format inside invoke tags"""
        text = """I'll help you with information about San Francisco and get its favorite tourist spot for you.

        <｜DSML｜function_calls>
            <｜DSML｜invoke name="get_favorite_tourist_spot">
            {
                "city": "San Francisco"
            }
        </｜DSML｜invoke>
            <｜DSML｜invoke name="search">
            {
                "query": "WebNav benchmark",
                "topn": 10,
                "source": "web"
            }
        </｜DSML｜invoke>
        </｜DSML｜function_calls>
        """
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertIn("I'll help you with information", result.normal_text)
        self.assertEqual(len(result.calls), 2)

        # Check first call
        call1 = result.calls[0]
        self.assertEqual(call1.name, "get_favorite_tourist_spot")
        params1 = json.loads(call1.parameters)
        self.assertEqual(params1["city"], "San Francisco")

        # Check second call
        call2 = result.calls[1]
        self.assertEqual(call2.name, "search")
        params2 = json.loads(call2.parameters)
        self.assertEqual(params2["query"], "WebNav benchmark")
        self.assertEqual(params2["topn"], 10)
        self.assertEqual(params2["source"], "web")

    def test_streaming_xml_format(self):
        """Test streaming parsing of XML format"""
        text = """<｜DSML｜function_calls>
            <｜DSML｜invoke name="get_favorite_tourist_spot">
                <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
                <｜DSML｜parameter name="another_city" string="true">London</｜DSML｜parameter>
                <｜DSML｜parameter name="topn" string="false">10</｜DSML｜parameter>
                <｜DSML｜parameter name="obj" string="false">{"name": "John", "age": 30}</｜DSML｜parameter>
            </｜DSML｜invoke>
        </｜DSML｜function_calls>"""

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(len(tool_calls_by_index), 1)
        self.assertEqual(tool_calls_by_index[0]["name"], "get_favorite_tourist_spot")
        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["city"], "San Francisco")
        self.assertEqual(params["another_city"], "London")
        self.assertEqual(params["topn"], 10)
        self.assertEqual(params["obj"]["name"], "John")
        self.assertEqual(params["obj"]["age"], 30)

    def test_streaming_json_format(self):
        """Test streaming parsing of JSON format"""
        text = """<｜DSML｜function_calls>
            <｜DSML｜invoke name="get_favorite_tourist_spot">
            {
                "city": "San Francisco"
            }
            </｜DSML｜invoke>
        </｜DSML｜function_calls>"""

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        self.assertEqual(len(tool_calls_by_index), 1)
        self.assertEqual(tool_calls_by_index[0]["name"], "get_favorite_tourist_spot")

        # Clean up parameters string if needed (trim whitespace)
        params_str = tool_calls_by_index[0]["parameters"].strip()
        params = json.loads(params_str)
        self.assertEqual(params["city"], "San Francisco")

    def test_detect_and_parse_no_parameters(self):
        """Test parsing function calls with no parameters (non-streaming)"""
        # Add a no-parameter tool
        tools_with_no_param = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_date",
                    description="Get the current date.",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

        text = """Let me get the current date for you.

<｜DSML｜function_calls>
<｜DSML｜invoke name="get_date">
</｜DSML｜invoke>
</｜DSML｜function_calls>"""

        result = self.detector.detect_and_parse(text, tools_with_no_param)

        self.assertIn("Let me get the current date", result.normal_text)
        self.assertEqual(len(result.calls), 1)

        call = result.calls[0]
        self.assertEqual(call.name, "get_date")
        params = json.loads(call.parameters)
        self.assertEqual(params, {})

    def test_streaming_no_parameters(self):
        """Test streaming parsing of function calls with no parameters.

        This test verifies the fix for the bug where functions with no parameters
        were being silently skipped in streaming mode.
        """
        # Add a no-parameter tool
        tools_with_no_param = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_date",
                    description="Get the current date.",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

        text = """<｜DSML｜function_calls>
<｜DSML｜invoke name="get_date">
</｜DSML｜invoke>
</｜DSML｜function_calls>"""

        # Reset detector state
        self.detector = DeepSeekV32Detector()

        # Simulate streaming by splitting into small chunks
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, tools_with_no_param)
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        # Verify that the no-parameter function was correctly parsed
        self.assertEqual(
            len(tool_calls_by_index), 1, "Should have exactly one tool call"
        )
        self.assertEqual(tool_calls_by_index[0]["name"], "get_date")

        # Parameters should be empty JSON object
        params_str = tool_calls_by_index[0]["parameters"].strip()
        params = json.loads(params_str)
        self.assertEqual(params, {})

    def test_streaming_no_parameters_with_whitespace(self):
        """Test streaming parsing when invoke content has only whitespace (newlines)."""
        tools_with_no_param = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_date",
                    description="Get the current date.",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

        # This format has newlines inside the invoke tag (common model output)
        text = """<｜DSML｜function_calls>
<｜DSML｜invoke name="get_date">

</｜DSML｜invoke>
</｜DSML｜function_calls>"""

        # Reset detector state
        self.detector = DeepSeekV32Detector()

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, tools_with_no_param)
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        # Should still parse correctly even with whitespace-only content
        self.assertEqual(
            len(tool_calls_by_index), 1, "Should have exactly one tool call"
        )
        self.assertEqual(tool_calls_by_index[0]["name"], "get_date")
        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params, {})


class TestQwen3CoderDetector(unittest.TestCase):
    """Test suite for Qwen3CoderDetector."""

    def setUp(self):
        """Initialize test fixtures before each test method."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_current_weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                            "days": {"type": "integer"},
                        },
                        "required": ["location"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="sql_interpreter",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "dry_run": {"type": "boolean"},
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="TodoWrite",
                    parameters={
                        "type": "object",
                        "properties": {
                            "todos": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string"},
                                        "status": {"type": "string"},
                                    },
                                    "required": ["content", "status"],
                                },
                            },
                        },
                    },
                ),
            ),
        ]
        self.detector = Qwen3CoderDetector()

    # ==================== Basic Functionality Tests ====================

    def test_plain_text_only(self):
        """
        Test parsing of plain text without any tool calls.

        Scenario: Input contains only plain text, no tool call markers.
        Purpose: Verify that plain text is correctly identified and no false tool calls are detected.
        """
        text = "This is plain text without any tool calls."
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, text)
        self.assertEqual(len(result.calls), 0)

    def test_single_tool_call(self):
        """
        Test parsing of a single tool call.

        Scenario: Input contains one complete tool call with parameters.
        Purpose: Verify correct extraction of tool name and parameters.
        """
        text = """<tool_call>
<function=get_current_weather>
<parameter=location>Boston</parameter>
<parameter=unit>celsius</parameter>
<parameter=days>3</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_current_weather")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "Boston")
        self.assertEqual(params["unit"], "celsius")
        self.assertEqual(params["days"], 3)

    def test_single_tool_call_with_text_prefix(self):
        """
        Test parsing of tool call with preceding text.

        Scenario: Input has plain text followed by a tool call.
        Purpose: Verify correct separation of text and tool call.
        """
        text = """Let me check the weather for you.

<tool_call>
<function=get_current_weather>
<parameter=location>New York</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertTrue(result.normal_text.startswith("Let me check"))
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_current_weather")

    def test_multiple_tool_calls(self):
        """
        Test parsing of multiple consecutive tool calls.

        Scenario: Input contains two tool calls one after another.
        Purpose: Verify that multiple tool calls are correctly identified and parsed.
        """
        text = """<tool_call>
<function=get_current_weather>
<parameter=location>New York</parameter>
</function>
</tool_call>
<tool_call>
<function=sql_interpreter>
<parameter=query>SELECT * FROM users</parameter>
<parameter=dry_run>True</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_current_weather")
        self.assertEqual(result.calls[1].name, "sql_interpreter")

        params1 = json.loads(result.calls[0].parameters)
        self.assertEqual(params1["location"], "New York")

        params2 = json.loads(result.calls[1].parameters)
        self.assertEqual(params2["query"], "SELECT * FROM users")
        self.assertEqual(params2["dry_run"], True)

    # ==================== Streaming Tests ====================

    def test_streaming_single_tool_call(self):
        """
        Test streaming parsing of a single tool call.

        Scenario: Tool call is fed incrementally in chunks.
        Purpose: Verify streaming parser correctly assembles tool call from chunks.
        """
        chunks = [
            "<tool_call>",
            "<function=get_current_weather>",
            "<parameter=location>",
            "Boston",
            "</parameter>",
            "<parameter=unit>celsius</parameter>",
            "</function>",
            "</tool_call>",
        ]

        detector = Qwen3CoderDetector()
        all_calls = []
        collected_params = ""

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            for call in result.calls:
                if call.parameters:
                    collected_params += call.parameters

        # Verify we got the tool call
        self.assertGreater(len(all_calls), 0)

        # Verify parameters were collected
        if collected_params:
            params = json.loads(collected_params)
            self.assertEqual(params["location"], "Boston")
            self.assertEqual(params["unit"], "celsius")

    def test_streaming_with_text_and_tool(self):
        """
        Test streaming parsing with mixed text and tool call.

        Scenario: Stream contains plain text followed by a tool call.
        Purpose: Verify correct separation in streaming mode.
        """
        chunks = [
            "Let me ",
            "help you.\n\n",
            "<tool_call>",
            "<function=get_current_weather>",
            "<parameter=location>Paris</parameter>",
            "</function>",
            "</tool_call>",
        ]

        detector = Qwen3CoderDetector()
        full_text = ""
        all_calls = []

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            if result.normal_text:
                full_text += result.normal_text
            all_calls.extend(result.calls)

        self.assertTrue(full_text.startswith("Let me"))
        self.assertGreater(len(all_calls), 0)

    # ==================== Parameter Type Tests ====================

    def test_integer_parameter_conversion(self):
        """
        Test correct type conversion for integer parameters.

        Scenario: Tool call with integer parameter.
        Purpose: Verify integer values are correctly parsed and typed.
        """
        text = """<tool_call>
<function=get_current_weather>
<parameter=location>Tokyo</parameter>
<parameter=days>5</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        params = json.loads(result.calls[0].parameters)
        self.assertIsInstance(params["days"], int)
        self.assertEqual(params["days"], 5)

    def test_boolean_parameter_conversion(self):
        """
        Test correct type conversion for boolean parameters.

        Scenario: Tool call with boolean parameter.
        Purpose: Verify boolean values are correctly parsed.
        """
        text = """<tool_call>
<function=sql_interpreter>
<parameter=query>SELECT 1</parameter>
<parameter=dry_run>True</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        params = json.loads(result.calls[0].parameters)
        self.assertIsInstance(params["dry_run"], bool)
        self.assertEqual(params["dry_run"], True)

    def test_complex_array_parameter(self):
        """
        Test parsing of complex array parameters.

        Scenario: Tool call with array of objects as parameter.
        Purpose: Verify complex nested structures are correctly parsed.
        """
        text = """<tool_call>
<function=TodoWrite>
<parameter=todos>
[
  {"content": "Buy groceries", "status": "pending"},
  {"content": "Finish report", "status": "completed"}
]
</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        params = json.loads(result.calls[0].parameters)
        self.assertIsInstance(params["todos"], list)
        self.assertEqual(len(params["todos"]), 2)
        self.assertEqual(params["todos"][0]["content"], "Buy groceries")
        self.assertEqual(params["todos"][1]["status"], "completed")

    # ==================== Edge Cases ====================

    def test_empty_parameter_value(self):
        """
        Test handling of empty parameter values.

        Scenario: Tool call with empty parameter value.
        Purpose: Verify empty values are handled gracefully.
        """
        text = """<tool_call>
<function=get_current_weather>
<parameter=location></parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "")

    def test_parameter_with_special_characters(self):
        """
        Test handling of parameters with special characters.

        Scenario: Parameter value contains special characters like quotes, newlines.
        Purpose: Verify special characters are correctly preserved.
        """
        text = """<tool_call>
<function=sql_interpreter>
<parameter=query>SELECT * FROM users WHERE name = 'John "Doe"'</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        params = json.loads(result.calls[0].parameters)
        self.assertIn("John", params["query"])
        self.assertIn("Doe", params["query"])

    def test_incomplete_tool_call(self):
        """
        Test handling of incomplete tool call at end of stream.

        Scenario: Stream ends with an incomplete tool call (missing closing tag).
        Purpose: Verify detector handles incomplete input gracefully without crashing.
        """
        text = """<tool_call>
<function=get_current_weather>
<parameter=location>London"""

        # Should not crash
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertIsInstance(result, StreamingParseResult)

    def test_has_tool_call_detection(self):
        """
        Test the has_tool_call method for detecting tool call markers.

        Scenario: Various inputs with and without tool call markers.
        Purpose: Verify correct detection of tool call presence.
        """
        self.assertTrue(self.detector.has_tool_call("<tool_call>"))
        self.assertTrue(self.detector.has_tool_call("text <tool_call> more"))
        self.assertFalse(self.detector.has_tool_call("plain text only"))
        self.assertFalse(self.detector.has_tool_call(""))


class TestGlm4MoeDetector(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "date": {"type": "string", "description": "Date"},
                        },
                        "required": ["city", "date"],
                    },
                ),
            ),
        ]
        self.detector = Glm4MoeDetector()

    def test_single_tool_call(self):
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n"
            "<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            result.calls[0].parameters, '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(result.normal_text, "")

    def test_multiple_tool_calls(self):
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n"
            "<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n"
            "</tool_call>"
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n<arg_value>Shanghai</arg_value>\n"
            "<arg_key>date</arg_key>\n<arg_value>2024-06-28</arg_value>\n"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            result.calls[0].parameters, '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(result.calls[1].name, "get_weather")
        self.assertEqual(
            result.calls[1].parameters, '{"city": "Shanghai", "date": "2024-06-28"}'
        )
        self.assertEqual(result.normal_text, "")

    def test_streaming_tool_call(self):
        """Test streaming incremental parsing of a tool call."""
        chunks = [
            "<tool_call>get_weather\n",
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n",
            "<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n",
            "</tool_call>",
        ]
        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if (
                    hasattr(tool_call_chunk, "tool_index")
                    and tool_call_chunk.tool_index is not None
                ):
                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})
                    tc = tool_calls[tool_call_chunk.tool_index]
                    if tool_call_chunk.name:
                        tc["name"] = tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(
            tool_calls[0]["parameters"], '{"city": "Beijing", "date": "2024-06-27"}'
        )

    def test_streaming_multiple_tool_calls(self):
        """Test streaming incremental parsing of multiple tool calls."""
        chunks = [
            "<tool_call>get_weather\n",
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n",
            "<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n",
            "</tool_call><tool_call>get_weather\n",
            "<arg_key>city</arg_key>\n<arg_value>Shanghai</arg_value>\n",
            "<arg_key>date</arg_key>\n<arg_value>2024-06-28</arg_value>\n",
            "</tool_call>",
        ]
        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if (
                    hasattr(tool_call_chunk, "tool_index")
                    and tool_call_chunk.tool_index is not None
                ):
                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})
                    tc = tool_calls[tool_call_chunk.tool_index]
                    if tool_call_chunk.name:
                        tc["name"] = tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(
            tool_calls[0]["parameters"], '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(tool_calls[1]["name"], "get_weather")
        self.assertEqual(
            tool_calls[1]["parameters"], '{"city": "Shanghai", "date": "2024-06-28"}'
        )

    def test_tool_call_id(self):
        """Test that the buffer and state are reset after a tool call is completed."""
        chunks = [
            "<tool_call>get_weather\n",
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n",
            "<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n",
            "</tool_call>",
        ]
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
        self.assertEqual(self.detector.current_tool_id, 1)

    def test_invalid_tool_call(self):
        """Test that invalid tool calls are handled correctly."""
        text = "<tool_call>invalid_func\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_partial_tool_call(self):
        """Test parsing a partial tool call that spans multiple chunks."""
        chunks = [
            "<tool_call>get_weather\n",
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n",
            "<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n</tool_call>",
        ]

        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if (
                    hasattr(tool_call_chunk, "tool_index")
                    and tool_call_chunk.tool_index is not None
                ):
                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})
                    tc = tool_calls[tool_call_chunk.tool_index]
                    if tool_call_chunk.name:
                        tc["name"] = tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(
            tool_calls[0]["parameters"], '{"city": "Beijing", "date": "2024-06-27"}'
        )

    def test_array_argument_with_escaped_json(self):
        """Test that array arguments with escaped JSON are properly handled without double-escaping."""
        # Add a tool with array parameter
        tools_with_array = [
            Tool(
                type="function",
                function=Function(
                    name="todo_write",
                    description="Write todos",
                    parameters={
                        "type": "object",
                        "properties": {
                            "todos": {
                                "type": "array",
                                "description": "The updated todo list",
                            }
                        },
                        "required": ["todos"],
                    },
                ),
            ),
        ]

        def check_params(result):
            self.assertEqual(1, len(result.calls))
            self.assertEqual("todo_write", result.calls[0].name)
            params = json.loads(result.calls[0].parameters)
            self.assertIsInstance(params["todos"], list)
            self.assertEqual(4, len(params["todos"]))
            self.assertEqual("1", params["todos"][0]["id"])
            self.assertEqual(
                "Check for hard-coded issues in the backend code",
                params["todos"][0]["task"],
            )
            self.assertEqual("in_progress", params["todos"][0]["status"])
            self.assertEqual("2", params["todos"][1]["id"])
            self.assertEqual(
                "Check for hard-coded issues in the frontend code",
                params["todos"][1]["task"],
            )
            self.assertEqual("pending", params["todos"][1]["status"])
            self.assertEqual("3", params["todos"][2]["id"])
            self.assertEqual(
                "Check for code violating the Single Responsibility Principle",
                params["todos"][2]["task"],
            )
            self.assertEqual("pending", params["todos"][2]["status"])
            self.assertEqual("4", params["todos"][3]["id"])
            self.assertEqual(
                "Generate a rectification proposal report", params["todos"][3]["task"]
            )
            self.assertEqual("pending", params["todos"][3]["status"])

        # Simulate the raw response from GLM-4.6 model with normal and escaped JSON in XML
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write\n<arg_key>todos</arg_key>\n<arg_value>[{\"id\": \"1\", \"task\": \"Check for hard-coded issues in the backend code\", \"status\": \"in_progress\"}, {\"id\": \"2\", \"task\": \"Check for hard-coded issues in the frontend code\", \"status\": \"pending\"}, {\"id\": \"3\", \"task\": \"Check for code violating the Single Responsibility Principle\", \"status\": \"pending\"}, {\"id\": \"4\", \"task\": \"Generate a rectification proposal report\", \"status\": \"pending\"}]</arg_value>
</tool_call>""",
            tools_with_array,
        )
        check_params(result)
        result = self.detector.detect_and_parse(
            r"""<tool_call>todo_write\n<arg_key>todos</arg_key>\n<arg_value>[{\"id\": \"1\", \"task\": \"Check for hard-coded issues in the backend code\", \"status\": \"in_progress\"}, {\"id\": \"2\", \"task\": \"Check for hard-coded issues in the frontend code\", \"status\": \"pending\"}, {\"id\": \"3\", \"task\": \"Check for code violating the Single Responsibility Principle\", \"status\": \"pending\"}, {\"id\": \"4\", \"task\": \"Generate a rectification proposal report\", \"status\": \"pending\"}]</arg_value>
</tool_call>""",
            tools_with_array,
        )
        check_params(result)

        def check_single_todos(tool_result, expected):
            self.assertEqual(1, len(tool_result.calls))
            self.assertEqual("todo_write", tool_result.calls[0].name)
            params = json.loads(tool_result.calls[0].parameters)
            self.assertIsInstance(params["todos"], list)
            self.assertEqual(1, len(params["todos"]))
            self.assertEqual("1", params["todos"][0]["id"])
            self.assertEqual(expected, params["todos"][0]["task"])
            self.assertEqual("pending", params["todos"][0]["status"])

        # Test with escaped JSON containing backslashes in content (e.g., Windows paths)
        expected_path = r"Check file at C:\Users\test.txt"
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write\n<arg_key>todos</arg_key>\n<arg_value>[{\"id\": \"1\", \"task\": \"Check file at C:\\\\Users\\\\test.txt\", \"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_path)
        result = self.detector.detect_and_parse(
            r"""<tool_call>todo_write\n<arg_key>todos</arg_key>\n<arg_value>[{\"id\": \"1\", \"task\": \"Check file at C:\\\\Users\\\\test.txt\", \"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_path)

        # Should contain literal \n, not actual newline
        expected_output = r"Print \n to see newline"
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write\n<arg_key>todos</arg_key>\n<arg_value>[{\"id\": \"1\", \"task\": \"Print \\\\n to see newline\",\"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_output)
        result = self.detector.detect_and_parse(
            r"""<tool_call>todo_write\n<arg_key>todos</arg_key>\n<arg_value>[{\"id\": \"1\", \"task\": \"Print \\\\n to see newline\",\"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_output)

    def test_empty_function_name_handling(self):
        """Test that empty function name is handled gracefully without assertion error."""
        # This test simulates the issue where the model outputs only the start token without a function name
        chunks = [
            "<tool_call>",  # Start token only, no function name yet
            "\n",  # More content without function name
        ]

        for chunk in chunks:
            # Should not raise AssertionError: func_name should not be empty
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            # Should return empty calls without error
            self.assertIsInstance(result, StreamingParseResult)
            self.assertEqual(result.calls, [])


class TestGlm47MoeDetector(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "date": {"type": "string", "description": "Date"},
                        },
                        "required": ["city", "date"],
                    },
                ),
            ),
        ]
        self.detector = Glm47MoeDetector()

    def test_single_tool_call(self):
        text = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            result.calls[0].parameters, '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(result.normal_text, "")

    def test_multiple_tool_calls(self):
        text = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>"
            "</tool_call>"
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Shanghai</arg_value>"
            "<arg_key>date</arg_key><arg_value>2024-06-28</arg_value>"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            result.calls[0].parameters, '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(result.calls[1].name, "get_weather")
        self.assertEqual(
            result.calls[1].parameters, '{"city": "Shanghai", "date": "2024-06-28"}'
        )
        self.assertEqual(result.normal_text, "")

    def test_streaming_tool_call(self):
        """Test streaming incremental parsing of a tool call."""
        chunks = [
            "<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>",
            "</tool_call>",
        ]
        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if (
                    hasattr(tool_call_chunk, "tool_index")
                    and tool_call_chunk.tool_index is not None
                ):
                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})
                    tc = tool_calls[tool_call_chunk.tool_index]
                    if tool_call_chunk.name:
                        tc["name"] = tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(
            tool_calls[0]["parameters"], '{"city": "Beijing", "date": "2024-06-27"}'
        )

    def test_streaming_multiple_tool_calls(self):
        """Test streaming incremental parsing of multiple tool calls."""
        chunks = [
            "<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>",
            "</tool_call><tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Shanghai</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-28</arg_value>",
            "</tool_call>",
        ]
        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if (
                    hasattr(tool_call_chunk, "tool_index")
                    and tool_call_chunk.tool_index is not None
                ):
                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})
                    tc = tool_calls[tool_call_chunk.tool_index]
                    if tool_call_chunk.name:
                        tc["name"] = tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(
            tool_calls[0]["parameters"], '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(tool_calls[1]["name"], "get_weather")
        self.assertEqual(
            tool_calls[1]["parameters"], '{"city": "Shanghai", "date": "2024-06-28"}'
        )

    def test_tool_call_id(self):
        """Test that the buffer and state are reset after a tool call is completed."""
        chunks = [
            "<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>",
            "</tool_call>",
        ]
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
        self.assertEqual(self.detector.current_tool_id, 1)

    def test_invalid_tool_call(self):
        """Test that invalid tool calls are handled correctly."""
        text = "<tool_call>invalid_func<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_partial_tool_call(self):
        """Test parsing a partial tool call that spans multiple chunks."""
        chunks = [
            "<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>",
        ]

        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if (
                    hasattr(tool_call_chunk, "tool_index")
                    and tool_call_chunk.tool_index is not None
                ):
                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})
                    tc = tool_calls[tool_call_chunk.tool_index]
                    if tool_call_chunk.name:
                        tc["name"] = tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(
            tool_calls[0]["parameters"], '{"city": "Beijing", "date": "2024-06-27"}'
        )

    def test_array_argument_with_escaped_json(self):
        """Test that array arguments with escaped JSON are properly handled without double-escaping."""
        # Add a tool with array parameter
        tools_with_array = [
            Tool(
                type="function",
                function=Function(
                    name="todo_write",
                    description="Write todos",
                    parameters={
                        "type": "object",
                        "properties": {
                            "todos": {
                                "type": "array",
                                "description": "The updated todo list",
                            }
                        },
                        "required": ["todos"],
                    },
                ),
            ),
        ]

        def check_params(result):
            self.assertEqual(1, len(result.calls))
            self.assertEqual("todo_write", result.calls[0].name)
            params = json.loads(result.calls[0].parameters)
            self.assertIsInstance(params["todos"], list)
            self.assertEqual(4, len(params["todos"]))
            self.assertEqual("1", params["todos"][0]["id"])
            self.assertEqual(
                "Check for hard-coded issues in the backend code",
                params["todos"][0]["task"],
            )
            self.assertEqual("in_progress", params["todos"][0]["status"])
            self.assertEqual("2", params["todos"][1]["id"])
            self.assertEqual(
                "Check for hard-coded issues in the frontend code",
                params["todos"][1]["task"],
            )
            self.assertEqual("pending", params["todos"][1]["status"])
            self.assertEqual("3", params["todos"][2]["id"])
            self.assertEqual(
                "Check for code violating the Single Responsibility Principle",
                params["todos"][2]["task"],
            )
            self.assertEqual("pending", params["todos"][2]["status"])
            self.assertEqual("4", params["todos"][3]["id"])
            self.assertEqual(
                "Generate a rectification proposal report", params["todos"][3]["task"]
            )
            self.assertEqual("pending", params["todos"][3]["status"])

        # Simulate the raw response from GLM-4.6 model with normal and escaped JSON in XML
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Check for hard-coded issues in the backend code\", \"status\": \"in_progress\"}, {\"id\": \"2\", \"task\": \"Check for hard-coded issues in the frontend code\", \"status\": \"pending\"}, {\"id\": \"3\", \"task\": \"Check for code violating the Single Responsibility Principle\", \"status\": \"pending\"}, {\"id\": \"4\", \"task\": \"Generate a rectification proposal report\", \"status\": \"pending\"}]</arg_value>
</tool_call>""",
            tools_with_array,
        )
        check_params(result)
        result = self.detector.detect_and_parse(
            r"""<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Check for hard-coded issues in the backend code\", \"status\": \"in_progress\"}, {\"id\": \"2\", \"task\": \"Check for hard-coded issues in the frontend code\", \"status\": \"pending\"}, {\"id\": \"3\", \"task\": \"Check for code violating the Single Responsibility Principle\", \"status\": \"pending\"}, {\"id\": \"4\", \"task\": \"Generate a rectification proposal report\", \"status\": \"pending\"}]</arg_value>
</tool_call>""",
            tools_with_array,
        )
        check_params(result)

        def check_single_todos(tool_result, expected):
            self.assertEqual(1, len(tool_result.calls))
            self.assertEqual("todo_write", tool_result.calls[0].name)
            params = json.loads(tool_result.calls[0].parameters)
            self.assertIsInstance(params["todos"], list)
            self.assertEqual(1, len(params["todos"]))
            self.assertEqual("1", params["todos"][0]["id"])
            self.assertEqual(expected, params["todos"][0]["task"])
            self.assertEqual("pending", params["todos"][0]["status"])

        # Test with escaped JSON containing backslashes in content (e.g., Windows paths)
        expected_path = r"Check file at C:\Users\test.txt"
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Check file at C:\\\\Users\\\\test.txt\", \"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_path)
        result = self.detector.detect_and_parse(
            r"""<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Check file at C:\\\\Users\\\\test.txt\", \"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_path)

        # Should contain literal \n, not actual newline
        expected_output = r"Print \n to see newline"
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Print \\\\n to see newline\",\"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_output)
        result = self.detector.detect_and_parse(
            r"""<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Print \\\\n to see newline\",\"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_output)


class TestJsonArrayParser(unittest.TestCase):
    def setUp(self):
        # Create sample tools for testing
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location to get weather for",
                            },
                            "unit": {
                                "type": "string",
                                "description": "Temperature unit",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search for information",
                    parameters={
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ),
        ]
        self.detector = JsonArrayParser()

    def test_json_detector_has_no_ebnf(self):
        """JsonArrayParser no longer exposes EBNF generation helpers."""
        self.assertFalse(
            hasattr(self.detector, "build_ebnf"),
            "JsonArrayParser should not expose EBNF helpers after cleanup",
        )

    def test_parse_streaming_increment_malformed_json(self):
        """Test parsing with malformed JSON"""
        # Test with malformed JSON
        text = '[{"name": "get_weather", "parameters": {"location": "Tokyo"'
        result = self.detector.parse_streaming_increment(text, self.tools)

        # Should not crash and return a valid result
        self.assertIsInstance(result, StreamingParseResult)

        text = "[{}}}]"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertIsInstance(result, StreamingParseResult)

    def test_parse_streaming_increment_empty_input(self):
        """Test parsing with empty input"""
        result = self.detector.parse_streaming_increment("", self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "")

    def test_parse_streaming_increment_whitespace_handling(self):
        """Test parsing with various whitespace scenarios"""
        # Test with leading/trailing whitespace split across chunks
        chunk1 = '  [{"name": "get_weather", "parameters": '
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = '{"location": "Tokyo"}}]  '
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        # The base class should handle this
        self.assertIsInstance(result2, StreamingParseResult)

    def test_parse_streaming_increment_nested_objects(self):
        """Test parsing with nested JSON objects"""
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tokyo", '
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = '"nested": {"key": "value"}}}]'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        # The base class should handle this
        self.assertIsInstance(result2, StreamingParseResult)

    def test_json_parsing_with_commas(self):
        """Test that JSON parsing works correctly with comma separators"""
        # Stream two complete objects, at least 2 chunks per tool call
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tok'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = 'yo"}},'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)

        chunk3 = '{"name": "get_weather", "parameters": {"location": "Par'
        result3 = self.detector.parse_streaming_increment(chunk3, self.tools)
        self.assertIsInstance(result3, StreamingParseResult)
        chunk4 = 'is"}}]'
        result4 = self.detector.parse_streaming_increment(chunk4, self.tools)
        self.assertIsInstance(result4, StreamingParseResult)
        self.assertGreater(
            len(result4.calls), 0, "Should parse tool calls from text with separators"
        )

    def test_braces_in_strings(self):
        """Test that JSON with } characters inside strings works correctly"""
        # Test case: JSON array with } inside string values - streamed across chunks
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "has } inside"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = "}}"
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)
        self.assertGreater(
            len(result2.calls), 0, "Should parse tool call with } in string"
        )

        # Test with separator (streaming in progress)
        chunk3 = '[{"name": "get_weather", "parameters": {"location": "has } inside"}'
        result3 = self.detector.parse_streaming_increment(chunk3, self.tools)
        self.assertIsInstance(result3, StreamingParseResult)
        chunk4 = "},"
        result4 = self.detector.parse_streaming_increment(chunk4, self.tools)
        self.assertIsInstance(result4, StreamingParseResult)
        chunk5 = '{"name": "get_weather"'
        result5 = self.detector.parse_streaming_increment(chunk5, self.tools)
        self.assertIsInstance(result5, StreamingParseResult)
        self.assertGreater(
            len(result5.calls),
            0,
            "Should parse tool calls with separator and } in string",
        )

    def test_separator_in_same_chunk(self):
        """Test that separator already present in chunk works correctly"""
        # Test case: separator already in the chunk (streaming in progress) with 2+ chunks per tool call
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tokyo"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = '}},{"name": "get_weather"'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)
        self.assertGreater(
            len(result2.calls),
            0,
            "Should parse tool calls with separator in same chunk",
        )

    def test_separator_in_separate_chunk(self):
        """Test that separator in separate chunk works correctly"""
        # Test case: separator in separate chunk - this tests streaming behavior
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tokyo"}}'
        chunk2 = ","
        chunk3 = '{"name": "get_weather", "parameters": {"location": "Paris"}}'

        # Process first chunk
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)

        # Process separator chunk
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)

        # Process second chunk (streaming in progress)
        result3 = self.detector.parse_streaming_increment(chunk3, self.tools)
        self.assertIsInstance(result3, StreamingParseResult)

    def test_incomplete_json_across_chunks(self):
        """Test that incomplete JSON across chunks works correctly"""
        # Test case: incomplete JSON across chunks - this tests streaming behavior
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tokyo"'
        chunk2 = '}},{"name": "get_weather"'

        # Process first chunk (incomplete)
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)

        # Process second chunk (completes first object and starts second, streaming in progress)
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)

    def test_malformed_json_recovery(self):
        """Test that malformed JSON recovers gracefully"""
        # Test with malformed JSON - should handle gracefully
        malformed_text = (
            '[{"name": "get_weather", "parameters": {"location": "unclosed string'
        )

        result1 = self.detector.parse_streaming_increment(malformed_text, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)

        # Test valid JSON after malformed - streamed across 2 chunks (streaming in progress)
        valid_chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tok'
        result2 = self.detector.parse_streaming_increment(valid_chunk1, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)
        valid_chunk2 = 'yo"}}'
        result3 = self.detector.parse_streaming_increment(valid_chunk2, self.tools)
        self.assertIsInstance(result3, StreamingParseResult)

    def test_nested_objects_with_commas(self):
        """Test that nested objects with commas inside work correctly"""
        # Test with nested objects that have commas - should work with json.loads()
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tok'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = 'yo", "unit": "celsius"}}'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)
        self.assertGreater(
            len(result2.calls), 0, "Should parse tool call with nested objects"
        )

    def test_empty_objects(self):
        """Test that empty objects work correctly"""
        # Test with empty objects - should work with json.loads()
        chunk1 = '[{"name": "get_weather", "parameters": '
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = "{}}"
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)

    def test_whitespace_handling(self):
        """Test that various whitespace scenarios work correctly"""
        # Test with various whitespace patterns - should work with json.loads()
        chunk1 = ' \n\n [{"name": "get_weather", "parameters": '
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = '{"location": "Tokyo"}}'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)

    def test_multiple_commas_in_chunk(self):
        """Test that multiple commas in a single chunk work correctly"""
        # Stream multiple tool calls ensuring at least 2 chunks per complete tool call
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "To'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = 'kyo"}},'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)

        chunk3 = '{"name": "get_weather", "parameters": {"location": "Pa'
        result3 = self.detector.parse_streaming_increment(chunk3, self.tools)
        self.assertIsInstance(result3, StreamingParseResult)
        chunk4 = 'ris"}},'
        result4 = self.detector.parse_streaming_increment(chunk4, self.tools)
        self.assertIsInstance(result4, StreamingParseResult)

        chunk5 = '{"name": "get_weather"'
        result5 = self.detector.parse_streaming_increment(chunk5, self.tools)
        self.assertIsInstance(result5, StreamingParseResult)
        self.assertGreater(
            len(result5.calls), 0, "Should parse tool calls with multiple commas"
        )

    def test_complete_tool_call_with_trailing_comma(self):
        """Test that complete tool call with trailing comma parses correctly"""
        # Test case: complete tool call followed by comma at end of chunk (split across 2 chunks)
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tokyo"}'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = "}, "
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)
        self.assertGreater(len(result2.calls), 0, "Should parse complete tool call")

        # Test that next chunk with opening brace gets the separator prepended
        next_chunk = '{"name": "get_weather", "parameters": {"location": "Paris"}}'
        result_next = self.detector.parse_streaming_increment(next_chunk, self.tools)
        self.assertIsInstance(result_next, StreamingParseResult)
        self.assertGreater(
            len(result_next.calls), 0, "Should parse subsequent tool call"
        )

    def test_three_tool_calls_separate_chunks_with_commas(self):
        """Test parsing 3 tool calls in separate chunks with commas at the end"""
        # First tool call: 2 chunks
        chunk1_1 = '[{"name": "get_weather", "parameters": '
        result1_1 = self.detector.parse_streaming_increment(chunk1_1, self.tools)
        chunk1_2 = '{"location": "Tokyo"}},'
        result1_2 = self.detector.parse_streaming_increment(chunk1_2, self.tools)
        self.assertIsInstance(result1_2, StreamingParseResult)
        self.assertGreater(len(result1_2.calls), 0, "Should parse first tool call")

        # Second tool call: 2 chunks
        chunk2_1 = '{"name": "search", "parameters": '
        result2_1 = self.detector.parse_streaming_increment(chunk2_1, self.tools)
        chunk2_2 = '{"query": "restaurants"}},'
        result2_2 = self.detector.parse_streaming_increment(chunk2_2, self.tools)
        self.assertIsInstance(result2_2, StreamingParseResult)
        self.assertGreater(len(result2_2.calls), 0, "Should parse second tool call")

        # Third tool call: 2 chunks
        chunk3_1 = '{"name": "get_weather", "parameters": '
        result3_1 = self.detector.parse_streaming_increment(chunk3_1, self.tools)
        chunk3_2 = '{"location": "Paris"}}]'
        result3_2 = self.detector.parse_streaming_increment(chunk3_2, self.tools)
        self.assertIsInstance(result3_2, StreamingParseResult)
        self.assertGreater(len(result3_2.calls), 0, "Should parse third tool call")
        # Verify all tool calls were parsed correctly
        total_calls = len(result1_2.calls) + len(result2_2.calls) + len(result3_2.calls)
        self.assertEqual(total_calls, 3, "Should have parsed exactly 3 tool calls")


if __name__ == "__main__":
    unittest.main()
