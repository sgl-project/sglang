import json
import unittest

from xgrammar import GrammarCompiler, TokenizerInfo

from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.srt.function_call.mistral_detector import MistralDetector
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.openai_api.protocol import Function, Tool
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


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


class TestEBNFGeneration(unittest.TestCase):
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

        self.tokenizer = get_tokenizer(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
        tokenizer_info = TokenizerInfo.from_huggingface(self.tokenizer)
        self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)

        # Initialize all detectors
        self.pythonic_detector = PythonicDetector()
        self.deepseekv3_detector = DeepSeekV3Detector()
        self.llama32_detector = Llama32Detector()
        self.mistral_detector = MistralDetector()
        self.qwen25_detector = Qwen25Detector()

    def test_pythonic_detector_ebnf(self):
        """Test that the PythonicDetector generates valid EBNF."""
        ebnf = self.pythonic_detector.build_ebnf(self.tools)
        self.assertIsNotNone(ebnf)

        # Check that the EBNF contains expected patterns
        self.assertIn('call_get_weather ::= "get_weather" "(" ', ebnf)
        self.assertIn('"location" "=" basic_string', ebnf)
        self.assertIn('[ "unit" "=" ("\\"celsius\\"" | "\\"fahrenheit\\"") ]', ebnf)

        # Validate that the EBNF can be compiled by GrammarCompiler
        try:
            ctx = self.grammar_compiler.compile_grammar(ebnf)
            self.assertIsNotNone(ctx, "EBNF should be valid and compile successfully")
        except RuntimeError as e:
            self.fail(f"Failed to compile EBNF: {e}")

    def test_deepseekv3_detector_ebnf(self):
        """Test that the DeepSeekV3Detector generates valid EBNF."""
        ebnf = self.deepseekv3_detector.build_ebnf(self.tools)
        self.assertIsNotNone(ebnf)

        # Check that the EBNF contains expected patterns
        self.assertIn("<｜tool▁calls▁begin｜>", ebnf)
        self.assertIn("<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather", ebnf)
        self.assertIn('\\"location\\"" ":" basic_string ', ebnf)

        # Validate that the EBNF can be compiled by GrammarCompiler
        try:
            ctx = self.grammar_compiler.compile_grammar(ebnf)
            self.assertIsNotNone(ctx, "EBNF should be valid and compile successfully")
        except RuntimeError as e:
            self.fail(f"Failed to compile EBNF: {e}")

    def test_llama32_detector_ebnf(self):
        """Test that the Llama32Detector generates valid EBNF."""
        ebnf = self.llama32_detector.build_ebnf(self.tools)
        self.assertIsNotNone(ebnf)

        # Check that the EBNF contains expected patterns
        self.assertIn('\\"name\\"" ":" "\\"get_weather\\"', ebnf)
        self.assertIn('"\\"arguments\\"" ":"', ebnf)

        # Validate that the EBNF can be compiled by GrammarCompiler
        try:
            ctx = self.grammar_compiler.compile_grammar(ebnf)
            self.assertIsNotNone(ctx, "EBNF should be valid and compile successfully")
        except RuntimeError as e:
            self.fail(f"Failed to compile EBNF: {e}")

    def test_mistral_detector_ebnf(self):
        """Test that the MistralDetector generates valid EBNF."""
        ebnf = self.mistral_detector.build_ebnf(self.tools)
        self.assertIsNotNone(ebnf)

        # Check that the EBNF contains expected patterns
        self.assertIn('"[TOOL_CALLS] ["', ebnf)
        self.assertIn("call_get_weather | call_search", ebnf)
        self.assertIn('"\\"arguments\\"" ":"', ebnf)

        # Validate that the EBNF can be compiled by GrammarCompiler
        try:
            ctx = self.grammar_compiler.compile_grammar(ebnf)
            self.assertIsNotNone(ctx, "EBNF should be valid and compile successfully")
        except RuntimeError as e:
            self.fail(f"Failed to compile EBNF: {e}")

    def test_qwen25_detector_ebnf(self):
        """Test that the Qwen25Detector generates valid EBNF."""
        ebnf = self.qwen25_detector.build_ebnf(self.tools)
        self.assertIsNotNone(ebnf)

        # Check that the EBNF contains expected patterns
        self.assertIn("<tool_call>", ebnf)
        self.assertIn('\\"name\\"" ":" "\\"get_weather\\"', ebnf)
        self.assertIn('"\\"arguments\\"" ":"', ebnf)

        # Validate that the EBNF can be compiled by GrammarCompiler
        try:
            ctx = self.grammar_compiler.compile_grammar(ebnf)
            self.assertIsNotNone(ctx, "EBNF should be valid and compile successfully")
        except RuntimeError as e:
            self.fail(f"Failed to compile EBNF: {e}")


if __name__ == "__main__":
    unittest.main()
