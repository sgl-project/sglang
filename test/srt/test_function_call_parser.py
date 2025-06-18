import json
import unittest

from xgrammar import GrammarCompiler, TokenizerInfo

from sglang.srt.function_call.base_format_detector import BaseFormatDetector
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
        self.assertIn('( "unit" "=" ("\\"celsius\\"" | "\\"fahrenheit\\"") )', ebnf)

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

    def test_weather_function_optional_parameter_handling(self):
        """Test that weather function with optional unit parameter generates correct EBNF without trailing commas."""
        # Create a weather tool with required location and optional unit
        weather_tool = Tool(
            type="function",
            function=Function(
                name="get_current_weather",
                description="Get the current weather in a given location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            ),
        )

        # Test all detectors with the weather tool
        detectors = {
            "pythonic": self.pythonic_detector,
            "deepseekv3": self.deepseekv3_detector,
            "llama32": self.llama32_detector,
            "mistral": self.mistral_detector,
            "qwen25": self.qwen25_detector,
        }

        for name, detector in detectors.items():
            with self.subTest(detector=name):
                ebnf = detector.build_ebnf([weather_tool])
                self.assertIsNotNone(ebnf, f"{name} detector should generate EBNF")

                # Check that the EBNF properly handles optional parameters
                if name == "pythonic":
                    # Pythonic format: location="Paris" ( , ( unit=("celsius" | "fahrenheit") )?
                    self.assertIn('"location" "=" basic_string', ebnf)
                    # The comma should be inside the optional brackets for unit
                    self.assertIn('( "," ( "unit" "=" ', ebnf)
                else:
                    # JSON format: "location": "Paris" ( , ( "unit": ("celsius" | "fahrenheit") )?
                    self.assertIn('"location\\"" ":" basic_string', ebnf)
                    # The comma should be part of the optional group
                    # This pattern ensures no trailing comma when unit is omitted
                    self.assertIn('( "," ( "\\"unit\\"" ":"', ebnf)

                # Validate that the EBNF can be compiled
                try:
                    ctx = self.grammar_compiler.compile_grammar(ebnf)
                    self.assertIsNotNone(
                        ctx, f"{name} EBNF should compile successfully"
                    )
                except RuntimeError as e:
                    self.fail(f"Failed to compile {name} EBNF: {e}")

    def test_multiple_optional_parameters_flexible_ordering(self):
        """Test that multiple optional parameters allow flexible ordering using llama.cpp approach."""
        # Create a tool with one required and multiple optional parameters
        test_tool = Tool(
            type="function",
            function=Function(
                name="test_func",
                description="Test function with multiple optional parameters",
                parameters={
                    "type": "object",
                    "properties": {
                        "required_field": {"type": "string"},
                        "opt1": {"type": "number"},
                        "opt2": {"type": "boolean"},
                        "opt3": {"type": "string"},
                    },
                    "required": ["required_field"],
                },
            ),
        )

        # Test JSON-based detectors (not pythonic)
        json_detectors = {
            "deepseekv3": self.deepseekv3_detector,
            "llama32": self.llama32_detector,
            "mistral": self.mistral_detector,
            "qwen25": self.qwen25_detector,
        }

        for name, detector in json_detectors.items():
            with self.subTest(detector=name):
                ebnf = detector.build_ebnf([test_tool])
                self.assertIsNotNone(ebnf, f"{name} detector should generate EBNF")

                # Print the arguments rule for debugging
                lines = ebnf.split("\n")
                args_rule = None
                for line in lines:
                    if line.startswith("arguments_test_func ::="):
                        args_rule = line
                        break

                self.assertIsNotNone(
                    args_rule, f"{name} should have arguments_test_func rule"
                )

                # Check required field
                self.assertIn('"required_field\\"" ":" basic_string', ebnf)

                # Check the structure for optional parameters
                # The pattern should be: required_field ( "," ( opt1 ... | opt2 ... | opt3 ... ) )?
                # This allows flexible ordering where any optional can be first

                # Check that optional parameters are in a group with comma
                if args_rule:  # Only check if args_rule was found
                    self.assertIn(
                        '( ","',
                        args_rule,
                        f"{name} should have comma grouped with optional parameters",
                    )

                    # Check for the alternation pattern that allows flexible ordering
                    # Should contain patterns like: opt1 ... | opt2 ... | opt3
                    self.assertIn('"opt1\\"" ":" basic_number', args_rule)
                    self.assertIn('"opt2\\"" ":" basic_boolean', args_rule)
                    self.assertIn('"opt3\\"" ":" basic_string', args_rule)

                    # Check for alternation (|) which allows skipping optional parameters
                    self.assertIn(
                        "|",
                        args_rule,
                        f"{name} should use alternation for flexible optional ordering",
                    )

                    # Check that the pattern ends properly with closing braces
                    self.assertTrue(
                        args_rule.endswith('"}"'),
                        f"{name} arguments rule should end with closing brace",
                    )

                # Validate compilation
                try:
                    ctx = self.grammar_compiler.compile_grammar(ebnf)
                    self.assertIsNotNone(
                        ctx, f"{name} EBNF should compile successfully"
                    )
                except RuntimeError as e:
                    self.fail(f"Failed to compile {name} EBNF: {e}")

    def test_all_optional_parameters_ordering(self):
        """Test the behavior when ALL parameters are optional - verifies ordering constraints."""
        # Create a tool with only optional parameters
        all_optional_tool = Tool(
            type="function",
            function=Function(
                name="optional_func",
                description="Function with all optional parameters",
                parameters={
                    "type": "object",
                    "properties": {
                        "opt1": {"type": "string"},
                        "opt2": {"type": "number"},
                        "opt3": {"type": "boolean"},
                    },
                    "required": [],  # No required parameters
                },
            ),
        )

        # Test JSON-based detectors
        json_detectors = {
            "deepseekv3": self.deepseekv3_detector,
            "llama32": self.llama32_detector,
            "mistral": self.mistral_detector,
            "qwen25": self.qwen25_detector,
        }

        for name, detector in json_detectors.items():
            with self.subTest(detector=name):
                ebnf = detector.build_ebnf([all_optional_tool])
                self.assertIsNotNone(ebnf, f"{name} detector should generate EBNF")

                # Extract the arguments rule
                lines = ebnf.split("\n")
                args_rule = None
                for line in lines:
                    if line.startswith("arguments_optional_func ::="):
                        args_rule = line
                        break

                self.assertIsNotNone(
                    args_rule, f"{name} should have arguments_optional_func rule"
                )

                if args_rule:
                    # When all parameters are optional, the pattern now uses alternation:
                    # "{" ( opt1 ... | opt2 ... | opt3 ... )? "}"
                    # This allows flexible ordering where any optional can appear first

                    # Check the structure
                    self.assertIn('"opt1\\"" ":" basic_string', args_rule)
                    self.assertIn('"opt2\\"" ":" basic_number', args_rule)
                    self.assertIn('"opt3\\"" ":" basic_boolean', args_rule)

                    # The pattern SHOULD have alternation (|) for flexible ordering
                    self.assertIn(
                        "|",
                        args_rule,
                        f"{name} should use alternation for flexible ordering even when all properties are optional",
                    )

                # Validate compilation
                try:
                    ctx = self.grammar_compiler.compile_grammar(ebnf)
                    self.assertIsNotNone(
                        ctx, f"{name} EBNF should compile successfully"
                    )
                except RuntimeError as e:
                    self.fail(f"Failed to compile {name} EBNF: {e}")


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

            def build_ebnf(self, tools):
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


if __name__ == "__main__":
    unittest.main()
