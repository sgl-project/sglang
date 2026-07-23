"""Unit tests for PythonicDetector — no server, no model loading."""

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestPythonicDetector(CustomTestCase):
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
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search the web",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                ),
            ),
        ]
        self.detector = PythonicDetector()

    # ==================== _text_strip Tests ====================

    def test_text_strip_removes_python_start_token(self):
        text = "<|python_start|>get_weather(city='Tokyo')"
        result = PythonicDetector._text_strip(text)
        self.assertEqual(result, "get_weather(city='Tokyo')")

    def test_text_strip_removes_python_end_token(self):
        text = "get_weather(city='Tokyo')<|python_end|>"
        result = PythonicDetector._text_strip(text)
        self.assertEqual(result, "get_weather(city='Tokyo')")

    def test_text_strip_removes_both_tokens(self):
        text = "<|python_start|>get_weather(city='Tokyo')<|python_end|>"
        result = PythonicDetector._text_strip(text)
        self.assertEqual(result, "get_weather(city='Tokyo')")

    def test_text_strip_no_op_without_tokens(self):
        text = "get_weather(city='Tokyo')"
        result = PythonicDetector._text_strip(text)
        self.assertEqual(result, text)

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_single(self):
        text = "[get_weather(city='Tokyo')]"
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_multiple(self):
        text = "[get_weather(city='Tokyo'), search(query='news')]"
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        text = "The weather in Tokyo is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    def test_has_tool_call_with_python_tokens(self):
        text = "<|python_start|>[get_weather(city='Tokyo')]<|python_end|>"
        self.assertTrue(self.detector.has_tool_call(text))

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call_string_arg(self):
        text = "[get_weather(city='Tokyo')]"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[0].parameters, '{"city": "Tokyo"}')
        self.assertEqual(result.normal_text, "")

    def test_single_tool_call_int_arg(self):
        text = "[get_weather(city=123)]"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_positional_arguments_ignored(self):
        # Current implementation only parses keywords; positional args are ignored
        text = "[get_weather('Tokyo')]"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, "{}")

    def test_multiple_tool_calls(self):
        text = "[get_weather(city='Tokyo'), search(query='restaurants')]"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[0].tool_index, 0)
        self.assertEqual(result.calls[0].parameters, '{"city": "Tokyo"}')
        self.assertEqual(result.calls[1].name, "search")
        self.assertEqual(result.calls[1].tool_index, 1)
        self.assertEqual(result.calls[1].parameters, '{"query": "restaurants"}')

    def test_tool_call_with_leading_text(self):
        text = "I will check the weather for you. [get_weather(city='Tokyo')]"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertIn("check the weather", result.normal_text)

    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_tool_call_with_dict_arg(self):
        text = "[search(query={'key': 'value', 'nested': {'a': 1}})]"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")

    def test_tool_call_with_list_arg(self):
        text = "[search(query=['a', 'b', 'c'])]"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")

    def test_unknown_tool_name_skipped(self):
        text = "[unknown_func(arg=1)]"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_malformed_python_returns_empty_calls(self):
        text = "[get_weather(city=)]"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_no_bracket_returns_empty_calls(self):
        text = "get_weather(city='Tokyo')"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    # ==================== _find_matching_bracket Tests ====================

    def test_find_matching_bracket_simple(self):
        self.assertEqual(self.detector._find_matching_bracket("[abc]", 0), 4)

    def test_find_matching_bracket_nested(self):
        self.assertEqual(
            self.detector._find_matching_bracket("[outer [inner] end]", 0), 18
        )

    def test_find_matching_bracket_in_string(self):
        # Note: The current implementation is not string-aware and will fail this test
        text = '[func(arg="]")]'
        self.assertEqual(self.detector._find_matching_bracket(text, 0), 14)

    def test_find_matching_bracket_no_match(self):
        self.assertEqual(self.detector._find_matching_bracket("[abc", 0), -1)

    def test_find_matching_bracket_not_at_open(self):
        self.assertEqual(self.detector._find_matching_bracket("x[abc]", 1), 5)

    # ==================== parse_streaming_increment Tests ====================

    def test_streaming_single_tool_call_chunked(self):
        detector = PythonicDetector()
        chunks = [
            "[get_",
            "weather(city=",
            "'Tokyo'",
            ")]",
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")

    def test_streaming_normal_text_before_tool(self):
        detector = PythonicDetector()
        result = detector.parse_streaming_increment("Hello! Let me help. ", self.tools)
        self.assertEqual(result.normal_text, "Hello! Let me help. ")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_text_then_tool_call(self):
        detector = PythonicDetector()
        chunks = [
            "Sure, let me check. ",
            "[get_weather(city=",
            "'Tokyo')]",
        ]
        all_calls = []
        all_normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal_text += result.normal_text

        self.assertIn("Sure, let me check. ", all_normal_text)
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")

    def test_streaming_with_python_tokens(self):
        detector = PythonicDetector()
        chunks = [
            "<|python_start|>[get_",
            "weather(city='Tokyo')]<|python_end|>",
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")

    def test_streaming_incomplete_tool_call(self):
        detector = PythonicDetector()
        result = detector.parse_streaming_increment("[get_weather(city=", self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "")

        # Test with leading text
        detector2 = PythonicDetector()
        result = detector2.parse_streaming_increment(
            "Hello! [search(query=", self.tools
        )
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "Hello! ")

    # ==================== structure_info / supports_structural_tag Tests ====================

    def test_supports_structural_tag_false(self):
        self.assertFalse(self.detector.supports_structural_tag())

    def test_structure_info_raises(self):
        with self.assertRaises(NotImplementedError):
            self.detector.structure_info()()


if __name__ == "__main__":
    import unittest

    unittest.main()
