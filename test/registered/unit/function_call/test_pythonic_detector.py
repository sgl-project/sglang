"""Unit tests for PythonicDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


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

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_true(self):
        text = '[get_weather(city="Beijing")]'
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false_plain_text(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    def test_has_tool_call_false_unmatched_bracket(self):
        text = '[get_weather(city="Beijing")'
        self.assertFalse(self.detector.has_tool_call(text))

    def test_has_tool_call_with_python_start_end_tokens(self):
        text = '<|python_start|>[get_weather(city="Beijing")]<|python_end|>'
        self.assertTrue(self.detector.has_tool_call(text))

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call(self):
        text = '[get_weather(city="Beijing")]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(result.normal_text, "")

    def test_multiple_tool_calls(self):
        text = '[get_weather(city="Beijing"), search(query="restaurants")]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")
        self.assertEqual(result.calls[0].tool_index, 0)
        self.assertEqual(result.calls[1].tool_index, 1)

    def test_tool_call_with_multiple_arguments(self):
        text = '[get_weather(city="London", unit="celsius")]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "London")
        self.assertEqual(args["unit"], "celsius")

    def test_python_start_end_tokens_are_stripped(self):
        text = '<|python_start|>[get_weather(city="Tokyo")]<|python_end|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Tokyo")

    def test_no_tool_call_returns_normal_text(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_unknown_tool_is_dropped_by_default(self):
        text = '[unknown_func(x="y"), get_weather(city="Paris")]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_non_literal_argument_returns_no_calls(self):
        text = "[get_weather(city=some_variable)]"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    # ==================== Argument type coverage ====================

    def test_argument_types_int_float_bool_none(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="set_config",
                    description="Configure values",
                    parameters={"type": "object", "properties": {}},
                ),
            )
        ]
        text = "[set_config(count=3, ratio=0.5, enabled=True, label=None)]"
        result = self.detector.detect_and_parse(text, tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["count"], 3)
        self.assertEqual(args["ratio"], 0.5)
        self.assertIs(args["enabled"], True)
        self.assertIsNone(args["label"])

    def test_argument_types_list_and_dict(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="batch",
                    description="Batch op",
                    parameters={"type": "object", "properties": {}},
                ),
            )
        ]
        text = '[batch(items=["a", "b"], meta={"k": 1})]'
        result = self.detector.detect_and_parse(text, tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["items"], ["a", "b"])
        self.assertEqual(args["meta"], {"k": 1})

    # ==================== Capability flags ====================

    def test_supports_structural_tag_is_false(self):
        self.assertFalse(self.detector.supports_structural_tag())

    def test_structure_info_raises(self):
        with self.assertRaises(NotImplementedError):
            self.detector.structure_info()

    # ==================== Streaming Tests ====================

    def test_streaming_single_tool_call(self):
        detector = PythonicDetector()
        chunks = ["[get_weat", 'her(city="Be', 'ijing")]']
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")

        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Beijing")

    def test_streaming_normal_text_before_tool(self):
        detector = PythonicDetector()
        result = detector.parse_streaming_increment(
            "Let me check the weather. ", self.tools
        )
        self.assertEqual(result.normal_text, "Let me check the weather. ")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_text_then_tool_call(self):
        detector = PythonicDetector()
        chunks = [
            "Sure, here you go: ",
            '[get_weather(city="Tokyo"',
            ', unit="celsius")]',
        ]
        all_calls = []
        all_normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal_text += result.normal_text or ""

        self.assertEqual(all_normal_text, "Sure, here you go: ")
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Tokyo")
        self.assertEqual(params["unit"], "celsius")

    def test_streaming_bracket_inside_string_argument(self):
        detector = PythonicDetector()
        text = '[search(query="hello]world")]'
        result = detector.parse_streaming_increment(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["query"], "hello]world")

    def test_streaming_balanced_brackets_inside_string_argument(self):
        detector = PythonicDetector()
        text = '[search(query="abc[def]ghi")]'
        result = detector.parse_streaming_increment(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["query"], "abc[def]ghi")

    def test_streaming_escaped_quote_inside_string_argument(self):
        detector = PythonicDetector()
        text = r'[search(query="he\"llo]world")]'
        result = detector.parse_streaming_increment(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["query"], 'he"llo]world')

    def test_streaming_single_quoted_string_argument(self):
        detector = PythonicDetector()
        text = "[search(query='single]quotes')]"
        result = detector.parse_streaming_increment(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["query"], "single]quotes")

    def test_streaming_partial_python_start_token_held_back(self):
        detector = PythonicDetector()
        result = detector.parse_streaming_increment("<|python_", self.tools)
        self.assertNotIn("<|python_", result.normal_text or "")
        result2 = detector.parse_streaming_increment(
            'start|>[get_weather(city="Paris")]<|python_end|>', self.tools
        )
        self.assertEqual(len(result2.calls), 1)
        self.assertEqual(result2.calls[0].name, "get_weather")


if __name__ == "__main__":
    import unittest

    unittest.main()
