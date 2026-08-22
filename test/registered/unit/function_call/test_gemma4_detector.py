"""Unit tests for Gemma4Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.gemma4_detector import (
    Gemma4Detector,
    _parse_gemma4_args,
    _parse_gemma4_array,
    _parse_gemma4_value,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestGemma4Detector(CustomTestCase):
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
                            "location": {"type": "string"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                ),
            )
        ]
        self.detector = Gemma4Detector()

    def test_detect_and_parse(self):
        text = 'Some text before <|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "Some text before ")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "Tokyo")

    def test_parse_streaming_increment(self):
        chunks = [
            "Some text ",
            "before <|tool",
            "_call>call:get_we",
            "ather{location:<|",  # codespell:ignore
            '"|>Tokyo<|"|>}<tool_',
            "call|> after",
        ]

        all_results = []
        for chunk in chunks:
            res = self.detector.parse_streaming_increment(chunk, self.tools)
            all_results.append(res)

        combined_normal_text = "".join(r.normal_text for r in all_results)
        self.assertEqual(combined_normal_text, "Some text before  after")

        found_name = False
        found_params = False
        for res in all_results:
            for call in res.calls:
                if call.name == "get_weather":
                    found_name = True
                if call.parameters:
                    params = json.loads(call.parameters)
                    if params == {"location": "Tokyo"}:
                        found_params = True

        self.assertTrue(found_name)
        self.assertTrue(found_params)

    def test_nested_array_streaming(self):
        # Additional coverage for complex structure
        chunks = [
            '<|tool_call>call:get_weather{location:<|"',
            '|>New York<|"|>,nested:[1, 2, {inner:<|"|>',
            'val<|"|>}]}<tool_call|>',
        ]

        all_results = []
        for chunk in chunks:
            res = self.detector.parse_streaming_increment(chunk, self.tools)
            all_results.append(res)

        found_params = False
        for res in all_results:
            for call in res.calls:
                if call.parameters:
                    params = json.loads(call.parameters)
                    if "location" in params and params["location"] == "New York":
                        if "nested" in params and params["nested"] == [
                            1,
                            2,
                            {"inner": "val"},
                        ]:
                            found_params = True

        self.assertTrue(found_params)

    def test_detect_and_parse_no_tool_call(self):
        text = "This is plain text without any tool calls."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(len(result.calls), 0)

    def test_detect_and_parse_unknown_tool_index(self):
        text = '<|tool_call>call:unknown_func{arg:<|"|>val<|"|>}<tool_call|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].tool_index, -1)

    def test_detect_and_parse_nested_object(self):
        text = '<|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>,details:{temp:25,unit:<|"|>celsius<|"|>}}<tool_call|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "Tokyo")
        self.assertIsInstance(params["details"], dict)
        self.assertEqual(params["details"]["temp"], 25)
        self.assertEqual(params["details"]["unit"], "celsius")

    def test_detect_and_parse_multiple_calls(self):
        extra_tools = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_time",
                    description="Get current time",
                    parameters={
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                    },
                ),
            )
        ]
        text = (
            'Some text <|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|>'
            ' more text <|tool_call>call:get_time{timezone:<|"|>UTC<|"|>}<tool_call|>'
        )
        result = self.detector.detect_and_parse(text, extra_tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "get_time")
        self.assertEqual(result.normal_text, "Some text ")

    def test_parse_gemma4_args_empty(self):
        self.assertEqual(_parse_gemma4_args(""), {})
        self.assertEqual(_parse_gemma4_args("   "), {})

    def test_parse_gemma4_args_booleans(self):
        result = _parse_gemma4_args("flag:true,other:false")
        self.assertIs(result["flag"], True)
        self.assertIs(result["other"], False)

    def test_parse_gemma4_args_string_with_colon(self):
        result = _parse_gemma4_args('url:<|"|>http://example.com<|"|>')
        self.assertEqual(result["url"], "http://example.com")

    def test_parse_gemma4_args_nested_object(self):
        result = _parse_gemma4_args('outer:{inner:<|"|>val<|"|>,num:5}')
        self.assertIsInstance(result["outer"], dict)
        self.assertEqual(result["outer"]["inner"], "val")
        self.assertEqual(result["outer"]["num"], 5)

    def test_parse_gemma4_array_mixed_types(self):
        result = _parse_gemma4_array('<|"|>hello<|"|>, 42, true, {key:<|"|>val<|"|>}')
        self.assertEqual(result[0], "hello")
        self.assertEqual(result[1], 42)
        self.assertIs(result[2], True)
        self.assertIsInstance(result[3], dict)
        self.assertEqual(result[3]["key"], "val")

    def test_parse_gemma4_value_types(self):
        self.assertIs(_parse_gemma4_value("true"), True)
        self.assertIs(_parse_gemma4_value("false"), False)
        self.assertEqual(_parse_gemma4_value("42"), 42)
        self.assertAlmostEqual(_parse_gemma4_value("3.14"), 3.14)
        self.assertEqual(_parse_gemma4_value("hello"), "hello")
        self.assertEqual(_parse_gemma4_value(""), "")

    def _collect_streaming(self, chunks):
        """Helper: feed chunks and collect normal text + tool calls by index."""
        normal_text = ""
        tool_calls_by_index = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            normal_text += result.normal_text
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
        return normal_text, tool_calls_by_index

    def test_streaming_multiple_tool_calls(self):
        """Test streaming with two consecutive tool calls."""
        extra_tools = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_time",
                    description="Get current time",
                    parameters={
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                    },
                ),
            )
        ]
        chunks = [
            '<|tool_call>call:get_weather{location:<|"|>',
            'Tokyo<|"|>}<tool_call|>',
            ' <|tool_call>call:get_time{timezone:<|"|>',
            'UTC<|"|>}<tool_call|>',
        ]
        normal_text = ""
        tool_calls_by_index = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, extra_tools)
            normal_text += result.normal_text
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

        self.assertEqual(len(tool_calls_by_index), 2)
        self.assertEqual(tool_calls_by_index[0]["name"], "get_weather")
        self.assertEqual(tool_calls_by_index[1]["name"], "get_time")
        params0 = json.loads(tool_calls_by_index[0]["parameters"])
        params1 = json.loads(tool_calls_by_index[1]["parameters"])
        self.assertEqual(params0["location"], "Tokyo")
        self.assertEqual(params1["timezone"], "UTC")

    def test_streaming_very_small_chunks(self):
        """Test streaming with character-by-character chunks."""
        full_text = '<|tool_call>call:get_weather{location:<|"|>Rome<|"|>}<tool_call|>'
        chunks = list(full_text)

        normal_text, tool_calls = self._collect_streaming(chunks)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        params = json.loads(tool_calls[0]["parameters"])
        self.assertEqual(params["location"], "Rome")

    def test_streaming_empty_args(self):
        """Test streaming a tool call with no arguments."""
        chunks = ["<|tool_call>call:get_weather{}", "<tool_call|>"]
        normal_text, tool_calls = self._collect_streaming(chunks)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "get_weather")

    def test_streaming_text_between_tool_calls(self):
        """Test streaming with normal text interleaved between two different tool calls."""
        extra_tools = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_time",
                    description="Get current time",
                    parameters={
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                    },
                ),
            )
        ]
        chunks = [
            "Hello! ",
            '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|>',
            " Let me also check ",
            '<|tool_call>call:get_time{timezone:<|"|>UTC<|"|>}<tool_call|>',
        ]
        normal_text = ""
        tool_calls_by_index = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, extra_tools)
            normal_text += result.normal_text
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
        self.assertIn("Hello!", normal_text)
        self.assertIn("Let me also check", normal_text)
        self.assertEqual(len(tool_calls_by_index), 2)
        self.assertEqual(tool_calls_by_index[0]["name"], "get_weather")
        self.assertEqual(tool_calls_by_index[1]["name"], "get_time")
        params0 = json.loads(tool_calls_by_index[0]["parameters"])
        params1 = json.loads(tool_calls_by_index[1]["parameters"])
        self.assertEqual(params0["location"], "Paris")
        self.assertEqual(params1["timezone"], "UTC")


if __name__ == "__main__":
    import unittest

    unittest.main()
