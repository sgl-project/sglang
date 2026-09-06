"""Unit tests for Gemma4Detector — no server, no model loading.

Gemma4 emits tool calls in its own custom wire format::

    <|tool_call>call:get_weather{city:<|"|>Beijing<|"|>, unit:celsius}<tool_call|>

where arguments use a ``key:value`` grammar (strings wrapped in ``<|"|>``),
not JSON. These tests pin the parser helpers and both the one-shot and
streaming detector entry points, including edge cases and negative branches.
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.gemma4_detector import (
    STRING_DELIM,
    TOOL_CALL_END,
    TOOL_CALL_START,
    Gemma4Detector,
    _find_matching_brace,
    _parse_gemma4_args,
    _parse_gemma4_array,
    _parse_gemma4_value,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tools() -> list:
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get weather information",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
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


class TestGemma4Tokens(CustomTestCase):
    def test_special_tokens(self):
        self.assertEqual(TOOL_CALL_START, "<|tool_call>")
        self.assertEqual(TOOL_CALL_END, "<tool_call|>")
        self.assertEqual(STRING_DELIM, '<|"|>')


class TestParseGemma4Value(CustomTestCase):
    def test_empty_string(self):
        self.assertEqual(_parse_gemma4_value(""), "")
        self.assertEqual(_parse_gemma4_value("   "), "")

    def test_booleans(self):
        self.assertIs(_parse_gemma4_value("true"), True)
        self.assertIs(_parse_gemma4_value("false"), False)

    def test_integers(self):
        self.assertEqual(_parse_gemma4_value("42"), 42)
        self.assertEqual(_parse_gemma4_value("-7"), -7)

    def test_floats(self):
        self.assertEqual(_parse_gemma4_value("3.14"), 3.14)
        self.assertEqual(_parse_gemma4_value("-0.5"), -0.5)

    def test_bare_string(self):
        self.assertEqual(_parse_gemma4_value("hello"), "hello")

    def test_whitespace_around_value(self):
        self.assertEqual(_parse_gemma4_value("  42 "), 42)


class TestParseGemma4Array(CustomTestCase):
    def test_empty(self):
        self.assertEqual(_parse_gemma4_array(""), [])
        self.assertEqual(_parse_gemma4_array("   "), [])

    def test_string_elements(self):
        arr = _parse_gemma4_array(
            f"{STRING_DELIM}a{STRING_DELIM}, {STRING_DELIM}b{STRING_DELIM}"
        )
        self.assertEqual(arr, ["a", "b"])

    def test_bare_elements(self):
        self.assertEqual(_parse_gemma4_array("1, 2, 3"), [1, 2, 3])
        self.assertEqual(_parse_gemma4_array("true, false"), [True, False])

    def test_nested_object(self):
        arr = _parse_gemma4_array("{lat:39.9, lng:116.4}")
        self.assertEqual(arr, [{"lat": 39.9, "lng": 116.4}])

    def test_nested_array(self):
        self.assertEqual(_parse_gemma4_array("[1, 2], [3]"), [[1, 2], [3]])

    def test_mixed_elements(self):
        arr = _parse_gemma4_array(f"1, {STRING_DELIM}x{STRING_DELIM}, {{k:v}}")
        self.assertEqual(arr, [1, "x", {"k": "v"}])

    def test_unterminated_string_element_takes_rest(self):
        arr = _parse_gemma4_array(f"{STRING_DELIM}abc")
        self.assertEqual(arr, ["abc"])


class TestParseGemma4Args(CustomTestCase):
    def test_empty(self):
        self.assertEqual(_parse_gemma4_args(""), {})
        self.assertEqual(_parse_gemma4_args("  "), {})

    def test_string_value(self):
        args = _parse_gemma4_args(f"city:{STRING_DELIM}Beijing{STRING_DELIM}")
        self.assertEqual(args, {"city": "Beijing"})

    def test_multiple_keys(self):
        args = _parse_gemma4_args(
            f"city:{STRING_DELIM}Beijing{STRING_DELIM}, unit:{STRING_DELIM}celsius{STRING_DELIM}"
        )
        self.assertEqual(args, {"city": "Beijing", "unit": "celsius"})

    def test_bare_numeric_and_boolean_values(self):
        args = _parse_gemma4_args("temperature:25, windy:true, rainy:false")
        self.assertEqual(args, {"temperature": 25, "windy": True, "rainy": False})

    def test_nested_object(self):
        args = _parse_gemma4_args("location:{lat:39.9, lng:116.4}")
        self.assertEqual(args, {"location": {"lat": 39.9, "lng": 116.4}})

    def test_array_value(self):
        args = _parse_gemma4_args(
            f"tags:[{STRING_DELIM}a{STRING_DELIM}, {STRING_DELIM}b{STRING_DELIM}]"
        )
        self.assertEqual(args, {"tags": ["a", "b"]})

    def test_unterminated_string_takes_rest(self):
        args = _parse_gemma4_args(f"city:{STRING_DELIM}Beijing")
        self.assertEqual(args, {"city": "Beijing"})

    def test_missing_value_after_colon(self):
        args = _parse_gemma4_args("city:")
        self.assertEqual(args, {"city": ""})

    def test_deeply_nested_mixed(self):
        args = _parse_gemma4_args(
            f"outer:{{inner:{{v:{STRING_DELIM}x{STRING_DELIM}}}, list:[1, 2]}}"
        )
        self.assertEqual(args, {"outer": {"inner": {"v": "x"}, "list": [1, 2]}})


class TestFindMatchingBrace(CustomTestCase):
    def test_simple(self):
        self.assertEqual(_find_matching_brace("abc}"), 3)

    def test_nested(self):
        self.assertEqual(_find_matching_brace("a{b}c}"), 5)

    def test_delim_braces_ignored(self):
        # Braces inside <|"|> delimiters must not affect brace balancing.
        text = "a" + STRING_DELIM + "}" + STRING_DELIM + "b}"
        self.assertEqual(_find_matching_brace(text), len(text) - 1)

    def test_unclosed_returns_minus_one(self):
        self.assertEqual(_find_matching_brace("abc"), -1)
        self.assertEqual(_find_matching_brace(f"a{STRING_DELIM}unclosed"), -1)


class TestExtractToolCalls(CustomTestCase):
    def test_no_tool_call(self):
        self.assertEqual(Gemma4Detector._extract_tool_calls("hello"), [])

    def test_single(self):
        text = f"{TOOL_CALL_START}call:get_weather{{city:{STRING_DELIM}Beijing{STRING_DELIM}}}{TOOL_CALL_END}"
        calls = Gemma4Detector._extract_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "get_weather")
        self.assertIn("city", calls[0][1])

    def test_multiple(self):
        text = (
            f"{TOOL_CALL_START}call:get_weather{{city:x}}{TOOL_CALL_END}"
            f"{TOOL_CALL_START}call:search{{query:y}}{TOOL_CALL_END}"
        )
        calls = Gemma4Detector._extract_tool_calls(text)
        self.assertEqual([c[0] for c in calls], ["get_weather", "search"])

    def test_missing_end_token(self):
        text = f"{TOOL_CALL_START}call:get_weather{{city:x}}"
        self.assertEqual(Gemma4Detector._extract_tool_calls(text), [])

    def test_non_call_inner_skipped(self):
        text = f"{TOOL_CALL_START}garbage{{x:1}}{TOOL_CALL_END}"
        self.assertEqual(Gemma4Detector._extract_tool_calls(text), [])

    def test_unclosed_brace_takes_rest(self):
        text = f"{TOOL_CALL_START}call:get_weather{{city:{STRING_DELIM}Beijing{TOOL_CALL_END}"
        calls = Gemma4Detector._extract_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "get_weather")


class TestGemma4Detector(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()
        self.detector = Gemma4Detector()

    # ==================== has_tool_call ====================

    def test_has_tool_call_true(self):
        self.assertTrue(self.detector.has_tool_call(f"hi {TOOL_CALL_START}..."))

    def test_has_tool_call_false(self):
        self.assertFalse(
            self.detector.has_tool_call("The weather in Beijing is sunny.")
        )

    # ==================== detect_and_parse ====================

    def test_no_tool_call_returns_original_text(self):
        result = self.detector.detect_and_parse(
            "The weather is nice today.", self.tools
        )
        self.assertIsInstance(result, StreamingParseResult)
        self.assertEqual(result.normal_text, "The weather is nice today.")
        self.assertEqual(result.calls, [])

    def test_single_tool_call(self):
        text = f"{TOOL_CALL_START}call:get_weather{{city:{STRING_DELIM}Beijing{STRING_DELIM}}}{TOOL_CALL_END}"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        call = result.calls[0]
        self.assertEqual(call.name, "get_weather")
        self.assertEqual(call.tool_index, 0)
        args = json.loads(call.parameters)
        self.assertEqual(args, {"city": "Beijing"})
        self.assertEqual(result.normal_text, "")

    def test_multiple_tool_calls(self):
        text = (
            f"{TOOL_CALL_START}call:get_weather{{city:x}}{TOOL_CALL_END}"
            f"{TOOL_CALL_START}call:search{{query:y}}{TOOL_CALL_END}"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual([c.name for c in result.calls], ["get_weather", "search"])
        self.assertEqual([c.tool_index for c in result.calls], [0, 1])

    def test_tool_call_with_leading_text(self):
        text = (
            f"I will check. {TOOL_CALL_START}call:get_weather{{city:x}}{TOOL_CALL_END}"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "I will check. ")
        self.assertEqual(len(result.calls), 1)

    def test_unknown_tool_name_index_is_minus_one(self):
        text = f"{TOOL_CALL_START}call:no_such_tool{{city:x}}{TOOL_CALL_END}"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "no_such_tool")
        self.assertEqual(result.calls[0].tool_index, -1)

    def test_rich_arguments_round_trip(self):
        text = (
            f"{TOOL_CALL_START}call:get_weather{{"
            f"city:{STRING_DELIM}Beijing{STRING_DELIM}, "
            f"temperature:25, windy:true, "
            f"tags:[{STRING_DELIM}a{STRING_DELIM}, {STRING_DELIM}b{STRING_DELIM}], "
            f"location:{{lat:39.9, lng:116.4}}"
            f"}}{TOOL_CALL_END}"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(args["temperature"], 25)
        self.assertIs(args["windy"], True)
        self.assertEqual(args["tags"], ["a", "b"])
        self.assertEqual(args["location"], {"lat": 39.9, "lng": 116.4})

    def test_malformed_tool_call_returns_original_text(self):
        text = f"{TOOL_CALL_START}not a call{TOOL_CALL_END}"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, text)

    def test_empty_normal_text_when_call_starts_at_beginning(self):
        text = f"{TOOL_CALL_START}call:get_weather{{city:x}}{TOOL_CALL_END}"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "")

    # ==================== parse_streaming_increment ====================

    def test_streaming_plain_text(self):
        result = self.detector.parse_streaming_increment(
            "Hello! Let me help. ", self.tools
        )
        self.assertEqual(result.normal_text, "Hello! Let me help. ")
        self.assertEqual(result.calls, [])

    def test_streaming_single_tool_call(self):
        chunks = [
            f"{TOOL_CALL_START}call:get_weather{{",
            f"city:{STRING_DELIM}Beijing{STRING_DELIM}",
            f"}}{TOOL_CALL_END}",
        ]
        all_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        named = [c for c in all_calls if c.name]
        self.assertEqual(len(named), 1)
        self.assertEqual(named[0].name, "get_weather")

        params = "".join(c.parameters for c in all_calls if c.parameters)
        self.assertEqual(json.loads(params), {"city": "Beijing"})

    def test_streaming_text_then_tool_call(self):
        chunks = [
            "Sure, let me check. ",
            f"{TOOL_CALL_START}call:get_weather{{",
            f"city:{STRING_DELIM}Tokyo{STRING_DELIM}",
            f"}}{TOOL_CALL_END}",
        ]
        all_calls = []
        all_normal = ""
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal += result.normal_text

        self.assertEqual(all_normal, "Sure, let me check. ")
        named = [c for c in all_calls if c.name]
        self.assertEqual(len(named), 1)
        self.assertEqual(named[0].name, "get_weather")
        params = "".join(c.parameters for c in all_calls if c.parameters)
        self.assertEqual(json.loads(params), {"city": "Tokyo"})

    def test_streaming_partial_start_token(self):
        # The start token split across chunks must not be emitted as text.
        results = []
        for chunk in ["<|tool", "_call>"]:
            results.append(self.detector.parse_streaming_increment(chunk, self.tools))
        # No normal text leaked from the partial token.
        self.assertEqual(results[0].normal_text, "")
        self.assertEqual(results[1].normal_text, "")

    def test_streaming_multiple_tool_calls(self):
        chunks = [
            f"{TOOL_CALL_START}call:get_weather{{city:x}}{TOOL_CALL_END}",
            f"{TOOL_CALL_START}call:search{{query:y}}{TOOL_CALL_END}",
        ]
        all_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
        named = [c for c in all_calls if c.name]
        self.assertEqual([c.name for c in named], ["get_weather", "search"])
        self.assertEqual([c.tool_index for c in named], [0, 1])

    def test_streaming_state_resets_between_buffers(self):
        # A fresh detector must start clean.
        d = Gemma4Detector()
        self.assertEqual(
            d.parse_streaming_increment("plain text", self.tools).normal_text,
            "plain text",
        )
        self.assertEqual(d._buffer, "")
        self.assertEqual(d.parsed_pos, 0)

    # ==================== structural tag ====================

    def test_supports_structural_tag_false(self):
        self.assertFalse(self.detector.supports_structural_tag())

    def test_structure_info_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.detector.structure_info()


if __name__ == "__main__":
    unittest.main()
