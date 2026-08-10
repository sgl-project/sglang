"""Unit tests for Glm47MoeDetector (GLM-4.7 / GLM-5) — no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _weather_call(*pairs):
    """Build a GLM-4.7 tool call (no separators between tags) for get_weather."""
    body = "".join(
        f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
        for key, value in pairs
    )
    return f"<tool_call>get_weather{body}</tool_call>"


class TestGlm47MoeDetector(CustomTestCase):
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
                            "city": {"type": "string"},
                            "count": {"type": "number"},
                            "filters": {"type": "object"},
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="noop",
                    description="Takes no arguments",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search the web",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                ),
            ),
        ]

    def _stream(self, detector, text, chunk_size):
        calls = []
        normal_text = ""
        for i in range(0, len(text), chunk_size):
            result = detector.parse_streaming_increment(
                text[i : i + chunk_size], self.tools
            )
            calls.extend(result.calls)
            normal_text += result.normal_text
        return calls, normal_text

    def _group_streamed_calls(self, calls):
        """Group increments by tool_index into (name, concatenated params)."""
        grouped = {}
        for item in calls:
            name, params = grouped.get(item.tool_index, (None, ""))
            if item.name is not None:
                name = item.name
            if item.parameters:
                params += item.parameters
            grouped[item.tool_index] = (name, params)
        return [grouped[index] for index in sorted(grouped)]

    # ==================== has_tool_call ====================

    def test_has_tool_call_detection(self):
        """Detects the exact bot marker, but not plain text."""
        self.assertTrue(Glm47MoeDetector().has_tool_call("<tool_call>"))
        self.assertFalse(
            Glm47MoeDetector().has_tool_call("The weather in Tokyo is sunny.")
        )

    # ==================== detect_and_parse ====================

    def test_schema_type_coercion(self):
        """Quoted digits in a number parameter are coerced to an int."""
        text = _weather_call(
            ("city", "北京"), ("count", '"42"'), ("filters", '{"a": 1}')
        )
        result = Glm47MoeDetector().detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        arguments = json.loads(result.calls[0].parameters)
        self.assertEqual(arguments, {"city": "北京", "count": 42, "filters": {"a": 1}})
        self.assertIs(type(arguments["count"]), int)

    def test_string_typed_digits_stay_string(self):
        """A digits-only value for a string parameter remains a string."""
        result = Glm47MoeDetector().detect_and_parse(
            _weather_call(("city", "123")), self.tools
        )
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "123"})

    def test_string_value_backslashes_preserved(self):
        """Invalid JSON escapes in string arguments are preserved literally."""
        result = Glm47MoeDetector().detect_and_parse(
            _weather_call(("city", "C:\\Users\\test")), self.tools
        )
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"city": "C:\\Users\\test"}
        )

    def test_no_arg_call_parses_with_empty_object(self):
        """A call without arguments parses with an empty parameter object."""
        result = Glm47MoeDetector().detect_and_parse(
            "<tool_call>noop</tool_call>", self.tools
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "noop")
        self.assertEqual(json.loads(result.calls[0].parameters), {})

    def test_leading_text_preserved(self):
        """Leading text is returned as normal text."""
        text = "Let me check. " + _weather_call(("city", "Tokyo"))
        result = Glm47MoeDetector().detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "Let me check.")
        self.assertEqual(len(result.calls), 1)

    def test_text_between_and_after_calls_preserved(self):
        """Normal text between and after calls is preserved."""
        text = (
            _weather_call(("city", "Tokyo"))
            + " middle <tool_call>noop</tool_call> tail"
        )
        result = Glm47MoeDetector().detect_and_parse(text, self.tools)
        self.assertEqual([c.name for c in result.calls], ["get_weather", "noop"])
        self.assertEqual(result.normal_text, "middle  tail")

    # ==================== streaming ====================

    def test_streaming_matches_non_streaming_at_small_chunk_sizes(self):
        """Streaming names and arguments match non-streaming parsing at sizes 1 and 7."""
        scenarios = {
            "typed_args": _weather_call(
                ("city", "北京"), ("filters", '{"a": 1}'), ("count", "42")
            ),
            "angle_bracket_in_value": _weather_call(("city", "a<b & c")),
            "quote_and_newline_in_value": _weather_call(("city", 'say "hi"\nok')),
            "two_calls": _weather_call(("city", "Tokyo"))
            + (
                "<tool_call>search<arg_key>query</arg_key>"
                "<arg_value>cafes</arg_value></tool_call>"
            ),
        }
        for label, text in scenarios.items():
            non_streaming_result = Glm47MoeDetector().detect_and_parse(text, self.tools)
            expected_calls = [
                (call.name, json.loads(call.parameters))
                for call in non_streaming_result.calls
            ]
            for chunk_size in (1, 7):
                with self.subTest(scenario=label, chunk_size=chunk_size):
                    calls, _ = self._stream(Glm47MoeDetector(), text, chunk_size)
                    actual_calls = [
                        (name, json.loads(params))
                        for name, params in self._group_streamed_calls(calls)
                    ]
                    self.assertEqual(actual_calls, expected_calls)

    def test_streaming_whole_call_in_single_increment(self):
        """A complete call in one increment emits its name and arguments immediately."""
        text = _weather_call(("city", "北京"), ("count", "42"))
        result = Glm47MoeDetector().parse_streaming_increment(text, self.tools)
        [(name, params)] = self._group_streamed_calls(result.calls)
        self.assertEqual(name, "get_weather")
        self.assertEqual(json.loads(params), {"city": "北京", "count": 42})

    def test_streaming_no_args_emits_empty_object(self):
        """No-argument calls stream a complete empty JSON object."""
        calls, _ = self._stream(Glm47MoeDetector(), "<tool_call>noop</tool_call>", 1)
        [(name, params)] = self._group_streamed_calls(calls)
        self.assertEqual(name, "noop")
        self.assertEqual(params, "{}")

    def test_streaming_sequential_calls_get_distinct_indices(self):
        """Sequential calls use distinct indices and valid arguments."""
        text = _weather_call(("city", "Tokyo")) + (
            "<tool_call>search<arg_key>query</arg_key>"
            "<arg_value>cafes</arg_value></tool_call>"
        )
        calls, normal_text = self._stream(Glm47MoeDetector(), text, 1)
        self.assertEqual(sorted({c.tool_index for c in calls}), [0, 1])
        streamed = self._group_streamed_calls(calls)
        self.assertEqual(streamed[0][0], "get_weather")
        self.assertEqual(json.loads(streamed[0][1]), {"city": "Tokyo"})
        self.assertEqual(streamed[1][0], "search")
        self.assertEqual(json.loads(streamed[1][1]), {"query": "cafes"})
        self.assertEqual(normal_text, "")

    def test_streaming_never_emits_truncated_tool_name(self):
        """A partial function name is not emitted."""
        detector = Glm47MoeDetector()
        result = detector.parse_streaming_increment("<tool_call>get_weat", self.tools)
        self.assertEqual(result.calls, [])
        result = detector.parse_streaming_increment(
            "her<arg_key>city</arg_key><arg_value>x</arg_value></tool_call>",
            self.tools,
        )
        names = [c.name for c in result.calls if c.name is not None]
        self.assertEqual(names, ["get_weather"])

    def test_streaming_plain_text_with_angle_brackets_passes_through(self):
        """Plain text containing angle brackets passes through unchanged."""
        text = "a < b and c > d. done"
        calls, normal_text = self._stream(Glm47MoeDetector(), text, 1)
        self.assertEqual(calls, [])
        self.assertEqual(normal_text, text)

    def test_streaming_leading_text_then_call(self):
        """Streaming separates leading text from the following tool call."""
        text = "Let me check. " + _weather_call(("city", "Tokyo"))
        calls, normal_text = self._stream(Glm47MoeDetector(), text, 1)
        self.assertEqual(normal_text, "Let me check. ")
        [(name, params)] = self._group_streamed_calls(calls)
        self.assertEqual(name, "get_weather")
        self.assertEqual(json.loads(params), {"city": "Tokyo"})


if __name__ == "__main__":
    unittest.main()
