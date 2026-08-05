"""Unit tests for Glm4MoeDetector (GLM-4.5 / GLM-4.6) — no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _weather_call(*pairs):
    """Build a GLM-4.5 newline-separated tool call for get_weather."""
    body = "".join(
        f"<arg_key>{key}</arg_key>\n<arg_value>{value}</arg_value>\n"
        for key, value in pairs
    )
    return f"<tool_call>get_weather\n{body}</tool_call>"


class TestGlm4MoeDetector(CustomTestCase):
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
        """Plain text must not trigger; the bot token must trigger."""
        self.assertTrue(
            Glm4MoeDetector().has_tool_call(_weather_call(("city", "Tokyo")))
        )
        self.assertFalse(
            Glm4MoeDetector().has_tool_call("The weather in Tokyo is sunny.")
        )

    # ==================== detect_and_parse ====================

    def test_schema_type_coercion(self):
        """Values are coerced per the tool schema: string stays a JSON string,
        number becomes int, object becomes a nested dict."""
        text = _weather_call(("city", "北京"), ("count", "42"), ("filters", '{"a": 1}'))
        result = Glm4MoeDetector().detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"city": "北京", "count": 42, "filters": {"a": 1}},
        )

    def test_string_typed_digits_stay_string(self):
        """A digits-only value for a string-typed parameter must not be
        re-coerced to a number, even though it parses as JSON int."""
        result = Glm4MoeDetector().detect_and_parse(
            _weather_call(("city", "123")), self.tools
        )
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "123"})

    def test_string_value_backslashes_preserved(self):
        """Backslashes that are not valid JSON escapes (e.g. Windows paths)
        must survive parsing byte-for-byte instead of being reinterpreted."""
        result = Glm4MoeDetector().detect_and_parse(
            _weather_call(("city", "C:\\Users\\test")), self.tools
        )
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"city": "C:\\Users\\test"}
        )

    def test_literal_backslash_n_separator_accepted(self):
        """GLM models emit either real newlines or the two-character sequence
        '\\n' between tags; both spellings must parse."""
        text = (
            "<tool_call>get_weather\\n<arg_key>city</arg_key>\\n"
            "<arg_value>Tokyo</arg_value>\\n</tool_call>"
        )
        result = Glm4MoeDetector().detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Tokyo"})

    def test_leading_text_preserved(self):
        text = "Let me check. " + _weather_call(("city", "Tokyo"))
        result = Glm4MoeDetector().detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "Let me check.")
        self.assertEqual(len(result.calls), 1)

    def test_multiple_calls_parsed_in_order(self):
        text = (
            _weather_call(("city", "Tokyo"))
            + "\n<tool_call>search\n<arg_key>query</arg_key>\n"
            "<arg_value>cafes</arg_value>\n</tool_call>"
        )
        result = Glm4MoeDetector().detect_and_parse(text, self.tools)
        self.assertEqual([c.name for c in result.calls], ["get_weather", "search"])
        self.assertEqual(
            json.loads(result.calls[1].parameters), {"query": "cafes"}
        )

    def test_unknown_tool_dropped(self):
        """A call to a function not in the tool list must not produce a call."""
        text = (
            "<tool_call>bogus_fn\n<arg_key>x</arg_key>\n"
            "<arg_value>1</arg_value>\n</tool_call>"
        )
        result = Glm4MoeDetector().detect_and_parse(text, self.tools)
        self.assertEqual(result.calls, [])

    def test_missing_name_separator_yields_no_calls(self):
        """A tool call without the newline after the function name is malformed;
        it must be rejected without raising."""
        result = Glm4MoeDetector().detect_and_parse(
            "<tool_call>get_weather</tool_call>", self.tools
        )
        self.assertEqual(result.calls, [])

    # ==================== streaming ====================

    def test_streaming_matches_oneshot_across_chunkings(self):
        """The concatenated streamed parameter increments must form valid JSON
        equal to the one-shot parse, for every chunk boundary position.
        Chunk size 1 places a boundary inside every XML tag and value."""
        scenarios = {
            "typed_args": _weather_call(
                ("city", "北京"), ("filters", '{"a": 1}'), ("count", "42")
            ),
            "angle_bracket_in_value": _weather_call(("city", "a<b & c")),
            "quote_and_newline_in_value": _weather_call(("city", 'say "hi"\nok')),
            "two_calls": (
                _weather_call(("city", "Tokyo"))
                + "\n<tool_call>search\n<arg_key>query</arg_key>\n"
                "<arg_value>cafes</arg_value>\n</tool_call>"
            ),
        }
        for label, text in scenarios.items():
            oneshot = Glm4MoeDetector().detect_and_parse(text, self.tools)
            expected = [
                (call.name, json.loads(call.parameters)) for call in oneshot.calls
            ]
            for chunk_size in (1, 7):
                with self.subTest(scenario=label, chunk_size=chunk_size):
                    calls, _ = self._stream(Glm4MoeDetector(), text, chunk_size)
                    streamed = [
                        (name, json.loads(params))
                        for name, params in self._group_streamed_calls(calls)
                    ]
                    self.assertEqual(streamed, expected)

    def test_streaming_no_args_emits_empty_object(self):
        """A tool call without arguments must stream exactly '{}' so the
        client always receives complete JSON."""
        calls, _ = self._stream(Glm4MoeDetector(), "<tool_call>noop\n</tool_call>", 1)
        [(name, params)] = self._group_streamed_calls(calls)
        self.assertEqual(name, "noop")
        self.assertEqual(params, "{}")

    def test_streaming_sequential_calls_get_distinct_indices(self):
        """State must reset between sequential calls: distinct tool_index per
        call and independently valid argument JSON."""
        text = (
            _weather_call(("city", "Tokyo"))
            + "\n<tool_call>search\n<arg_key>query</arg_key>\n"
            "<arg_value>cafes</arg_value>\n</tool_call>"
        )
        calls, normal_text = self._stream(Glm4MoeDetector(), text, 1)
        self.assertEqual(sorted({c.tool_index for c in calls}), [0, 1])
        streamed = self._group_streamed_calls(calls)
        self.assertEqual(streamed[0][0], "get_weather")
        self.assertEqual(json.loads(streamed[0][1]), {"city": "Tokyo"})
        self.assertEqual(streamed[1][0], "search")
        self.assertEqual(json.loads(streamed[1][1]), {"query": "cafes"})
        self.assertEqual(normal_text.strip(), "")

    def test_streaming_never_emits_truncated_tool_name(self):
        """No call may be emitted while the function name is still partial;
        the first emitted name must be the complete one."""
        detector = Glm4MoeDetector()
        result = detector.parse_streaming_increment("<tool_call>get_wea", self.tools)
        self.assertEqual(result.calls, [])
        result = detector.parse_streaming_increment(
            "ther\n<arg_key>city</arg_key>\n<arg_value>x</arg_value>\n</tool_call>",
            self.tools,
        )
        names = [c.name for c in result.calls if c.name is not None]
        self.assertEqual(names, ["get_weather"])

    def test_streaming_plain_text_with_angle_brackets_passes_through(self):
        """Text containing '<' that never becomes a tool call must be emitted
        in full — buffered prefix characters must not be swallowed."""
        text = "a < b and c > d. done"
        calls, normal_text = self._stream(Glm4MoeDetector(), text, 1)
        self.assertEqual(calls, [])
        self.assertEqual(normal_text, text)

    def test_streaming_leading_text_then_call(self):
        """The text/tool boundary must be exact: all leading text emitted as
        normal text, none of it leaking into the tool call (or vice versa)."""
        text = "Let me check. " + _weather_call(("city", "Tokyo"))
        calls, normal_text = self._stream(Glm4MoeDetector(), text, 1)
        self.assertEqual(normal_text, "Let me check. ")
        [(name, params)] = self._group_streamed_calls(calls)
        self.assertEqual(name, "get_weather")
        self.assertEqual(json.loads(params), {"city": "Tokyo"})


if __name__ == "__main__":
    unittest.main()
