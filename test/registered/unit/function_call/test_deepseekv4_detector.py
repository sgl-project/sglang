"""Unit tests for DeepSeekV4Detector DSML streaming — no server, no model loading."""

from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

DSML = "｜DSML｜"


def _wrapped(invoke: str) -> str:
    return f"<{DSML}tool_calls>\n{invoke}\n</{DSML}tool_calls>"


def _invoke(name: str, params: str = "") -> str:
    return f'<{DSML}invoke name="{name}">\n{params}\n</{DSML}invoke>'


def _param(name: str, is_string: str, value: str) -> str:
    return (
        f'<{DSML}parameter name="{name}" string="{is_string}">{value}</{DSML}parameter>'
    )


def _weather_call(city: str = "SF") -> str:
    return _wrapped(_invoke("get_weather", _param("city", "true", city)))


class TestDeepSeekV4Streaming(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            )
        ]

    def _feed(self, chunks):
        """Returns (normal_text, calls) accumulated over the chunks."""
        detector = DeepSeekV4Detector()
        normal, calls = "", []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        return normal, calls

    def test_preamble_in_same_delta_as_tool_call(self):
        """Prose sharing a delta with the tool call must not be dropped, and the
        streaming and one-shot paths must agree on it."""
        text = "Let me check.\n" + _weather_call()
        normal, calls = self._feed([text])

        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
        self.assertEqual(
            normal, DeepSeekV4Detector().detect_and_parse(text, self.tools).normal_text
        )

    def test_preamble_before_bare_invoke_without_wrapper(self):
        """The bare `<｜DSML｜invoke …>` form has no tool_calls wrapper to walk
        back to, so the preamble is computed from the invoke itself."""
        text = "Checking.\n" + _invoke("get_weather", _param("city", "true", "SF"))
        normal, calls = self._feed([text])

        self.assertIn("Checking.", normal)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])

    def test_no_dsml_markers_leak_into_normal_text(self):
        text = "Prose.\n" + _weather_call()
        normal, _ = self._feed([text[i : i + 4] for i in range(0, len(text), 4)])

        self.assertNotIn(DSML, normal)

    def test_malformed_partial_json_falls_back_to_raw_value(self):
        """A partial non-string parameter must not escape as MalformedJSON."""
        detector = DeepSeekV4Detector()
        result = detector.parse_streaming_increment(
            f'<{DSML}tool_calls>\n<{DSML}invoke name="get_weather">\n'
            f'<{DSML}parameter name="city" string="false">{{"a"',
            self.tools,
        )

        self.assertEqual([c.name for c in result.calls if c.name], ["get_weather"])

    def test_non_streaming_parses_every_tool_calls_section(self):
        """A turn with two tool_calls sections must yield both calls."""
        result = DeepSeekV4Detector().detect_and_parse(
            f"{_weather_call('SF')}\n{_weather_call('NY')}", self.tools
        )

        self.assertEqual(len(result.calls), 2)

    def test_parse_error_neither_swallows_nor_duplicates(self):
        """An unexpected parse error must not empty the turn, and the dropped
        buffer must not come back on the next delta."""
        detector = DeepSeekV4Detector()

        with patch.object(
            DeepSeekV4Detector,
            "_parse_parameters_from_xml",
            side_effect=RuntimeError("boom"),
        ):
            first = detector.parse_streaming_increment(_weather_call(), self.tools)
            self.assertEqual(detector._buffer, "")
            second = detector.parse_streaming_increment(" tail", self.tools)

        self.assertIn("get_weather", first.normal_text)
        self.assertNotIn("get_weather", second.normal_text)
        # No half-formed call: the failure can land between a tool's name and its
        # arguments, so an argument-less named call must not reach the client.
        self.assertEqual(first.calls, [])


class TestDeepSeekV4NonStreamingLeak(CustomTestCase):
    """Non-streaming turns must never return DSML markup as content.

    When a generation carries tool markup the detector cannot convert, the
    response comes back with no `tool_calls` and `finish_reason: "stop"`.
    An OpenAI-compatible client reads that as "the model is done", ends the
    turn, and the requested calls are dropped with no error anywhere.
    """

    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="record_note",
                    description="Record a note",
                    parameters={
                        "type": "object",
                        "properties": {"targets": {"type": "array"}},
                    },
                ),
            ),
        ]

    def _parse(self, text):
        return DeepSeekV4Detector().detect_and_parse(text, self.tools)

    def test_invoke_without_tool_calls_section(self):
        """A bare invoke block with no section wrapper is still a tool call."""
        result = self._parse(
            "Let me check.\n\n" + _invoke("get_weather", _param("city", "true", "SF"))
        )

        self.assertEqual([c.name for c in result.calls], ["get_weather"])
        self.assertNotIn(DSML, result.normal_text)

    def test_unterminated_tool_calls_section(self):
        """An opened-but-unclosed section must not swallow its calls."""
        result = self._parse(
            f"<{DSML}tool_calls>\n"
            + _invoke("get_weather", _param("city", "true", "SF"))
        )

        self.assertEqual([c.name for c in result.calls], ["get_weather"])
        self.assertNotIn(DSML, result.normal_text)

    def test_malformed_invoke_keeps_sibling_calls(self):
        """One unparsable invoke must not discard the calls next to it."""
        malformed = (
            f'<{DSML}invoke name="record_note">\n{{"targets": ["a",}}\n</{DSML}invoke>'
        )
        result = self._parse(
            _wrapped(
                malformed + "\n" + _invoke("get_weather", _param("city", "true", "SF"))
            )
        )

        self.assertEqual([c.name for c in result.calls], ["get_weather"])
        self.assertNotIn(DSML, result.normal_text)

    def test_unknown_tool_markup_is_not_returned_as_content(self):
        """An unmatched tool name yields no call, but also no raw markup."""
        result = self._parse(
            _wrapped(_invoke("no_such_tool", _param("city", "true", "SF")))
        )

        self.assertEqual(result.calls, [])
        self.assertNotIn(DSML, result.normal_text)

    def test_prose_is_preserved_alongside_tool_call(self):
        """Stripping markup must not eat the model's ordinary prose."""
        result = self._parse("Checking the weather.\n\n" + _weather_call("SF"))

        self.assertEqual([c.name for c in result.calls], ["get_weather"])
        self.assertIn("Checking the weather.", result.normal_text)


if __name__ == "__main__":
    import unittest

    unittest.main()
