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

    def test_prose_between_two_tool_calls_is_streamed(self):
        """Prose sitting between two calls used to be dropped outright: the lead
        was only recovered for the first call, and consuming the second call
        advanced the buffer straight past the text in front of it."""
        normal, calls = self._feed(
            [_weather_call("SF"), "\n\nNow the other city.\n\n", _weather_call("NY")]
        )

        self.assertIn("Now the other city.", normal)
        self.assertNotIn(DSML, normal)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"] * 2)

    def test_prose_after_the_last_tool_call_is_streamed(self):
        """Trailing prose used to be stranded: the section closer keeps the DSML
        guard true for the rest of the turn, so the buffer holding the text was
        never released and nothing flushes it at end of turn."""
        detector = DeepSeekV4Detector()
        normal = ""
        for chunk in [_weather_call("SF"), "\n\nThat's the forecast."]:
            normal += detector.parse_streaming_increment(chunk, self.tools).normal_text

        self.assertIn("That's the forecast.", normal)
        self.assertNotIn(DSML, normal)
        self.assertEqual(detector._buffer, "")

    def test_trailing_prose_is_chunk_invariant(self):
        """Where the deltas happen to split must not change what the client sees.
        The lead of a call spanning several chunks is emitted once, and text after
        a call reads the same whether or not it shares a delta with the closer."""
        text = _weather_call("SF") + "\n\nAnd that is that."
        one_shot, _ = self._feed([text])

        for size in (1, 2, 3, 7, 13):
            with self.subTest(chunk_size=size):
                split, _ = self._feed(
                    [text[i : i + size] for i in range(0, len(text), size)]
                )
                self.assertEqual(split, one_shot)

    def test_prose_with_angle_bracket_is_not_held_back(self):
        """Only a suffix that could still grow into a DSML marker may be withheld.
        Holding every `<` would strand ordinary prose until end of turn, and no
        end-of-turn flush exists to release it."""
        normal, _ = self._feed([_weather_call("SF"), "\n\n5 < 6 and a < b."])

        self.assertIn("5 < 6 and a < b.", normal)

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


if __name__ == "__main__":
    import unittest

    unittest.main()
