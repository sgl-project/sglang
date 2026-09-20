"""Unit tests for DeepSeekV4Detector DSML streaming — no server, no model loading."""

import json
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
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

    def test_non_string_parameter_is_not_rewritten_after_streaming(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="set_config",
                    parameters={
                        "type": "object",
                        "properties": {"value": {}, "label": {"type": "string"}},
                    },
                ),
            )
        ]
        for value in [
            "12345e-3",
            '{"ratio":12345e-3,"enabled":true}',
            '[{"step":"write code","ratio":12345e-3},null,false]',
        ]:
            text = _wrapped(
                _invoke(
                    "set_config",
                    _param("value", "false", value)
                    + _param("label", "true", "complete"),
                )
            )
            expected = {"value": json.loads(value), "label": "complete"}
            for detector_class in [DeepSeekV4Detector, DeepSeekV32Detector]:
                source = text
                if detector_class is DeepSeekV32Detector:
                    source = text.replace("tool_calls", "function_calls")
                for chunk_size in [1, 2, 3, 5, 7, 11, 23, len(source)]:
                    with self.subTest(
                        detector=detector_class.__name__,
                        value=value,
                        chunk_size=chunk_size,
                    ):
                        detector = detector_class()
                        arguments = ""
                        for start in range(0, len(source), chunk_size):
                            result = detector.parse_streaming_increment(
                                source[start : start + chunk_size], tools
                            )
                            arguments += "".join(c.parameters for c in result.calls)
                            self.assertTrue(
                                json.dumps(expected, ensure_ascii=False).startswith(
                                    arguments
                                )
                            )
                        self.assertEqual(json.loads(arguments), expected)
                        self.assertEqual(
                            arguments, detector.prev_tool_call_arr[0]["arguments"]
                        )

    def test_incomplete_non_string_parameter_does_not_emit_a_guessed_value(self):
        detector = DeepSeekV4Detector()
        chunks = [
            f'<{DSML}tool_calls><{DSML}invoke name="get_weather">'
            f'<{DSML}parameter name="city" string="false">',
            "123",
            "45",
            "e-",
            "3",
        ]
        calls = []
        for chunk in chunks:
            calls.extend(detector.parse_streaming_increment(chunk, self.tools).calls)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
        self.assertEqual("".join(c.parameters for c in calls), "")
        result = detector.parse_streaming_increment(
            f"</{DSML}parameter></{DSML}invoke></{DSML}tool_calls>", self.tools
        )
        self.assertEqual(
            json.loads("".join(c.parameters for c in result.calls)), {"city": 12.345}
        )

    def test_string_parameter_still_streams_incrementally(self):
        detector = DeepSeekV4Detector()
        start = (
            f'<{DSML}tool_calls><{DSML}invoke name="get_weather">'
            f'<{DSML}parameter name="city" string="true">San Fran'
        )
        first = detector.parse_streaming_increment(start, self.tools)
        second = detector.parse_streaming_increment("cisco", self.tools)
        prefix = "".join(c.parameters for c in first.calls + second.calls)
        self.assertIn("San Fran", prefix)
        final = detector.parse_streaming_increment(
            f"</{DSML}parameter></{DSML}invoke></{DSML}tool_calls>", self.tools
        )
        self.assertEqual(
            json.loads(prefix + "".join(c.parameters for c in final.calls)),
            {"city": "San Francisco"},
        )

    def test_truncated_non_string_parameter_is_an_explicit_error(self):
        detector = DeepSeekV4Detector()
        detector.parse_streaming_increment(
            f'<{DSML}tool_calls><{DSML}invoke name="get_weather">'
            f'<{DSML}parameter name="city" string="false">[1,2',
            self.tools,
        )
        with self.assertRaisesRegex(ValueError, "Incomplete DSML non-string parameter"):
            detector.finish(self.tools)

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
