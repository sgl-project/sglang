"""Unit tests for Qwen3CoderDetector -- no server, no model loading."""

import json
import time

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="get_weather",
            description="Get weather information",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city"],
            },
        ),
    )
]


def _call(city_block: str) -> str:
    return (
        "<tool_call>\n<function=get_weather>\n"
        f"<parameter=city>\n{city_block}\n"
        "<parameter=days>\n3\n</parameter>\n"
        "</function>\n</tool_call>"
    )


def _stream(detector: Qwen3CoderDetector, text: str, chunk: int = 3) -> str:
    args = ""
    for i in range(0, len(text), chunk):
        result = detector.parse_streaming_increment(text[i : i + chunk], TOOLS)
        for item in result.calls:
            args += item.parameters
    return args


class TestQwen3CoderDetectorMalformedParameterEnd(CustomTestCase):
    """A close marker the grammar let drift ("</parameter1>", "</parameter_x>",
    a bare "</parameter") must not leak into the argument value."""

    def test_non_streaming_strips_drifted_end_marker(self):
        for marker in ("</parameter1>", "</parameter_function>", "</parameter"):
            with self.subTest(marker=marker):
                result = Qwen3CoderDetector().detect_and_parse(
                    _call(f"Shanghai\n{marker}"), TOOLS
                )
                self.assertEqual(len(result.calls), 1)
                self.assertEqual(
                    json.loads(result.calls[0].parameters),
                    {"city": "Shanghai", "days": 3},
                )

    def test_streaming_strips_drifted_end_marker(self):
        args = _stream(Qwen3CoderDetector(), _call("Shanghai\n</parameter1>"))
        self.assertEqual(json.loads(args), {"city": "Shanghai", "days": 3})

    def test_well_formed_value_unchanged(self):
        result = Qwen3CoderDetector().detect_and_parse(
            _call("Shanghai\n</parameter>"), TOOLS
        )
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"city": "Shanghai", "days": 3}
        )

    def test_angle_brackets_inside_value_are_kept(self):
        result = Qwen3CoderDetector().detect_and_parse(
            _call("a <b>tag</b> city\n</parameter>"), TOOLS
        )
        self.assertEqual(
            json.loads(result.calls[0].parameters)["city"], "a <b>tag</b> city"
        )

    def test_closed_value_ending_like_a_marker_is_kept(self):
        """A value closed by a real </parameter> may itself end in "</parameters>"
        or "</parameter_x>" (e.g. XML a coder model writes); only an unclosed
        value can carry a drifted marker."""
        for value in ("<config>\n  <p>1</p>\n</parameters>", "x </parameterList>"):
            with self.subTest(value=value):
                text = _call(f"{value}\n</parameter>")
                parsed = Qwen3CoderDetector().detect_and_parse(text, TOOLS)
                self.assertEqual(json.loads(parsed.calls[0].parameters)["city"], value)
                self.assertEqual(
                    json.loads(_stream(Qwen3CoderDetector(), text))["city"], value
                )

    def test_long_whitespace_run_in_unclosed_value_parses_in_linear_time(self):
        """Finding a drifted marker at the end of an unclosed value must not rescan
        from every whitespace position (quadratic on long whitespace runs)."""
        value = "a" + " " * 100_000 + "b"
        text = (
            "<tool_call>\n<function=get_weather>\n"
            f"<parameter=city>\n{value}\n"
            "<parameter=days>\n3\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        start = time.perf_counter()
        result = Qwen3CoderDetector().detect_and_parse(text, TOOLS)
        self.assertLess(time.perf_counter() - start, 2.0)
        self.assertEqual(json.loads(result.calls[0].parameters)["city"], value)


if __name__ == "__main__":
    import unittest

    unittest.main()
