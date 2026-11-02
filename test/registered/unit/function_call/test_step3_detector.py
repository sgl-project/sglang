"""Unit tests for Step3Detector — no server, no model loading.

Covers parsing of calls that take no parameters, and streaming/non-streaming
equivalence when several calls arrive in the same increment.
"""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.step3_detector import Step3Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

TOOL_CALLS_BEGIN = "<｜tool_calls_begin｜>"
TOOL_CALLS_END = "<｜tool_calls_end｜>"
TOOL_CALL_BEGIN = "<｜tool_call_begin｜>"
TOOL_CALL_END = "<｜tool_call_end｜>"
TOOL_SEP = "<｜tool_sep｜>"


def _block(name, params=""):
    return (
        f"{TOOL_CALL_BEGIN}function{TOOL_SEP}"
        f'<steptml:invoke name="{name}">{params}</steptml:invoke>'
        f"{TOOL_CALL_END}"
    )


def _param(name, value):
    return f'<steptml:parameter name="{name}">{value}</steptml:parameter>'


def _wrap(*blocks):
    return TOOL_CALLS_BEGIN + "".join(blocks) + TOOL_CALLS_END


class TestStep3Detector(CustomTestCase):
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
                            "days": {"type": "integer"},
                        },
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
        ]

    # ---------------- non-streaming ----------------

    def test_parses_call_without_parameters(self):
        # Regression: the invoke regex required at least one character between
        # the tags, so a parameterless call matched nothing and was dropped.
        result = Step3Detector().detect_and_parse(
            _wrap(_block("get_weather")), self.tools
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {})

    def test_parses_mix_of_parameterless_and_parameterized_calls(self):
        result = Step3Detector().detect_and_parse(
            _wrap(
                _block("get_weather"),
                _block("get_weather", _param("city", "Beijing")),
            ),
            self.tools,
        )
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(json.loads(result.calls[0].parameters), {})
        self.assertEqual(json.loads(result.calls[1].parameters), {"city": "Beijing"})

    # ---------------- streaming equivalence ----------------

    def _stream(self, chunks):
        detector = Step3Detector()
        seq, normal, cur = [], "", None

        def absorb(result):
            nonlocal normal, cur
            normal += result.normal_text or ""
            for call in result.calls:
                if call.name:
                    cur = [call.name, ""]
                    seq.append(cur)
                if call.parameters:
                    if cur is None:
                        cur = ["", ""]
                        seq.append(cur)
                    cur[1] += call.parameters

        for chunk in chunks:
            absorb(detector.parse_streaming_increment(chunk, self.tools))
        # Drain at end of stream; serving keeps invoking the detector per token.
        for _ in range(200):
            result = detector.parse_streaming_increment("", self.tools)
            if not result.calls and not result.normal_text:
                break
            absorb(result)

        return [tuple(x) for x in seq], normal

    def assert_stream_matches_final(self, text):
        reference = Step3Detector().detect_and_parse(text, self.tools)
        want = [(c.name, c.parameters) for c in reference.calls]
        want_normal = (reference.normal_text or "").strip()

        chunkings = {
            "char": list(text),
            "whole": [text],
            "halves": [text[: len(text) // 2], text[len(text) // 2 :]],
        }
        for label, chunks in chunkings.items():
            got, got_normal = self._stream(chunks)

            self.assertEqual(
                len(got), len(want), f"[{label}] call count differs for {text!r}"
            )
            for (g_name, g_args), (w_name, w_args) in zip(got, want):
                self.assertEqual(g_name, w_name, f"[{label}] name differs")
                self.assertEqual(
                    json.loads(g_args) if g_args else None,
                    json.loads(w_args) if w_args else None,
                    f"[{label}] arguments differ for {text!r}",
                )

            for marker in ("steptml", "tool_call"):
                self.assertNotIn(
                    marker, got_normal, f"[{label}] {marker!r} leaked into content"
                )
            self.assertEqual(
                got_normal.strip(), want_normal, f"[{label}] normal text differs"
            )

    def test_stream_single_call_no_parameters(self):
        self.assert_stream_matches_final(_wrap(_block("get_weather")))

    def test_stream_single_call_one_parameter(self):
        self.assert_stream_matches_final(
            _wrap(_block("get_weather", _param("city", "Beijing")))
        )

    def test_stream_single_call_two_parameters(self):
        self.assert_stream_matches_final(
            _wrap(
                _block("get_weather", _param("city", "Beijing") + _param("days", "3"))
            )
        )

    def test_stream_integer_parameter_keeps_type(self):
        self.assert_stream_matches_final(
            _wrap(_block("get_weather", _param("days", "7")))
        )

    def test_stream_unicode_parameter(self):
        self.assert_stream_matches_final(
            _wrap(_block("get_weather", _param("city", "北京")))
        )

    def test_stream_tool_with_empty_schema(self):
        self.assert_stream_matches_final(_wrap(_block("noop")))

    def test_stream_two_parameterless_calls(self):
        self.assert_stream_matches_final(
            _wrap(_block("get_weather"), _block("get_weather"))
        )

    def test_stream_two_parameterless_calls_different_tools(self):
        self.assert_stream_matches_final(_wrap(_block("get_weather"), _block("noop")))

    def test_stream_parameterless_then_parameterized(self):
        self.assert_stream_matches_final(
            _wrap(_block("get_weather"), _block("get_weather", _param("city", "x")))
        )

    def test_stream_parameterized_then_parameterless(self):
        self.assert_stream_matches_final(
            _wrap(_block("get_weather", _param("city", "x")), _block("get_weather"))
        )

    def test_stream_two_calls_distinct_parameters(self):
        # Regression: with the whole response in one increment, the first call
        # absorbed the second call's parameters.
        self.assert_stream_matches_final(
            _wrap(
                _block("get_weather", _param("city", "a")),
                _block("get_weather", _param("city", "b")),
            )
        )

    def test_stream_three_mixed_calls(self):
        self.assert_stream_matches_final(
            _wrap(
                _block("get_weather"),
                _block("get_weather", _param("city", "a")),
                _block("noop"),
            )
        )

    def test_stream_normal_text_before_tool_block(self):
        self.assert_stream_matches_final("Sure. " + _wrap(_block("get_weather")))

    def test_stream_plain_text_without_tool_call(self):
        self.assert_stream_matches_final("The weather is nice today.")


if __name__ == "__main__":
    import unittest

    unittest.main()
