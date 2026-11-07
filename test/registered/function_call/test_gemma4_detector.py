import json
import signal
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.gemma4_detector import (
    STRING_DELIM,
    Gemma4Detector,
    _parse_gemma4_args,
    _parse_gemma4_array,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")

D = STRING_DELIM


class _ParseTimeout(Exception):
    pass


def _raise_timeout(*_):
    raise _ParseTimeout()


def _bounded(fn, *args, seconds=2):
    """Runs fn and fails fast instead of hanging if it never returns."""
    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        return fn(*args)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _tool(name, parameters):
    return Tool(
        type="function",
        function=Function(name=name, description=name, parameters=parameters),
    )


class TestGemma4ArrayParsing(unittest.TestCase):
    def test_flat_string_array(self):
        self.assertEqual(
            _bounded(_parse_gemma4_args, f"items:[{D}a{D},{D}b{D}]"),
            {"items": ["a", "b"]},
        )

    def test_nested_number_array(self):
        self.assertEqual(
            _bounded(_parse_gemma4_args, "m:[[1,2],[3]]"), {"m": [[1, 2], [3]]}
        )

    def test_nested_array_with_bracket_inside_string(self):
        # Used to loop forever: the nested-array scan did not skip strings, so
        # the "]" inside the string closed the inner array early.
        self.assertEqual(
            _bounded(_parse_gemma4_args, f"m:[[{D}a]b{D}]]"), {"m": [["a]b"]]}
        )

    def test_stray_closing_bracket(self):
        # Used to loop forever: the bare-value branch made no progress on "]".
        self.assertEqual(_bounded(_parse_gemma4_array, "1,],2"), [1, 2])

    def test_object_in_array_with_bracket_in_string(self):
        self.assertEqual(
            _bounded(_parse_gemma4_args, f"x:[{{k:{D}v]{D}}}]"),
            {"x": [{"k": "v]"}]},
        )

    def test_malformed_inputs_terminate(self):
        for s in [
            "]",
            "]]]",
            "[",
            "[[",
            f"[{D}",
            f"[{D}]",
            ",,]",
            "[]]",
            "[[]]]",
            f"[[{D}]]{D}",
            "a]b]c",
            "{]",
            "[{]}",
        ]:
            with self.subTest(s=s):
                _bounded(_parse_gemma4_array, s)
                _bounded(_parse_gemma4_args, f"k:[{s}]")


class TestGemma4DetectorEndToEnd(unittest.TestCase):
    def test_tool_call_with_bracket_in_nested_string_array(self):
        tools = [
            _tool(
                "run",
                {
                    "type": "object",
                    "properties": {
                        "rows": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "string"}},
                        }
                    },
                },
            )
        ]
        text = f"<|tool_call>call:run{{rows:[[{D}a]b{D},{D}c{D}]]}}<tool_call|>"
        result = _bounded(Gemma4Detector().detect_and_parse, text, tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "run")
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"rows": [["a]b", "c"]]}
        )


if __name__ == "__main__":
    unittest.main()
