"""Unit tests for Gemma4Detector array parsing - no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.gemma4_detector import (
    Gemma4Detector,
    _parse_gemma4_args,
    _parse_gemma4_array,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="f",
                parameters={
                    "type": "object",
                    "properties": {"paths": {"type": "array"}},
                },
            ),
        )
    ]


class TestGemma4ArrayParsing(CustomTestCase):
    def test_nested_array_with_close_bracket_in_string(self):
        # A "]" inside a string used to close the nested array early; the
        # leftover bracket then looped the parser forever.
        result = _parse_gemma4_args('paths: [[<|"|>a]b<|"|>, <|"|>c<|"|>]]')
        self.assertEqual(result, {"paths": [["a]b", "c"]]})

    def test_nested_array_with_open_bracket_in_string(self):
        result = _parse_gemma4_args('paths: [[<|"|>a[b<|"|>, <|"|>c<|"|>]]')
        self.assertEqual(result, {"paths": [["a[b", "c"]]})

    def test_deeply_nested_arrays_with_strings(self):
        result = _parse_gemma4_args('m: [[<|"|>x]y<|"|>], [2, [<|"|>p]q<|"|>]]]')
        self.assertEqual(result, {"m": [["x]y"], [2, ["p]q"]]]})

    def test_stray_close_bracket_terminates(self):
        # The bare-value scan cannot consume a stray "]"; the parser must
        # still make progress and return.
        self.assertEqual(_parse_gemma4_array("1, 2]"), [1, 2])
        self.assertEqual(_parse_gemma4_array("]]"), [])

    def test_plain_nested_arrays_unchanged(self):
        result = _parse_gemma4_args("paths: [[1, 2], [3, 4]]")
        self.assertEqual(result, {"paths": [[1, 2], [3, 4]]})

    def test_flat_strings_with_brackets_unchanged(self):
        result = _parse_gemma4_args('paths: [<|"|>a]b<|"|>, <|"|>c<|"|>]')
        self.assertEqual(result, {"paths": ["a]b", "c"]})

    def test_detect_and_parse_nested_bracket_string(self):
        detector = Gemma4Detector()
        result = detector.detect_and_parse(
            '<|tool_call>call:f{paths: [[<|"|>a]b<|"|>, <|"|>c<|"|>]]}<tool_call|>',
            _tools(),
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "f")
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"paths": [["a]b", "c"]]}
        )

    def test_streaming_nested_bracket_string(self):
        detector = Gemma4Detector()
        text = '<|tool_call>call:f{paths: [[<|"|>a]b<|"|>, <|"|>c<|"|>]]}<tool_call|>'
        calls = []
        for i in range(0, len(text), 7):
            calls.extend(
                detector.parse_streaming_increment(text[i : i + 7], _tools()).calls
            )
        self.assertEqual(calls[0].name, "f")
        parameters = "".join(c.parameters for c in calls if c.parameters)
        self.assertEqual(json.loads(parameters), {"paths": [["a]b", "c"]]})


if __name__ == "__main__":
    unittest.main()
