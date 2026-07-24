"""Unit tests for PythonicDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestPythonicDetector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="set_temperature",
                    description="Set the target temperature",
                    parameters={
                        "type": "object",
                        "properties": {
                            "value": {"type": "number"},
                        },
                        "required": ["value"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="move",
                    description="Move to coordinates",
                    parameters={
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "deltas": {"type": "array"},
                        },
                        "required": ["x", "y"],
                    },
                ),
            ),
        ]
        self.detector = PythonicDetector()

    def test_positive_number_argument(self):
        result = self.detector.detect_and_parse(
            "[set_temperature(value=5)]", self.tools
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters), {"value": 5})

    def test_negative_integer_argument(self):
        result = self.detector.detect_and_parse(
            "[set_temperature(value=-5)]", self.tools
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters), {"value": -5})

    def test_negative_float_argument(self):
        result = self.detector.detect_and_parse("[move(x=-1.5, y=2.0)]", self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"x": -1.5, "y": 2.0}
        )

    def test_negative_numbers_in_list(self):
        result = self.detector.detect_and_parse(
            "[move(x=0, y=0, deltas=[-1, -2, 3])]", self.tools
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"x": 0, "y": 0, "deltas": [-1, -2, 3]},
        )

    def test_negative_does_not_drop_sibling_calls(self):
        result = self.detector.detect_and_parse(
            "[set_temperature(value=-5), move(x=1, y=2)]", self.tools
        )
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(json.loads(result.calls[0].parameters), {"value": -5})
        self.assertEqual(json.loads(result.calls[1].parameters), {"x": 1, "y": 2})

    def test_non_numeric_unary_does_not_crash_batch(self):
        # A malformed call with a non-numeric unary should be dropped, not crash
        result = self.detector.detect_and_parse(
            '[move(x=-"hello", y=0)]', self.tools
        )
        self.assertEqual(len(result.calls), 0)

    def test_bool_unary_does_not_produce_numeric(self):
        # -True / +False must not silently produce -1 / 0 as a numeric arg
        result = self.detector.detect_and_parse(
            "[move(x=-True, y=0)]", self.tools
        )
        self.assertEqual(len(result.calls), 0)
