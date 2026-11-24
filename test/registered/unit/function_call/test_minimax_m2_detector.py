import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="f",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {"type": ["integer", "null"]},
                    "tag": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "ratio": {"type": "number"},
                    "count": {"type": "integer"},
                    "nothing": {"type": "null"},
                    "config": {"type": "object"},
                    "items": {"type": "array"},
                },
            },
        ),
    )
]


def _wire(name, value):
    return (
        '<minimax:tool_call><invoke name="f">'
        f'<parameter name="{name}">{value}</parameter>'
        "</invoke></minimax:tool_call>"
    )


def _parse_both(name, value):
    """Return arguments from the one-shot and streaming paths."""
    text = _wire(name, value)
    one_shot = MinimaxM2Detector().detect_and_parse(text, TOOLS).calls[0].parameters
    calls = MinimaxM2Detector().parse_streaming_increment(text, TOOLS).calls
    streamed = "".join(call.parameters for call in calls)
    return json.loads(one_shot), json.loads(streamed)


class TestMinimaxM2ParamConversion(CustomTestCase):
    def test_nullable_schema_keeps_real_values(self):
        """Any schema allowing null, such as Optional[str], used to erase the value."""
        for name, raw, expected in (("limit", "5", 5), ("tag", "abc", "abc")):
            for arguments in _parse_both(name, raw):
                self.assertEqual(arguments, {name: expected})

    def test_null_still_maps_to_none(self):
        for name, raw in (("limit", "null"), ("tag", "none"), ("nothing", "x")):
            for arguments in _parse_both(name, raw):
                self.assertEqual(arguments, {name: None})

    def test_non_finite_number_does_not_crash_or_emit_invalid_json(self):
        """1e999 overflowed int() and crashed both paths; serializing inf or NaN
        would emit tokens that strict JSON clients reject."""
        for name, raw in (
            ("ratio", "1e999"),
            ("count", "1e999"),
            ("ratio", "NaN"),
            ("config", '{"x": NaN}'),
            ("items", "[1e999]"),
        ):
            for arguments in _parse_both(name, raw):
                self.assertEqual(arguments, {name: raw})


if __name__ == "__main__":
    unittest.main()
