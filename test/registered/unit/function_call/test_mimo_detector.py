import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _union_tool() -> Tool:
    return Tool(
        type="function",
        function=Function(
            name="record",
            parameters={
                "type": "object",
                "properties": {
                    "value": {
                        "anyOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        ]
                    }
                },
            },
        ),
    )


def _parse_value(raw: str):
    text = (
        "<tool_call><function=record><parameter=value>"
        f"{raw}</parameter></function></tool_call>"
    )
    calls = MiMoDetector().detect_and_parse(text, [_union_tool()]).calls
    return json.loads(calls[0].parameters)["value"]


class TestMiMoDetectorStringUnion(CustomTestCase):
    def test_union_object_value_stays_structured(self):
        """An object allowed by an anyOf string union must not be emitted as a JSON string."""
        self.assertEqual(_parse_value('{"city": "Paris"}'), {"city": "Paris"})

    def test_union_value_not_matching_schema_stays_string(self):
        """JSON-looking text the union does not allow as structured stays the raw string."""
        self.assertEqual(_parse_value('{"town": "Paris"}'), '{"town": "Paris"}')
        self.assertEqual(_parse_value("Paris"), "Paris")


if __name__ == "__main__":
    unittest.main()
