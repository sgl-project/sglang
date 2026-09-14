"""DeepSeekV41Detector structural tag under tool_choice required / named: the
non-strict invoke body is constrained to a JSON object. No server, no model."""

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv41_detector import (
    ANY_OBJECT_BODY,
    DeepSeekV41Detector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _tools():
    return [
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
                name="lookup",
                description="Look up a value",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                        "flags": {"type": "array"},
                    },
                },
            ),
        ),
    ]


class TestDeepSeekV41InvokeBody(CustomTestCase):
    def setUp(self):
        self.tools = _tools()
        self.detector = DeepSeekV41Detector()

    def test_non_strict_body_is_any_json_object(self):
        """The grammar must not admit `"Haifa"`, `[1]` or `null` as an invoke
        body: the detector reads only objects and the forced call would come
        back with `{}` arguments."""
        tag = self.detector.get_structural_tag(tools=self.tools, tool_choice="required")
        _, calls, _ = tag.format.elements
        self.assertEqual(
            [t.content.json_schema for t in calls.tags],
            [ANY_OBJECT_BODY, ANY_OBJECT_BODY],
        )
        self.assertEqual(
            ANY_OBJECT_BODY, {"type": "object", "additionalProperties": True}
        )

    def test_strict_tool_keeps_its_parameters_schema(self):
        tools = _tools()
        tools[0].function.strict = True
        tag = self.detector.get_structural_tag(tools=tools, tool_choice="required")
        _, calls, _ = tag.format.elements
        self.assertEqual(
            calls.tags[0].content.json_schema, tools[0].function.parameters
        )
        self.assertEqual(calls.tags[1].content.json_schema, ANY_OBJECT_BODY)


if __name__ == "__main__":
    import unittest

    unittest.main()
