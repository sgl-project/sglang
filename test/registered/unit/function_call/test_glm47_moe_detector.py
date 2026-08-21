"""Unit tests for nullable arguments in the GLM-4.7 tool-call detector."""

import json
import unittest
from typing import Any

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1.0, suite="base-a-test-cpu")


_NULLABLE_STRING_CASES = (
    (
        "nullable-null",
        {"enum": ["value", None], "type": ["string", "null"]},
        "null",
        None,
    ),
    ("string-literal-null", {"type": "string"}, "null", "null"),
    (
        "nullable-string",
        {"enum": ["value", None], "type": ["string", "null"]},
        "value",
        "value",
    ),
)


def _make_tools(value_schema: dict[str, Any]) -> list[Tool]:
    return [
        Tool(
            type="function",
            function=Function(
                name="record_value",
                parameters={
                    "type": "object",
                    "properties": {"value": value_schema},
                    "required": ["value"],
                },
            ),
        )
    ]


def _make_tool_call(value: str) -> str:
    return (
        "<tool_call>record_value"
        "<arg_key>value</arg_key>"
        f"<arg_value>{value}</arg_value>"
        "</tool_call>"
    )


class TestGlm47MoeDetector(CustomTestCase):
    def test_complete_tool_call_preserves_nullable_string_semantics(self) -> None:
        for (
            case_name,
            value_schema,
            wire_value,
            expected_value,
        ) in _NULLABLE_STRING_CASES:
            with self.subTest(case=case_name):
                result = Glm47MoeDetector().detect_and_parse(
                    _make_tool_call(wire_value), _make_tools(value_schema)
                )

                self.assertEqual(len(result.calls), 1)
                self.assertEqual(result.calls[0].name, "record_value")
                self.assertEqual(
                    json.loads(result.calls[0].parameters),
                    {"value": expected_value},
                )

    def test_incremental_tool_call_preserves_nullable_string_semantics(self) -> None:
        for (
            case_name,
            value_schema,
            wire_value,
            expected_value,
        ) in _NULLABLE_STRING_CASES:
            with self.subTest(case=case_name):
                detector = Glm47MoeDetector()
                tools = _make_tools(value_schema)
                split_index = max(1, len(wire_value) // 2)
                increments = (
                    "<tool_call>record_value<arg_key>value</arg_key><arg_value>"
                    + wire_value[:split_index],
                    wire_value[split_index:],
                    "</arg_value></tool_call>",
                )

                calls = []
                for increment in increments:
                    calls.extend(
                        detector.parse_streaming_increment(increment, tools).calls
                    )

                self.assertEqual(
                    [call.name for call in calls if call.name], ["record_value"]
                )
                arguments = "".join(
                    call.parameters for call in calls if call.parameters
                )
                self.assertEqual(json.loads(arguments), {"value": expected_value})


if __name__ == "__main__":
    unittest.main()
