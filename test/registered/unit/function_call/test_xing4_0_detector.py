"""Unit tests for Xing4_0Detector -- no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.xing4_0_detector import Xing4_0Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _tool(name: str, properties: dict) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            parameters={"type": "object", "properties": properties},
        ),
    )


class TestXing4_0Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            _tool("get_time", {"tz": {"type": "string"}}),
            _tool(
                "lookup",
                {
                    # Pydantic's rendering of Optional[str] / Optional[int].
                    "zip_code": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                },
            ),
        ]

    def _stream(self, text: str, chunk_size: int = 7):
        detector = Xing4_0Detector()
        calls = []
        for i in range(0, len(text), chunk_size):
            calls.extend(
                detector.parse_streaming_increment(
                    text[i : i + chunk_size], self.tools
                ).calls
            )
        return calls

    def test_optional_string_param_stays_string(self):
        """An Optional[str] value that looks like a number stays a string,
        while an Optional[int] value is still decoded."""
        text = (
            "<tool_call>lookup"
            "<param_key>zip_code</param_key><param_value>10001</param_value>"
            "<param_key>limit</param_key><param_value>5</param_value>"
            "</tool_call>"
        )
        expected = {"zip_code": "10001", "limit": 5}

        result = Xing4_0Detector().detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters), expected)

        streamed = self._stream(text)
        self.assertEqual(len(streamed), 1)
        self.assertEqual(json.loads(streamed[0].parameters), expected)

    def test_streaming_tool_index_is_call_ordinal(self):
        """Streamed calls are numbered 0, 1, ... in emission order, so two calls
        to the same tool are not merged by the client into one call."""
        text = (
            '<tool_call>{"name": "lookup", "arguments": {"limit": 1}}</tool_call>'
            '<tool_call>{"name": "lookup", "arguments": {"limit": 2}}</tool_call>'
        )
        calls = self._stream(text)
        self.assertEqual([c.tool_index for c in calls], [0, 1])
        self.assertEqual([c.name for c in calls], ["lookup", "lookup"])
        self.assertEqual(
            [json.loads(c.parameters) for c in calls], [{"limit": 1}, {"limit": 2}]
        )


if __name__ == "__main__":
    unittest.main()
