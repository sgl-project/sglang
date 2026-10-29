"""Unit tests for MiMoDetector - no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="search",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "site": {"type": ["string", "null"]},
                        "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                        "threshold": {"type": "number"},
                    },
                },
            ),
        ),
    ]


def _call(**params):
    body = "".join(f"<parameter={k}>{v}</parameter>" for k, v in params.items())
    return f"<tool_call>\n<function=search>\n{body}\n</function>\n</tool_call>"


class TestMiMoDetectorParamTypes(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()

    def _parse_args(self, text):
        result = MiMoDetector().detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        return json.loads(result.calls[0].parameters)

    def test_type_array_schema_parses(self):
        """A `"type": [..., "null"]` property must not abort parsing of the whole call."""
        args = self._parse_args(_call(query="sglang", site="github.com"))
        self.assertEqual(args, {"query": "sglang", "site": "github.com"})

    def test_any_of_schema_converts_to_inner_type(self):
        """An Optional[int] (`anyOf` with null) property is converted to an int, not left a string."""
        args = self._parse_args(_call(query="sglang", limit="5"))
        self.assertEqual(args["limit"], 5)

    def test_out_of_range_number_does_not_abort_parse(self):
        """A number that overflows to inf must not abort parsing of the whole call."""
        args = self._parse_args(_call(query="sglang", threshold="1e400"))
        self.assertEqual(args["query"], "sglang")


class TestMiMoDetectorStreaming(CustomTestCase):
    def test_multiple_calls_in_one_chunk(self):
        """Every complete call in a chunk is emitted, not only the first one."""
        detector = MiMoDetector()
        text = _call(query="a") + "\n" + _call(query="b")
        result = detector.parse_streaming_increment(text, _make_tools())

        self.assertEqual([c.tool_index for c in result.calls], [0, 1])
        self.assertEqual(
            [json.loads(c.parameters)["query"] for c in result.calls], ["a", "b"]
        )


if __name__ == "__main__":
    unittest.main()
