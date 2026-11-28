import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.granite_toolcall_detector import GraniteToolCallDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGraniteLiteralNull(CustomTestCase):
    def test_string_null_is_literal_and_nullable_values_still_convert(self):
        tool = Tool(
            type="function",
            function=Function(
                name="search",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "optional": {"type": ["string", "null"]},
                        "count": {"type": "integer"},
                    },
                },
            ),
        )
        for raw in ("null", "NULL"):
            with self.subTest(raw=raw):
                text = (
                    "<tool_call><function=search>"
                    f"<parameter=query>{raw}</parameter>"
                    "<parameter=optional>null</parameter>"
                    "<parameter=count>null</parameter>"
                    "</function></tool_call>"
                )
                expected = {"query": raw, "optional": None, "count": None}
                result = GraniteToolCallDetector().detect_and_parse(text, [tool])
                self.assertEqual(len(result.calls), 1)
                self.assertEqual(json.loads(result.calls[0].parameters), expected)
                for size in (1, 7, len(text)):
                    parser = GraniteToolCallDetector()
                    arguments = ""
                    for start in range(0, len(text), size):
                        result = parser.parse_streaming_increment(
                            text[start : start + size], [tool]
                        )
                        arguments += "".join(
                            item.parameters or "" for item in result.calls
                        )
                    self.assertEqual(json.loads(arguments), expected)


if __name__ == "__main__":
    unittest.main()
