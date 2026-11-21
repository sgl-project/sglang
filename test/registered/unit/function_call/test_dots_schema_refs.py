import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.dots_detector import DotsToolDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _tool(name, properties):
    return Tool(
        type="function",
        function=Function(
            name=name, parameters={"type": "object", "properties": properties}
        ),
    )


class TestDotsSchemaRefs(CustomTestCase):
    def test_non_stream_resolves_escaped_definition_names(self):
        for name in ("Count", "Count/Value", "Count~Value", "Count~1Value"):
            with self.subTest(name=name):
                pointer = name.replace("~", "~0").replace("/", "~1")
                tool = _tool("count", {"value": {"$ref": f"#/$defs/{pointer}"}})
                tool.function.parameters["$defs"] = {name: {"type": "integer"}}
                parser = DotsToolDetector()
                text = (
                    '<dots_function_call><invoke name="count">'
                    '<parameter name="value">42</parameter>'
                    "</invoke></dots_function_call>"
                )

                calls = parser.detect_and_parse(text, [tool]).calls
                self.assertEqual(len(calls), 1)
                self.assertEqual(json.loads(calls[0].parameters), {"value": 42})

                parser = DotsToolDetector()
                streamed = []
                for chunk in (text[:31], text[31:65], text[65:]):
                    streamed.extend(
                        parser.parse_streaming_increment(chunk, [tool]).calls
                    )
                self.assertEqual(len(streamed), 1)
                self.assertEqual(json.loads(streamed[0].parameters), {"value": 42})


if __name__ == "__main__":
    unittest.main()
