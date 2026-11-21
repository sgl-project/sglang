import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_tools(field):
    return [
        Tool(
            type="function",
            function=Function(
                name="inspect",
                parameters={"type": "object", "properties": {"value": field}},
            ),
        )
    ]


class TestGlm47StreamingBuffer(CustomTestCase):
    def test_emitted_values_are_not_retained_in_argument_buffer(self):
        for value_type, value in (
            ("string", 'line\n"quoted" \\ café <tag> ' * 256),
            ("array", list(range(256))),
            ("object", {str(i): i for i in range(128)}),
            ("boolean", True),
        ):
            with self.subTest(value_type=value_type):
                tools = make_tools({"type": value_type})
                raw = value if value_type == "string" else json.dumps(value)
                detector = Glm47MoeDetector()
                chunks = ["<tool_call>inspect<arg_key>value</arg_key><arg_value>"]
                chunks.extend(raw[i : i + 31] for i in range(0, len(raw), 31))
                calls = []
                for chunk in chunks:
                    calls.extend(detector.parse_streaming_increment(chunk, tools).calls)
                    # The XML tail can retain a partial closing tag. Already
                    # emitted argument content must not accumulate a second copy.
                    self.assertEqual(detector._current_value, "")
                for char in "</arg_value></tool_call>":
                    calls.extend(detector.parse_streaming_increment(char, tools).calls)
                self.assertEqual(
                    json.loads("".join(call.parameters for call in calls)),
                    {"value": value},
                )

    def test_empty_values_and_closing_tag_boundaries(self):
        for raw in ("", "a", "<tag>", "</arg_val suffix", 'x\n"y"\\z', "é"):
            text = (
                "<tool_call>inspect<arg_key>value</arg_key><arg_value>"
                + raw
                + "</arg_value></tool_call>"
            )
            for size in (1, 7, len(text)):
                with self.subTest(raw=raw, chunk_size=size):
                    detector = Glm47MoeDetector()
                    calls = []
                    for offset in range(0, len(text), size):
                        calls.extend(
                            detector.parse_streaming_increment(
                                text[offset : offset + size],
                                make_tools({"type": "string"}),
                            ).calls
                        )
                    self.assertEqual(
                        json.loads("".join(call.parameters for call in calls)),
                        {"value": raw},
                    )

    def test_buffered_numeric_and_inferred_values_are_preserved(self):
        for field, raw, expected in (
            ({"type": "integer"}, "123456", 123456),
            ({"type": "number"}, "1.25e2", 125.0),
            ({}, '{"ok": true}', {"ok": True}),
        ):
            with self.subTest(field=field, raw=raw):
                detector = Glm47MoeDetector()
                tools = make_tools(field)
                text = (
                    "<tool_call>inspect<arg_key>value</arg_key><arg_value>"
                    + raw
                    + "</arg_value></tool_call>"
                )
                calls = []
                for char in text:
                    calls.extend(detector.parse_streaming_increment(char, tools).calls)
                self.assertEqual(
                    json.loads("".join(call.parameters for call in calls)),
                    {"value": expected},
                )


if __name__ == "__main__":
    unittest.main()
