"""Unit tests for RippleDetector against the checkpoint's chat template layout."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.ripple_detector import RippleDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def call(name, arguments):
    return f'<tool_call>\n{{"name": "{name}", "arguments": {json.dumps(arguments)}}}\n</tool_call>'


ADD = call("add", {"a": 1, "b": 2})
SEARCH = call("search", {"query": "x"})


class TestRippleDetector(CustomTestCase):
    TOOLS = [
        Tool(
            type="function", function=Function(name=name, parameters={"type": "object"})
        )
        for name in ("add", "search")
    ]

    def _stream(self, text, size):
        detector = RippleDetector()
        parts = [
            detector.parse_streaming_increment(text[i : i + size], self.TOOLS)
            for i in range(0, len(text), size)
        ]
        parts.append(detector.finish(self.TOOLS))
        calls = {}
        for part in parts:
            for item in part.calls:
                entry = calls.setdefault(item.tool_index, {"name": "", "arguments": ""})
                entry["name"] += item.name or ""
                entry["arguments"] += item.parameters or ""
        return "".join(part.normal_text for part in parts), [
            (calls[i]["name"], json.loads(calls[i]["arguments"])) for i in sorted(calls)
        ]

    def test_streaming_matches_one_shot_at_any_chunk_size(self):
        cases = {
            "\n" + ADD: "",
            "Let me add.\n" + ADD: "Let me add.",
            "Keeps\n\ninner lines.\n" + SEARCH + "\n" + ADD: "Keeps\n\ninner lines.",
            "Only text, trailing newline.\n": "Only text, trailing newline.\n",
            "ends in a partial tag <tool_c": "ends in a partial tag <tool_c",
        }
        for text, content in cases.items():
            result = RippleDetector().detect_and_parse(text, self.TOOLS)
            self.assertEqual(result.normal_text, content)
            expected = [(c.name, json.loads(c.parameters)) for c in result.calls]
            for size in (1, 2, 3, 5, 8, 13, 1000):
                self.assertEqual(
                    self._stream(text, size), (content, expected), (text, size)
                )

    def test_constrained_call_is_in_the_template_layout(self):
        info = RippleDetector().structure_info()("add")
        self.assertEqual(info.begin + '{"a": 1, "b": 2}' + info.end, ADD)


if __name__ == "__main__":
    unittest.main()
