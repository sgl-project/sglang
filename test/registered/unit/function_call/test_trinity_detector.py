"""Unit tests for TrinityDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.trinity_detector import TrinityDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestTrinityDetector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="write_file",
                    parameters={
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                    },
                ),
            )
        ]
        self.arguments = {"content": "Qwen emits <think> and </think> tags"}
        self.tool_call = (
            "<tool_call>\n"
            + json.dumps({"name": "write_file", "arguments": self.arguments})
            + "\n</tool_call>"
        )
        self.detector = TrinityDetector()

    def _stream(self, chunks):
        normal_text, names, parameters = "", [], ""
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            normal_text += result.normal_text
            for call in result.calls:
                if call.name:
                    names.append(call.name)
                parameters += call.parameters
        return normal_text, names, parameters

    def test_think_tags_in_arguments_preserved(self):
        result = self.detector.detect_and_parse(self.tool_call, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "write_file")
        self.assertEqual(json.loads(result.calls[0].parameters), self.arguments)

    def test_think_tags_outside_tool_call_stripped(self):
        text = "<think> Write the file </think>\n" + self.tool_call
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "Write the file")
        self.assertEqual(json.loads(result.calls[0].parameters), self.arguments)

    def test_tool_call_inside_think_section(self):
        text = "<think>plan\n" + self.tool_call + "</think>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "plan")
        self.assertEqual(json.loads(result.calls[0].parameters), self.arguments)

    def test_no_tool_call_strips_think_tags(self):
        result = self.detector.detect_and_parse(
            "<think>reasoning</think>answer", self.tools
        )
        self.assertEqual(result.normal_text, "reasoninganswer")
        self.assertEqual(result.calls, [])

    def test_streaming_think_tags_in_arguments_preserved(self):
        # Think tags arrive as whole tokens, both outside and inside the call.
        chunks = [
            "<think>",
            "Write the file",
            "</think>",
            "\n",
            '<tool_call>\n{"name": "write_file", "arguments": ',
            '{"content": "Qwen emits ',
            "<think>",
            " and ",
            "</think>",
            ' tags"}}',
            "\n</tool_call>",
        ]
        normal_text, names, parameters = self._stream(chunks)
        self.assertEqual(normal_text, "Write the file\n")
        self.assertEqual(names, ["write_file"])
        self.assertEqual(json.loads(parameters), self.arguments)


if __name__ == "__main__":
    import unittest

    unittest.main()
