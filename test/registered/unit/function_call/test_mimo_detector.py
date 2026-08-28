"""CPU-only tests for the MiMo tool-call streaming parser."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMiMoDetector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="file",
                    description="Write a file",
                    parameters={
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "content": {"type": "string"},
                            "path": {"type": "string"},
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="typed",
                    description="Exercise typed parameters",
                    parameters={
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "ratio": {"type": "number"},
                            "enabled": {"type": "boolean"},
                            "payload": {"type": "object"},
                            "text": {"type": "string"},
                        },
                    },
                ),
            ),
        ]

    @staticmethod
    def _collect(detector, chunks, tools):
        normal_text = ""
        calls_by_index = {}
        calls_per_chunk = []

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, tools)
            normal_text += result.normal_text
            calls_per_chunk.append(result.calls)
            for call in result.calls:
                assembled = calls_by_index.setdefault(
                    call.tool_index, {"name": None, "arguments": ""}
                )
                if call.name is not None:
                    assembled["name"] = call.name
                assembled["arguments"] += call.parameters

        return normal_text, calls_by_index, calls_per_chunk

    def test_streams_long_string_before_tool_call_closes(self):
        detector = MiMoDetector()
        chunks = [
            "before<tool",
            "_call><function=file><parameter=action>wri",
            "te</parameter><parameter=content><!DOCTYPE html>\n",
            '<div title="quoted">中文 &am',
            "p; more</para",
            "meter></function></tool_call>after",
        ]

        normal, calls, calls_per_chunk = self._collect(detector, chunks, self.tools)

        self.assertEqual(normal, "beforeafter")
        self.assertEqual(calls[0]["name"], "file")
        self.assertEqual(
            json.loads(calls[0]["arguments"]),
            {
                "action": "write",
                "content": '<!DOCTYPE html>\n<div title="quoted">中文 & more',
            },
        )
        self.assertEqual(calls_per_chunk[1][0].name, "file")
        self.assertTrue(
            any(call.parameters for call in calls_per_chunk[1]),
            "arguments should start streaming before </parameter>",
        )
        self.assertTrue(
            any(call.parameters for call in calls_per_chunk[3]),
            "long string content should produce intermediate deltas",
        )
        self.assertFalse(
            any("</para" in call.parameters for call in calls_per_chunk[4])
        )

    def test_typed_parameters_remain_compatible_with_non_streaming(self):
        model_output = (
            "<tool_call><function=typed>"
            "<parameter=count>42</parameter>"
            "<parameter=ratio>3.5</parameter>"
            "<parameter=enabled>true</parameter>"
            '<parameter=payload>{"key": [1, 2]}</parameter>'
            "</function></tool_call>"
        )
        detector = MiMoDetector()

        _, calls, _ = self._collect(
            detector,
            [model_output[:80], model_output[80:140], model_output[140:]],
            self.tools,
        )
        non_stream = MiMoDetector().detect_and_parse(model_output, self.tools)

        self.assertEqual(calls[0]["name"], "typed")
        self.assertEqual(
            json.loads(calls[0]["arguments"]),
            json.loads(non_stream.calls[0].parameters),
        )
        self.assertEqual(
            json.loads(calls[0]["arguments"]),
            {
                "count": 42,
                "ratio": 3.5,
                "enabled": True,
                "payload": {"key": [1, 2]},
            },
        )

    def test_character_by_character_boundaries_preserve_json_escaping(self):
        detector = MiMoDetector()
        model_output = (
            "prefix<tool_call><function=file><parameter=content>"
            'line 1\n"quoted" \\ slash &amp; 中文'
            "</parameter></function></tool_call>suffix"
        )

        normal, calls, _ = self._collect(detector, list(model_output), self.tools)

        self.assertEqual(normal, "prefixsuffix")
        self.assertEqual(
            json.loads(calls[0]["arguments"]),
            {"content": 'line 1\n"quoted" \\ slash & 中文'},
        )

    def test_string_null_literal_is_not_streamed_as_a_string(self):
        detector = MiMoDetector()
        chunks = [
            "<tool_call><function=typed><parameter=text>n",
            "ul",
            "l</parameter></function></tool_call>",
        ]

        _, calls, calls_per_chunk = self._collect(detector, chunks, self.tools)

        self.assertEqual(len(calls_per_chunk[0]), 1)
        self.assertEqual(calls_per_chunk[0][0].name, "typed")
        self.assertEqual(calls_per_chunk[0][0].parameters, "")
        self.assertEqual(calls_per_chunk[1], [])
        self.assertEqual(json.loads(calls[0]["arguments"]), {"text": None})

    def test_drains_multiple_calls_and_interleaved_text_from_one_chunk(self):
        detector = MiMoDetector()
        chunk = (
            "<tool_call><function=file>"
            "<parameter=path>a.txt</parameter>"
            "</function></tool_call>middle"
            "<tool_call><function=file>"
            "<parameter=path>b.txt</parameter>"
            "</function></tool_call>tail"
        )

        normal, calls, calls_per_chunk = self._collect(detector, [chunk], self.tools)

        self.assertEqual(normal, "middletail")
        self.assertEqual(sorted(calls), [0, 1])
        self.assertEqual(json.loads(calls[0]["arguments"]), {"path": "a.txt"})
        self.assertEqual(json.loads(calls[1]["arguments"]), {"path": "b.txt"})
        self.assertEqual(
            [call.name for call in calls_per_chunk[0] if call.name],
            ["file", "file"],
        )

    def test_empty_arguments_are_emitted_as_valid_json(self):
        detector = MiMoDetector()

        _, calls, _ = self._collect(
            detector,
            ["<tool_call><function=file></function></tool_call>"],
            self.tools,
        )

        self.assertEqual(calls[0], {"name": "file", "arguments": "{}"})

    def test_unknown_tool_is_forwarded_as_normal_text_by_default(self):
        detector = MiMoDetector()
        tool_block = (
            "<tool_call><function=missing>"
            "<parameter=value>partial content</parameter>"
            "</function></tool_call>"
        )

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
            normal, calls, _ = self._collect(
                detector,
                [tool_block[:35], tool_block[35:] + "tail"],
                self.tools,
            )

        self.assertEqual(normal, tool_block + "tail")
        self.assertEqual(calls, {})


if __name__ == "__main__":
    unittest.main()
