"""Unit tests for AgensDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.agens_detector import AgensDetector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="get_weather",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city"],
            },
        ),
    )
]

CALL = (
    "<|call|>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n"
    "<parameter=days>\n3\n</parameter>\n</function>\n<|/call|>"
)


class TestAgensDetector(CustomTestCase):
    def setUp(self):
        self.detector = AgensDetector()

    def test_registered(self):
        self.assertIs(FunctionCallParser.ToolCallParserEnum["agens"], AgensDetector)

    def test_plain_text(self):
        result = self.detector.detect_and_parse("No tools needed.", TOOLS)
        self.assertEqual(result.normal_text, "No tools needed.")
        self.assertEqual(result.calls, [])

    def test_single_call_with_prefix(self):
        result = self.detector.detect_and_parse("Checking.\n" + CALL, TOOLS)
        self.assertTrue(result.normal_text.startswith("Checking."))
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"city": "Tokyo", "days": 3}
        )

    def test_two_calls(self):
        result = self.detector.detect_and_parse(CALL + "\n" + CALL, TOOLS)
        self.assertEqual([c.name for c in result.calls], ["get_weather"] * 2)

    def test_qwen_sentinels_are_not_agens_calls(self):
        text = CALL.replace("<|call|>", "<tool_call>").replace(
            "<|/call|>", "</tool_call>"
        )
        self.assertFalse(self.detector.has_tool_call(text))

    def test_streaming_matches_one_shot(self):
        detector = AgensDetector()
        names, params = [], ""
        for i in range(0, len(CALL), 7):
            res = detector.parse_streaming_increment(CALL[i : i + 7], TOOLS)
            for call in res.calls:
                if call.name:
                    names.append(call.name)
                params += call.parameters or ""
        self.assertEqual(names, ["get_weather"])
        self.assertEqual(json.loads(params), {"city": "Tokyo", "days": 3})

    def test_structure_info(self):
        info = self.detector.structure_info()("get_weather")
        self.assertEqual(info.begin, "<|call|>\n<function=get_weather>\n")
        self.assertEqual(info.end, "</function>\n<|/call|>")
        self.assertEqual(info.trigger, "<|call|>")
        self.assertIsNone(self.detector.get_structural_tag_name())


if __name__ == "__main__":
    import unittest

    unittest.main()
