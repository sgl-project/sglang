"""GPT-OSS native tool headers must parse without an assistant start prefix."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.gpt_oss_detector import GptOssDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestGptOssAbbreviatedHeader(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                ),
            )
        ]
        self.header = '<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"city":"Paris"}<|call|><|call|>'

    def test_abbreviated_and_full_header_parse(self):
        for prefix in ("", "<|start|>assistant"):
            with self.subTest(prefix=prefix):
                text = prefix + self.header
                detector = GptOssDetector()
                self.assertTrue(detector.has_tool_call(text))
                result = detector.detect_and_parse(text, self.tools)
                self.assertEqual(result.normal_text, "")
                self.assertEqual(len(result.calls), 1)
                self.assertEqual(result.calls[0].name, "get_weather")
                self.assertEqual(
                    json.loads(result.calls[0].parameters), {"city": "Paris"}
                )
                streamed = GptOssDetector().parse_streaming_increment(text, self.tools)
                self.assertEqual(result.calls, streamed.calls)

    def test_parser_entry_for_tool_choices(self):
        for choice in (
            "auto",
            "required",
            ToolChoice(function=ToolChoiceFuncName(name="get_weather")),
        ):
            with self.subTest(choice=choice):
                parser = FunctionCallParser(self.tools, "gpt-oss", tool_choice=choice)
                self.assertTrue(parser.has_tool_call(self.header))
                normal, calls = parser.parse_non_stream(self.header)
                self.assertEqual(normal, "")
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].name, "get_weather")
                self.assertEqual(json.loads(calls[0].parameters), {"city": "Paris"})

    def test_plain_and_unrelated_headers_are_not_calls(self):
        for text in (
            "ordinary commentary to=functions.get_weather",
            '{"city":"Paris"}',
            "<|channel|>final<|message|>The weather is sunny.<|return|>",
            "<|channel|>analysis<|message|>Think about weather.<|end|>",
            "<|channel|>commentary<|message|>Checking the weather.<|end|>",
        ):
            with self.subTest(text=text):
                detector = GptOssDetector()
                self.assertFalse(detector.has_tool_call(text))
                result = detector.detect_and_parse(text, self.tools)
                self.assertEqual(result.normal_text, text)
                self.assertEqual(result.calls, [])


if __name__ == "__main__":
    unittest.main()
