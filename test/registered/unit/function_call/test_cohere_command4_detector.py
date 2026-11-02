import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCohereCommand4Detector(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(name="get_weather", parameters={"type": "object"}),
            )
        ]
        self.body = json.dumps(
            [
                {
                    "tool_call_id": str(index),
                    "tool_name": "get_weather",
                    "parameters": {"city": city},
                }
                for index, city in enumerate(("SF", "Paris"))
            ]
        )
        self.block = "<|START_ACTION|>" + self.body + "<|END_ACTION|>"

    def assert_stream_matches(self, text, chunks):
        expected_text, expected_calls = FunctionCallParser(
            self.tools, "cohere_command4"
        ).parse_non_stream(text)
        parser = FunctionCallParser(self.tools, "cohere_command4")
        normal = ""
        calls = []
        for chunk in chunks:
            output, delta = parser.parse_stream_chunk(chunk)
            normal += output
            calls.extend(delta)
        output, delta = parser.parse_stream_end()
        normal += output
        calls.extend(delta)
        self.assertEqual(normal, expected_text)
        self.assertEqual(calls, expected_calls)

    def test_complete_block_with_preamble_in_one_chunk(self):
        text = "Let me check." + self.block
        self.assert_stream_matches(text, [text])

    def test_every_two_chunk_boundary(self):
        text = "Let me check." + self.block
        for split in range(1, len(text)):
            with self.subTest(split=split):
                self.assert_stream_matches(text, [text[:split], text[split:]])

    def test_character_chunks(self):
        text = "Let me check." + self.block
        self.assert_stream_matches(text, list(text))

    def test_block_without_preamble_and_plain_text(self):
        for text in (self.block, "The weather is sunny."):
            with self.subTest(text=text):
                self.assert_stream_matches(text, [text])


if __name__ == "__main__":
    unittest.main()
