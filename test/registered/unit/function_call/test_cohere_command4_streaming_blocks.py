import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestCohereCommand4StreamingBlocks(CustomTestCase):
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
        self.block = (
            '<|START_ACTION|>[{"tool_call_id":"0","tool_name":"get_weather",'
            '"parameters":{"city":"SF"}}]<|END_ACTION|>'
        )

    def parse_chunks(self, chunks):
        parser = FunctionCallParser(
            tools=self.tools, tool_call_parser="cohere_command4"
        )
        text = ""
        calls = []
        for chunk in chunks:
            normal_text, new_calls = parser.parse_stream_chunk(chunk)
            text += normal_text
            calls.extend(new_calls)
        end_text, end_calls = parser.parse_stream_end()
        return text + end_text, calls + end_calls

    def assert_weather(self, calls, count=1):
        self.assertEqual(len(calls), count)
        for call in calls:
            self.assertEqual(call.name, "get_weather")
            self.assertEqual(json.loads(call.parameters), {"city": "SF"})

    def test_leading_text_and_complete_call_in_final_chunk(self):
        text, calls = self.parse_chunks(["Let me check." + self.block])
        self.assertEqual(text, "Let me check.")
        self.assert_weather(calls)

    def test_every_two_chunk_split_matches_non_streaming(self):
        source = "Let me check." + self.block
        expected = FunctionCallParser(
            tools=self.tools, tool_call_parser="cohere_command4"
        ).parse_non_stream(source)
        for split in range(len(source) + 1):
            with self.subTest(split=split):
                self.assertEqual(
                    self.parse_chunks([source[:split], source[split:]]), expected
                )

    def test_character_chunks(self):
        text, calls = self.parse_chunks(list("Let me check." + self.block))
        self.assertEqual(text, "Let me check.")
        self.assert_weather(calls)

    def test_trailing_text_is_preserved_for_next_increment(self):
        parser = FunctionCallParser(
            tools=self.tools, tool_call_parser="cohere_command4"
        )
        text, calls = parser.parse_stream_chunk(
            "Let me check." + self.block + "All set."
        )
        self.assertEqual(text, "Let me check.")
        self.assert_weather(calls)
        self.assertEqual(parser.parse_stream_chunk(""), ("All set.", []))

    def test_multiple_calls_in_one_action_block(self):
        action = json.loads(
            self.block[len("<|START_ACTION|>") : -len("<|END_ACTION|>")]
        )
        block = "<|START_ACTION|>" + json.dumps(action * 2) + "<|END_ACTION|>"
        text, calls = self.parse_chunks(["Let me check." + block])
        self.assertEqual(text, "Let me check.")
        self.assert_weather(calls, count=2)

    def test_complete_call_without_leading_text(self):
        text, calls = self.parse_chunks([self.block])
        self.assertEqual(text, "")
        self.assert_weather(calls)

    def test_unknown_tool_is_dropped(self):
        text, calls = self.parse_chunks(
            ["Let me check." + self.block.replace("get_weather", "unknown_tool")]
        )
        self.assertEqual((text, calls), ("Let me check.", []))

    def test_incomplete_block_waits_for_closing_marker(self):
        parser = FunctionCallParser(
            tools=self.tools, tool_call_parser="cohere_command4"
        )
        first = parser.parse_stream_chunk("Let me check." + self.block[:-1])
        self.assertEqual(first, ("Let me check.", []))
        text, calls = parser.parse_stream_chunk(self.block[-1:])
        self.assertEqual(text, "")
        self.assert_weather(calls)
        self.assertEqual(parser.parse_stream_end(), ("", []))

    def test_plain_text(self):
        self.assertEqual(self.parse_chunks(["Hello", " world."]), ("Hello world.", []))


if __name__ == "__main__":
    unittest.main()
