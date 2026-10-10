import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepSeekV31Detector(unittest.TestCase):
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
        self.parser = FunctionCallParser(self.tools, "deepseekv31")

    def _call(self, city):
        info = self.parser.detector.structure_info()("get_weather")
        return info.begin + json.dumps({"city": city}) + info.end

    def test_structural_tag_output_round_trips(self):
        text = self._call("Paris")
        self.assertTrue(self.parser.has_tool_call(text))
        normal, calls = self.parser.parse_non_stream(text)
        self.assertEqual(normal, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters), {"city": "Paris"})

    def test_wrapped_and_unwrapped_calls_preserve_preamble(self):
        body = self._call("Paris") + self._call("Tokyo")
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                text = body
                if wrapped:
                    text = "<｜tool▁calls▁begin｜>" + text + "<｜tool▁calls▁end｜>"
                normal, calls = self.parser.parse_non_stream("Checking.\n" + text)
                self.assertEqual(normal, "Checking.")
                self.assertEqual([call.name for call in calls], ["get_weather"] * 2)
                self.assertEqual(
                    [json.loads(call.parameters) for call in calls],
                    [{"city": "Paris"}, {"city": "Tokyo"}],
                )

    def test_plain_text_is_unchanged(self):
        text = "The weather in Paris is sunny."
        self.assertFalse(self.parser.has_tool_call(text))
        self.assertEqual(self.parser.parse_non_stream(text), (text, []))

    def test_structural_tag_streaming_matches_non_streaming(self):
        info = self.parser.detector.structure_info()("get_weather")
        arguments = json.dumps({"city": "Paris"})
        normal, calls = self.parser.parse_non_stream(info.begin + arguments + info.end)
        stream_normal = ""
        stream_calls = []
        for chunk in (info.begin, arguments, info.end):
            text, delta = self.parser.parse_stream_chunk(chunk)
            stream_normal += text
            stream_calls.extend(delta)
        self.assertEqual(stream_normal, normal)
        self.assertEqual(
            [call.name for call in stream_calls if call.name], [calls[0].name]
        )
        self.assertEqual(
            json.loads("".join(call.parameters for call in stream_calls)),
            json.loads(calls[0].parameters),
        )

    def test_invalid_arguments_fall_back_to_plain_text(self):
        text = self._call("Paris").replace('{"city": "Paris"}', "not JSON")
        self.assertEqual(self.parser.parse_non_stream(text), (text, []))


if __name__ == "__main__":
    unittest.main()
