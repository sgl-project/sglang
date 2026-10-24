import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "default")

SECTION_BEGIN = "<｜tool▁calls▁begin｜>"
SECTION_END = "<｜tool▁calls▁end｜>"
CALL_BEGIN = "<｜tool▁call▁begin｜>"
CALL_END = "<｜tool▁call▁end｜>"
SEP = "<｜tool▁sep｜>"


class TestDeepSeekV31Streaming(unittest.TestCase):
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

    def _stream(self, detector, deltas):
        items = []
        for delta in deltas:
            items.extend(detector.parse_streaming_increment(delta, self.tools).calls)
        return items

    def test_delta_spanning_two_calls_keeps_both_arguments(self):
        # One delta carries the end of the first call and the start of the
        # second. The first call's arguments must still stream, and the
        # second call must stream under its own index.
        deltas = [
            SECTION_BEGIN + CALL_BEGIN + "get_weather" + SEP,
            '{"city": "Tokyo"}' + CALL_END + CALL_BEGIN + "get_weather" + SEP,
            '{"city": "Paris"}' + CALL_END + SECTION_END,
        ]
        items = self._stream(DeepSeekV31Detector(), deltas)
        names = [item for item in items if item.name is not None]
        args = [item for item in items if item.name is None]
        self.assertEqual([item.tool_index for item in names], [0, 1])
        self.assertEqual([item.tool_index for item in args], [0, 1])
        self.assertEqual(args[0].parameters, '{"city": "Tokyo"}')
        self.assertEqual(args[1].parameters, '{"city": "Paris"}')

    def test_complete_call_in_single_delta_streams_arguments(self):
        # With a large stream interval the header, the full arguments and the
        # end marker can all arrive in one delta. The arguments must stream in
        # the same pass as the name, and nothing may be left for the serving
        # layer's end-of-stream flush.
        detector = DeepSeekV31Detector()
        deltas = [
            SECTION_BEGIN
            + CALL_BEGIN
            + "get_weather"
            + SEP
            + '{"city": "Tokyo"}'
            + CALL_END
            + SECTION_END
        ]
        items = self._stream(detector, deltas)
        self.assertEqual(items[0].name, "get_weather")
        self.assertEqual(items[0].parameters, "")
        self.assertEqual(items[1].name, None)
        self.assertEqual(items[1].parameters, '{"city": "Tokyo"}')
        self.assertEqual(items[1].tool_index, 0)
        # Flush-visible state: stored arguments equal what was streamed.
        self.assertEqual(detector.prev_tool_call_arr[0]["arguments"], {"city": "Tokyo"})
        self.assertEqual(detector.streamed_args_for_tool[0], '{"city": "Tokyo"}')

    def test_arguments_streamed_progressively(self):
        deltas = [
            SECTION_BEGIN + CALL_BEGIN + "get_weather" + SEP,
            '{"city": "Tok',
            'yo"}' + CALL_END + SECTION_END,
        ]
        items = self._stream(DeepSeekV31Detector(), deltas)
        args = [item.parameters for item in items if item.name is None]
        self.assertEqual(args, ['{"city": "Tok', 'yo"}'])

    def test_one_shot_parse_unchanged(self):
        text = (
            SECTION_BEGIN
            + CALL_BEGIN
            + "get_weather"
            + SEP
            + '{"city": "Tokyo"}'
            + CALL_END
            + CALL_BEGIN
            + "get_weather"
            + SEP
            + '{"city": "Paris"}'
            + CALL_END
            + SECTION_END
        )
        result = DeepSeekV31Detector().detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Tokyo"})
        self.assertEqual(json.loads(result.calls[1].parameters), {"city": "Paris"})


if __name__ == "__main__":
    unittest.main()
