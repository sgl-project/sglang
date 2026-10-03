import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _tool(name: str, properties: dict) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=name,
            parameters={"type": "object", "properties": properties},
        ),
    )


class TestZeroArgumentToolCallStreaming(unittest.TestCase):
    """A tool call that carries no ``arguments`` key must still stream ``{}``.

    A zero-parameter tool call leaves ``arguments`` absent. Streaming used to
    skip its whole bookkeeping block, so the call streamed an empty argument
    string (``json.loads("")`` raises client-side), the buffer was never
    drained, and ``current_tool_id`` never advanced -- which silently dropped
    every later tool call in the same response.
    """

    def setUp(self):
        self.tools = [
            _tool("get_time", {}),
            _tool("get_weather", {"city": {"type": "string"}}),
        ]

    def _stream(self, detector, text):
        events = []
        for ch in text:
            result = detector.parse_streaming_increment(ch, self.tools)
            for call in result.calls or []:
                events.append((call.tool_index, call.name, call.parameters))
        return events

    def test_zero_argument_call_streams_empty_object(self):
        events = self._stream(
            Qwen25Detector(), '<tool_call>\n{"name": "get_time"}\n</tool_call>'
        )
        names = [name for _, name, _ in events if name]
        self.assertEqual(names, ["get_time"])
        streamed_args = "".join(args for _, _, args in events if args)
        self.assertEqual(streamed_args, "{}")

    def test_zero_argument_call_does_not_drop_the_next_call(self):
        events = self._stream(
            Qwen25Detector(),
            '<tool_call>\n{"name": "get_time"}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
            "</tool_call>",
        )
        names = [name for _, name, _ in events if name]
        self.assertEqual(names, ["get_time", "get_weather"])
        per_tool = {}
        for index, _, args in events:
            if args:
                per_tool[index] = per_tool.get(index, "") + args
        self.assertEqual(per_tool.get(0), "{}")
        self.assertEqual(per_tool.get(1), '{"city": "Paris"}')

    def test_zero_argument_call_drains_the_buffer(self):
        detector = Qwen25Detector()
        self._stream(detector, '<tool_call>\n{"name": "get_time"}\n</tool_call>')
        self.assertNotIn("get_time", detector._buffer)

    def test_json_array_parser_streams_empty_object(self):
        # tool_choice="required" constrains output to a JSON array and is
        # model-independent, so it reaches this path for every model.
        events = self._stream(JsonArrayParser(), '[{"name": "get_time"}]')
        streamed_args = "".join(args for _, _, args in events if args)
        self.assertEqual(streamed_args, "{}")


if __name__ == "__main__":
    unittest.main()
