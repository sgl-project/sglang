"""DeepSeek V3.1 streaming tool calls across coalesced chunks; no server."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

BOT = "<｜tool▁calls▁begin｜>"
EOT = "<｜tool▁calls▁end｜>"
CALL_BEGIN = "<｜tool▁call▁begin｜>"
CALL_END = "<｜tool▁call▁end｜>"
SEP = "<｜tool▁sep｜>"


class TestDeepSeekV31Streaming(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(type="function", function=Function(name=name, parameters={}))
            for name in ["weather", "search"]
        ]
        self.names = ["weather", "search"]
        self.arguments = [
            '{"city":"Tokyo","options":{"days":2},"note":"晴れ"}',
            '{"query":"cafes","filters":["open",true,null]}',
        ]
        self.headers = [CALL_BEGIN + name + SEP for name in self.names]

    def _assert_stream(self, chunks, names=None, arguments=None):
        names = self.names if names is None else names
        arguments = self.arguments if arguments is None else arguments
        detector = DeepSeekV31Detector()
        calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            self.assertEqual(result.normal_text, "")
            calls.extend(result.calls)
        self.assertEqual(
            [(call.tool_index, call.name) for call in calls if call.name],
            list(enumerate(names)),
        )
        self.assertEqual({call.tool_index for call in calls}, set(range(len(names))))
        for i, arguments_json in enumerate(arguments):
            streamed = "".join(
                call.parameters for call in calls if call.tool_index == i
            )
            self.assertEqual(json.loads(streamed), json.loads(arguments_json))
            self.assertEqual(detector.streamed_args_for_tool[i], streamed)
            self.assertEqual(
                detector.prev_tool_call_arr[i],
                {"name": names[i], "arguments": json.loads(arguments_json)},
            )
        # Once arguments have been sent, end-of-stream cannot duplicate them.
        self.assertEqual(detector.finish(self.tools).calls, [])
        return calls

    def test_single_call_name_and_arguments_in_one_chunk(self):
        self._assert_stream(
            [BOT + self.headers[0] + self.arguments[0] + CALL_END + EOT],
            self.names[:1],
            self.arguments[:1],
        )

    def test_multiple_calls_in_one_chunk(self):
        self._assert_stream(
            [
                BOT
                + "".join(
                    h + a + CALL_END for h, a in zip(self.headers, self.arguments)
                )
                + EOT
            ]
        )

    def test_first_header_then_coalesced_calls(self):
        self._assert_stream(
            [
                BOT + self.headers[0],
                self.arguments[0]
                + CALL_END
                + self.headers[1]
                + self.arguments[1]
                + CALL_END
                + EOT,
            ]
        )

    def test_sequential_header_argument_chunks(self):
        self._assert_stream(
            [
                BOT + self.headers[0],
                self.arguments[0] + CALL_END,
                self.headers[1],
                self.arguments[1] + CALL_END + EOT,
            ]
        )

    def test_first_complete_call_and_partial_second(self):
        for split in range(1, len(self.arguments[1])):
            with self.subTest(split=split):
                self._assert_stream(
                    [
                        BOT
                        + self.headers[0]
                        + self.arguments[0]
                        + CALL_END
                        + self.headers[1]
                        + self.arguments[1][:split],
                        self.arguments[1][split:] + CALL_END + EOT,
                    ]
                )

    def test_json_complete_before_end_marker(self):
        self._assert_stream(
            [
                BOT + self.headers[0] + self.arguments[0],
                "  " + CALL_END + self.headers[1] + self.arguments[1],
                CALL_END + EOT,
            ]
        )

    def test_repeated_same_tool_uses_new_call_index(self):
        self._assert_stream(
            [
                BOT
                + self.headers[0]
                + self.arguments[0]
                + CALL_END
                + self.headers[0]
                + self.arguments[1]
                + CALL_END
                + EOT
            ],
            ["weather", "weather"],
        )

    def test_empty_arguments_and_stop_before_end_marker(self):
        self._assert_stream([BOT + self.headers[0] + "{}"], ["weather"], ["{}"])

    def test_plain_text_and_nonstream_control(self):
        detector = DeepSeekV31Detector()
        self.assertEqual(
            detector.parse_streaming_increment("hello", self.tools).normal_text, "hello"
        )
        text = (
            BOT
            + "".join(h + a + CALL_END for h, a in zip(self.headers, self.arguments))
            + EOT
        )
        parsed = detector.detect_and_parse(text, self.tools)
        self.assertEqual([call.name for call in parsed.calls], self.names)
        self.assertEqual(
            [json.loads(call.parameters) for call in parsed.calls],
            [json.loads(a) for a in self.arguments],
        )


if __name__ == "__main__":
    unittest.main()
