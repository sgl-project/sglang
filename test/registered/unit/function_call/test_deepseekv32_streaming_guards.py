"""Unit tests for DeepSeek DSML streaming unknown-tool filtering and tag prefix guards."""

import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.deepseekv41_detector import DeepSeekV41Detector
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")

DSML = "｜DSML｜"


class TestDeepSeekStreamingUnknownToolsAndGuards(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                ),
            )
        ]

    @staticmethod
    def _detector_classes():
        from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
        from sglang.srt.function_call.deepseekv41_detector import DeepSeekV41Detector
        from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector

        return (DeepSeekV32Detector, DeepSeekV4Detector, DeepSeekV41Detector)

    def _feed_all(self, chunks, detector_cls=None):
        if detector_cls is None:
            detector_cls = self._detector_classes()[1]
        """Returns (normal_text, calls) accumulated over chunks plus finish()."""
        detector = detector_cls()
        normal, calls = "", []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        end = detector.finish(self.tools)
        normal += end.normal_text
        calls.extend(end.calls)
        return normal, calls

    def test_streaming_drops_unknown_tool_unless_forward_unknown_enabled(self):
        """Streaming and non-streaming must agree on unknown tool calls across
        V3.2, V4, and V4.1 detectors and chunk sizes."""
        for cls in self._detector_classes():
            d = cls()
            p = f"{d.dsml_token}{cls.parameter_tag_name}"
            text = (
                f"Checking.\n\n{d.bot_token}\n"
                f'{d.invoke_start_token} name="unknown_tool">\n'
                f'<{p} name="city" string="true">SF</{p}>\n'
                f"{d.invoke_end_token}\n"
                f'{d.invoke_start_token} name="get_weather">\n'
                f'<{p} name="city" string="true">NY</{p}>\n'
                f"{d.invoke_end_token}\n"
                f"{d.eot_token}"
            )
            one_shot = cls().detect_and_parse(text, self.tools)
            self.assertEqual(one_shot.normal_text, "Checking.")
            self.assertEqual(
                [(call.name, call.parameters) for call in one_shot.calls],
                [("get_weather", '{"city": "NY"}')],
            )
            for width in (1, 5, 13, len(text)):
                chunks = [text[i : i + width] for i in range(0, len(text), width)]
                with self.subTest(cls=cls.__name__, width=width):
                    normal, calls = self._feed_all(chunks, cls)
                    self.assertEqual(normal, one_shot.normal_text)
                    self.assertEqual(
                        [
                            (call.tool_index, call.name, call.parameters)
                            for call in calls
                        ],
                        [(0, "get_weather", '{"city": "NY"}')],
                    )

            with (
                patch(
                    "sglang.srt.function_call.deepseekv32_detector.envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get",
                    return_value=True,
                ),
                patch(
                    "sglang.srt.function_call.base_format_detector.envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get",
                    return_value=True,
                ),
            ):
                one_shot_fwd = cls().detect_and_parse(text, self.tools)
                for width in (1, 7, len(text)):
                    chunks = [text[i : i + width] for i in range(0, len(text), width)]
                    with self.subTest(cls=cls.__name__, forward=True, width=width):
                        normal_fwd, calls_fwd = self._feed_all(chunks, cls)
                        self.assertEqual(normal_fwd, one_shot_fwd.normal_text)
                        self.assertEqual(
                            [
                                (call.tool_index, call.name, call.parameters)
                                for call in calls_fwd
                            ],
                            [
                                (0, "unknown_tool", '{"city": "SF"}'),
                                (1, "get_weather", '{"city": "NY"}'),
                            ],
                        )
                        self.assertEqual(
                            [(call.name, call.parameters) for call in calls_fwd],
                            [
                                (call.name, call.parameters)
                                for call in one_shot_fwd.calls
                            ],
                        )

    def test_unknown_only_invoke_withholds_trailing_dsml_on_finish(self):
        """When the only invoke in a block is for an unknown tool, the preamble
        is emitted once and finish() does not leak the trailing closing tag."""
        v32_cls, v4_cls, v41_cls = self._detector_classes()
        for parser_name, cls in (
            ("deepseekv32", v32_cls),
            ("deepseekv4", v4_cls),
            ("deepseekv41", v41_cls),
        ):
            d = cls()
            p = f"{d.dsml_token}{cls.parameter_tag_name}"
            text = (
                f"Checking.\n\n{d.bot_token}\n"
                f'{d.invoke_start_token} name="unknown_tool">\n'
                f'<{p} name="city" string="true">SF</{p}>\n'
                f"{d.invoke_end_token}\n"
                f"{d.eot_token}"
            )
            for width in (1, 5, len(text)):
                with self.subTest(parser=parser_name, width=width):
                    parser = FunctionCallParser(self.tools, parser_name)
                    normal, calls = "", []
                    for idx in range(0, len(text), width):
                        chunk_normal, chunk_calls = parser.parse_stream_chunk(
                            text[idx : idx + width]
                        )
                        normal += chunk_normal
                        calls.extend(chunk_calls)
                    end_normal, end_calls = parser.parse_stream_end()
                    normal += end_normal
                    calls.extend(end_calls)

                    self.assertEqual(normal, "Checking.")
                    self.assertEqual(calls, [])

    def test_non_dsml_fullwidth_bar_tag_does_not_stall_streaming(self):
        """A fullwidth bar tag like `<|end|>` is not a DSML tool tag and must
        stream immediately instead of stalling in the buffer until finish()."""
        for cls in self._detector_classes():
            detector = cls()
            first = detector.parse_streaming_increment("hello <｜end｜>", self.tools)
            second = detector.parse_streaming_increment(" world", self.tools)

            self.assertEqual(first.normal_text, "hello <｜end｜>")
            self.assertEqual(second.normal_text, " world")
            self.assertEqual(detector.finish(self.tools).normal_text, "")

    def test_literal_dsml_token_in_plain_prose_streams_without_truncation(self):
        """Prose mentioning the bare DSML marker without `<` or `</` must
        stream immediately and never be truncated by _text_before_dsml."""
        text = f"The tag {DSML} marks tool calls."
        for cls in self._detector_classes():
            d = cls()
            p = f"{d.dsml_token}{cls.parameter_tag_name}"

            # Must emit on increment instead of stalling in _buffer until finish().
            detector = cls()
            inc = detector.parse_streaming_increment(text, self.tools)
            self.assertEqual(inc.normal_text, text)
            self.assertEqual(detector.finish(self.tools).normal_text, "")

            # When a trailing partial prefix or incomplete invoke holds the buffer
            # until finish(), _text_before_dsml runs on prose containing bare DSML.
            held_chunks = [f"The tag {DSML} marks a < b\n\n", "<"]
            held_text = "".join(held_chunks)
            self.assertEqual(
                cls().detect_and_parse(held_text, self.tools).normal_text, held_text
            )
            self.assertEqual(self._feed_all(held_chunks, cls), (held_text, []))

            incomplete_invoke = [
                f"The tag {DSML} marks.\n\n",
                f'{d.bot_token}\n{d.invoke_start_token} name="get_weather">',
            ]
            self.assertEqual(
                self._feed_all(incomplete_invoke, cls),
                (f"The tag {DSML} marks.", []),
            )

            # Bare invoke after prose containing bare DSML exercises
            # _text_before_dsml in both detect_and_parse and streaming.
            bare_invoke_text = (
                f"The tag {DSML} marks tool calls.\n\n"
                f'{d.invoke_start_token} name="get_weather">'
                f'<{p} name="city" string="true">SF</{p}>'
                f"{d.invoke_end_token}"
            )
            one_shot = cls().detect_and_parse(bare_invoke_text, self.tools)
            self.assertEqual(one_shot.normal_text, text)
            self.assertEqual(
                [(c.name, c.parameters) for c in one_shot.calls],
                [("get_weather", '{"city": "SF"}')],
            )
            stream_normal, stream_calls = self._feed_all(
                [bare_invoke_text[:18], bare_invoke_text[18:]], cls
            )
            self.assertEqual(stream_normal, text)
            self.assertEqual(
                [(c.name, c.parameters) for c in stream_calls],
                [("get_weather", '{"city": "SF"}')],
            )

    def test_trailing_partial_dsml_prefix_preserved_at_stream_end(self):
        """A reply ending with `<` or a partial DSML prefix at EOF must flush
        unchanged in finish() and match detect_and_parse."""
        for cls in self._detector_classes():
            for chunks in (
                ["for (int i = 0; i ", "<"],
                ["hello\n\n", "<"],
                ["hello\n\n", "<｜"],
                ["hello\n\n", "<｜DSML"],
                ["hello\n\n", "</"],
                ["hello\n\n", "</｜DSML｜"],
            ):
                text = "".join(chunks)
                with self.subTest(cls=cls.__name__, chunks=chunks):
                    one_shot = cls().detect_and_parse(text, self.tools)
                    self.assertEqual((one_shot.normal_text, one_shot.calls), (text, []))
                    self.assertEqual(self._feed_all(chunks, cls), (text, []))

    def test_stray_dsml_fragment_after_invoke_does_not_leak(self):
        """Stray DSML opening or closing fragments after an invoke must remain
        suppressed in streaming mode just as in detect_and_parse."""
        for cls in self._detector_classes():
            d = cls()
            p = f"{d.dsml_token}{cls.parameter_tag_name}"
            inv = (
                f'{d.invoke_start_token} name="get_weather">'
                f'<{p} name="city" string="true">SF</{p}>'
                f"{d.invoke_end_token}"
            )
            for stray in (
                f'\n<{p} name="city" string="true">SF\n',
                f"\n</{p}>\n",
            ):
                chunks = [
                    f"{d.bot_token}\n{inv}",
                    stray,
                    f"{inv}\n</{DSML}function_calls>\n{d.eot_token}",
                ]
                text = "".join(chunks)
                with self.subTest(cls=cls.__name__, stray=stray):
                    one_shot = cls().detect_and_parse(text, self.tools)
                    stream_normal, stream_calls = self._feed_all(chunks, cls)
                    self.assertEqual(stream_normal, "")
                    self.assertEqual(stream_normal, one_shot.normal_text)
                    self.assertEqual(
                        [(c.name, c.parameters) for c in stream_calls],
                        [
                            ("get_weather", '{"city": "SF"}'),
                            ("get_weather", '{"city": "SF"}'),
                        ],
                    )
                    self.assertEqual(
                        [(c.name, c.parameters) for c in stream_calls],
                        [(c.name, c.parameters) for c in one_shot.calls],
                    )


if __name__ == "__main__":
    unittest.main()
