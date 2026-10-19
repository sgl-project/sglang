"""Unit tests for DeepSeekV4Detector DSML streaming — no server, no model loading."""

import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.deepseekv41_detector import DeepSeekV41Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")

DSML = "｜DSML｜"


def _wrapped(invoke: str) -> str:
    return f"<{DSML}tool_calls>\n{invoke}\n</{DSML}tool_calls>"


def _invoke(name: str, params: str = "") -> str:
    return f'<{DSML}invoke name="{name}">\n{params}\n</{DSML}invoke>'


def _param(name: str, is_string: str, value: str) -> str:
    return (
        f'<{DSML}parameter name="{name}" string="{is_string}">{value}</{DSML}parameter>'
    )


def _weather_call(city: str = "SF") -> str:
    return _wrapped(_invoke("get_weather", _param("city", "true", city)))


class TestDeepSeekV4Streaming(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            )
        ]

    def _feed(self, chunks):
        """Returns (normal_text, calls) accumulated over the chunks."""
        detector = DeepSeekV4Detector()
        normal, calls = "", []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        return normal, calls

    def test_preamble_in_same_delta_as_tool_call(self):
        """Prose sharing a delta with the tool call must not be dropped, and the
        streaming and one-shot paths must agree on it."""
        text = "Let me check.\n" + _weather_call()
        normal, calls = self._feed([text])

        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
        self.assertEqual(
            normal, DeepSeekV4Detector().detect_and_parse(text, self.tools).normal_text
        )

    def test_preamble_before_bare_invoke_without_wrapper(self):
        """The bare `<｜DSML｜invoke …>` form has no tool_calls wrapper to walk
        back to, so the preamble is computed from the invoke itself."""
        text = "Checking.\n" + _invoke("get_weather", _param("city", "true", "SF"))
        normal, calls = self._feed([text])

        self.assertIn("Checking.", normal)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])

    def test_no_dsml_markers_leak_into_normal_text(self):
        text = "Prose.\n" + _weather_call()
        normal, _ = self._feed([text[i : i + 4] for i in range(0, len(text), 4)])

        self.assertNotIn(DSML, normal)

    def test_streaming_waits_for_complete_invoke(self):
        detector = DeepSeekV4Detector()
        partial = detector.parse_streaming_increment(
            f'<{DSML}tool_calls>\n<{DSML}invoke name="get_weather">\n'
            f'<{DSML}parameter name="city" string="false">{{"a"',
            self.tools,
        )

        self.assertEqual(partial.calls, [])

        complete = detector.parse_streaming_increment(
            f"</{DSML}parameter></{DSML}invoke></{DSML}tool_calls>",
            self.tools,
        )
        self.assertEqual(len(complete.calls), 1)
        self.assertEqual(complete.calls[0].name, "get_weather")
        self.assertEqual(complete.calls[0].parameters, '{"city": "{\\"a\\""}')
        self.assertEqual(
            detector.prev_tool_call_arr,
            [{"name": "get_weather", "arguments": complete.calls[0].parameters}],
        )
        self.assertEqual(
            detector.streamed_args_for_tool, [complete.calls[0].parameters]
        )

    def test_streaming_matches_complete_parse_at_every_chunk_width(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="create_tasks",
                    description="Create tasks",
                    parameters={
                        "type": "object",
                        "properties": {"tasks": {"type": "array"}},
                    },
                ),
            )
        ]
        values = (
            '[{"description":"first","priority":"medium"}]',
            '[{"description":"unterminated,"priority":"medium"}]',
            "[{]",
            "[}",
        )

        for value in values:
            text = _wrapped(_invoke("create_tasks", _param("tasks", "false", value)))
            expected = DeepSeekV4Detector().detect_and_parse(text, tools).calls[0]
            for width in range(1, len(text) + 1):
                with self.subTest(value=value, width=width):
                    chunks = [
                        text[index : index + width]
                        for index in range(0, len(text), width)
                    ]
                    normal, calls = self._feed_with_tools(chunks, tools)

                    self.assertEqual(normal.strip(), "")
                    self.assertNotIn(DSML, normal)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0].name, "create_tasks")
                    self.assertEqual(calls[0].parameters, expected.parameters)

    def test_complete_invoke_emits_name_and_arguments_together(self):
        text = _weather_call()

        _, calls = self._feed([text[:-1], text[-1:]])

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[0].parameters, '{"city": "SF"}')

    def test_multiple_invokes_emit_one_complete_item_each(self):
        text = _wrapped(
            _invoke("get_weather", _param("city", "true", "SF"))
            + "\n"
            + _invoke("get_weather", _param("city", "true", "NY"))
        )

        _, calls = self._feed([text])

        self.assertEqual(
            [(call.tool_index, call.name, call.parameters) for call in calls],
            [
                (0, "get_weather", '{"city": "SF"}'),
                (1, "get_weather", '{"city": "NY"}'),
            ],
        )

    def test_closed_invoke_rejects_incomplete_parameter(self):
        text = _wrapped(
            _invoke(
                "get_weather",
                f'<{DSML}parameter name="city" string="true">SF',
            )
        )

        normal, calls = self._feed([text])

        self.assertEqual((normal, calls), ("", []))

    def test_stray_prose_inside_invoke_is_ignored(self):
        """Prose around the parameter tags cannot hide a lost argument, so the
        call is still emitted, matching the one-shot parser and the previous
        streaming behavior."""
        text = _wrapped(
            _invoke(
                "get_weather",
                "thinking aloud\n" + _param("city", "true", "SF") + "\nnote to self",
            )
        )

        for width in (4, len(text)):
            with self.subTest(width=width):
                normal, calls = self._feed(
                    [text[i : i + width] for i in range(0, len(text), width)]
                )
                self.assertEqual(normal, "")
                self.assertEqual(
                    [(call.name, call.parameters) for call in calls],
                    [("get_weather", '{"city": "SF"}')],
                )

        result = DeepSeekV4Detector().detect_and_parse(text, self.tools)
        self.assertEqual(
            [(call.name, call.parameters) for call in result.calls],
            [("get_weather", '{"city": "SF"}')],
        )

    def test_closed_invoke_rejects_unmatched_parameter_tag(self):
        """Leftover text containing a DSML fragment is a parameter that failed
        to match; the call is dropped rather than emitted without it."""
        bodies = {
            "missing string attr": f'<{DSML}parameter name="city">SF</{DSML}parameter>',
            "self-closing": f'<{DSML}parameter name="city" string="true"/>',
            "unclosed after valid": _param("city", "true", "SF")
            + f'<{DSML}parameter name="units" string="true">C',
        }
        for label, body in bodies.items():
            with self.subTest(label=label):
                normal, calls = self._feed([_wrapped(_invoke("get_weather", body))])
                self.assertEqual((normal, calls), ("", []))

    def test_closed_invoke_rejects_body_without_parameters(self):
        text = _wrapped(_invoke("get_weather", "just prose, no parameters"))

        normal, calls = self._feed([text])

        self.assertEqual((normal, calls), ("", []))

    def test_closed_invoke_rejects_malformed_direct_json(self):
        text = _wrapped(_invoke("get_weather", '{"city": }'))

        normal, calls = self._feed([text])

        self.assertEqual((normal, calls), ("", []))

    def test_closed_invoke_rejects_nonstandard_json_constant(self):
        text = _wrapped(_invoke("get_weather", '{"city": NaN}'))

        normal, calls = self._feed([text])

        self.assertEqual((normal, calls), ("", []))

    def test_nonstandard_xml_constant_falls_back_to_string(self):
        text = _wrapped(_invoke("get_weather", _param("city", "false", "NaN")))

        _, calls = self._feed([text])

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].parameters, '{"city": "NaN"}')

    def test_malformed_first_invoke_does_not_create_flush_state(self):
        detector = DeepSeekV4Detector()

        result = detector.parse_streaming_increment(
            _wrapped(_invoke("get_weather", '{"city": }')), self.tools
        )

        self.assertEqual(result.calls, [])
        self.assertEqual(detector.prev_tool_call_arr, [])
        self.assertEqual(detector.streamed_args_for_tool, [])

    def test_malformed_later_invoke_preserves_completed_call(self):
        text = _wrapped(
            _invoke("get_weather", _param("city", "true", "SF"))
            + "\n"
            + _invoke("get_weather", '{"city": }')
        )

        normal, calls = self._feed([text])

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[0].parameters, '{"city": "SF"}')
        self.assertEqual(normal, "")

    def test_malformed_first_invoke_keeps_prose_and_later_call(self):
        """A malformed invoke is discarded on its own: the prose ahead of the
        block survives, the next well-formed invoke still becomes call 0, and
        no DSML reaches the client."""
        text = "Checking.\n\n" + _wrapped(
            _invoke("get_weather", f'<{DSML}parameter name="city">SF')
            + "\n"
            + _invoke("get_weather", _param("city", "true", "NY"))
        )

        normal, calls = self._feed([text])

        self.assertEqual(normal, "Checking.")
        self.assertEqual(
            [(call.tool_index, call.name, call.parameters) for call in calls],
            [(0, "get_weather", '{"city": "NY"}')],
        )

    def test_malformed_invoke_fails_closed_at_every_chunk_width(self):
        """Whatever the delta boundaries, a malformed invoke yields the prose,
        no call, and no DSML, including the trailing tool_calls close."""
        text = "Checking.\n\n" + _wrapped(
            _invoke("get_weather", f'<{DSML}parameter name="city" string="true">SF')
        )

        for width in range(1, len(text) + 1):
            with self.subTest(width=width):
                parser = FunctionCallParser(self.tools, "deepseekv4")
                normal, calls = "", []
                for index in range(0, len(text), width):
                    chunk_normal, chunk_calls = parser.parse_stream_chunk(
                        text[index : index + width]
                    )
                    normal += chunk_normal
                    calls.extend(chunk_calls)
                end_normal, end_calls = parser.parse_stream_end()
                normal += end_normal
                calls.extend(end_calls)

                self.assertEqual(normal, "Checking.")
                self.assertEqual(calls, [])

    def test_non_streaming_drops_only_the_malformed_invoke(self):
        text = "Checking.\n\n" + _wrapped(
            _invoke("get_weather", '{"city": }')
            + "\n"
            + _invoke("get_weather", _param("city", "true", "NY"))
        )

        result = DeepSeekV4Detector().detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "Checking.")
        self.assertEqual(
            [(call.name, call.parameters) for call in result.calls],
            [("get_weather", '{"city": "NY"}')],
        )

    def test_non_streaming_drops_unclosed_invoke_in_closed_wrapper(self):
        text = (
            f"<{DSML}tool_calls>\n"
            f'<{DSML}invoke name="get_weather">\n{{"city": "SF"}}\n'
            f"</{DSML}tool_calls>"
        )

        result = DeepSeekV4Detector().detect_and_parse(text, self.tools)

        self.assertEqual(result.calls, [])
        self.assertNotIn(DSML, result.normal_text)

    def test_non_streaming_parses_bare_invoke_without_wrapper(self):
        result = DeepSeekV4Detector().detect_and_parse(
            "Checking.\n\n" + _invoke("get_weather", '{"city": "SF"}'), self.tools
        )

        self.assertEqual(result.normal_text, "Checking.")
        self.assertEqual(
            [(call.name, call.parameters) for call in result.calls],
            [("get_weather", '{"city": "SF"}')],
        )

    def test_non_streaming_malformed_bare_invoke_fails_closed(self):
        result = DeepSeekV4Detector().detect_and_parse(
            _invoke("get_weather", '{"city": }'), self.tools
        )

        self.assertEqual(result.calls, [])
        self.assertNotIn(DSML, result.normal_text)

    def test_stream_end_drops_incomplete_invoke(self):
        parser = FunctionCallParser(self.tools, "deepseekv4")
        normal, calls = parser.parse_stream_chunk(
            "Checking.\n\n"
            f'<{DSML}tool_calls><{DSML}invoke name="get_weather">'
            f'<{DSML}parameter name="city" string="true">SF'
        )

        self.assertEqual((normal, calls), ("", []))
        self.assertEqual(parser.parse_stream_end(), ("Checking.", []))

    def _feed_with_tools(self, chunks, tools):
        detector = DeepSeekV4Detector()
        normal, calls = "", []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, tools)
            normal += result.normal_text
            calls.extend(result.calls)
        return normal, calls

    def test_non_streaming_parses_every_tool_calls_section(self):
        """A turn with two tool_calls sections must yield both calls."""
        result = DeepSeekV4Detector().detect_and_parse(
            f"{_weather_call('SF')}\n{_weather_call('NY')}", self.tools
        )

        self.assertEqual(len(result.calls), 2)

    def test_unexpected_parse_error_fails_closed(self):
        """An unexpected parse error keeps the prose, drops the DSML and the
        call, and the dropped buffer must not come back on the next delta."""
        detector = DeepSeekV4Detector()

        with patch.object(
            DeepSeekV4Detector,
            "_parse_parameters_from_xml",
            side_effect=RuntimeError("boom"),
        ):
            first = detector.parse_streaming_increment(
                "Checking.\n\n" + _weather_call(), self.tools
            )
            self.assertEqual(detector._buffer, "")
            second = detector.parse_streaming_increment(" tail", self.tools)

        self.assertEqual(first.normal_text, "Checking.")
        self.assertEqual(first.calls, [])
        self.assertEqual(second.normal_text, " tail")

    def _feed_all(self, chunks, detector_cls=DeepSeekV4Detector):
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
        for cls in (DeepSeekV32Detector, DeepSeekV4Detector, DeepSeekV41Detector):
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
        for parser_name, cls in (
            ("deepseekv32", DeepSeekV32Detector),
            ("deepseekv4", DeepSeekV4Detector),
            ("deepseekv41", DeepSeekV41Detector),
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
        for cls in (DeepSeekV32Detector, DeepSeekV4Detector, DeepSeekV41Detector):
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
        for cls in (DeepSeekV32Detector, DeepSeekV4Detector, DeepSeekV41Detector):
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
        for cls in (DeepSeekV32Detector, DeepSeekV4Detector, DeepSeekV41Detector):
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
        for cls in (DeepSeekV32Detector, DeepSeekV4Detector, DeepSeekV41Detector):
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
