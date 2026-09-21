"""Unit tests for DeepSeekV4Detector DSML streaming — no server, no model loading."""

import json
import os
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

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


class TestDeepSeekV4Streaming(CustomTestCase):
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

    def test_strict_reasoning_boundary_preserves_tool_argument_literals(self):
        value = "Example:\n<think>keep this</think>"
        text = "Inspect.</think>Now call.\n</think>\n" + _weather_call(value)
        for size in [1, 2, 7, 23, len(text)]:
            with self.subTest(size=size), patch.dict(
                os.environ, {"SGLANG_DSV4_STRICT_TOOL_OUTPUT": "1"}
            ):
                reasoning = ReasoningParser(
                    "deepseek-v4",
                    force_reasoning=True,
                    tool_call_parser_active=True,
                )
                detector = DeepSeekV4Detector()
                thought, normal, arguments, names = "", "", "", []
                for start in range(0, len(text), size):
                    delta_thought, delta = reasoning.parse_stream_chunk(
                        text[start : start + size]
                    )
                    thought += delta_thought
                    result = detector.parse_streaming_increment(delta, self.tools)
                    normal += result.normal_text
                    for call in result.calls:
                        if call.name:
                            names.append(call.name)
                        arguments += call.parameters
                self.assertEqual(reasoning.parse_stream_end(), ("", ""))
                normal += detector.finish(self.tools).normal_text
                self.assertEqual(thought, "Inspect.")
                self.assertEqual(normal.strip(), "Now call.")
                self.assertEqual(names, ["get_weather"])
                self.assertEqual(json.loads(arguments), {"city": value})

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

    def test_malformed_partial_json_falls_back_to_raw_value(self):
        """A partial non-string parameter must not escape as MalformedJSON."""
        detector = DeepSeekV4Detector()
        result = detector.parse_streaming_increment(
            f'<{DSML}tool_calls>\n<{DSML}invoke name="get_weather">\n'
            f'<{DSML}parameter name="city" string="false">{{"a"',
            self.tools,
        )

        self.assertEqual([c.name for c in result.calls if c.name], ["get_weather"])

    def test_non_streaming_parses_every_tool_calls_section(self):
        """A turn with two tool_calls sections must yield both calls."""
        result = DeepSeekV4Detector().detect_and_parse(
            f"{_weather_call('SF')}\n{_weather_call('NY')}", self.tools
        )

        self.assertEqual(len(result.calls), 2)

    def test_non_string_parameter_is_not_rewritten_after_streaming(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="set_config",
                    parameters={
                        "type": "object",
                        "properties": {"value": {}, "label": {"type": "string"}},
                    },
                ),
            )
        ]
        for value in [
            "12345e-3",
            '{"ratio":12345e-3,"enabled":true}',
            '[{"step":"write code","ratio":12345e-3},null,false]',
        ]:
            text = _wrapped(
                _invoke(
                    "set_config",
                    _param("value", "false", value)
                    + _param("label", "true", "complete"),
                )
            )
            expected = {"value": json.loads(value), "label": "complete"}
            for detector_class in [DeepSeekV4Detector, DeepSeekV32Detector]:
                source = text
                if detector_class is DeepSeekV32Detector:
                    source = text.replace("tool_calls", "function_calls")
                for chunk_size in [1, 2, 3, 5, 7, 11, 23, len(source)]:
                    with self.subTest(
                        detector=detector_class.__name__,
                        value=value,
                        chunk_size=chunk_size,
                    ):
                        detector = detector_class()
                        arguments = ""
                        for start in range(0, len(source), chunk_size):
                            result = detector.parse_streaming_increment(
                                source[start : start + chunk_size], tools
                            )
                            arguments += "".join(c.parameters for c in result.calls)
                            self.assertTrue(
                                json.dumps(expected, ensure_ascii=False).startswith(
                                    arguments
                                )
                            )
                        self.assertEqual(json.loads(arguments), expected)
                        self.assertEqual(
                            arguments, detector.prev_tool_call_arr[0]["arguments"]
                        )

    def test_incomplete_non_string_parameter_does_not_emit_a_guessed_value(self):
        detector = DeepSeekV4Detector()
        chunks = [
            f'<{DSML}tool_calls><{DSML}invoke name="get_weather">'
            f'<{DSML}parameter name="city" string="false">',
            "123",
            "45",
            "e-",
            "3",
        ]
        calls = []
        for chunk in chunks:
            calls.extend(detector.parse_streaming_increment(chunk, self.tools).calls)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
        self.assertEqual("".join(c.parameters for c in calls), "")
        result = detector.parse_streaming_increment(
            f"</{DSML}parameter></{DSML}invoke></{DSML}tool_calls>", self.tools
        )
        self.assertEqual(
            json.loads("".join(c.parameters for c in result.calls)), {"city": 12.345}
        )

    def test_string_parameter_still_streams_incrementally(self):
        detector = DeepSeekV4Detector()
        start = (
            f'<{DSML}tool_calls><{DSML}invoke name="get_weather">'
            f'<{DSML}parameter name="city" string="true">San Fran'
        )
        first = detector.parse_streaming_increment(start, self.tools)
        second = detector.parse_streaming_increment("cisco", self.tools)
        prefix = "".join(c.parameters for c in first.calls + second.calls)
        self.assertIn("San Fran", prefix)
        final = detector.parse_streaming_increment(
            f"</{DSML}parameter></{DSML}invoke></{DSML}tool_calls>", self.tools
        )
        self.assertEqual(
            json.loads(prefix + "".join(c.parameters for c in final.calls)),
            {"city": "San Francisco"},
        )

    def test_truncated_non_string_parameter_is_an_explicit_error(self):
        detector = DeepSeekV4Detector()
        detector.parse_streaming_increment(
            f'<{DSML}tool_calls><{DSML}invoke name="get_weather">'
            f'<{DSML}parameter name="city" string="false">[1,2',
            self.tools,
        )
        with self.assertRaisesRegex(ValueError, "Incomplete DSML non-string parameter"):
            detector.finish(self.tools)

    def test_parse_error_neither_swallows_nor_duplicates(self):
        """An unexpected parse error must not empty the turn, and the dropped
        buffer must not come back on the next delta."""
        detector = DeepSeekV4Detector()

        with patch.object(
            DeepSeekV4Detector,
            "_parse_parameters_from_xml",
            side_effect=RuntimeError("boom"),
        ):
            first = detector.parse_streaming_increment(_weather_call(), self.tools)
            self.assertEqual(detector._buffer, "")
            second = detector.parse_streaming_increment(" tail", self.tools)

        self.assertIn("get_weather", first.normal_text)
        self.assertNotIn("get_weather", second.normal_text)
        # No half-formed call: the failure can land between a tool's name and its
        # arguments, so an argument-less named call must not reach the client.
        self.assertEqual(first.calls, [])

    def test_damaged_wrappers_do_not_become_assistant_preamble(self):
        prefixes = [
            f"\n\n<{DSML}toolcalls>\n\n",
            f"\n\n<{DSML}tool_calls\n",
            f"\n\n<{DSML}tool_calls<tool_calls>\n\n",
            f"Checking files.\n\n<{DSML} SugarPanel>\n",
            f"\n\n<{DSML}tool_calls...\n\n",
            f"Final scan:\n\n<{DSML}tool_calls-ok Let's scan the outputs.\n",
        ]
        for prefix in prefixes:
            expected = prefix.split(f"<{DSML}", 1)[0].strip()
            for call in [
                _weather_call(),
                _invoke("get_weather", _param("city", "true", "SF")),
            ]:
                text = prefix + call
                for width in [1, 2, 4, 7, 23, len(text)]:
                    with self.subTest(prefix=prefix, width=width):
                        detector = DeepSeekV4Detector()
                        normal, calls = "", []
                        for start in range(0, len(text), width):
                            result = detector.parse_streaming_increment(
                                text[start : start + width], self.tools
                            )
                            normal += result.normal_text
                            calls.extend(result.calls)
                        tail = detector.finish(self.tools)
                        normal += tail.normal_text
                        calls.extend(tail.calls)
                        self.assertEqual(normal.strip(), expected)
                        self.assertEqual(
                            [c.name for c in calls if c.name], ["get_weather"]
                        )
                        self.assertEqual(
                            json.loads("".join(c.parameters for c in calls)),
                            {"city": "SF"},
                        )
                parsed = DeepSeekV4Detector().detect_and_parse(text, self.tools)
                self.assertEqual(parsed.normal_text.strip(), expected)
                self.assertEqual(len(parsed.calls), 1)
                self.assertEqual(json.loads(parsed.calls[0].parameters), {"city": "SF"})

    def test_quoted_dsml_is_not_executed_or_removed(self):
        for quote in [
            f"Example:\n```xml\n{_weather_call('literal')}\n```\n",
            f"Example:\n~~~xml\n{_weather_call('literal')}\n~~~\n",
            f"Example: `{_weather_call('literal')}`\n",
            f"Example: '<{DSML}toolcalls>'\n",
            f"Example: “<{DSML}toolcalls>”\n",
        ]:
            text = quote + _weather_call()
            for width in [1, 3, 11, len(text)]:
                with self.subTest(quote=quote, width=width):
                    detector = DeepSeekV4Detector()
                    normal, calls = "", []
                    for start in range(0, len(text), width):
                        result = detector.parse_streaming_increment(
                            text[start : start + width], self.tools
                        )
                        normal += result.normal_text
                        calls.extend(result.calls)
                    tail = detector.finish(self.tools)
                    normal += tail.normal_text
                    calls.extend(tail.calls)
                    self.assertEqual(normal.strip(), quote.strip())
                    self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
                    self.assertEqual(
                        json.loads("".join(c.parameters for c in calls)), {"city": "SF"}
                    )
            quoted_only = DeepSeekV4Detector().detect_and_parse(quote, self.tools)
            self.assertEqual(quoted_only.normal_text, quote)
            self.assertEqual(quoted_only.calls, [])

    def test_strict_mode_rejects_orphan_markup_before_content_is_emitted(self):
        for text in [
            "Now let me update the result.\n\n</parameter>\n",
            '\n\n</parameter>\n</invoke>\n<invoke name="exec_command">\n',
            '</parameter></invoke><invoke name="exec_command">\n',
        ]:
            with self.subTest(text=text), patch.dict(
                os.environ, {"SGLANG_DSV4_STRICT_TOOL_OUTPUT": "1"}
            ):
                detector = DeepSeekV4Detector()
                normal = ""
                for start in range(0, len(text), 3):
                    result = detector.parse_streaming_increment(
                        text[start : start + 3], self.tools
                    )
                    normal += result.normal_text
                self.assertEqual(normal, "")
                with self.assertRaisesRegex(ValueError, "Orphan"):
                    detector.finish(self.tools)
                with self.assertRaisesRegex(ValueError, "Orphan"):
                    DeepSeekV4Detector().detect_and_parse(text, self.tools)

    def test_strict_mode_preserves_literal_markup_and_balanced_xml(self):
        for text in [
            "Literal `</parameter>`.",
            "Example:\n```xml\n</parameter>\n</invoke>\n```",
            'XML:\n<parameter description="a > b">\nvalue\n</parameter>',
            "</parameter>",
            "Text with an unrelated SGML fragment: <![unexpected[.",
        ]:
            with self.subTest(text=text), patch.dict(
                os.environ, {"SGLANG_DSV4_STRICT_TOOL_OUTPUT": "1"}
            ):
                detector = DeepSeekV4Detector()
                for part in text:
                    result = detector.parse_streaming_increment(part, self.tools)
                    self.assertEqual(result.normal_text, "")
                self.assertEqual(detector.finish(self.tools).normal_text, text)
                self.assertEqual(
                    DeepSeekV4Detector().detect_and_parse(text, self.tools).normal_text,
                    text,
                )

    def test_strict_mode_never_falls_back_to_raw_dsml_on_parse_failure(self):
        with patch.dict(os.environ, {"SGLANG_DSV4_STRICT_TOOL_OUTPUT": "1"}):
            detector = DeepSeekV4Detector()
            with patch.object(
                detector, "_parse_parameters_from_xml", side_effect=RuntimeError("boom")
            ):
                with self.assertRaisesRegex(ValueError, "Failed to parse"):
                    detector.parse_streaming_increment(_weather_call(), self.tools)

    def test_strict_mode_rejects_invalid_non_string_json_instead_of_coercion(self):
        text = _wrapped(_invoke("get_weather", _param("city", "false", "[1,broken]")))
        with patch.dict(os.environ, {"SGLANG_DSV4_STRICT_TOOL_OUTPUT": "1"}):
            with self.assertRaisesRegex(ValueError, "Invalid JSON"):
                DeepSeekV4Detector().detect_and_parse(text, self.tools)

    def test_strict_mode_keeps_tool_argument_streaming(self):
        with patch.dict(os.environ, {"SGLANG_DSV4_STRICT_TOOL_OUTPUT": "1"}):
            detector = DeepSeekV4Detector()
            start = (
                f'<{DSML}tool_calls><{DSML}invoke name="get_weather">'
                f'<{DSML}parameter name="city" string="true">San Fran'
            )
            calls = detector.parse_streaming_increment(start, self.tools).calls
            calls += detector.parse_streaming_increment("cisco", self.tools).calls
            self.assertIn("San Fran", "".join(c.parameters for c in calls))
            tail = detector.parse_streaming_increment(
                f"</{DSML}parameter></{DSML}invoke></{DSML}tool_calls>", self.tools
            )
            calls += tail.calls + detector.finish(self.tools).calls
            self.assertEqual(
                json.loads("".join(c.parameters for c in calls)),
                {"city": "San Francisco"},
            )

    def test_incomplete_dsml_invocation_is_an_explicit_error(self):
        detector = DeepSeekV4Detector()
        detector.parse_streaming_increment(
            f'<{DSML}tool_calls><{DSML}invoke name="get_weather">',
            self.tools,
        )
        with self.assertRaisesRegex(ValueError, "Incomplete DSML invoke"):
            detector.finish(self.tools)

    def test_indented_native_calls_and_parameter_literals_are_preserved(self):
        literal = f"Echo `<{DSML}tool_calls>` without interpreting it"
        first = _weather_call(literal)
        second = _weather_call("NY")
        text = "Checking.\n" + "\n".join(
            "        " + line for line in (first + "\n" + second).splitlines()
        )
        for width in [1, 3, 13, len(text)]:
            with self.subTest(width=width):
                parser = FunctionCallParser(self.tools, "deepseekv4")
                names, arguments = [], {}
                for start in range(0, len(text), width):
                    _, calls = parser.parse_stream_chunk(text[start : start + width])
                    for call in calls:
                        if call.name:
                            names.append(call.name)
                        arguments[call.tool_index] = (
                            arguments.get(call.tool_index, "") + call.parameters
                        )
                parser.parse_stream_end()
                self.assertEqual(names, ["get_weather", "get_weather"])
                self.assertEqual(
                    [json.loads(value) for value in arguments.values()],
                    [{"city": literal}, {"city": "NY"}],
                )
        _, calls = FunctionCallParser(self.tools, "deepseekv4").parse_non_stream(text)
        self.assertEqual(
            [json.loads(call.parameters) for call in calls],
            [{"city": literal}, {"city": "NY"}],
        )

    def test_public_parser_flush_rejects_orphan_output_in_strict_mode(self):
        with patch.dict(os.environ, {"SGLANG_DSV4_STRICT_TOOL_OUTPUT": "1"}):
            parser = FunctionCallParser(self.tools, "deepseekv4")
            for chunk in ["Working.\n", "</para", "meter>\n"]:
                self.assertEqual(parser.parse_stream_chunk(chunk), ("", []))
            with self.assertRaisesRegex(ValueError, "Orphan"):
                parser.parse_stream_end()

    def test_strict_string_parameter_rejects_malformed_closer_before_dispatch(self):
        bad = f"</{DSML}parameter |"
        command = (
            "cat <<'JSONEOF'\n{}\nJSONEOF\n"
            f'echo "created"{bad}\npython3 -c "print(1)"'
        )
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="exec_command",
                    parameters={
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                ),
            )
        ]
        source = _wrapped(_invoke("exec_command", _param("cmd", "true", command)))
        for width in [1, 2, 7, 31, len(source)]:
            with self.subTest(width=width):
                detector = DeepSeekV4Detector(strict_output=True)
                arguments = ""
                with self.assertRaisesRegex(
                    ValueError, "Malformed DSML parameter terminator"
                ):
                    for start in range(0, len(source), width):
                        result = detector.parse_streaming_increment(
                            source[start : start + width], tools
                        )
                        arguments += "".join(item.parameters for item in result.calls)
                    detector.finish(tools)
                self.assertNotIn(bad, arguments)
                with self.assertRaises(json.JSONDecodeError):
                    json.loads(arguments)
        with self.assertRaisesRegex(ValueError, "Malformed DSML parameter terminator"):
            DeepSeekV4Detector(strict_output=True).detect_and_parse(source, tools)
        direct_json = _wrapped(_invoke("exec_command", json.dumps({"cmd": command})))
        with self.assertRaisesRegex(ValueError, "Malformed DSML parameter terminator"):
            DeepSeekV4Detector(strict_output=True).detect_and_parse(direct_json, tools)

    def test_string_parameter_preserves_quoted_heredoc_and_escaped_marker_literals(
        self,
    ):
        marker = f"</{DSML}parameter |"
        values = [
            f'print("{marker}")',
            f"Example: `{marker}`",
            f"```xml\n{marker}\n```",
            f"cat <<'EOF'\n{marker}\nEOF",
            f"cat <<EOF\n{marker}\nEOF",
            f"cat <<-EOF\n\t{marker}\n\tEOF",
            f"cat <<'A' <<'B'\n{marker}\nA\n{marker}\nB",
            f"# Literal example: {marker}\nprint(1)",
            f"echo \\{marker}",
        ]
        for value in values:
            source = _wrapped(_invoke("get_weather", _param("city", "true", value)))
            for width in [1, 3, 13, len(source)]:
                with self.subTest(value=value, width=width):
                    detector = DeepSeekV4Detector(strict_output=True)
                    arguments = ""
                    for start in range(0, len(source), width):
                        result = detector.parse_streaming_increment(
                            source[start : start + width], self.tools
                        )
                        arguments += "".join(item.parameters for item in result.calls)
                    detector.finish(self.tools)
                    self.assertEqual(json.loads(arguments), {"city": value})
            parsed = DeepSeekV4Detector(strict_output=True).detect_and_parse(
                source, self.tools
            )
            self.assertEqual(json.loads(parsed.calls[0].parameters), {"city": value})

    def test_new_parameter_guard_does_not_change_non_strict_behavior(self):
        value = f'echo "created"</{DSML}parameter |\nprintf done'
        source = _wrapped(_invoke("get_weather", _param("city", "true", value)))
        parsed = DeepSeekV4Detector(strict_output=False).detect_and_parse(
            source, self.tools
        )
        self.assertEqual(json.loads(parsed.calls[0].parameters), {"city": value})

    def test_strict_direct_json_waits_for_complete_validated_invoke(self):
        detector = DeepSeekV4Detector(strict_output=True)
        opening = f'<{DSML}tool_calls><{DSML}invoke name="get_weather">'
        calls = detector.parse_streaming_increment(
            opening + '{"city":', self.tools
        ).calls
        calls += detector.parse_streaming_increment('"SF"}', self.tools).calls
        self.assertEqual([call.name for call in calls if call.name], ["get_weather"])
        self.assertEqual("".join(call.parameters for call in calls), "")
        calls += detector.parse_streaming_increment(
            f"</{DSML}invoke></{DSML}tool_calls>", self.tools
        ).calls
        calls += detector.finish(self.tools).calls
        self.assertEqual(
            json.loads("".join(call.parameters for call in calls)), {"city": "SF"}
        )

    def test_strict_rejects_nested_invoke_instead_of_rewriting_streamed_arguments(self):
        source = _wrapped(
            f'<{DSML}invoke name="get_weather">'
            + _param("city", "true", "San Francisco")
            + "\n</"
            + _invoke("get_weather", _param("city", "true", "NY"))
        )
        for width in [1, 2, 7, 31, len(source)]:
            with self.subTest(width=width):
                detector = DeepSeekV4Detector(strict_output=True)
                with self.assertRaisesRegex(ValueError, "Malformed DSML parameter"):
                    for start in range(0, len(source), width):
                        detector.parse_streaming_increment(
                            source[start : start + width], self.tools
                        )
                    detector.finish(self.tools)
        with self.assertRaisesRegex(ValueError, "Malformed DSML parameter"):
            DeepSeekV4Detector(strict_output=True).detect_and_parse(source, self.tools)

    def test_strict_rejects_unclosed_string_parameter_at_invoke_end(self):
        source = _wrapped(
            f'<{DSML}invoke name="get_weather">'
            f'<{DSML}parameter name="city" string="true">SF</{DSML}invoke>'
        )
        for width in [1, 2, 7, 31, len(source)]:
            with self.subTest(width=width):
                detector = DeepSeekV4Detector(strict_output=True)
                with self.assertRaisesRegex(ValueError, "Incomplete DSML parameter"):
                    for start in range(0, len(source), width):
                        detector.parse_streaming_increment(
                            source[start : start + width], self.tools
                        )
                    detector.finish(self.tools)
        with self.assertRaisesRegex(ValueError, "Incomplete DSML parameter"):
            DeepSeekV4Detector(strict_output=True).detect_and_parse(source, self.tools)

    def test_strict_incomplete_later_call_does_not_rewrite_previous_calls(self):
        detector = DeepSeekV4Detector(strict_output=True)
        calls = detector.parse_streaming_increment(
            _weather_call("SF"), self.tools
        ).calls
        calls += detector.parse_streaming_increment(
            _weather_call("NY"), self.tools
        ).calls
        self.assertEqual(
            [json.loads(call.parameters) for call in calls if call.parameters],
            [{"city": "SF"}, {"city": "NY"}],
        )
        with self.assertRaisesRegex(ValueError, "Incomplete DSML parameter"):
            detector.parse_streaming_increment(
                _wrapped(
                    f'<{DSML}invoke name="get_weather">'
                    f'<{DSML}parameter name="city" string="true">LA</{DSML}invoke>'
                ),
                self.tools,
            )

    def test_strict_rejects_duplicate_xml_parameters_and_invalid_string_flags(self):
        for body, message in [
            (_param("city", "true", "SF") + _param("city", "true", "NY"), "Duplicate"),
            (_param("city", "maybe", '"SF"'), "Invalid DSML string flag"),
        ]:
            source = _wrapped(_invoke("get_weather", body))
            for width in [1, 5, len(source)]:
                with self.subTest(body=body, width=width):
                    detector = DeepSeekV4Detector(strict_output=True)
                    with self.assertRaisesRegex(ValueError, message):
                        for start in range(0, len(source), width):
                            detector.parse_streaming_increment(
                                source[start : start + width], self.tools
                            )
                        detector.finish(self.tools)

    def test_strict_argument_stream_cannot_change_emitted_prefix(self):
        detector = DeepSeekV4Detector(strict_output=True)
        detector.parse_streaming_increment(
            f'<{DSML}invoke name="get_weather">'
            f'<{DSML}parameter name="city" string="true">San Fran',
            self.tools,
        )
        detector.parse_streaming_increment("cisco", self.tools)
        self.assertIn("San Fran", detector.streamed_args_for_tool[0])
        with patch.object(
            detector, "_parse_parameters_from_xml", return_value='{"city": "NY"}'
        ):
            with self.assertRaisesRegex(ValueError, "Non-monotonic DSML"):
                detector.parse_streaming_increment(
                    f"</{DSML}parameter></{DSML}invoke>", self.tools
                )

    def test_strict_completed_arguments_must_be_a_json_object(self):
        for arguments in ['{"city":', "[]", '{"city": NaN}']:
            with self.subTest(arguments=arguments):
                detector = DeepSeekV4Detector(strict_output=True)
                with patch.object(
                    detector, "_parse_parameters_from_xml", return_value=arguments
                ):
                    with self.assertRaisesRegex(ValueError, "JSON|must be an object"):
                        detector.parse_streaming_increment(_weather_call(), self.tools)

    def test_strict_direct_json_rejects_duplicates_and_non_json_constants(self):
        for arguments in [
            '{"city": "SF", "city": "NY"}',
            '{"city": NaN}',
            '{"city": Infinity}',
            '{"city": {"value": 1, "value": 2}}',
        ]:
            with self.subTest(arguments=arguments):
                source = _wrapped(_invoke("get_weather", arguments))
                with self.assertRaisesRegex(ValueError, "Duplicate|Invalid DSML JSON"):
                    DeepSeekV4Detector(strict_output=True).detect_and_parse(
                        source, self.tools
                    )

    def test_non_strict_duplicate_parameters_retain_legacy_behavior(self):
        source = _wrapped(
            _invoke(
                "get_weather",
                _param("city", "true", "SF") + _param("city", "true", "NY"),
            )
        )
        parsed = DeepSeekV4Detector(strict_output=False).detect_and_parse(
            source, self.tools
        )
        self.assertEqual(json.loads(parsed.calls[0].parameters), {"city": "NY"})

    def test_optional_schema_validation_rejects_missing_unknown_and_wrong_types(self):
        self.tools[0].function.parameters["additionalProperties"] = False
        for arguments, constraint in [
            ({}, "required"),
            ({"city": "SF", "other": 1}, "additionalProperties"),
            ({"city": 7}, "type"),
        ]:
            for body in [
                json.dumps(arguments),
                "".join(
                    _param(name, "false", json.dumps(value))
                    for name, value in arguments.items()
                ),
            ]:
                source = _wrapped(_invoke("get_weather", body))
                for width in [1, 7, len(source)]:
                    with self.subTest(arguments=arguments, width=width, body=body):
                        detector = DeepSeekV4Detector(
                            strict_output=True, validate_tool_schema=True
                        )
                        with self.assertRaisesRegex(ValueError, constraint):
                            for start in range(0, len(source), width):
                                detector.parse_streaming_increment(
                                    source[start : start + width], self.tools
                                )
                            detector.finish(self.tools)
                with self.assertRaisesRegex(ValueError, constraint):
                    DeepSeekV4Detector(
                        strict_output=True, validate_tool_schema=True
                    ).detect_and_parse(source, self.tools)

    def test_optional_schema_validation_keeps_valid_arguments_unchanged(self):
        for source in [
            _weather_call("SF"),
            _wrapped(_invoke("get_weather", '{"city": "SF"}')),
        ]:
            detector = DeepSeekV4Detector(strict_output=True, validate_tool_schema=True)
            arguments = ""
            for character in source:
                result = detector.parse_streaming_increment(character, self.tools)
                arguments += "".join(call.parameters for call in result.calls)
            detector.finish(self.tools)
            self.assertEqual(json.loads(arguments), {"city": "SF"})
            self.assertEqual(len(detector._schema_validators), 1)

    def test_optional_schema_validation_supports_local_references(self):
        self.tools[0].function.parameters = {
            "type": "object",
            "properties": {"city": {"$ref": "#/$defs/city"}},
            "$defs": {"city": {"type": "string", "enum": ["SF"]}},
            "required": ["city"],
        }
        parsed = DeepSeekV4Detector(
            strict_output=True, validate_tool_schema=True
        ).detect_and_parse(_weather_call("SF"), self.tools)
        self.assertEqual(json.loads(parsed.calls[0].parameters), {"city": "SF"})
        with self.assertRaisesRegex(ValueError, "enum"):
            DeepSeekV4Detector(
                strict_output=True, validate_tool_schema=True
            ).detect_and_parse(_weather_call("NY"), self.tools)

    def test_optional_schema_validation_never_fetches_remote_references(self):
        self.tools[0].function.parameters["properties"]["city"] = {
            "$ref": "https://example.invalid/city.json"
        }
        with patch(
            "urllib.request.urlopen", side_effect=AssertionError("network forbidden")
        ):
            with self.assertRaisesRegex(ValueError, "external retrieval is disabled"):
                DeepSeekV4Detector(
                    strict_output=True, validate_tool_schema=True
                ).detect_and_parse(_weather_call(), self.tools)

    def test_optional_schema_validation_reuses_normalization_without_mutation(self):
        schema = self.tools[0].function.parameters
        schema["properties"]["city"]["type"] = "varchar"
        parsed = DeepSeekV4Detector(
            strict_output=True, validate_tool_schema=True
        ).detect_and_parse(_weather_call(), self.tools)
        self.assertEqual(json.loads(parsed.calls[0].parameters), {"city": "SF"})
        self.assertEqual(schema["properties"]["city"]["type"], "varchar")

    def test_optional_schema_validation_preserves_unknown_tool_forwarding_policy(self):
        source = _wrapped(_invoke("unknown_tool", "{}"))
        with patch.dict(os.environ, {"SGLANG_FORWARD_UNKNOWN_TOOLS": "0"}):
            with self.assertRaisesRegex(ValueError, "Undefined DSML tool"):
                DeepSeekV4Detector(
                    strict_output=True, validate_tool_schema=True
                ).detect_and_parse(source, self.tools)
        with patch.dict(os.environ, {"SGLANG_FORWARD_UNKNOWN_TOOLS": "1"}):
            parsed = DeepSeekV4Detector(
                strict_output=True, validate_tool_schema=True
            ).detect_and_parse(source, self.tools)
            self.assertEqual(parsed.calls[0].name, "unknown_tool")

    def test_optional_schema_validation_is_off_by_default_and_requires_strict(self):
        source = _wrapped(_invoke("get_weather", "{}"))
        parsed = DeepSeekV4Detector(
            strict_output=True, validate_tool_schema=False
        ).detect_and_parse(source, self.tools)
        self.assertEqual(json.loads(parsed.calls[0].parameters), {})
        with self.assertRaisesRegex(ValueError, "requires strict"):
            DeepSeekV4Detector(strict_output=False, validate_tool_schema=True)
        with patch.dict(
            os.environ,
            {
                "SGLANG_DSV4_STRICT_TOOL_OUTPUT": "1",
                "SGLANG_DSV4_VALIDATE_TOOL_SCHEMA": "1",
            },
        ):
            with self.assertRaisesRegex(ValueError, "required"):
                FunctionCallParser(self.tools, "deepseekv4").parse_non_stream(source)


if __name__ == "__main__":
    import unittest

    unittest.main()
