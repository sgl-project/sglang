"""Unit tests for DeepSeekV4Detector DSML streaming — no server, no model loading."""

import json
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
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


class TestDeepSeekV4NonStreamingLeak(CustomTestCase):
    """Non-streaming turns must never return DSML markup as content.

    When a generation carries tool markup the detector cannot convert, the
    response comes back with no `tool_calls` and `finish_reason: "stop"`.
    An OpenAI-compatible client reads that as "the model is done", ends the
    turn, and the requested calls are dropped with no error anywhere.
    """

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
            ),
            Tool(
                type="function",
                function=Function(
                    name="record_note",
                    description="Record a note",
                    parameters={
                        "type": "object",
                        "properties": {"targets": {"type": "array"}},
                    },
                ),
            ),
        ]

    def _parse(self, text):
        return DeepSeekV4Detector().detect_and_parse(text, self.tools)

    def test_invoke_without_tool_calls_section(self):
        """A bare invoke block with no section wrapper is still a tool call."""
        result = self._parse(
            "Let me check.\n\n" + _invoke("get_weather", _param("city", "true", "SF"))
        )

        self.assertEqual([c.name for c in result.calls], ["get_weather"])
        self.assertNotIn(DSML, result.normal_text)

    def test_unterminated_tool_calls_section(self):
        """An opened-but-unclosed section must not swallow its calls."""
        result = self._parse(
            f"<{DSML}tool_calls>\n"
            + _invoke("get_weather", _param("city", "true", "SF"))
        )

        self.assertEqual([c.name for c in result.calls], ["get_weather"])
        self.assertNotIn(DSML, result.normal_text)

    def test_malformed_invoke_keeps_sibling_calls(self):
        """One unparsable invoke must not discard the calls next to it."""
        malformed = (
            f'<{DSML}invoke name="record_note">\n{{"targets": ["a",}}\n</{DSML}invoke>'
        )
        result = self._parse(
            _wrapped(
                malformed + "\n" + _invoke("get_weather", _param("city", "true", "SF"))
            )
        )

        self.assertEqual([c.name for c in result.calls], ["get_weather"])
        self.assertNotIn(DSML, result.normal_text)

    def test_unknown_tool_markup_is_not_returned_as_content(self):
        """An unmatched tool name yields no call, but also no raw markup."""
        result = self._parse(
            _wrapped(_invoke("no_such_tool", _param("city", "true", "SF")))
        )

        self.assertEqual(result.calls, [])
        self.assertNotIn(DSML, result.normal_text)

    def test_prose_is_preserved_alongside_tool_call(self):
        """Stripping markup must not eat the model's ordinary prose."""
        result = self._parse("Checking the weather.\n\n" + _weather_call("SF"))

        self.assertEqual([c.name for c in result.calls], ["get_weather"])
        self.assertIn("Checking the weather.", result.normal_text)

    def test_v32_function_calls_block_is_not_leaked(self):
        """V4 sometimes emits the older `function_calls` block name.

        The V4 detector keys on `tool_calls`, so before the fix the whole
        section came back as content. Anchoring on the invoke marker
        recovers the call regardless of which block name wraps it.
        """
        result = self._parse(
            f"<{DSML}function_calls>\n"
            + _invoke("get_weather", _param("city", "true", "SF"))
            + f"\n</{DSML}function_calls>"
        )

        self.assertEqual([c.name for c in result.calls], ["get_weather"])
        self.assertNotIn(DSML, result.normal_text)

    def test_generated_markup_round_trips_through_the_encoder(self):
        """Arguments built by the model's own DSML encoder must come back intact.

        Uses `encoding_dsv4.encode_arguments_to_dsml` rather than hand-written
        markup, so the parser is checked against the format the model is
        actually prompted to emit, including the string/JSON type split.
        """
        from sglang.srt.entrypoints.openai.encoding_dsv4 import (
            encode_arguments_to_dsml,
        )

        payloads = [
            {"city": "SF"},
            {"statement": "multi\nline\nvalue", "source": "alert"},
            {"targets": [{"id": "pr_001", "polarity": "supports"}]},
            {"count": 7, "enabled": True, "ratio": 0.5},
            {"text": 'quotes " and <angle> inside'},
            {"unicode": "日本語 café"},
            {"nested": {"a": [1, 2, {"b": None}]}},
        ]
        for arguments in payloads:
            with self.subTest(arguments=arguments):
                body = encode_arguments_to_dsml(
                    {"name": "get_weather", "arguments": arguments}
                )
                result = self._parse(
                    _wrapped(
                        f'<{DSML}invoke name="get_weather">\n{body}\n</{DSML}invoke>'
                    )
                )

                self.assertEqual(len(result.calls), 1)
                self.assertEqual(json.loads(result.calls[0].parameters), arguments)
                self.assertNotIn(DSML, result.normal_text)

    def test_truncated_or_malformed_tag_is_stripped(self):
        """A tag the model mangles must not survive into content.

        Observed from a live server: the model emitted
        `<｜DSML｜tool_calls|` (ASCII pipe, no closing `>`) alongside a
        valid invoke. The call parses, but the broken opener was still
        being returned as content.
        """
        mangled = f"<{DSML}tool_calls|"
        result = self._parse(
            f"{mangled}\n"
            + _invoke("get_weather", _param("city", "true", "SF"))
            + f"\n</{DSML}tool_calls>"
        )

        self.assertEqual([c.name for c in result.calls], ["get_weather"])
        self.assertNotIn(DSML, result.normal_text)

    def test_stripping_does_not_eat_surrounding_prose(self):
        """The strip must remove tags only, never the text around them."""
        result = self._parse(
            f"Before. <{DSML}tool_calls| After.\n"
            + _invoke("get_weather", _param("city", "true", "SF"))
        )

        self.assertNotIn(DSML, result.normal_text)
        self.assertIn("Before.", result.normal_text)
        self.assertIn("After.", result.normal_text)

    def test_bracketless_marker_is_stripped(self):
        """The marker is one special token, so a bare occurrence is markup.

        `｜DSML｜` encodes to a single token id, so the model emitting it
        without surrounding brackets is still leaked markup rather than
        prose the user wrote.
        """
        result = self._parse(
            f"The {DSML} marker.\n"
            + _invoke("get_weather", _param("city", "true", "SF"))
        )

        self.assertEqual([c.name for c in result.calls], ["get_weather"])
        self.assertNotIn(DSML, result.normal_text)
        self.assertIn("marker.", result.normal_text)

    def test_streaming_strips_a_malformed_opener(self):
        """Streaming must not emit a mangled opener as content either.

        Observed against a live server: the model emitted
        `<｜DSML｜tool_calls|` (ASCII pipe, no closing `>`) before a valid
        invoke. `preamble` is built by backing up over a well-formed
        `bot_token`, which does not match the mangled form, so it used to
        reach `delta.content`.
        """
        text = (
            f"<{DSML}tool_calls|\n"
            + _invoke("get_weather", _param("city", "true", "SF"))
            + f"\n</{DSML}tool_calls>"
        )
        for size in (1, 8, len(text)):
            with self.subTest(chunk=size):
                detector = DeepSeekV4Detector()
                normal, names = "", []
                for start in range(0, len(text), size):
                    result = detector.parse_streaming_increment(
                        text[start : start + size], self.tools
                    )
                    normal += result.normal_text
                    names += [c.name for c in result.calls if c.name]

                self.assertNotIn(DSML, normal)
                self.assertEqual(names, ["get_weather"])

        # The non-streaming path agrees.
        self.assertNotIn(DSML, self._parse(text).normal_text)

    def test_streaming_strip_keeps_prose_containing_angle_brackets(self):
        """A mangled marker must consume the tag only, never the sentence."""
        text = (
            f"Before. <{DSML}tool_calls| 5 > 3 After.\n"
            + _invoke("get_weather", _param("city", "true", "SF"))
            + f"\n</{DSML}tool_calls>"
        )
        detector = DeepSeekV4Detector()
        normal = ""
        for start in range(0, len(text), 8):
            normal += detector.parse_streaming_increment(
                text[start : start + 8], self.tools
            ).normal_text

        self.assertNotIn(DSML, normal)
        self.assertIn("Before.", normal)
        self.assertIn("5 > 3 After.", normal)


class TestDeepSeekV4StreamFinish(CustomTestCase):
    """End-of-stream flush for the DSML detector.

    The streaming guard holds back any buffer containing the marker while it
    waits for a closer. Without a `finish()` override that text is discarded,
    so a turn whose prose follows the tool calls comes back empty.
    """

    D = DSML

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
            ),
            Tool(
                type="function",
                function=Function(
                    name="ping",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

    def _invoke(self, closed=True):
        body = (
            f'<{self.D}invoke name="get_weather">\n'
            f'<{self.D}parameter name="city" string="true">SF</{self.D}parameter>\n'
        )
        return body + (f"</{self.D}invoke>" if closed else "")

    def _feed(self, text, size):
        """Stream `text`, then flush. Returns (normal_text, names, arguments)."""
        parser = FunctionCallParser(self.tools, "deepseekv4")
        normal, names, arguments = "", [], ""
        for start in range(0, len(text), size):
            chunk_text, calls = parser.parse_stream_chunk(text[start : start + size])
            normal += chunk_text
            for call in calls:
                if call.name:
                    names.append(call.name)
                if call.parameters:
                    arguments += call.parameters
        tail, calls = parser.parse_stream_end()
        normal += tail
        for call in calls:
            if call.name:
                names.append(call.name)
            if call.parameters:
                arguments += call.parameters
        return normal, names, arguments

    def test_arguments_are_chunk_invariant_when_cut_mid_arguments(self):
        """A body cut mid-arguments must never complete as a zero-arg call.

        Re-parsing a truncated body yields "{}", so completing the call from
        it would dispatch `get_weather({})` the model never asked for — and
        only at the chunk sizes where nothing had been streamed yet.
        """
        text = f'Let me check.\n<{self.D}invoke name="get_weather">\n' '{"city": "S'
        for size in (1, 8, len(text)):
            with self.subTest(chunk=size):
                _, _, arguments = self._feed(text, size)

                # The exact prefix depends on how much streamed before the
                # cut, but no chunking may turn a truncated body into a
                # dispatchable zero-argument call.
                self.assertNotEqual(arguments, "{}")
                if arguments:
                    with self.assertRaises(json.JSONDecodeError):
                        json.loads(arguments)

    def test_zero_argument_invoke_without_closer_still_completes(self):
        """An empty body legitimately means no arguments, so it must finish."""
        text = f'<{self.D}tool_calls>\n<{self.D}invoke name="ping">\n'
        for size in (8, len(text)):
            with self.subTest(chunk=size):
                _, names, arguments = self._feed(text, size)

                self.assertEqual(names, ["ping"])
                self.assertEqual(json.loads(arguments), {})

    def test_partial_marker_tail_is_not_released(self):
        """A generation cut inside the marker leaves residue, not prose."""
        _, _, _ = self._feed("Answer done.", 4)
        normal, _, _ = self._feed(f"Answer done <{self.D[:3]}", 4)

        self.assertNotIn(self.D[:3], normal)
        self.assertIn("Answer done", normal)

    def test_prose_after_the_call_is_released(self):
        """Text following a completed call must not die in the buffer."""
        text = (
            f"<{self.D}tool_calls>\n{self._invoke()}\n</{self.D}tool_calls>\n"
            "Here you go!"
        )
        for size in (1, 8, len(text)):
            with self.subTest(chunk=size):
                normal, names, arguments = self._feed(text, size)

                self.assertIn("Here you go!", normal)
                self.assertNotIn(self.D, normal)
                self.assertEqual(names, ["get_weather"])
                self.assertEqual(json.loads(arguments), {"city": "SF"})

    def test_invoke_without_closer_still_completes_arguments(self):
        """A missing `</invoke>` must not strand truncated arguments."""
        text = (
            f"<{self.D}tool_calls>\n{self._invoke(closed=False)}"
            f"</{self.D}tool_calls>"
        )
        for size in (1, 8, len(text)):
            with self.subTest(chunk=size):
                normal, names, arguments = self._feed(text, size)

                self.assertEqual(names, ["get_weather"])
                self.assertEqual(json.loads(arguments), {"city": "SF"})
                self.assertNotIn(self.D, normal)

    def test_refusal_after_a_mangled_opener_is_released(self):
        """Markup with no parseable call must still yield the model's prose."""
        text = f"<{self.D}tool_calls|\nI cannot do that."
        for size in (1, 8, len(text)):
            with self.subTest(chunk=size):
                normal, names, _ = self._feed(text, size)

                self.assertIn("I cannot do that.", normal)
                self.assertNotIn(self.D, normal)
                self.assertEqual(names, [])

    def test_preamble_is_not_resent_at_finish(self):
        """A chunk boundary inside the preamble must not duplicate it.

        The buffer can still hold a partial copy of prose already streamed,
        so releasing the buffer wholesale would emit it twice.
        """
        text = f'Let me check.\n<{self.D}invoke name="get_weather">\n' '{"city": "S'
        for size in (1, 8, len(text)):
            with self.subTest(chunk=size):
                normal, _, _ = self._feed(text, size)

                self.assertEqual(normal.count("Let me check."), 1)
                self.assertNotIn(self.D, normal)

    def test_finish_is_idempotent(self):
        """A repeated terminal flush must be a no-op."""
        text = f"<{self.D}tool_calls>\n{self._invoke()}\n</{self.D}tool_calls>\nTail"
        parser = FunctionCallParser(self.tools, "deepseekv4")
        for start in range(0, len(text), 8):
            parser.parse_stream_chunk(text[start : start + 8])

        first = parser.parse_stream_end()
        second = parser.parse_stream_end()

        self.assertIn("Tail", first[0])
        self.assertEqual(second, ("", []))

    def test_plain_text_needs_no_flush(self):
        normal, names, _ = self._feed("Just a plain answer.", 8)

        self.assertEqual(normal, "Just a plain answer.")
        self.assertEqual(names, [])


class TestDeepSeekV32SharesTheFix(CustomTestCase):
    """`deepseekv32` inherits the same base, so it must gain the fix too.

    The V3.2 detector is the parent class being changed, so these guard
    both directions: the V3.2 dialect keeps working, and the shapes that
    used to leak no longer do.
    """

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
        self.detector = DeepSeekV32Detector()

    def _invoke(self):
        return _invoke("get_weather", _param("city", "true", "SF"))

    def _parse(self, text):
        return self.detector.detect_and_parse(text, self.tools)

    def test_v32_native_block_still_parses(self):
        result = self._parse(
            f"Sure.\n\n<{DSML}function_calls>\n{self._invoke()}\n</{DSML}function_calls>"
        )

        self.assertEqual([c.name for c in result.calls], ["get_weather"])
        self.assertIn("Sure.", result.normal_text)

    def test_v32_leaking_shapes_are_fixed(self):
        cases = {
            "bare invoke": f"Sure.\n\n{self._invoke()}",
            "unterminated": f"<{DSML}function_calls>\n{self._invoke()}",
            "v4 block name": f"<{DSML}tool_calls>\n{self._invoke()}\n</{DSML}tool_calls>",
        }
        for label, text in cases.items():
            with self.subTest(payload=label):
                result = self._parse(text)

                self.assertEqual([c.name for c in result.calls], ["get_weather"])
                self.assertNotIn(DSML, result.normal_text)

    def test_plain_text_is_untouched(self):
        result = self._parse("plain answer")

        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, "plain answer")


if __name__ == "__main__":
    import unittest

    unittest.main()
