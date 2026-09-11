"""Unit tests for DeepSeekV4Detector DSML streaming — no server, no model loading."""

from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
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


# Turn shapes the streaming and one-shot paths must agree on, whatever the chunk
# size. Anything the one-shot path cannot parse at all -- a bare invoke with no
# section around it -- is tested on its own instead.
TURN_SHAPES = {
    "no tool call at all": "Just talking.\n\n",
    "preamble": "Hi\n\n" + _weather_call(),
    "preamble, single newline": "Hi\n" + _weather_call(),
    "call alone": _weather_call(),
    "prose after the call": _weather_call() + "\n\nSunny in SF.",
    "newlines after the call": _weather_call() + "\n\n",
    "prose between two calls": (
        _weather_call("SF") + "\n\nNow the other city.\n\n" + _weather_call("NY")
    ),
    "sections back to back": _weather_call("SF") + "\n" + _weather_call("NY"),
    "parallel invokes in one section": _wrapped(
        _invoke("get_weather", _param("city", "true", "SF"))
        + "\n"
        + _invoke("get_weather", _param("city", "true", "NY"))
    ),
    "json body": "Lead.\n\n" + _wrapped(_invoke("get_weather", '{"city": "SF"}')),
    "self-closing invoke": _wrapped(f'<{DSML}invoke name="get_weather"/>')
    + "\n\nDone.",
    "angle bracket in the prose": _weather_call() + "\n\n5 < 6 and a < b.",
    "several paragraphs after the call": (
        _weather_call() + "\n\nFirst line.\nSecond line.\n\nThird."
    ),
    "indented line after the call": "Result:\n" + _weather_call() + "\n    x = True\n",
    "markdown list after the call": (
        "Here you go:\n\n" + _weather_call() + "\n- SF: sunny\n- LA: hot\n"
    ),
}

# What the turn should read as once the calls are taken out of it, for the shapes
# where getting that wrong is invisible to the one-shot comparison.
JOINED_TURNS = {
    "Let me look it up.\n\n"
    + _weather_call()
    + "\nIt is sunny in SF.": "Let me look it up.\nIt is sunny in SF.",
    "我查一下。\n\n"
    + _weather_call()
    + "\n旧金山是晴天。": "我查一下。\n旧金山是晴天。",
    _weather_call()
    + "\nAnd now the other one:\n"
    + _weather_call("NY")
    + "\nDone.": "And now the other one:\nDone.",
    "Here you go:\n\n"
    + _weather_call()
    + "\n- SF: sunny\n- LA: hot\n": "Here you go:\n- SF: sunny\n- LA: hot\n",
    "Result:\n" + _weather_call() + "\n    x = True\n": "Result:\n    x = True\n",
}


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

    def _stream(self, text, chunk_size):
        """Returns (normal_text, [(name, arguments)]) for `text` fed in deltas of
        `chunk_size` characters (the whole text at once when it is None), end of
        turn included.

        Calls are compared as an ordered list rather than by `tool_index`: the
        one-shot path numbers them by position in the tools list, the streaming
        path by call ordinal, so the indices themselves do not line up.
        """
        detector = DeepSeekV4Detector()
        chunks = (
            [text]
            if chunk_size is None
            else [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        )
        normal, merged = "", {}
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            for call in result.calls:
                item = merged.setdefault(call.tool_index, {"name": None, "args": ""})
                if call.name:
                    item["name"] = call.name
                item["args"] += call.parameters
        normal += detector.finish(self.tools).normal_text
        self.assertEqual(detector._buffer, "", "end of turn left the buffer behind")
        return normal, [(v["name"], v["args"]) for _, v in sorted(merged.items())]

    def _one_shot(self, text):
        result = DeepSeekV4Detector().detect_and_parse(text, self.tools)
        return result.normal_text, [(c.name, c.parameters) for c in result.calls]

    def test_streaming_matches_one_shot_at_every_chunk_size(self):
        """A turn must read the same however it was cut into deltas, and the same
        as if it had never been streamed at all.

        Both halves matter. Where the deltas fall is not the model's choice, so
        anything that depends on it is a bug by construction; and a client that
        switches `stream=` must not get different content out of the same
        generation. Down to one character per delta, because that is where a
        marker or its indent gets split across two chunks -- how the duplicated
        preamble and the stray newlines this file guards were all found.
        """
        for label, text in TURN_SHAPES.items():
            expected = self._one_shot(text)
            for size in (None, 1, 2, 3, 5, 13):
                with self.subTest(shape=label, chunk_size=size):
                    self.assertEqual(self._stream(text, size), expected)

    def test_removing_a_call_leaves_the_line_it_stood_on(self):
        """Taking the call out has to leave the line break it was standing on --
        and only that.

        Dropping the whitespace outright runs the sentence before the call into
        the one after it, with no word boundary to recover in CJK, and takes the
        indent off the line that follows: a list item stops starting its line, a
        code line loses the spaces the newline was carrying. None of that is
        visible to the one-shot comparison, which agreed with the streaming path
        on all five of these while both were wrong, so the text is pinned here.
        """
        for text, expected in JOINED_TURNS.items():
            for size in (None, 16, 8, 4, 1):
                with self.subTest(turn=text[:24], chunk_size=size):
                    normal, _ = self._stream(text, size)
                    self.assertEqual(normal, expected)

    def test_turn_cut_off_mid_generation_repeats_nothing_and_leaks_no_markup(self):
        """A turn that stops inside a tool call is what a length cap looks like.

        The lead used to come out twice -- once from the delta that matched the
        invoke, once more from the end-of-turn flush, because an invoke that
        never completes never advances the buffer past it -- and the half-written
        tag and its partial JSON went to the client as assistant content.

        Only the text is compared with the one-shot path here. The calls cannot
        be: streaming a cut-off call is exactly what the incremental protocol is
        for, while `detect_and_parse` has no complete section to report.
        """
        full = "Checking.\n" + _weather_call() + "\nSunny."

        for length in range(1, len(full) + 1):
            text = full[:length]
            expected, _ = self._one_shot(text)
            for size in (None, 1, 3, 7):
                with self.subTest(cut_at=length, chunk_size=size):
                    normal, _ = self._stream(text, size)
                    self.assertNotIn(DSML, normal)
                    self.assertEqual(normal, expected)

    def test_preamble_before_bare_invoke_without_wrapper(self):
        """The bare `<｜DSML｜invoke …>` form has no tool_calls wrapper to walk
        back to, so the preamble is computed from the invoke itself."""
        text = "Checking.\n" + _invoke("get_weather", _param("city", "true", "SF"))
        normal, calls = self._feed([text])

        self.assertIn("Checking.", normal)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])

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

    def test_prose_around_tool_calls_reaches_the_client(self):
        """Prose between two calls used to be dropped outright -- the lead was
        only recovered for the first call, and consuming the second advanced the
        buffer straight past the text in front of it -- and prose after the last
        call was stranded, because the section closer keeps the DSML guard true
        for the rest of the turn.

        Both are asserted here rather than left to the one-shot comparison: that
        comparison would still hold if a change dropped this text on both paths.
        """
        normal, calls = self._feed(
            [
                _weather_call("SF"),
                "\n\nNow the other city.\n\n",
                _weather_call("NY"),
                "\n\nThat's the forecast.",
            ]
        )

        self.assertIn("Now the other city.", normal)
        self.assertIn("That's the forecast.", normal)
        self.assertNotIn(DSML, normal)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"] * 2)

    def test_prose_with_angle_bracket_is_not_held_back(self):
        """Only a suffix that could still grow into a DSML marker may be withheld.
        Holding every `<` would keep ordinary prose out of the stream until the
        turn ended, long after the client should have seen it."""
        normal, _ = self._feed([_weather_call("SF"), "\n\n5 < 6 and a < b."])

        self.assertIn("5 < 6 and a < b.", normal)

    def test_parallel_calls_do_not_leak_their_separator(self):
        """Invokes inside one section are separated by layout, not by content.
        Collecting a lead for every call must not turn that whitespace into an
        assistant message -- `detect_and_parse` never surfaces it either."""
        section = _wrapped(
            _invoke("get_weather", _param("city", "true", "SF"))
            + "\n"
            + _invoke("get_weather", _param("city", "true", "NY"))
        )

        for size in (None, 1, 5, 17):
            with self.subTest(chunk_size=size):
                chunks = (
                    [section]
                    if size is None
                    else [section[i : i + size] for i in range(0, len(section), size)]
                )
                normal, calls = self._feed(chunks)

                self.assertEqual(normal, "")
                self.assertEqual([c.name for c in calls if c.name], ["get_weather"] * 2)

    def test_error_on_a_later_call_does_not_duplicate_its_lead(self):
        """The lead of the call that failed is still at the head of the buffer the
        error path dumps verbatim, so it must not also be emitted from the parts
        collected so far. Only leads whose call completed are gone from it."""
        detector = DeepSeekV4Detector()
        text = _weather_call("SF") + "\n\nNow the other one.\n\n" + _weather_call("NY")
        real = DeepSeekV4Detector._parse_parameters_from_xml
        seen = []

        def fail_on_second(self, *args, **kwargs):
            seen.append(1)
            if len(seen) == 2:
                raise RuntimeError("boom")
            return real(self, *args, **kwargs)

        with patch.object(
            DeepSeekV4Detector, "_parse_parameters_from_xml", fail_on_second
        ):
            result = detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text.count("Now the other one."), 1)

    def test_finish_releases_text_held_for_a_marker_that_never_came(self):
        """Trailing newlines are withheld in case they indent a tag still to come.
        When the stream ends instead, they were ordinary text all along, and the
        end-of-turn flush is what owes them to the client."""
        detector = DeepSeekV4Detector()
        normal = ""
        for chunk in [_weather_call("SF"), "\n\nBye.\n\n"]:
            normal += detector.parse_streaming_increment(chunk, self.tools).normal_text
        normal += detector.finish(self.tools).normal_text

        self.assertTrue(normal.endswith("Bye.\n\n"), repr(normal))
        self.assertEqual(detector._buffer, "")

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


if __name__ == "__main__":
    import unittest

    unittest.main()
