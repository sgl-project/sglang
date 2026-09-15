"""Unit tests for DeepSeekV4Detector DSML streaming — no server, no model loading."""

import json
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


def _tasks_call(tasks_literal: str, notes: str = "see the plan") -> str:
    """A create_tasks call: an array-of-objects parameter (string="false") plus a
    string one. The array is the largest delimiter surface a tool argument has,
    which is where the streaming corruption showed up."""
    return _wrapped(
        _invoke(
            "create_tasks",
            _param("tasks", "false", tasks_literal)
            + "\n"
            + _param("notes", "true", notes),
        )
    )


_TASKS = (
    '[{"id": "T1", "title": "Review the template", '
    '"description": "He said \\"no problem\\", but check the delimiters.", '
    '"dependencies": [], "priority": "high", "estimated_minutes": 120}, '
    '{"id": "T2", "title": "压测吞吐与首 token 延迟", '
    '"description": "在预发环境对比两套引擎。", '
    '"dependencies": ["T1"], "priority": "medium", "estimated_minutes": 240}]'
)

# How the model plausibly gets the INNER json slightly wrong. Nothing constrains
# a parameter's interior — the structural tag pins the DSML tags and the function
# name — so these are ordinary outputs, and they are what used to make the
# serialisation change shape between chunks.
ARGUMENT_FIXTURES = {
    "well_formed": _tasks_call(_TASKS),
    "trailing_comma": _tasks_call(_TASKS[:-1] + ",]"),
    "unescaped_quote": _tasks_call(_TASKS.replace("压测吞吐", '压测"吞吐"')),
    "single_quoted_key": _tasks_call(_TASKS.replace('"priority"', "'priority'")),
    "value_ends_in_tag_letters": _tasks_call(_TASKS, notes="please read the note"),
}


def _chunkings(text):
    """Real chunk boundaries are token boundaries; the parser has to be correct at
    all of them, so sweep rather than sample."""
    yield "per_char", [text[i : i + 1] for i in range(len(text))]
    for n in (2, 3, 5, 7, 11, 16, 32, 64):
        yield f"fixed_{n}", [text[i : i + n] for i in range(0, len(text), n)]


class TestDeepSeekV4StreamingArguments(CustomTestCase):
    """Streaming must never hand the client argument bytes it then contradicts.

    parse_streaming_increment() rebuilds the whole arguments string on every
    chunk. It used to emit common_prefix(current, previous)[sent_len:] and then
    current[sent_len:] on the closing tag, which assumes every byte already sent
    is still a prefix of the finished serialisation. A non-string parameter
    breaks that: it serialises as a structure while _partial_json_loads succeeds
    and as a quote-escaped string when it falls back, so the closing-tag slice
    started at a stale offset and dropped a run out of the middle.
    """

    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="create_tasks",
                    description="Create a task plan",
                    parameters={
                        "type": "object",
                        "properties": {
                            "tasks": {"type": "array", "items": {"type": "object"}},
                            "notes": {"type": "string"},
                        },
                        "required": ["tasks"],
                    },
                ),
            )
        ]

    def _stream_arguments(self, chunks):
        """Reassemble arguments the way an OpenAI client does, and count deltas."""
        detector = DeepSeekV4Detector()
        arguments, deltas = "", 0
        for chunk in chunks:
            for call in detector.parse_streaming_increment(chunk, self.tools).calls:
                if call.parameters:
                    arguments += call.parameters
                    deltas += 1
        return arguments, deltas

    def test_streamed_arguments_match_non_streaming(self):
        for name, text in ARGUMENT_FIXTURES.items():
            expected = DeepSeekV4Detector().detect_and_parse(text, self.tools).calls
            self.assertEqual(len(expected), 1, name)
            for label, chunks in _chunkings(text):
                with self.subTest(fixture=name, chunking=label):
                    arguments, _ = self._stream_arguments(chunks)
                    self.assertEqual(
                        json.loads(arguments), json.loads(expected[0].parameters)
                    )

    def test_streamed_arguments_are_always_a_prefix_of_the_final_value(self):
        """The model can stop anywhere and a delta cannot be recalled, so at every
        truncation point what has been streamed must still be a prefix of what the
        finished stream produces."""
        for name, text in ARGUMENT_FIXTURES.items():
            final, _ = self._stream_arguments([text])
            for cut in range(1, len(text), 3):
                partial, _ = self._stream_arguments(
                    [text[:cut][i : i + 4] for i in range(0, cut, 4)]
                )
                with self.subTest(fixture=name, cut=cut):
                    self.assertTrue(final.startswith(partial), partial[-80:])

    def test_string_parameter_arguments_still_stream_incrementally(self):
        """Holding back the unsettled tail must not degrade into one final chunk:
        a string parameter is prefix-stable and has to keep flowing (#11888)."""
        value = "paragraph " * 300
        text = _wrapped(_invoke("create_tasks", _param("notes", "true", value)))
        arguments, deltas = self._stream_arguments(
            [text[i : i + 8] for i in range(0, len(text), 8)]
        )

        self.assertEqual(json.loads(arguments)["notes"], value.strip())
        self.assertGreater(deltas, 10)

    def test_partial_tag_suffix_trim_does_not_eat_value_characters(self):
        """str.rstrip takes a character set, not a suffix: rstrip("oke") turned a
        value ending in "note" into "not"."""
        trim = DeepSeekV4Detector()._strip_partial_tag_suffix
        end = f"</{DSML}parameter>"

        self.assertEqual(trim("read the note", end), "read the note")
        self.assertEqual(trim("value", end), "value")
        self.assertEqual(trim("value</", end), "value")
        self.assertEqual(trim(f"value</{DSML}par", end), "value")


if __name__ == "__main__":
    import unittest

    unittest.main()
