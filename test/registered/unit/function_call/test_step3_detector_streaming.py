"""Streaming regression tests for Step3Detector -- no server, no model loading.

Every case here feeds the *same* model output at several chunk sizes and
compares the merged stream against ``detect_and_parse``. The bugs guarded are
described in each docstring in black-box terms; all of them fail before the
accompanying detector change and pass after it.
"""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.step3_detector import Step3Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

BOT = "<｜tool_calls_begin｜>"
EOT = "<｜tool_calls_end｜>"
CALL_BEGIN = "<｜tool_call_begin｜>"
CALL_END = "<｜tool_call_end｜>"
SEP = "<｜tool_sep｜>"

# Markup that must never reach the client as assistant content.
MARKUP = ("<steptml", "<｜tool_call", "<｜tool_sep")

# Chunk sizes every case is replayed at. ``None`` means "the whole output in a
# single increment", which is what the last increment of a stream looks like and
# what ``stream_interval > 1`` or speculative decoding can produce mid-stream.
CHUNK_SIZES = (None, 16, 8, 4, 1)


def _call(name: str, params: str) -> str:
    return (
        f'{CALL_BEGIN}function{SEP}<steptml:invoke name="{name}">\n'
        f"{params}"
        f"</steptml:invoke>{CALL_END}\n"
    )


def _param(name: str, value: str) -> str:
    return f'<steptml:parameter name="{name}">{value}</steptml:parameter>\n'


class TestStep3DetectorStreaming(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer"},
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_time",
                    description="Get the current time",
                    parameters={
                        "type": "object",
                        "properties": {"tz": {"type": "string"}},
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="ping",
                    description="Takes no argument",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

    def _stream(self, text, chunk_size=None):
        """Replay ``text`` through the streaming API and merge the deltas.

        Returns ``(normal_text, [(name, arguments), ...])`` in the same shape
        ``detect_and_parse`` produces, so the two can be compared directly.
        """
        if chunk_size is None:
            chunks = [text]
        else:
            chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        return self._stream_parts(chunks)

    def _stream_parts(self, parts):
        """Same as ``_stream`` but with explicitly chosen increment boundaries."""
        detector = Step3Detector()

        normal_text = ""
        names = {}
        arguments = {}
        order = []

        def absorb(result):
            nonlocal normal_text
            normal_text += result.normal_text or ""
            for item in result.calls:
                if item.tool_index not in arguments:
                    arguments[item.tool_index] = ""
                    order.append(item.tool_index)
                if item.name:
                    names[item.tool_index] = item.name
                if item.parameters:
                    arguments[item.tool_index] += item.parameters

        for chunk in parts:
            absorb(detector.parse_streaming_increment(chunk, self.tools))
        # The serving layer calls finish() once the stream is over.
        absorb(detector.finish(self.tools))

        return normal_text, [(names.get(i), arguments.get(i, "")) for i in order]

    def _assert_no_markup(self, text, label):
        for markup in MARKUP:
            self.assertNotIn(markup, text, f"{markup!r} leaked into content ({label})")

    def _assert_matches_non_streaming(self, text):
        """Content and calls must be chunk-invariant and match one-shot parsing."""
        expected = Step3Detector().detect_and_parse(text, self.tools)
        expected_calls = [(c.name, c.parameters) for c in expected.calls]
        self._assert_no_markup(expected.normal_text or "", "non-streaming")
        for chunk_size in CHUNK_SIZES:
            with self.subTest(chunk_size=chunk_size):
                normal_text, calls = self._stream(text, chunk_size)
                self._assert_no_markup(normal_text, f"chunk_size={chunk_size}")
                self.assertEqual(normal_text, expected.normal_text or "")
                self.assertEqual(calls, expected_calls)
                for _, params in calls:
                    json.loads(params)  # arguments must always be valid JSON
        return expected_calls

    # ==================== One increment, several things ====================

    def test_streaming_text_call_and_text_in_one_increment(self):
        """A single increment carrying preamble + a call + trailing prose.

        The detector used to perform one action per increment and defer the rest
        to the next one, so an increment that also contained the tool call
        emitted only the preamble and dropped the call entirely -- and the last
        increment of a stream has no next increment to defer to.
        """
        text = (
            "Sure.\n"
            + BOT
            + "\n"
            + _call("get_weather", _param("city", "SF") + _param("days", "3"))
            + EOT
            + "\nDone."
        )
        calls = self._assert_matches_non_streaming(text)
        self.assertEqual(calls, [("get_weather", '{"city": "SF", "days": 3}')])
        normal_text, _ = self._stream(text)
        self.assertEqual(normal_text, "Sure.\n\nDone.")

    def test_streaming_two_calls_in_one_increment_keep_their_own_parameters(self):
        """Two complete calls inside one increment.

        Parameters were collected by scanning the whole buffer, so the second
        call donated its parameters to the first one and was then dropped:
        ``get_weather{"city": "SF", "tz": "UTC"}`` instead of two calls. The
        increment boundaries here are explicit because this is what speculative
        decoding produces: preamble, then several accepted calls at once.
        """
        parts = [
            "Checking.\n" + BOT + "\n",
            _call("get_weather", _param("city", "SF"))
            + _call("get_time", _param("tz", "UTC")),
            EOT + "\nBoth done.",
        ]
        normal_text, calls = self._stream_parts(parts)
        self._assert_no_markup(normal_text, "explicit increments")
        self.assertEqual(normal_text, "Checking.\n\nBoth done.")
        self.assertEqual(
            calls,
            [("get_weather", '{"city": "SF"}'), ("get_time", '{"tz": "UTC"}')],
        )
        self._assert_matches_non_streaming("".join(parts))

    def test_streaming_trailing_text_is_chunk_invariant(self):
        """Prose after the tool block must not depend on the chunk boundaries.

        With the whole answer in one increment the trailing sentence was lost,
        while smaller chunks kept it -- the same answer yielded different
        content depending on ``stream_interval``.
        """
        text = (
            BOT
            + "\n"
            + _call("get_weather", _param("city", "SF"))
            + EOT
            + "\nIt is sunny in SF."
        )
        self._assert_matches_non_streaming(text)
        for chunk_size in CHUNK_SIZES:
            with self.subTest(chunk_size=chunk_size):
                normal_text, _ = self._stream(text, chunk_size)
                self.assertEqual(normal_text, "\nIt is sunny in SF.")

    # ==================== Parameter-less calls ====================

    def test_streaming_zero_parameter_call_has_valid_arguments(self):
        """A call with no parameters must still stream ``{}``.

        Nothing was streamed for such a call and the closing brace was only sent
        when something had been streamed before, so ``arguments`` stayed the
        empty string -- which is not valid JSON for the client.
        """
        text = BOT + "\n" + _call("ping", "") + EOT + "\nok"
        calls = self._assert_matches_non_streaming(text)
        self.assertEqual(calls, [("ping", "{}")])

    def test_detect_and_parse_keeps_call_with_empty_invoke_body(self):
        """``<steptml:invoke name="ping"></steptml:invoke>`` is a real call.

        The invoke pattern required a non-empty body, so a parameter-less call
        written without even a newline was not matched and the whole call was
        silently dropped by one-shot parsing.
        """
        text = (
            BOT
            + "\n"
            + f'{CALL_BEGIN}function{SEP}<steptml:invoke name="ping">'
            + f"</steptml:invoke>{CALL_END}\n"
            + EOT
        )
        result = Step3Detector().detect_and_parse(text, self.tools)
        self.assertEqual(
            [(c.name, c.parameters) for c in result.calls], [("ping", "{}")]
        )
        self._assert_matches_non_streaming(text)

    # ==================== Calls that must be dropped ====================

    def test_streaming_rejected_calls_keep_trailing_text(self):
        """An unknown tool name, or a non-``function`` call type, is dropped.

        Both are dropped by one-shot parsing, so streaming must drop them too --
        without leaking the raw markup as assistant content and without eating
        the prose that follows the tool block.
        """
        unknown = BOT + "\n" + _call("get_wether", _param("city", "SF")) + EOT + "\nhmm"
        self.assertEqual(self._assert_matches_non_streaming(unknown), [])
        self.assertEqual(self._stream(unknown)[0], "\nhmm")

        bad_type = (
            BOT
            + "\n"
            + f'{CALL_BEGIN}custom{SEP}<steptml:invoke name="get_weather">'
            + _param("city", "SF")
            + f"</steptml:invoke>{CALL_END}\n"
            + EOT
            + "\ntail"
        )
        self.assertEqual(self._assert_matches_non_streaming(bad_type), [])
        self.assertEqual(self._stream(bad_type)[0], "\ntail")

    def test_truncated_tool_block_is_markup_not_content(self):
        """A block cut short by the token limit must not be echoed as content.

        One-shot parsing returned the whole raw text -- ``<｜tool_calls_begin｜>``,
        ``<steptml:invoke ...>`` and all -- as assistant content whenever the end
        token was missing, and dropped the calls that had completed before the
        cut. Streaming keeps the completed call and closes its arguments, since a
        name already sent to the client cannot be taken back.
        """
        text = (
            "Let me check.\n"
            + BOT
            + "\n"
            + _call("get_weather", _param("city", "SF"))
            + f'{CALL_BEGIN}function{SEP}<steptml:invoke name="get_ti'
        )
        result = Step3Detector().detect_and_parse(text, self.tools)
        self._assert_no_markup(result.normal_text or "", "non-streaming")
        self.assertEqual(result.normal_text, "Let me check.\n")
        self.assertEqual(
            [(c.name, c.parameters) for c in result.calls],
            [("get_weather", '{"city": "SF"}')],
        )

        for chunk_size in CHUNK_SIZES:
            with self.subTest(chunk_size=chunk_size):
                normal_text, calls = self._stream(text, chunk_size)
                self._assert_no_markup(normal_text, f"chunk_size={chunk_size}")
                self.assertEqual(normal_text, "Let me check.\n")
                self.assertEqual(calls, [("get_weather", '{"city": "SF"}')])

    def test_streaming_releases_text_that_looked_like_a_tool_marker(self):
        """Text ending in the first characters of the begin token is content.

        Such a tail is held back in case the marker completes; once the stream is
        over it cannot, so ``finish()`` has to release it instead of swallowing
        it.
        """
        text = "The price is 10<｜"
        for chunk_size in CHUNK_SIZES:
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(self._stream(text, chunk_size), (text, []))


if __name__ == "__main__":
    import unittest

    unittest.main()
