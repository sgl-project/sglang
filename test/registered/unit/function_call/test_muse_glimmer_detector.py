"""Unit tests for the Muse Glimmer ATEM tool-call detector — no server, no model loading.

The expectations here are pinned to the checkpoint's own ``response_template``
(``MUSE_GLIMMER_RESPONSE_SCHEMA`` in ``tokenizer_config.json``) and to the vendor's
reference parser, with particular attention to channel scoping: an
``<atem:invoke>`` that only appears inside a reasoning block or a final answer
must never become a real tool call.
"""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.muse_glimmer_detector import MuseGlimmerDetector
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

DOUBLED = "get_weather.get_weather"


def atem(name: str, **params: str) -> str:
    body = "".join(
        f'<atem:parameter name="{k}">{v}</atem:parameter>\n' for k, v in params.items()
    )
    return (
        f'<atem:function_calls>\n<atem:invoke name="{name}">\n{body}'
        f"</atem:invoke>\n</atem:function_calls>"
    )


class TestMuseGlimmerDetector(CustomTestCase):
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
                        "required": ["city"],
                    },
                ),
            ),
        ]

    # ---- helpers ----------------------------------------------------------

    def parse(self, text):
        """Non-streaming parse -> (normal_text, [(name, args), ...])."""
        result = MuseGlimmerDetector().detect_and_parse(text, self.tools)
        return result.normal_text, [
            (c.name, json.loads(c.parameters)) for c in result.calls if c.name
        ]

    def parse_streaming(self, text, chunk_size):
        detector = MuseGlimmerDetector()
        normal, calls = [], []
        for i in range(0, len(text), chunk_size):
            result = detector.parse_streaming_increment(
                text[i : i + chunk_size], self.tools
            )
            normal.append(result.normal_text)
            calls.extend(
                (c.name, json.loads(c.parameters)) for c in result.calls if c.name
            )
        return "".join(normal), calls

    def assert_streaming_matches(self, text):
        """Streaming must agree with one-shot parsing at every chunk boundary."""
        expected = self.parse(text)
        for chunk_size in (1, 2, 3, 5, 7, 13, 29, 100):
            self.assertEqual(
                self.parse_streaming(text, chunk_size),
                expected,
                f"streaming diverged at chunk_size={chunk_size}",
            )

    # ---- tool extraction --------------------------------------------------

    def test_single_tool_call(self):
        text = (
            f" to=self<|message|>Need weather.<|eom|>"
            f"<|start|>assistant to={DOUBLED}<|message|>{atem(DOUBLED, city='Paris')}"
        )
        normal, calls = self.parse(text)
        self.assertEqual(calls, [("get_weather", {"city": "Paris"})])
        self.assertEqual(normal, "Need weather.")
        self.assert_streaming_matches(text)

    def test_parallel_tool_calls(self):
        text = (
            f" to=self<|message|>Two cities.<|eom|>"
            f"<|start|>assistant to={DOUBLED}<|message|>"
            f"{atem(DOUBLED, city='Paris')}<|eom|>"
            f"<|start|>assistant to={DOUBLED}<|message|>{atem(DOUBLED, city='Tokyo')}"
        )
        _, calls = self.parse(text)
        self.assertEqual(
            calls,
            [("get_weather", {"city": "Paris"}), ("get_weather", {"city": "Tokyo"})],
        )
        self.assert_streaming_matches(text)

    def test_tool_call_then_final_answer(self):
        text = (
            f" to=self<|message|>r<|eom|>"
            f"<|start|>assistant to={DOUBLED}<|message|>"
            f"{atem(DOUBLED, city='Paris')}<|eom|>"
            f"<|start|>assistant to=user<|message|>It is sunny."
        )
        normal, calls = self.parse(text)
        self.assertEqual(calls, [("get_weather", {"city": "Paris"})])
        self.assertIn("It is sunny.", normal)
        self.assert_streaming_matches(text)

    def test_namespaced_name_passes_through(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="weather.get",
                    description="d",
                    parameters={"type": "object", "properties": {}},
                ),
            )
        ]
        text = (
            f" to=self<|message|>r<|eom|><|start|>assistant to=weather.get<|message|>"
            f"{atem('weather.get', city='Paris')}"
        )
        result = MuseGlimmerDetector().detect_and_parse(text, tools)
        self.assertEqual([c.name for c in result.calls], ["weather.get"])

    def test_parameter_value_typing(self):
        """``allow_non_json: True`` — JSON literals decode, bare strings do not."""
        invoke = (
            '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
            '<atem:parameter name="s">hello world</atem:parameter>\n'
            '<atem:parameter name="i">42</atem:parameter>\n'
            '<atem:parameter name="b">true</atem:parameter>\n'
            '<atem:parameter name="n">null</atem:parameter>\n'
            '<atem:parameter name="o">{"a": 1}</atem:parameter>\n'
            '<atem:parameter name="l">[1, 2]</atem:parameter>\n'
            "</atem:invoke>\n</atem:function_calls>"
        )
        text = f"<|start|>assistant to=get_weather<|message|>{invoke}"
        _, calls = self.parse(text)
        self.assertEqual(
            calls[0][1],
            {
                "s": "hello world",
                "i": 42,
                "b": True,
                "n": None,
                "o": {"a": 1},
                "l": [1, 2],
            },
        )

    def test_multiline_parameter_value(self):
        value = 'line1\nline2\n"quoted"\n'
        text = (
            f"<|start|>assistant to=get_weather<|message|>"
            f'<atem:function_calls>\n<atem:invoke name="get_weather">\n'
            f'<atem:parameter name="code">{value}</atem:parameter>\n'
            f"</atem:invoke>\n</atem:function_calls>"
        )
        _, calls = self.parse(text)
        self.assertEqual(calls[0][1], {"code": value})
        self.assert_streaming_matches(text)

    # ---- channel scoping (safety) -----------------------------------------

    def test_invoke_inside_reasoning_is_not_a_call(self):
        text = (
            f" to=self<|message|>Maybe I call {atem(DOUBLED, city='X')} — no.<|eom|>"
            f"<|start|>assistant to=user<|message|>I will not call it."
        )
        _, calls = self.parse(text)
        self.assertEqual(calls, [])
        self.assert_streaming_matches(text)

    def test_invoke_inside_final_answer_is_not_a_call(self):
        text = (
            f" to=self<|message|>r<|eom|><|start|>assistant to=user<|message|>"
            f"You would write:\n{atem(DOUBLED, city='X')}"
        )
        _, calls = self.parse(text)
        self.assertEqual(calls, [])
        self.assert_streaming_matches(text)

    def test_invoke_inside_truncated_reasoning_is_not_a_call(self):
        """Generation cut mid-CoT leaves no closing ``<|eom|>`` to anchor on."""
        text = f" to=self<|message|>I could call {atem(DOUBLED, city='X')} but"
        _, calls = self.parse(text)
        self.assertEqual(calls, [])
        self.assert_streaming_matches(text)

    def test_truncated_tool_channel_drops_partial_invoke(self):
        """A token cap mid-invoke must not fabricate a call from partial
        arguments, and ATEM scaffolding must not leak into content."""
        text = (
            f" to=self<|message|>Need weather.<|eom|>"
            f"<|start|>assistant to={DOUBLED}<|message|>"
            f'<atem:function_calls>\n<atem:invoke name="{DOUBLED}">\n'
            f'<atem:parameter name="city">Par'
        )
        normal, calls = self.parse(text)
        self.assertEqual(calls, [])
        self.assertEqual(normal, "Need weather.")
        self.assert_streaming_matches(text)

    def test_prose_opening_with_to_equals_does_not_stall(self):
        """A bare ``to=...`` header opens the stream, so prose that happens to
        start the same way is ambiguous. Mis-reading it as a header parks the
        parser waiting for a ``<|message|>`` that never arrives and strands the
        whole response in the buffer."""
        for text in ("to=x is the syntax.", "to= is an assignment", "to=a<b is false"):
            detector = MuseGlimmerDetector()
            streamed = "".join(
                detector.parse_streaming_increment(ch, self.tools).normal_text
                for ch in text
            )
            self.assertEqual(streamed, text)
            self.assertEqual(detector._buffer, "", "text stranded in the buffer")

    def test_unframed_atem_is_content_not_a_call(self):
        """Deliberately stricter than the vendor; see ``_is_tool_channel``."""
        text = atem(DOUBLED, city="Paris")
        normal, calls = self.parse(text)
        self.assertEqual(calls, [])
        self.assertEqual(normal, text)

    # ---- integration with the reasoning parser ----------------------------

    def test_pipeline_with_reasoning_parser(self):
        """The real serving order: reasoning parser first, then this detector."""
        raw = (
            f" to=self<|message|>Need weather.<|eom|>"
            f"<|start|>assistant to={DOUBLED}<|message|>"
            f"{atem(DOUBLED, city='Paris')}<|eom|>"
            f"<|start|>assistant to=user<|message|>It is sunny in Paris."
        )
        reasoning, remainder = ReasoningParser("muse").parse_non_stream(raw)
        content, calls = FunctionCallParser(self.tools, "muse").parse_non_stream(
            remainder
        )
        self.assertEqual(reasoning, "Need weather.")
        self.assertEqual(content, "It is sunny in Paris.")
        self.assertEqual(
            [(c.name, json.loads(c.parameters)) for c in calls],
            [("get_weather", {"city": "Paris"})],
        )

    def test_pipeline_quoted_invoke_stays_content(self):
        """A quoted ATEM block must survive as text, not become a call."""
        raw = (
            f" to=self<|message|>r<|eom|><|start|>assistant to=user<|message|>"
            f"Example:\n{atem(DOUBLED, city='X')}"
        )
        _, remainder = ReasoningParser("muse").parse_non_stream(raw)
        content, calls = FunctionCallParser(self.tools, "muse").parse_non_stream(
            remainder
        )
        self.assertEqual(calls, [])
        self.assertIn("<atem:invoke", content)

    def pipeline_stream(self, raw, chunk_size, tool_call_parser_active=True):
        """The real streaming order: reasoning deltas feed the tool parser."""
        rp = ReasoningParser("muse", tool_call_parser_active=tool_call_parser_active)
        fp = FunctionCallParser(self.tools, "muse")
        reasoning_parts, content_parts, calls = [], [], []
        chunks = [raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)]
        for i, chunk in enumerate(chunks):
            reasoning, normal = rp.parse_stream_chunk(chunk)
            if i == len(chunks) - 1:
                end_reasoning, end_normal = rp.parse_stream_end()
                reasoning = (reasoning or "") + (end_reasoning or "")
                normal = (normal or "") + (end_normal or "")
            if reasoning:
                reasoning_parts.append(reasoning)
            if normal:
                content, chunk_calls = fp.parse_stream_chunk(normal)
                content_parts.append(content)
                calls.extend(
                    (c.name, json.loads(c.parameters)) for c in chunk_calls if c.name
                )
        end_content, end_calls = fp.parse_stream_end()
        content_parts.append(end_content)
        calls.extend((c.name, json.loads(c.parameters)) for c in end_calls if c.name)
        return "".join(reasoning_parts), "".join(content_parts), calls

    def test_pipeline_quoted_header_in_answer_is_not_a_call(self):
        """The answer keeps its channel framing on the way to this detector, so
        a quoted header inside it stays inside the ``to=user`` body. Goes red if
        the reasoning parser unwraps the answer before the tool parser runs, or
        if ``detect_and_parse`` regains a scan-ahead ``<|start|>`` search."""
        for quoted_at in ("after prose:\n", ""):
            raw = (
                f" to=self<|message|>r<|eom|><|start|>assistant to=user<|message|>"
                f"{quoted_at}<|start|>assistant to={DOUBLED}<|message|>"
                f"{atem(DOUBLED, city='X')}"
            )
            _, remainder = ReasoningParser(
                "muse", tool_call_parser_active=True
            ).parse_non_stream(raw)
            content, calls = FunctionCallParser(self.tools, "muse").parse_non_stream(
                remainder
            )
            self.assertEqual(
                calls, [], f"quoted header parsed as a call ({quoted_at!r})"
            )
            self.assertIn("<atem:invoke", content)
            for chunk_size in (1, 7, 100):
                _, s_content, s_calls = self.pipeline_stream(raw, chunk_size)
                self.assertEqual(s_calls, [])
                self.assertIn("<atem:invoke", s_content)

    def test_pipeline_preamble_before_tool_call_streams(self):
        """A real ``to=user`` message may precede the tool channel; its
        terminator is what re-arms the header state. Goes red if the hand-off
        drops the ``to=user`` terminator again."""
        raw = (
            f" to=self<|message|>r<|eom|><|start|>assistant to=user<|message|>"
            f"Let me check.<|eom|><|start|>assistant to={DOUBLED}<|message|>"
            f"{atem(DOUBLED, city='Paris')}"
        )
        want_calls = [("get_weather", {"city": "Paris"})]
        _, remainder = ReasoningParser(
            "muse", tool_call_parser_active=True
        ).parse_non_stream(raw)
        content, calls = FunctionCallParser(self.tools, "muse").parse_non_stream(
            remainder
        )
        self.assertEqual(
            [(c.name, json.loads(c.parameters)) for c in calls], want_calls
        )
        self.assertIn("Let me check.", content)
        for chunk_size in (1, 7, 100):
            _, s_content, s_calls = self.pipeline_stream(raw, chunk_size)
            self.assertEqual(s_calls, want_calls, f"chunk_size={chunk_size}")
            self.assertIn("Let me check.", s_content)

    def test_pipeline_plain_answer_stays_clean_without_tool_parse(self):
        """Serving skips the tool detector when ``has_tool_call()`` is false, so
        a turn with no ATEM block must come out of the reasoning parser already
        unwrapped. Goes red if framing is preserved unconditionally."""
        raw = (
            " to=self<|message|>r<|eom|>"
            "<|start|>assistant to=user<|message|>Hello there."
        )
        reasoning, remainder = ReasoningParser(
            "muse", tool_call_parser_active=True
        ).parse_non_stream(raw)
        self.assertFalse(
            FunctionCallParser(self.tools, "muse").has_tool_call(remainder)
        )
        self.assertEqual(reasoning, "r")
        self.assertEqual(remainder, "Hello there.")

    def test_reasoning_finish_flushes_unframed_text(self):
        """An unframed turn never emits ``<|message|>``, so nothing leaves the
        buffer until the end-of-stream flush. Goes red if the Muse Glimmer reasoning
        detector loses its ``finish()`` override."""
        rp = ReasoningParser("muse")
        _, streamed = rp.parse_stream_chunk("Just plain text.")
        _, flushed = rp.parse_stream_end()
        self.assertEqual((streamed or "") + (flushed or ""), "Just plain text.")

    def test_interleaved_reasoning_blocks_join_with_newline(self):
        """A turn may reason, call a tool, then reason again; the reference
        schema joins the blocks with a newline. Goes red if the reasoning
        detector concatenates the bodies directly, gluing two thoughts into
        one word ("...thoughtsecond...")."""
        raw = (
            f" to=self<|message|>first thought<|eom|>"
            f"<|start|>assistant to={DOUBLED}<|message|>"
            f"{atem(DOUBLED, city='Paris')}<|eom|>"
            f"<|start|>assistant to=self<|message|>second thought<|eom|>"
            f"<|start|>assistant to=user<|message|>done"
        )
        reasoning, _ = ReasoningParser(
            "muse", tool_call_parser_active=True
        ).parse_non_stream(raw)
        self.assertEqual(reasoning, "first thought\nsecond thought")
        for chunk_size in (1, 7, 100):
            s_reasoning, s_content, s_calls = self.pipeline_stream(raw, chunk_size)
            self.assertEqual(s_reasoning, "first thought\nsecond thought")
            self.assertEqual(s_calls, [("get_weather", {"city": "Paris"})])
            self.assertIn("done", s_content)

    def test_stream_end_flushes_partial_marker(self):
        """An answer ending in a marker prefix (``<|st``) is held back while
        streaming in case it grows into ``<|start|>``; the stream's end proves
        it never will. Goes red if the tool parser loses its stream-end flush
        (``parse_stream_end`` / detector ``finish``)."""
        raw = (
            " to=self<|message|>r<|eom|>"
            "<|start|>assistant to=user<|message|>answer<|st"
        )
        for chunk_size in (1, 7, 100):
            _, content, calls = self.pipeline_stream(raw, chunk_size)
            self.assertEqual(calls, [])
            self.assertEqual(content, "answer<|st", f"chunk_size={chunk_size}")

    def test_whitespace_before_header_is_tolerated(self):
        """The model may put whitespace between ``<|eom|>`` and the next
        ``<|start|>``; it travels with the header text. Goes red if the header
        state requires ``<|start|>`` at exactly the first byte again."""
        text = (
            f" to=self<|message|>r<|eom|>\n<|start|>assistant to={DOUBLED}"
            f"<|message|>{atem(DOUBLED, city='Paris')}"
        )
        _, calls = self.parse(text)
        self.assertEqual(calls, [("get_weather", {"city": "Paris"})])
        self.assert_streaming_matches(text)

    def test_registered_in_parser_enum(self):
        self.assertIs(
            FunctionCallParser.ToolCallParserEnum["muse"], MuseGlimmerDetector
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
