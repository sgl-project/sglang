"""Unit tests for InklingDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.inkling_detector import InklingDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestInklingDetector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="weather",
                    description="Lookup weather",
                    parameters={"type": "object"},
                ),
            )
        ]

    def test_canonical_header_is_not_visible_content(self):
        detector = InklingDetector()
        source = (
            "<|message_model|>weather<|content_invoke_tool_json|>"
            '{"name":"weather","args":{"city":"SF"}}<|end_message|>'
        )
        result = detector.detect_and_parse(source, self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "SF"})

    def test_streaming_header_is_buffered_until_the_tool_kind(self):
        detector = InklingDetector()
        chunks = [
            "<|message_model|>",
            "weat",
            "her",
            "<|content_invoke_tool_json|>",
            '{"name":"weather",',
            '"args":{"city":"SF"}}',
            "<|end_message|>",
        ]
        normal_text = ""
        name = None
        parameters = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_text += result.normal_text
            for call in result.calls:
                name = call.name or name
                parameters += call.parameters
        self.assertEqual(normal_text, "")
        self.assertEqual(name, "weather")
        self.assertEqual(json.loads(parameters), {"city": "SF"})

    def test_header_name_is_ignored_and_payload_name_wins(self):
        """The message header is author metadata, not a name check: a header
        that differs from the payload name still yields a call named by the
        payload."""
        detector = InklingDetector()
        source = (
            "<|message_model|>other<|content_invoke_tool_json|>"
            '{"name":"weather","args":{}}<|end_message|>'
        )
        result = detector.detect_and_parse(source, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "weather")
        self.assertEqual(result.normal_text, "")

    def test_raw_fallback_strips_protocol_tokens(self):
        """When a payload cannot be parsed or recovered, the visible text is
        surfaced as content with the <|...|> special tokens stripped."""
        detector = InklingDetector()
        source = (
            "<|message_model|>weather<|content_invoke_tool_json|>"
            "{not json at all<|end_message|>"
        )
        result = detector.detect_and_parse(source, self.tools)
        self.assertEqual(result.calls, [])
        self.assertNotIn("<|", result.normal_text)
        self.assertIn("{not json at all", result.normal_text)

    def test_headerless_legacy_tool_call_still_parses(self):
        """Spec tolerance: a bare <|content_invoke_tool_json|> block with no
        <|message_model|>name header (the pre-canonical form) must keep
        parsing in both modes."""
        source = '<|content_invoke_tool_json|>{"name":"weather","args":{"city":"SF"}}<|end_message|>'
        result = InklingDetector().detect_and_parse(source, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "weather")

        streaming = InklingDetector()
        name = None
        parameters = ""
        for char in source:
            for call in streaming.parse_streaming_increment(char, self.tools).calls:
                name = call.name or name
                parameters += call.parameters
        self.assertEqual(name, "weather")
        self.assertEqual(json.loads(parameters), {"city": "SF"})

    def test_streaming_two_sequential_tool_calls_get_distinct_indices(self):
        """Coverage for multi-call responses: two back-to-back canonical tool
        calls must stream as tool_index 0 and 1 with per-call args."""
        detector = InklingDetector()
        source = (
            "<|message_model|>weather<|content_invoke_tool_json|>"
            '{"name":"weather","args":{"city":"SF"}}<|end_message|>'
            "<|message_model|>weather<|content_invoke_tool_json|>"
            '{"name":"weather","args":{"city":"NY"}}<|end_message|>'
        )
        args_by_index: dict = {}
        for char in source:
            for call in detector.parse_streaming_increment(char, self.tools).calls:
                args_by_index[call.tool_index] = (
                    args_by_index.get(call.tool_index, "") + call.parameters
                )
        self.assertEqual(sorted(args_by_index), [0, 1])
        self.assertEqual(json.loads(args_by_index[0]), {"city": "SF"})
        self.assertEqual(json.loads(args_by_index[1]), {"city": "NY"})

    def test_streaming_two_complete_tool_calls_in_one_delta_both_emit(self):
        """Bug regression: parse_streaming_increment parsed one call per delta
        and re-buffered the rest, relying on the NEXT delta to drain it. Two
        complete calls arriving in a single (e.g. final) delta left the second
        stranded in the buffer with no stream-end flush, so only the first was
        emitted."""
        detector = InklingDetector()
        source = (
            "<|message_model|>weather<|content_invoke_tool_json|>"
            '{"name":"weather","args":{"city":"SF"}}<|end_message|>'
            "<|message_model|>weather<|content_invoke_tool_json|>"
            '{"name":"weather","args":{"city":"NY"}}<|end_message|>'
        )
        args_by_index: dict = {}
        for call in detector.parse_streaming_increment(source, self.tools).calls:
            args_by_index[call.tool_index] = (
                args_by_index.get(call.tool_index, "") + call.parameters
            )
        self.assertEqual(sorted(args_by_index), [0, 1])
        self.assertEqual(json.loads(args_by_index[0]), {"city": "SF"})
        self.assertEqual(json.loads(args_by_index[1]), {"city": "NY"})

    def test_streaming_text_then_tool_call_in_one_delta_emits_both(self):
        """Bug regression: a delta carrying visible text followed by a complete
        tool call emitted only the text and stranded the call in the buffer
        (the drain loop stopped after the leading-text run), so a final such
        delta dropped the call. The drain must continue past leading text."""
        detector = InklingDetector()
        source = (
            "Sure, let me check.<|message_model|>weather<|content_invoke_tool_json|>"
            '{"name":"weather","args":{"city":"SF"}}<|end_message|>'
        )
        result = detector.parse_streaming_increment(source, self.tools)
        self.assertIn("Sure, let me check.", result.normal_text)
        names = [c.name for c in result.calls if c.name]
        self.assertEqual(names, ["weather"])
        args = "".join(c.parameters for c in result.calls)
        self.assertEqual(json.loads(args), {"city": "SF"})

    def test_streaming_differing_headers_all_stream(self):
        """The header is author metadata, not a name gate: three calls with
        differing headers all stream, indexed 0/1/2 by payload name."""
        detector = InklingDetector()
        source = (
            "<|message_model|>weather<|content_invoke_tool_json|>"
            '{"name":"weather","args":{"city":"SF"}}<|end_message|>'
            "<|message_model|>other<|content_invoke_tool_json|>"
            '{"name":"weather","args":{"city":"XX"}}<|end_message|>'
            "<|message_model|>weather<|content_invoke_tool_json|>"
            '{"name":"weather","args":{"city":"NY"}}<|end_message|>'
        )
        args_by_index: dict = {}
        for call in detector.parse_streaming_increment(source, self.tools).calls:
            args_by_index[call.tool_index] = (
                args_by_index.get(call.tool_index, "") + call.parameters
            )
        self.assertEqual(sorted(args_by_index), [0, 1, 2])
        self.assertEqual(json.loads(args_by_index[0]), {"city": "SF"})
        self.assertEqual(json.loads(args_by_index[1]), {"city": "XX"})
        self.assertEqual(json.loads(args_by_index[2]), {"city": "NY"})

    def test_streaming_malformed_call_switches_to_raw_passthrough(self):
        """A call that fails to frame switches the stream to raw passthrough:
        earlier calls stay emitted (streaming cannot un-emit), and everything
        after the failure is surfaced as content, never as further calls."""
        detector = InklingDetector()
        chunks = [
            "<|message_model|>weather<|content_invoke_tool_json|>",
            '{"name":"weather","args":{"city":"SF"}}<|end_message|>',
            # unrecoverable -> raw passthrough from here on
            "<|message_model|>weather<|content_invoke_tool_json|>",
            "{not json at all<|end_message|>",
            # would-be call, now passthrough text
            "<|message_model|>weather<|content_invoke_tool_json|>",
            '{"name":"weather","args":{"city":"LA"}}<|end_message|>',
        ]
        calls: list = []
        normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_text += result.normal_text
            calls.extend(result.calls)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "weather")
        self.assertEqual(json.loads(calls[0].parameters), {"city": "SF"})
        self.assertNotIn("<|", normal_text)
        self.assertIn("{not json at all", normal_text)
        self.assertIn("LA", normal_text)

    def test_undeclared_tool_name_is_surfaced(self):
        """A call to a tool absent from the request's tool list surfaces as a
        structured tool_call (OpenAI behavior for hallucinated tools) so agent
        harnesses can return a tool error and let the model self-correct,
        instead of the serialized invocation becoming terminal answer text."""
        detector = InklingDetector()
        source = (
            "<|message_model|>document_search<|content_invoke_tool_json|>"
            '{"name":"document_search","args":{"query":"q"}}<|end_message|>'
        )
        result = detector.detect_and_parse(source, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "document_search")
        self.assertEqual(json.loads(result.calls[0].parameters), {"query": "q"})
        self.assertNotIn("<|", result.normal_text)

    def test_undeclared_tool_name_surfaces_in_streaming(self):
        detector = InklingDetector()
        source = (
            "<|message_model|>document_search<|content_invoke_tool_json|>"
            '{"name":"document_search","args":{"query":"q"}}<|end_message|>'
        )
        normal_text = ""
        name = None
        parameters = ""
        for char in source:
            result = detector.parse_streaming_increment(char, self.tools)
            normal_text += result.normal_text
            for call in result.calls:
                name = call.name or name
                parameters += call.parameters
        self.assertEqual(name, "document_search")
        self.assertEqual(json.loads(parameters), {"query": "q"})
        self.assertNotIn("<|", normal_text)

    def test_malformed_json_surfaces_as_raw_fallback(self):
        """Malformed JSON that also fails recovery surfaces the visible payload
        as content (special tokens stripped), not a tool call."""
        detector = InklingDetector()
        source = (
            "<|message_model|>weather<|content_invoke_tool_json|>"
            "{not json at all<|end_message|>"
        )
        result = detector.detect_and_parse(source, self.tools)
        self.assertEqual(result.calls, [])
        self.assertNotIn("<|", result.normal_text)
        self.assertIn("{not json at all", result.normal_text)

    def test_parser_preserves_raw_fallback_text(self):
        """The parser wrapper preserves the detector's raw fallback, so the
        visible prefix plus the failed payload reach the caller as content."""
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        source = (
            "Visible prefix."
            "<|message_model|>weather<|content_invoke_tool_json|>"
            "{not json at all<|end_message|>"
        )
        normal_text, calls = FunctionCallParser(self.tools, "inkling").parse_non_stream(
            source
        )
        self.assertEqual(calls, [])
        self.assertTrue(normal_text.startswith("Visible prefix."))
        self.assertIn("{not json at all", normal_text)

    def test_parser_preserves_text_without_tool_call_marker(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        source = "  Ordinary assistant text.  "
        normal_text, calls = FunctionCallParser(self.tools, "inkling").parse_non_stream(
            source
        )
        self.assertEqual(normal_text, source)
        self.assertEqual(calls, [])

    def test_one_malformed_call_fails_the_whole_batch(self):
        """All-or-nothing: a single unrecoverable call fails canonical framing
        for the whole response, so even an earlier valid call is discarded and
        the visible text is surfaced as content."""
        source = (
            "<|message_model|>weather<|content_invoke_tool_json|>"
            '{"name":"weather","args":{"city":"SF"}}<|end_message|>'
            "<|message_model|>weather<|content_invoke_tool_json|>"
            "{not json at all<|end_message|>"
        )
        result = InklingDetector().detect_and_parse(source, self.tools)
        self.assertEqual(result.calls, [])
        self.assertNotIn("<|", result.normal_text)
        self.assertIn('{"name":"weather","args":{"city":"SF"}}', result.normal_text)
        self.assertIn("{not json at all", result.normal_text)

    def test_clean_normal_text_strips_the_full_control_alphabet(self):
        """Fall-through text is cleaned against the whole shared control-token
        alphabet, not a hand-picked subset."""
        detector = InklingDetector()
        source = (
            "<|message_model|><|content_thinking|>leak<|end_message|>"
            "<|message_user|>x<|content_audio_input|><|audio_end|>"
        )
        result = detector.detect_and_parse(source, self.tools)
        self.assertNotIn("<|", result.normal_text)

    def test_structural_tag_uses_the_canonical_header(self):
        info = InklingDetector().structure_info()("weather")
        header = "<|message_model|>weather<|content_invoke_tool_json|>"
        self.assertEqual(info.trigger, header)
        self.assertTrue(info.begin.startswith(header + '{"name":"weather"'))

    def test_content_after_tool_call_is_preserved(self):
        """A tool call followed by a text block returns both: the call plus the
        trailing visible content, not just the prefix before the marker."""
        source = (
            "<|message_model|>weather<|content_invoke_tool_json|>"
            '{"name":"weather","args":{"city":"SF"}}<|end_message|>'
            "<|message_model|><|content_text|>Here you go.<|end_message|>"
        )
        result = InklingDetector().detect_and_parse(source, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "weather")
        self.assertEqual(result.normal_text, "Here you go.")

    def test_empty_name_is_allowed_on_the_canonical_path(self):
        source = "<|content_invoke_tool_json|>" '{"name":"","args":{}}<|end_message|>'
        result = InklingDetector().detect_and_parse(source, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "")

    def test_recovery_uses_only_the_last_marker(self):
        """Canonical framing fails on the garbage payload; recovery reads only
        the payload after the LAST marker."""
        source = (
            "<|message_model|>weather<|content_invoke_tool_json|>garbage"
            "<|content_invoke_tool_json|>"
            '{"name":"weather","args":{"city":"SF"}}<|end_message|>'
        )
        result = InklingDetector().detect_and_parse(source, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "SF"})

    def test_recovery_requires_a_nonempty_name(self):
        """Recovery (unlike the canonical path) rejects an empty name, falling
        through to raw text."""
        source = (
            "<|message_model|>weather<|content_invoke_tool_json|>bad"
            '<|content_invoke_tool_json|>{"name":"","args":{}}<|end_message|>'
        )
        result = InklingDetector().detect_and_parse(source, self.tools)
        self.assertEqual(result.calls, [])
        self.assertNotIn("<|", result.normal_text)

    def test_nonfinite_numbers_rejected_canonically_but_recovered(self):
        """NaN/Infinity are not valid canonical JSON, so the strict pass fails;
        recovery accepts them."""
        source = (
            "<|content_invoke_tool_json|>"
            '{"name":"weather","args":{"v":NaN}}<|end_message|>'
        )
        result = InklingDetector().detect_and_parse(source, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "weather")

    def test_streaming_name_not_emitted_before_end_message(self):
        """Atomicity: the tool name is withheld until the closing marker, so a
        call that never completes never leaks an orphan name delta."""
        detector = InklingDetector()
        pre = detector.parse_streaming_increment(
            '<|message_model|>weather<|content_invoke_tool_json|>{"name":"wea',
            self.tools,
        )
        self.assertEqual(pre.calls, [])
        post = detector.parse_streaming_increment(
            'ther","args":{"city":"SF"}}<|end_message|>', self.tools
        )
        self.assertEqual(len(post.calls), 1)
        self.assertEqual(post.calls[0].name, "weather")
        self.assertEqual(json.loads(post.calls[0].parameters), {"city": "SF"})

    def test_raw_text_tool_invocation_surfaces_as_a_call(self):
        """A headerless <|content_invoke_tool_text|> block reaches the tool loop
        as a call carrying the raw body, instead of being dropped."""
        source = "<|content_invoke_tool_text|>search the web<|end_message|>"
        result = InklingDetector().detect_and_parse(source, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"text": "search the web"}
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
