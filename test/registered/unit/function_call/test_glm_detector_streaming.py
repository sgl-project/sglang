"""Unit tests for the GLM tool-call detectors — no server, no model loading.

These focus on streaming/non-streaming equivalence: the concatenation of the
streamed argument deltas must parse to the same JSON that `detect_and_parse`
produces for the same text.
"""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _mk_tool(name, properties):
    return Tool(
        type="function",
        function=Function(
            name=name, parameters={"type": "object", "properties": properties}
        ),
    )


# "typed" declares a schema for every argument; "untyped" declares none, which
# is what exposes the value-type inference path.
TOOLS = [
    _mk_tool(
        "typed",
        {
            "s": {"type": "string"},
            "n": {"type": "number"},
            "b": {"type": "boolean"},
            "o": {"type": "object"},
            "a": {"type": "array"},
        },
    ),
    _mk_tool("untyped", {}),
]


class _GlmStreamingEquivalenceMixin:
    """Shared assertions; subclasses provide detector_cls and build_text."""

    detector_cls = None

    def build_text(self, func_name, pairs):
        raise NotImplementedError

    def _stream_args(self, text):
        """Feed text one character at a time, return {tool_index: (name, args)}."""
        detector = self.detector_cls()
        acc, order = {}, []
        for ch in list(text) + ["", ""]:  # trailing flushes, as serving does
            for call in detector.parse_streaming_increment(ch, TOOLS).calls:
                if call.tool_index not in acc:
                    acc[call.tool_index] = ["", ""]
                    order.append(call.tool_index)
                if call.name:
                    acc[call.tool_index][0] = call.name
                if call.parameters:
                    acc[call.tool_index][1] += call.parameters
        return [(acc[i][0], acc[i][1]) for i in order]

    def assert_stream_matches_final(self, func_name, pairs):
        text = self.build_text(func_name, pairs)

        final = [
            (c.name, c.parameters)
            for c in self.detector_cls().detect_and_parse(text, TOOLS).calls
        ]
        streamed = self._stream_args(text)

        self.assertEqual(len(streamed), len(final), f"call count differs for {text!r}")
        for (s_name, s_args), (f_name, f_args) in zip(streamed, final):
            self.assertEqual(s_name, f_name)
            # Streamed arguments must be valid JSON on their own ...
            parsed_streamed = json.loads(s_args) if s_args else None
            parsed_final = json.loads(f_args) if f_args else None
            # ... and must agree with the non-streaming result, types included
            # (123 must not become "123").
            self.assertEqual(
                parsed_streamed,
                parsed_final,
                f"streamed args {s_args!r} != final args {f_args!r} for {text!r}",
            )

    # ---- schema-declared types: must keep char-by-char streaming behavior ----

    def test_typed_string(self):
        self.assert_stream_matches_final("typed", [("s", "hello")])

    def test_typed_number(self):
        self.assert_stream_matches_final("typed", [("n", "42")])

    def test_typed_number_negative(self):
        self.assert_stream_matches_final("typed", [("n", "-7")])

    def test_typed_number_float(self):
        self.assert_stream_matches_final("typed", [("n", "3.14")])

    def test_typed_boolean(self):
        self.assert_stream_matches_final("typed", [("b", "true")])

    def test_typed_array(self):
        self.assert_stream_matches_final("typed", [("a", "[1, 2]")])

    def test_typed_empty_string(self):
        self.assert_stream_matches_final("typed", [("s", "")])

    def test_typed_multiple_arguments(self):
        self.assert_stream_matches_final("typed", [("s", "abc"), ("n", "5")])

    def test_typed_object_closing_brace(self):
        # An object-valued last argument makes the streamed buffer end with "}"
        # while the outer argument object is still open, so the outer closing
        # brace must not be skipped.
        self.assert_stream_matches_final("typed", [("o", '{"k": 1}')])

    # ---- no schema type: value must be inferred from the complete value ----

    def test_untyped_int_stays_int(self):
        self.assert_stream_matches_final("untyped", [("x", "123")])

    def test_untyped_negative_int_stays_int(self):
        self.assert_stream_matches_final("untyped", [("x", "-123")])

    def test_untyped_float_stays_float(self):
        self.assert_stream_matches_final("untyped", [("x", "1.5")])

    def test_untyped_bool_stays_bool(self):
        self.assert_stream_matches_final("untyped", [("x", "true")])

    def test_untyped_plain_text_stays_string(self):
        self.assert_stream_matches_final("untyped", [("x", "hello world")])

    def test_untyped_text_starting_with_digit_stays_string(self):
        # Must not be mistaken for a number just because it starts with "1".
        self.assert_stream_matches_final("untyped", [("x", "123abc")])

    def test_untyped_object(self):
        self.assert_stream_matches_final("untyped", [("x", '{"k": 1}')])

    def test_untyped_array(self):
        self.assert_stream_matches_final("untyped", [("x", "[1, 2]")])

    def test_untyped_empty_value(self):
        self.assert_stream_matches_final("untyped", [("x", "")])

    def test_untyped_unicode(self):
        self.assert_stream_matches_final("untyped", [("x", "北京")])

    def test_untyped_mixed_arguments(self):
        self.assert_stream_matches_final("untyped", [("x", "7"), ("y", "text")])


class TestGlm4MoeDetectorStreaming(_GlmStreamingEquivalenceMixin, CustomTestCase):
    """Covers the parsers registered as "glm" and "glm45"."""

    detector_cls = Glm4MoeDetector

    def build_text(self, func_name, pairs):
        body = "".join(
            f"<arg_key>{k}</arg_key>\n<arg_value>{v}</arg_value>\n" for k, v in pairs
        )
        return f"<tool_call>{func_name}\n{body}</tool_call>"


class TestGlm47MoeDetectorStreaming(_GlmStreamingEquivalenceMixin, CustomTestCase):
    """Covers the parser registered as "glm47"."""

    detector_cls = Glm47MoeDetector

    def build_text(self, func_name, pairs):
        body = "".join(
            f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in pairs
        )
        return f"<tool_call>{func_name}{body}</tool_call>"


if __name__ == "__main__":
    import unittest

    unittest.main()
