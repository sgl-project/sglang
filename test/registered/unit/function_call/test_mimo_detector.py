"""Unit tests for MiMoDetector — no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mimo_detector import (
    MiMoDetector,
    _convert_param_value,
    _get_param_type,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


def _tool(name: str, properties: dict, required: list | None = None) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters={
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        ),
    )


class TestGetParamType(CustomTestCase):
    """_get_param_type looks up a param's declared type from tool schema."""

    def test_returns_declared_type(self):
        tools = [_tool("f", {"x": {"type": "integer"}})]
        self.assertEqual(_get_param_type("f", "x", tools), "integer")

    def test_defaults_to_string_for_missing_tool(self):
        tools = [_tool("f", {"x": {"type": "integer"}})]
        self.assertEqual(_get_param_type("other_func", "x", tools), "string")

    def test_defaults_to_string_for_missing_param(self):
        tools = [_tool("f", {"x": {"type": "integer"}})]
        self.assertEqual(_get_param_type("f", "missing_param", tools), "string")

    def test_defaults_to_string_when_type_field_absent(self):
        tools = [_tool("f", {"x": {"description": "no type field"}})]
        self.assertEqual(_get_param_type("f", "x", tools), "string")


class TestConvertParamValue(CustomTestCase):
    """_convert_param_value coerces a raw XML-extracted string into a typed value.

    The schema lookup happens inside; these tests cover every type branch plus
    the silent-degradation fallbacks that apply when a value can't be coerced.
    """

    def setUp(self):
        self.tools = [
            _tool(
                "f",
                {
                    "s": {"type": "string"},
                    "i": {"type": "integer"},
                    "n": {"type": "number"},
                    "fl": {"type": "float"},
                    "b": {"type": "boolean"},
                    "o": {"type": "object"},
                    "a": {"type": "array"},
                    "untyped": {"description": "no type"},
                },
            ),
        ]

    # --- null handling (applies before type dispatch) ---
    def test_null_lowercase_returns_none(self):
        self.assertIsNone(_convert_param_value("null", "s", "f", self.tools))

    def test_null_uppercase_returns_none(self):
        # Case-insensitive: "NULL"/"Null"/"nUll" all match
        self.assertIsNone(_convert_param_value("NULL", "i", "f", self.tools))
        self.assertIsNone(_convert_param_value("Null", "b", "f", self.tools))

    def test_null_short_circuits_before_type_coercion(self):
        # Even for integer type, the literal "null" should still become None
        # rather than raising / degrading.
        self.assertIsNone(_convert_param_value("null", "i", "f", self.tools))

    # --- HTML unescaping runs before everything else ---
    def test_html_entities_are_unescaped(self):
        self.assertEqual(
            _convert_param_value("a &amp; b", "s", "f", self.tools), "a & b"
        )

    def test_html_unescape_applies_to_quotes(self):
        self.assertEqual(
            _convert_param_value("&quot;hi&quot;", "s", "f", self.tools), '"hi"'
        )

    # --- string type ---
    def test_string_passthrough(self):
        self.assertEqual(_convert_param_value("hello", "s", "f", self.tools), "hello")

    # --- integer family ---
    def test_integer_valid(self):
        self.assertEqual(_convert_param_value("42", "i", "f", self.tools), 42)

    def test_integer_negative(self):
        self.assertEqual(_convert_param_value("-7", "i", "f", self.tools), -7)

    def test_integer_invalid_degenerates_to_string(self):
        # "abc" can't be int() — code logs warning and returns the raw string
        self.assertEqual(_convert_param_value("abc", "i", "f", self.tools), "abc")

    def test_uint_prefix_dispatches_to_integer(self):
        tools = [_tool("f", {"x": {"type": "uint32"}})]
        self.assertEqual(_convert_param_value("42", "x", "f", tools), 42)

    def test_long_type_dispatches_to_integer(self):
        tools = [_tool("f", {"x": {"type": "long"}})]
        self.assertEqual(_convert_param_value("42", "x", "f", tools), 42)

    # --- number / float family ---
    def test_number_with_fraction_stays_float(self):
        self.assertEqual(_convert_param_value("3.14", "n", "f", self.tools), 3.14)

    def test_number_whole_value_collapses_to_int(self):
        # "3.0" has zero fractional part → returned as int(3), not float(3.0).
        # This matters because downstream JSON serialization will then emit `3`
        # instead of `3.0`, which can surprise callers.
        result = _convert_param_value("3.0", "n", "f", self.tools)
        self.assertEqual(result, 3)
        self.assertIsInstance(result, int)

    def test_number_bare_integer_literal_collapses_to_int(self):
        # "3" → float(3) → 3.0 → int(3)
        result = _convert_param_value("3", "n", "f", self.tools)
        self.assertEqual(result, 3)
        self.assertIsInstance(result, int)

    def test_number_invalid_degenerates_to_string(self):
        self.assertEqual(_convert_param_value("abc", "n", "f", self.tools), "abc")

    def test_float_type_dispatches_to_number_path(self):
        self.assertEqual(_convert_param_value("2.5", "fl", "f", self.tools), 2.5)

    # --- boolean ---
    def test_boolean_true(self):
        self.assertIs(_convert_param_value("true", "b", "f", self.tools), True)

    def test_boolean_true_uppercase(self):
        self.assertIs(_convert_param_value("TRUE", "b", "f", self.tools), True)

    def test_boolean_false(self):
        self.assertIs(_convert_param_value("false", "b", "f", self.tools), False)

    def test_boolean_invalid_silently_becomes_false(self):
        # Current behavior: anything not in {"true","false"} (after lowercasing)
        # logs a warning and returns False. Guards against regressions that
        # accidentally raise or return the raw string.
        self.assertIs(_convert_param_value("yes", "b", "f", self.tools), False)

    # --- object / array / dict / list — JSON first, ast fallback ---
    def test_object_valid_json_parses(self):
        result = _convert_param_value('{"k": "v"}', "o", "f", self.tools)
        self.assertEqual(result, {"k": "v"})

    def test_array_valid_json_parses(self):
        result = _convert_param_value("[1, 2, 3]", "a", "f", self.tools)
        self.assertEqual(result, [1, 2, 3])

    def test_object_falls_back_to_ast_literal_for_python_repr(self):
        # Single-quoted Python repr is not valid JSON; ast.literal_eval catches it.
        # Regression: if the JSON-then-ast fallback breaks, this is the canary.
        result = _convert_param_value("{'k': 'v'}", "o", "f", self.tools)
        self.assertEqual(result, {"k": "v"})

    def test_object_degenerates_to_string_when_both_parsers_fail(self):
        # Neither json.loads nor ast.literal_eval can parse this
        result = _convert_param_value("{not valid", "o", "f", self.tools)
        self.assertEqual(result, "{not valid")

    # --- unknown type falls through to ast.literal_eval ---
    def test_unknown_type_uses_ast_literal_eval(self):
        tools = [_tool("f", {"x": {"type": "tuple"}})]
        result = _convert_param_value("(1, 2)", "x", "f", tools)
        self.assertEqual(result, (1, 2))

    def test_unknown_type_with_ast_failure_returns_original_string(self):
        tools = [_tool("f", {"x": {"type": "tuple"}})]
        # "abc" isn't a valid Python literal — ast.literal_eval raises,
        # code logs a warning and returns the raw (already html-unescaped) value.
        self.assertEqual(_convert_param_value("abc", "x", "f", tools), "abc")

    def test_untyped_param_defaults_to_string_path(self):
        # No "type" field in schema → _get_param_type returns "string" →
        # string path is taken and value passes through unchanged.
        self.assertEqual(
            _convert_param_value("raw value", "untyped", "f", self.tools),
            "raw value",
        )


class TestMiMoDetectorHasToolCall(CustomTestCase):
    def setUp(self):
        self.detector = MiMoDetector()

    def test_detects_start_token(self):
        self.assertTrue(
            self.detector.has_tool_call(
                "<tool_call><function=foo></function></tool_call>"
            )
        )

    def test_plain_text_is_not_a_tool_call(self):
        self.assertFalse(self.detector.has_tool_call("Just a plain answer."))

    def test_similar_but_wrong_tag_is_rejected(self):
        # Only the exact `<tool_call>` start token counts
        self.assertFalse(self.detector.has_tool_call("<tool>inner</tool>"))

    def test_start_token_alone_is_enough(self):
        # has_tool_call only checks for the start token; end token is not required
        # at this stage — useful for streaming where the end hasn't arrived yet.
        self.assertTrue(self.detector.has_tool_call("prefix <tool_call><function=f>"))


class TestMiMoDetectorDetectAndParse(CustomTestCase):
    def setUp(self):
        self.tools = [
            _tool(
                "execute_bash",
                {"command": {"type": "string"}},
                required=["command"],
            ),
            _tool(
                "get_weather",
                {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "verbose": {"type": "boolean"},
                },
            ),
        ]
        self.detector = MiMoDetector()

    def test_single_call_is_parsed(self):
        text = (
            "<tool_call>\n<function=execute_bash>\n"
            "<parameter=command>pwd && ls</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "execute_bash")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["command"], "pwd && ls")

    def test_multiple_calls_accumulate(self):
        text = (
            "<tool_call><function=execute_bash>"
            "<parameter=command>ls</parameter>"
            "</function></tool_call>"
            "<tool_call><function=get_weather>"
            "<parameter=city>Tokyo</parameter>"
            "</function></tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(
            [c.name for c in result.calls], ["execute_bash", "get_weather"]
        )

    def test_leading_normal_text_preserved(self):
        text = (
            "Sure, running now. "
            "<tool_call><function=execute_bash>"
            "<parameter=command>ls</parameter>"
            "</function></tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "Sure, running now. ")
        self.assertEqual(len(result.calls), 1)

    def test_plain_text_yields_no_calls(self):
        result = self.detector.detect_and_parse("Just prose.", self.tools)
        self.assertEqual(result.normal_text, "Just prose.")
        self.assertEqual(len(result.calls), 0)

    def test_parameter_types_are_coerced_per_schema(self):
        text = (
            "<tool_call><function=get_weather>"
            "<parameter=city>Berlin</parameter>"
            "<parameter=days>5</parameter>"
            "<parameter=verbose>true</parameter>"
            "</function></tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Berlin")
        self.assertEqual(args["days"], 5)  # integer coerced from "5"
        self.assertIs(args["verbose"], True)  # boolean coerced from "true"

    def test_unknown_function_is_dropped_from_calls_by_default(self):
        # SGLANG_FORWARD_UNKNOWN_TOOLS defaults to False → unknown funcs skipped
        text = (
            "<tool_call><function=unknown_fn>"
            "<parameter=x>1</parameter>"
            "</function></tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_html_entities_in_values_are_decoded(self):
        text = (
            "<tool_call><function=execute_bash>"
            "<parameter=command>echo &quot;hi&quot; &amp;&amp; exit</parameter>"
            "</function></tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["command"], 'echo "hi" && exit')

    def test_function_tag_missing_yields_no_call(self):
        # <tool_call>...</tool_call> with no inner <function=...> block —
        # _parse_tool_call returns None and the call is silently skipped.
        text = "<tool_call>no function here</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_parameterless_function_call(self):
        # <function=name></function> with no <parameter=...> — valid call, empty args.
        text = "<tool_call><function=execute_bash></function></tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "execute_bash")
        self.assertEqual(json.loads(result.calls[0].parameters), {})


class TestMiMoDetectorStreaming(CustomTestCase):
    def setUp(self):
        self.tools = [
            _tool("execute_bash", {"command": {"type": "string"}}),
        ]

    def test_plain_text_passes_through(self):
        detector = MiMoDetector()
        result = detector.parse_streaming_increment("Hello world.", self.tools)
        self.assertEqual(result.normal_text, "Hello world.")
        self.assertEqual(len(result.calls), 0)

    def test_complete_call_in_single_chunk(self):
        detector = MiMoDetector()
        text = (
            "<tool_call><function=execute_bash>"
            "<parameter=command>ls</parameter>"
            "</function></tool_call>"
        )
        result = detector.parse_streaming_increment(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "execute_bash")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["command"], "ls")

    def test_call_split_across_chunks_reassembles(self):
        detector = MiMoDetector()
        chunks = [
            "<tool_call><function=execute",
            "_bash><parameter=command>ls",
            " -la</parameter></function>",
            "</tool_call>",
        ]
        all_calls = []
        all_normal = ""
        for chunk in chunks:
            r = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(r.calls)
            all_normal += r.normal_text
        self.assertEqual(len(all_calls), 1)
        self.assertEqual(all_calls[0].name, "execute_bash")
        args = json.loads(all_calls[0].parameters)
        self.assertEqual(args["command"], "ls -la")
        self.assertEqual(all_normal, "")

    def test_leading_text_emitted_separately_from_call(self):
        detector = MiMoDetector()
        chunks = [
            "Let me run that. ",
            (
                "<tool_call><function=execute_bash>"
                "<parameter=command>ls</parameter>"
                "</function></tool_call>"
            ),
        ]
        all_calls = []
        all_normal = ""
        for chunk in chunks:
            r = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(r.calls)
            all_normal += r.normal_text
        self.assertEqual(all_normal, "Let me run that. ")
        self.assertEqual(len(all_calls), 1)

    def test_incomplete_call_is_buffered_not_emitted(self):
        # Start token arrives but end token doesn't — detector should emit
        # nothing (empty calls, empty normal_text) and hold the partial in
        # its internal buffer for the next chunk.
        detector = MiMoDetector()
        result = detector.parse_streaming_increment(
            "<tool_call><function=execute_bash>", self.tools
        )
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "")

    def test_incomplete_call_completes_on_next_chunk(self):
        detector = MiMoDetector()
        r1 = detector.parse_streaming_increment(
            "<tool_call><function=execute_bash>" "<parameter=command>ls</parameter>",
            self.tools,
        )
        self.assertEqual(len(r1.calls), 0)

        r2 = detector.parse_streaming_increment("</function></tool_call>", self.tools)
        self.assertEqual(len(r2.calls), 1)
        self.assertEqual(r2.calls[0].name, "execute_bash")

    def test_tool_index_advances_across_streamed_calls(self):
        # First call gets tool_index=0, second call gets tool_index=1 —
        # regression guard on the current_tool_id bookkeeping.
        detector = MiMoDetector()
        first = (
            "<tool_call><function=execute_bash>"
            "<parameter=command>ls</parameter>"
            "</function></tool_call>"
        )
        second = (
            "<tool_call><function=execute_bash>"
            "<parameter=command>pwd</parameter>"
            "</function></tool_call>"
        )
        r1 = detector.parse_streaming_increment(first, self.tools)
        r2 = detector.parse_streaming_increment(second, self.tools)
        self.assertEqual(len(r1.calls), 1)
        self.assertEqual(len(r2.calls), 1)
        self.assertEqual(r1.calls[0].tool_index, 0)
        self.assertEqual(r2.calls[0].tool_index, 1)


class TestMiMoDetectorStructureInfo(CustomTestCase):
    def test_structure_info_not_implemented(self):
        # MiMo relies on its bespoke XML-ish grammar, not structural tags.
        with self.assertRaises(NotImplementedError):
            MiMoDetector().structure_info()

    def test_supports_structural_tag_is_false(self):
        self.assertFalse(MiMoDetector().supports_structural_tag())


if __name__ == "__main__":
    unittest.main()
