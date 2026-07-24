"""Unit tests for Step3Detector — no server, no model loading.

Focus: numeric-looking string argument IDs (e.g. "007", long digit strings)
must round-trip unchanged, while real numbers still parse to numbers.
"""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.step3_detector import Step3Detector, parse_arguments
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _wrap(func_name: str, params: dict) -> str:
    """Build a Step3 tool-call payload for the given function/params."""
    inner = "".join(
        f'<steptml:parameter name="{k}">{v}</steptml:parameter>' for k, v in params.items()
    )
    return (
        "<｜tool_calls_begin｜>"
        "<｜tool_call_begin｜>function<｜tool_sep｜>"
        f'<steptml:invoke name="{func_name}">{inner}</steptml:invoke>'
        "<｜tool_call_end｜>"
        "<｜tool_calls_end｜>"
    )


class TestStep3Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="lookup",
                    description="Look something up",
                    parameters={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "order_id": {"type": "string"},
                            "count": {"type": "number"},
                            "index": {"type": "integer"},
                        },
                        "required": ["user_id"],
                    },
                ),
            ),
        ]
        self.detector = Step3Detector()

    # -------- parse_arguments unit tests (table-driven) --------

    def test_parse_arguments_preserves_numeric_strings_without_type(self):
        # No arg_type / unknown type -> numeric-looking strings stay strings.
        # Two distinct paths are exercised here:
        #   - "7", "12345678901234567890", "1e5" parse cleanly to a number and
        #     are kept as strings by the new numeric-type guard (is_numeric=False).
        #   - "007" and "0042" have leading zeros: json.loads rejects them and
        #     ast.literal_eval raises SyntaxError in Python 3, so they never reach
        #     the guard and fall through to the ``return value, False`` fallback.
        #     They are covered here to lock in that both paths keep the string.
        cases = ["007", "12345678901234567890", "7", "1e5", "0042"]
        for value in cases:
            with self.subTest(value=value):
                parsed, ok = parse_arguments(value)
                self.assertEqual(parsed, value)
                self.assertIsInstance(parsed, str)

    def test_parse_arguments_non_numeric_typename_not_coerced(self):
        # Guard against prefix false positives: a bogus type name that merely
        # starts with a numeric prefix (e.g. "interval" / "internal_id") must
        # NOT coerce a numeric-looking string to a number.
        for bogus in ("interval", "internal_id", "string", "duration"):
            with self.subTest(arg_type=bogus):
                parsed, _ = parse_arguments("42", bogus)
                self.assertEqual(parsed, "42")
                self.assertIsInstance(parsed, str)

    def test_parse_arguments_coerces_when_type_numeric(self):
        parsed, _ = parse_arguments("7", "number")
        self.assertEqual(parsed, 7)
        self.assertIsInstance(parsed, int)
        parsed, _ = parse_arguments("3.5", "number")
        self.assertEqual(parsed, 3.5)
        parsed, _ = parse_arguments("42", "integer")
        self.assertEqual(parsed, 42)

    def test_parse_arguments_handles_list_type(self):
        # Union/nullable types are lists in JSON Schema, e.g. ["integer", "null"].
        parsed, _ = parse_arguments("7", ["integer", "null"])
        self.assertEqual(parsed, 7)
        self.assertIsInstance(parsed, int)
        # Non-numeric union -> numeric-looking string preserved.
        parsed, _ = parse_arguments("7", ["string", "null"])
        self.assertEqual(parsed, "7")
        self.assertIsInstance(parsed, str)

    def test_parse_arguments_still_parses_structured_and_bool(self):
        self.assertEqual(parse_arguments("[1, 2, 3]")[0], [1, 2, 3])
        self.assertEqual(parse_arguments('{"a": 1}')[0], {"a": 1})
        self.assertEqual(parse_arguments("true")[0], True)
        self.assertEqual(parse_arguments("hello")[0], "hello")

    def test_parse_arguments_pathological_nesting_does_not_crash(self):
        # Security: deeply-nested untrusted input makes json.loads raise
        # RecursionError, which escapes a narrow (ValueError, SyntaxError,
        # TypeError) tuple and would crash the worker (DoS). The broad
        # ``except Exception`` must catch it and fall through to the
        # ``return value, False`` fallback, preserving the raw string.
        value = "[" * 10000 + "]" * 10000
        parsed, ok = parse_arguments(value)
        self.assertEqual(parsed, value)
        self.assertFalse(ok)

    # -------- end-to-end detect_and_parse --------

    def test_string_typed_numeric_ids_preserved(self):
        text = _wrap(
            "lookup", {"user_id": "007", "order_id": "12345678901234567890"}
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["user_id"], "007")
        self.assertEqual(args["order_id"], "12345678901234567890")

    def test_number_typed_still_parses(self):
        text = _wrap("lookup", {"user_id": "abc", "count": "42", "index": "5"})
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["count"], 42)
        self.assertEqual(args["index"], 5)

    def test_no_tools_fallback_preserves_numeric_id(self):
        # Empty tools -> the generic (no-schema) parsing path; must not coerce.
        text = _wrap("lookup", {"user_id": "12345678901234567890"})
        result = self.detector.detect_and_parse(text, [])
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["user_id"], "12345678901234567890")

    # -------- streaming path --------

    def _collect_streaming_params(self, text: str, chunk_size: int = 5) -> dict:
        """Feed ``text`` to parse_streaming_increment in chunks, reassemble args."""
        params_buf = ""
        for i in range(0, len(text), chunk_size):
            result = self.detector.parse_streaming_increment(
                text[i : i + chunk_size], self.tools
            )
            for tc in result.calls:
                if tc.parameters:
                    params_buf += tc.parameters
        return json.loads(params_buf)

    def test_streaming_preserves_numeric_string_ids(self):
        # The streaming path (_parse_partial_tool_call) calls the same schema-aware
        # parse_arguments; a string-typed numeric ID must survive incremental parsing
        # while a number-typed field is still coerced.
        text = _wrap(
            "lookup", {"user_id": "007", "order_id": "12345678901234567890", "index": "5"}
        )
        args = self._collect_streaming_params(text)
        self.assertEqual(args["user_id"], "007")
        self.assertEqual(args["order_id"], "12345678901234567890")
        self.assertEqual(args["index"], 5)


if __name__ == "__main__":
    import unittest

    unittest.main()
