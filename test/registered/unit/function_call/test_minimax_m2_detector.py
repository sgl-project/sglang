"""Unit tests for MinimaxM2Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _minimax_call(func_name: str, params: dict[str, str]) -> str:
    """Build a well-formed MiniMax M2 tool-call block from raw string values."""
    param_lines = "".join(
        f'\n<parameter name="{k}">{v}</parameter>' for k, v in params.items()
    )
    return (
        f"<minimax:tool_call>\n"
        f'<invoke name="{func_name}">'
        f"{param_lines}\n"
        f"</invoke>\n"
        f"</minimax:tool_call>"
    )


class TestHasToolCall(CustomTestCase):
    def setUp(self):
        self.detector = MinimaxM2Detector()

    def test_returns_true_when_start_token_present(self):
        self.assertTrue(self.detector.has_tool_call("<minimax:tool_call>something"))

    def test_returns_false_for_plain_text(self):
        self.assertFalse(self.detector.has_tool_call("The weather is nice today."))

    def test_returns_false_for_empty_string(self):
        self.assertFalse(self.detector.has_tool_call(""))


class TestDetectAndParse(CustomTestCase):
    def setUp(self):
        self.detector = MinimaxM2Detector()
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
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search the web",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="calculate",
                    description="Run a calculation",
                    parameters={
                        "type": "object",
                        "properties": {
                            "value": {"type": "number"},
                            "count": {"type": "integer"},
                            "verbose": {"type": "boolean"},
                            "tags": {"type": "array"},
                            "meta": {"type": "object"},
                        },
                    },
                ),
            ),
        ]

    def test_single_call_single_param(self):
        text = _minimax_call("get_weather", {"city": "Beijing"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(result.normal_text, "")

    def test_single_call_multiple_params(self):
        text = _minimax_call("get_weather", {"city": "London", "unit": "celsius"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "London")
        self.assertEqual(args["unit"], "celsius")

    def test_leading_normal_text_is_preserved(self):
        prefix = "Let me look that up. "
        text = prefix + _minimax_call("search", {"query": "sglang"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        self.assertEqual(result.normal_text, prefix)

    def test_no_tool_call_returns_text_unchanged(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_multiple_sequential_calls(self):
        text = _minimax_call("get_weather", {"city": "Tokyo"}) + _minimax_call(
            "search", {"query": "restaurants"}
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        names = {c.name for c in result.calls}
        self.assertIn("get_weather", names)
        self.assertIn("search", names)

    def test_unknown_function_name_is_dropped(self):
        text = _minimax_call("nonexistent_func", {"x": "1"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_incomplete_block_no_closing_tag(self):
        # No </minimax:tool_call> — _extract treats the dangling block as normal text.
        text = '<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="city">Beijing</parameter>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertIn("<minimax:tool_call>", result.normal_text)

    def test_typed_params_integer_and_boolean(self):
        text = _minimax_call("calculate", {"count": "7", "verbose": "true"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["count"], 7)
        self.assertIs(args["verbose"], True)


class TestTypeConversion(CustomTestCase):
    def setUp(self):
        self.detector = MinimaxM2Detector()

    def test_string_type(self):
        self.assertEqual(
            self.detector._convert_param_value_with_types("hello", ["string"]), "hello"
        )

    def test_integer_type(self):
        self.assertEqual(
            self.detector._convert_param_value_with_types("42", ["integer"]), 42
        )

    def test_number_type_float(self):
        self.assertAlmostEqual(
            self.detector._convert_param_value_with_types("3.14", ["number"]), 3.14
        )

    def test_number_type_whole_returns_int(self):
        # 3.0 collapses to int(3) per the implementation.
        result = self.detector._convert_param_value_with_types("3.0", ["number"])
        self.assertEqual(result, 3)
        self.assertIsInstance(result, int)

    def test_boolean_true_variants(self):
        for val in ("true", "True", "TRUE", "1", "yes", "on"):
            with self.subTest(val=val):
                self.assertIs(
                    self.detector._convert_param_value_with_types(val, ["boolean"]),
                    True,
                )

    def test_boolean_false_variants(self):
        for val in ("false", "False", "FALSE", "0", "no", "off"):
            with self.subTest(val=val):
                self.assertIs(
                    self.detector._convert_param_value_with_types(val, ["boolean"]),
                    False,
                )

    def test_null_value(self):
        self.assertIsNone(
            self.detector._convert_param_value_with_types("null", ["string"])
        )

    def test_object_type_parses_json(self):
        result = self.detector._convert_param_value_with_types(
            '{"key": "val"}', ["object"]
        )
        self.assertEqual(result, {"key": "val"})

    def test_array_type_parses_json(self):
        result = self.detector._convert_param_value_with_types("[1, 2, 3]", ["array"])
        self.assertEqual(result, [1, 2, 3])

    def test_invalid_integer_falls_back_to_string(self):
        result = self.detector._convert_param_value_with_types(
            "not_a_number", ["integer"]
        )
        self.assertEqual(result, "not_a_number")


class TestExtractTypesFromSchema(CustomTestCase):
    def setUp(self):
        self.detector = MinimaxM2Detector()

    def test_direct_string_type(self):
        self.assertEqual(
            self.detector._extract_types_from_schema({"type": "string"}), ["string"]
        )

    def test_type_array(self):
        types = self.detector._extract_types_from_schema({"type": ["string", "null"]})
        self.assertIn("string", types)
        self.assertIn("null", types)

    def test_any_of(self):
        types = self.detector._extract_types_from_schema(
            {"anyOf": [{"type": "integer"}, {"type": "null"}]}
        )
        self.assertIn("integer", types)
        self.assertIn("null", types)

    def test_enum_infers_types(self):
        types = self.detector._extract_types_from_schema({"enum": ["a", 1, None, True]})
        self.assertIn("string", types)
        self.assertIn("integer", types)
        self.assertIn("null", types)
        self.assertIn("boolean", types)

    def test_none_schema_defaults_to_string(self):
        self.assertEqual(self.detector._extract_types_from_schema(None), ["string"])

    def test_empty_schema_defaults_to_string(self):
        self.assertEqual(self.detector._extract_types_from_schema({}), ["string"])


class TestStreamingIncrement(CustomTestCase):
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
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="calculate",
                    description="Run a calculation",
                    parameters={
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "verbose": {"type": "boolean"},
                        },
                    },
                ),
            ),
        ]

    def test_streaming_complete_call_delivered_in_chunks(self):
        detector = MinimaxM2Detector()
        full = _minimax_call("get_weather", {"city": "Beijing"})
        # Split at the start token boundary to exercise buffering.
        mid = full.index(">") + 1
        chunks = [full[:mid], full[mid:]]

        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        name_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(name_calls), 1)
        self.assertEqual(name_calls[0].name, "get_weather")

        full_params = "".join(c.parameters for c in all_calls)
        args = json.loads(full_params)
        self.assertEqual(args["city"], "Beijing")

    def test_streaming_normal_text_before_tool_call(self):
        detector = MinimaxM2Detector()
        result = detector.parse_streaming_increment("Here is the info: ", self.tools)
        self.assertEqual(result.normal_text, "Here is the info: ")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_invalid_function_name_resets_state(self):
        detector = MinimaxM2Detector()
        text = _minimax_call("no_such_func", {"x": "1"})
        all_calls = []
        for chunk in [text]:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
        name_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(name_calls), 0)

    def test_streaming_multiple_params_incremental(self):
        # Exercises the "additional parameters" diff branch: first param opens
        # the JSON object, second param extends it via the new_keys path.
        detector = MinimaxM2Detector()
        full = _minimax_call("get_weather", {"city": "London", "unit": "celsius"})
        # Deliver one character at a time to maximally stress the buffer.
        all_calls = []
        for char in full:
            all_calls.extend(detector.parse_streaming_increment(char, self.tools).calls)

        name_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(name_calls), 1)
        self.assertEqual(name_calls[0].name, "get_weather")

        full_params = "".join(c.parameters for c in all_calls)
        args = json.loads(full_params)
        self.assertEqual(args["city"], "London")
        self.assertEqual(args["unit"], "celsius")

    def test_streaming_matches_non_streaming(self):
        # Both paths must produce identical parsed arguments.
        cases = [
            ("get_weather", {"city": "Tokyo"}),
            ("get_weather", {"city": "Paris", "unit": "fahrenheit"}),
            ("calculate", {"count": "3", "verbose": "false"}),
        ]
        for func, raw_params in cases:
            with self.subTest(func=func, params=raw_params):
                text = _minimax_call(func, raw_params)

                batch = MinimaxM2Detector()
                batch_result = batch.detect_and_parse(text, self.tools)
                batch_args = json.loads(batch_result.calls[0].parameters)

                stream = MinimaxM2Detector()
                stream_calls = []
                for char in text:
                    stream_calls.extend(
                        stream.parse_streaming_increment(char, self.tools).calls
                    )
                stream_args = json.loads("".join(c.parameters for c in stream_calls))

                self.assertEqual(batch_args, stream_args)

    def test_streaming_sequential_tool_calls_tool_index(self):
        # Two back-to-back tool-call blocks must get tool_index 0 and 1.
        detector = MinimaxM2Detector()
        text = _minimax_call("get_weather", {"city": "Berlin"}) + _minimax_call(
            "calculate", {"count": "5"}
        )
        all_calls = []
        for char in text:
            all_calls.extend(detector.parse_streaming_increment(char, self.tools).calls)

        indices = sorted({c.tool_index for c in all_calls})
        self.assertEqual(indices, [0, 1])

        by_index = {i: [] for i in indices}
        for c in all_calls:
            by_index[c.tool_index].append(c)

        args0 = json.loads("".join(c.parameters for c in by_index[0]))
        args1 = json.loads("".join(c.parameters for c in by_index[1]))
        self.assertEqual(args0["city"], "Berlin")
        self.assertEqual(args1["count"], 5)


class TestMiscInterface(CustomTestCase):
    def setUp(self):
        self.detector = MinimaxM2Detector()

    def test_supports_structural_tag_is_false(self):
        self.assertFalse(self.detector.supports_structural_tag())

    def test_structure_info_raises(self):
        with self.assertRaises(NotImplementedError):
            self.detector.structure_info()


if __name__ == "__main__":
    import unittest

    unittest.main()
