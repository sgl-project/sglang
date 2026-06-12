"""Unit tests for function_call/utils.py helpers — pure functions, no GPU/server."""

from partial_json_parser.core.options import Allow

from sglang.srt.function_call.utils import (
    _partial_json_loads,
    infer_type_from_json_schema,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestInferTypeFromJsonSchema(CustomTestCase):
    def test_direct_type(self):
        self.assertEqual(infer_type_from_json_schema({"type": "string"}), "string")
        self.assertEqual(infer_type_from_json_schema({"type": "integer"}), "integer")

    def test_type_array_filters_null(self):
        self.assertEqual(
            infer_type_from_json_schema({"type": ["null", "integer"]}), "integer"
        )
        # only null -> defaults to string
        self.assertEqual(infer_type_from_json_schema({"type": ["null"]}), "string")

    def test_any_of(self):
        # all the same -> that type
        self.assertEqual(
            infer_type_from_json_schema(
                {"anyOf": [{"type": "integer"}, {"type": "integer"}]}
            ),
            "integer",
        )
        # mixed -> string preferred
        self.assertEqual(
            infer_type_from_json_schema(
                {"anyOf": [{"type": "integer"}, {"type": "string"}]}
            ),
            "string",
        )
        # oneOf treated like anyOf
        self.assertEqual(
            infer_type_from_json_schema(
                {"oneOf": [{"type": "number"}, {"type": "number"}]}
            ),
            "number",
        )

    def test_enum_value_type_inference(self):
        self.assertEqual(infer_type_from_json_schema({"enum": [1, 2, 3]}), "integer")
        self.assertEqual(infer_type_from_json_schema({"enum": ["a", "b"]}), "string")
        # bool must be detected before int (bool is a subclass of int)
        self.assertEqual(
            infer_type_from_json_schema({"enum": [True, False]}), "boolean"
        )
        # mixed enum -> string
        self.assertEqual(infer_type_from_json_schema({"enum": [1, "a"]}), "string")
        # empty enum -> string
        self.assertEqual(infer_type_from_json_schema({"enum": []}), "string")

    def test_all_of(self):
        # first non-string type wins
        self.assertEqual(
            infer_type_from_json_schema(
                {"allOf": [{"type": "string"}, {"type": "object"}]}
            ),
            "object",
        )
        # all string -> string
        self.assertEqual(
            infer_type_from_json_schema({"allOf": [{"type": "string"}]}),
            "string",
        )

    def test_properties_and_items(self):
        self.assertEqual(
            infer_type_from_json_schema({"properties": {"a": {"type": "string"}}}),
            "object",
        )
        self.assertEqual(
            infer_type_from_json_schema({"items": {"type": "integer"}}), "array"
        )

    def test_non_dict_and_empty(self):
        self.assertIsNone(infer_type_from_json_schema("not a dict"))
        self.assertIsNone(infer_type_from_json_schema({}))

    def test_priority_type_over_enum(self):
        # explicit type field takes priority over enum value inference
        self.assertEqual(
            infer_type_from_json_schema({"type": "string", "enum": [1, 2]}),
            "string",
        )


class TestPartialJsonLoads(CustomTestCase):
    def test_complete_object_consumes_all(self):
        obj, end = _partial_json_loads('{"a": 1, "b": 2}', Allow.ALL)
        self.assertEqual(obj, {"a": 1, "b": 2})
        self.assertEqual(end, len('{"a": 1, "b": 2}'))

    def test_extra_data_falls_back_to_raw_decode(self):
        # trailing junk after a complete object -> raw_decode returns consumed < len
        s = '{"a": 1} trailing junk'
        obj, end = _partial_json_loads(s, Allow.ALL)
        self.assertEqual(obj, {"a": 1})
        self.assertEqual(end, len('{"a": 1}'))
        self.assertLess(end, len(s))

    def test_partial_object(self):
        # an incomplete object is tolerated under Allow.ALL
        obj, end = _partial_json_loads('{"a": 1, "b":', Allow.ALL)
        self.assertIsInstance(obj, dict)
        self.assertEqual(obj.get("a"), 1)

    def test_leading_whitespace_with_extra_data(self):
        s = '   {"x": 5} more'
        obj, end = _partial_json_loads(s, Allow.ALL)
        self.assertEqual(obj, {"x": 5})
        self.assertLess(end, len(s))


if __name__ == "__main__":
    import unittest

    unittest.main()
