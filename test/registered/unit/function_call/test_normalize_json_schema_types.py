"""Unit tests for tool-parameter schema alias normalization."""

import json
import unittest

from jsonschema import Draft202012Validator, SchemaError

from sglang.srt.function_call.utils import normalize_json_schema_types
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


class TestNormalizeJsonSchemaTypes(CustomTestCase):
    def _assert_accepts(self, schema: dict) -> None:
        Draft202012Validator.check_schema(schema)

    def test_enum_alias_becomes_string(self):
        schema = {
            "type": "object",
            "properties": {"color": {"type": "enum", "enum": ["red", "green", "blue"]}},
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["color"]["type"], "string")
        self.assertEqual(
            schema["properties"]["color"]["enum"], ["red", "green", "blue"]
        )
        self._assert_accepts(schema)

    def test_varchar_alias_becomes_string(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "varchar"},
                "short_name": {"type": "VARCHAR(255)"},
            },
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["name"]["type"], "string")
        self.assertEqual(schema["properties"]["short_name"]["type"], "string")
        self._assert_accepts(schema)

    def test_numeric_aliases(self):
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "int"},
                "big": {"type": "bigint"},
                "price": {"type": "decimal(10,2)"},
                "ratio": {"type": "float"},
            },
        }
        normalize_json_schema_types(schema)
        props = schema["properties"]
        self.assertEqual(props["age"]["type"], "integer")
        self.assertEqual(props["big"]["type"], "integer")
        self.assertEqual(props["price"]["type"], "number")
        self.assertEqual(props["ratio"]["type"], "number")
        self._assert_accepts(schema)

    def test_prefix_matched_numeric_types(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "int32"},
                "b": {"type": "int64"},
                "c": {"type": "uint"},
                "d": {"type": "unsigned"},
                "e": {"type": "long"},
                "f": {"type": "short"},
                "g": {"type": "float32"},
                "h": {"type": "float64"},
                "i": {"type": "num"},
                "j": {"type": "numeric"},
            },
        }
        normalize_json_schema_types(schema)
        p = schema["properties"]
        for k in ("a", "b", "c", "d", "e", "f"):
            self.assertEqual(p[k]["type"], "integer")
        for k in ("g", "h", "i", "j"):
            self.assertEqual(p[k]["type"], "number")
        self._assert_accepts(schema)

    def test_prefix_matched_compound_types(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "list[str]"},
                "b": {"type": "list<int>"},
                "c": {"type": "dict"},
                "d": {"type": "dict[str, int]"},
                "e": {"type": "long long"},
            },
        }
        normalize_json_schema_types(schema)
        p = schema["properties"]
        self.assertEqual(p["a"]["type"], "array")
        self.assertEqual(p["b"]["type"], "array")
        self.assertEqual(p["c"]["type"], "object")
        self.assertEqual(p["d"]["type"], "object")
        self.assertEqual(p["e"]["type"], "integer")
        self._assert_accepts(schema)

    def test_word_boundary_prevents_false_positives(self):
        """Prefixes must end at a non-identifier char, so custom type names
        that merely start with a known prefix are left alone."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "internal"},
                "b": {"type": "list_price"},
                "c": {"type": "integer_enum"},
                "d": {"type": "dictionary_entry"},
                "e": {"type": "floating"},
            },
        }
        normalize_json_schema_types(schema)
        p = schema["properties"]
        for key, expected in [
            ("a", "internal"),
            ("b", "list_price"),
            ("c", "integer_enum"),
            ("d", "dictionary_entry"),
            ("e", "floating"),
        ]:
            self.assertEqual(p[key]["type"], expected)

    def test_recurses_into_draft_2020_12_keywords(self):
        schema = {
            "type": "object",
            "properties": {
                "row": {
                    "dependentSchemas": {
                        "kind": {
                            "properties": {"sku": {"type": "varchar"}},
                        },
                    },
                    "propertyNames": {"type": "str"},
                    "unevaluatedProperties": {"type": "bigint"},
                },
                "rows": {
                    "type": "arr",
                    "unevaluatedItems": {"type": "int32"},
                },
            },
        }
        normalize_json_schema_types(schema)
        row = schema["properties"]["row"]
        rows = schema["properties"]["rows"]
        self.assertEqual(
            row["dependentSchemas"]["kind"]["properties"]["sku"]["type"], "string"
        )
        self.assertEqual(row["propertyNames"]["type"], "string")
        self.assertEqual(row["unevaluatedProperties"]["type"], "integer")
        self.assertEqual(rows["type"], "array")
        self.assertEqual(rows["unevaluatedItems"]["type"], "integer")
        self._assert_accepts(schema)

    def test_binary_and_arr_aliases(self):
        schema = {
            "type": "object",
            "properties": {
                "flag": {"type": "binary"},
                "items": {"type": "arr"},
            },
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["flag"]["type"], "boolean")
        self.assertEqual(schema["properties"]["items"]["type"], "array")
        self._assert_accepts(schema)

    def test_case_insensitive(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "VARCHAR"},
                "b": {"type": "INT"},
                "c": {"type": "String"},
            },
        }
        normalize_json_schema_types(schema)
        p = schema["properties"]
        self.assertEqual(p["a"]["type"], "string")
        self.assertEqual(p["b"]["type"], "integer")
        self.assertEqual(p["c"]["type"], "string")
        self._assert_accepts(schema)

    def test_array_and_object_aliases(self):
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "list", "items": {"type": "str"}},
                "meta": {"type": "dict"},
            },
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["tags"]["type"], "array")
        self.assertEqual(schema["properties"]["tags"]["items"]["type"], "string")
        self.assertEqual(schema["properties"]["meta"]["type"], "object")
        self._assert_accepts(schema)

    def test_nested_anyof_and_defs(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "int"},
                        {"type": "varchar"},
                    ]
                }
            },
            "$defs": {
                "Row": {"type": "object", "properties": {"id": {"type": "bigint"}}}
            },
        }
        normalize_json_schema_types(schema)
        any_of = schema["properties"]["value"]["anyOf"]
        self.assertEqual(any_of[0]["type"], "integer")
        self.assertEqual(any_of[1]["type"], "string")
        self.assertEqual(schema["$defs"]["Row"]["properties"]["id"]["type"], "integer")
        self._assert_accepts(schema)

    def test_type_list_member_normalized(self):
        schema = {"type": ["varchar", "null"]}
        normalize_json_schema_types(schema)
        self.assertEqual(schema["type"], ["string", "null"])
        self._assert_accepts(schema)

    def test_standard_types_untouched(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
                "c": {"type": "boolean"},
            },
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["a"]["type"], "string")
        self.assertEqual(schema["properties"]["b"]["type"], "integer")
        self.assertEqual(schema["properties"]["c"]["type"], "boolean")

    def test_unknown_type_left_alone(self):
        schema = {"type": "object", "properties": {"x": {"type": "geometry"}}}
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["x"]["type"], "geometry")
        with self.assertRaises(SchemaError):
            self._assert_accepts(schema)

    def test_common_db_orm_type_names_accepted(self):
        """Common non-standard DB/ORM type names all survive validation."""
        recognized = [
            # string family
            "string",
            "str",
            "text",
            "varchar",
            "char",
            "enum",
            # integer via prefix
            "int",
            "int32",
            "int64",
            "uint",
            "uint8",
            "long",
            "long long",
            "short",
            "unsigned",
            # number via prefix
            "num",
            "numeric",
            "float",
            "float32",
            "float64",
            # boolean
            "boolean",
            "bool",
            "binary",
            # compound
            "object",
            "array",
            "arr",
            "dict",
            "dict[str, int]",
            "list",
            "list[str]",
        ]
        for t in recognized:
            schema = {"type": "object", "properties": {"x": {"type": t}}}
            normalize_json_schema_types(schema)
            try:
                self._assert_accepts(schema)
            except SchemaError as e:
                self.fail(f"type {t!r} → {schema['properties']['x']['type']!r}: {e}")

    def test_pre_existing_400_schema_now_accepted(self):
        schema = {
            "type": "object",
            "properties": {
                "sql": {"type": "varchar"},
                "mode": {"type": "enum", "enum": ["read", "write"]},
            },
            "required": ["sql", "mode"],
        }
        normalize_json_schema_types(schema)
        self._assert_accepts(schema)

    def test_idempotent(self):
        """Running normalize twice produces the same result as running it once."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "varchar"},
                "b": {"type": "int32"},
                "c": {"type": ["bigint", "null"]},
                "d": {
                    "anyOf": [{"type": "enum"}, {"type": "decimal(10,2)"}],
                },
            },
        }
        normalize_json_schema_types(schema)
        once = json.loads(json.dumps(schema))
        normalize_json_schema_types(schema)
        self.assertEqual(schema, once)

    def test_non_string_type_values_pass_through(self):
        """``type`` that isn't str/list is left for the real validator to reject."""
        for bad in (None, 42, {"$ref": "#/$defs/Foo"}, ["string", 1, None]):
            schema = {"properties": {"x": {"type": bad}}}
            normalize_json_schema_types(schema)
            self.assertEqual(schema["properties"]["x"]["type"], bad)

    def test_recurses_into_all_walked_keywords(self):
        """Every keyword the walker recurses into must actually rewrite nested aliases."""
        schema = {
            "patternProperties": {"^x_": {"type": "varchar"}},
            "definitions": {"Row": {"type": "bigint"}},
            "prefixItems": [{"type": "int32"}, {"type": "float64"}],
            "if": {"type": "bool"},
            "then": {"type": "str"},
            "else": {"type": "decimal"},
            "not": {"type": "uuid"},
            "contains": {"type": "enum"},
            "additionalProperties": {"type": "binary"},
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["patternProperties"]["^x_"]["type"], "string")
        self.assertEqual(schema["definitions"]["Row"]["type"], "integer")
        self.assertEqual(schema["prefixItems"][0]["type"], "integer")
        self.assertEqual(schema["prefixItems"][1]["type"], "number")
        self.assertEqual(schema["if"]["type"], "boolean")
        self.assertEqual(schema["then"]["type"], "string")
        self.assertEqual(schema["else"]["type"], "number")
        self.assertEqual(schema["not"]["type"], "string")
        self.assertEqual(schema["contains"]["type"], "string")
        self.assertEqual(schema["additionalProperties"]["type"], "boolean")

    def test_cyclic_schema_raises_recursion_error(self):
        """A pathological cyclic schema surfaces as RecursionError — caller
        (``_validate_request``) converts it to a 400, not a 500."""
        schema = {"type": "object"}
        schema["items"] = schema
        with self.assertRaises(RecursionError):
            normalize_json_schema_types(schema)

    def test_boolean_subschema_does_not_crash(self):
        """``additionalProperties: True`` and ``items: false`` are valid 2020-12
        forms; the walker must pass through without raising."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "varchar"},
                "b": {"type": "array", "items": False},
            },
            "additionalProperties": True,
            "unevaluatedProperties": False,
        }
        normalize_json_schema_types(schema)
        self.assertEqual(schema["properties"]["a"]["type"], "string")
        self.assertIs(schema["properties"]["b"]["items"], False)
        self.assertIs(schema["additionalProperties"], True)
        self.assertIs(schema["unevaluatedProperties"], False)
        self._assert_accepts(schema)


if __name__ == "__main__":
    unittest.main()
