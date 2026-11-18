import unittest
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel

from sglang.srt.entrypoints.openai.tool_server import trim_schema
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Mode(str, Enum):
    FAST = "fast"
    SLOW = "slow"


class Address(BaseModel):
    city: str


class NullableEnumArgs(BaseModel):
    mode: Optional[Mode] = None


class NullableObjectArgs(BaseModel):
    address: Optional[Address] = None


class NullableArrayArgs(BaseModel):
    modes: Optional[list[Mode]] = None


class StructuredUnionArgs(BaseModel):
    value: Union[Mode, Address]


class RecursiveArgs(BaseModel):
    child: Optional["RecursiveArgs"] = None


class TestMCPToolSchema(CustomTestCase):
    def test_nullable_enum_keeps_allowed_values(self):
        schema = trim_schema(NullableEnumArgs.model_json_schema())
        self.assertEqual(
            schema["properties"]["mode"],
            {"type": "string", "enum": ["fast", "slow"]},
        )

    def test_nullable_object_keeps_properties_and_required_fields(self):
        schema = trim_schema(NullableObjectArgs.model_json_schema())
        self.assertEqual(
            schema["properties"]["address"],
            {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

    def test_nullable_array_resolves_item_references(self):
        schema = trim_schema(NullableArrayArgs.model_json_schema())
        self.assertEqual(
            schema["properties"]["modes"],
            {
                "type": "array",
                "items": {"type": "string", "enum": ["fast", "slow"]},
            },
        )

    def test_structured_union_keeps_each_branch(self):
        schema = trim_schema(StructuredUnionArgs.model_json_schema())
        variants = schema["properties"]["value"]["oneOf"]
        self.assertEqual(variants[0], {"type": "string", "enum": ["fast", "slow"]})
        self.assertEqual(variants[1]["properties"], {"city": {"type": "string"}})
        self.assertEqual(variants[1]["required"], ["city"])

    def test_primitive_union_preserves_existing_harmony_shape(self):
        self.assertEqual(
            trim_schema({"anyOf": [{"type": "string"}, {"type": "null"}]}),
            {"type": ["string"]},
        )
        self.assertEqual(
            trim_schema({"anyOf": [{"type": "integer"}, {"type": "string"}]}),
            {"type": ["integer", "string"]},
        )

    def test_local_reference_decodes_json_pointer_escapes(self):
        schema = trim_schema(
            {
                "$defs": {"a/b~c": {"type": "string", "enum": ["x"]}},
                "type": "object",
                "properties": {"value": {"$ref": "#/$defs/a~1b~0c"}},
            }
        )
        self.assertEqual(
            schema["properties"]["value"], {"type": "string", "enum": ["x"]}
        )

    def test_recursive_reference_reports_a_bounded_error(self):
        with self.assertRaisesRegex(ValueError, "recursive.*reference"):
            trim_schema(RecursiveArgs.model_json_schema())

    def test_deep_reference_chain_reports_a_bounded_error(self):
        definitions = {
            str(index): {"$ref": f"#/$defs/{index + 1}"} for index in range(65)
        }
        definitions["65"] = {"type": "string"}
        with self.assertRaisesRegex(ValueError, "reference depth exceeds 64"):
            trim_schema({"$defs": definitions, "$ref": "#/$defs/0"})

    def test_external_reference_is_not_fetched(self):
        with self.assertRaisesRegex(ValueError, "local.*references"):
            trim_schema({"$ref": "https://example.com/schema.json"})


if __name__ == "__main__":
    unittest.main()
