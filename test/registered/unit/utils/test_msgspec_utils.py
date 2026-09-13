"""Unit tests for msgspec utilities -- no server, model, or GPU required."""

import base64
import unittest
from typing import Annotated, Any

import msgspec
from pydantic import TypeAdapter, ValidationError

from sglang.srt.utils.msgspec_utils import (
    Base64Bytes,
    msgspec_struct_pydantic_core_schema,
    msgspec_to_builtins,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ChildStruct(msgspec.Struct):
    value: int


class _NestedStruct(msgspec.Struct):
    child: _ChildStruct
    mapping: dict[str, Any]
    sequence: list[Any]
    coordinates: tuple[Any, ...]
    labels: set[str]
    enabled: bool


class _SchemaStruct(msgspec.Struct, kw_only=True):
    required_count: int
    name: str = "default-name"
    items: list[int] = msgspec.field(default_factory=list)

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class _TensorPayload(msgspec.Struct, kw_only=True):
    serialized_named_tensors: Annotated[list[bytes], Base64Bytes()]

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class TestBase64Bytes(CustomTestCase):
    def test_decodes_scalar_and_list_values(self):
        scalar_adapter = TypeAdapter(Annotated[bytes, Base64Bytes()])
        list_adapter = TypeAdapter(Annotated[list[bytes], Base64Bytes()])

        self.assertEqual(
            scalar_adapter.validate_python(base64.b64encode(b"tensor-data").decode()),
            b"tensor-data",
        )
        self.assertEqual(
            list_adapter.validate_python(
                [
                    base64.b64encode(b"weight-0").decode(),
                    base64.b64encode(b"weight-1").decode(),
                ]
            ),
            [b"weight-0", b"weight-1"],
        )

    def test_recurses_through_nested_lists_and_tuples(self):
        nested_adapter = TypeAdapter(Annotated[list[tuple[bytes, ...]], Base64Bytes()])
        encoded = base64.b64encode(b"nested").decode()

        self.assertEqual(
            nested_adapter.validate_python([(encoded,), (encoded, encoded)]),
            [(b"nested",), (b"nested", b"nested")],
        )

    def test_rejects_malformed_base64(self):
        adapter = TypeAdapter(Annotated[bytes, Base64Bytes()])

        for value in ("not-base64!", "abc"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValidationError, "Expected base64-encoded bytes"
                ):
                    adapter.validate_python(value)

    def test_preserves_python_bytes(self):
        adapter = TypeAdapter(Annotated[list[bytes], Base64Bytes()])
        values = [b"already-decoded", b"\x00\xff"]

        self.assertEqual(adapter.validate_python(values), values)

    def test_decodes_multiple_serialized_tensor_values_from_json(self):
        adapter = TypeAdapter(_TensorPayload)
        encoded_values = [
            base64.b64encode(b"tensor-a").decode(),
            base64.b64encode(b"tensor-b").decode(),
        ]

        payload = adapter.validate_json(
            msgspec.json.encode({"serialized_named_tensors": encoded_values})
        )

        self.assertEqual(
            payload.serialized_named_tensors,
            [b"tensor-a", b"tensor-b"],
        )


class TestMsgspecToBuiltins(CustomTestCase):
    def test_recursively_converts_realistic_nested_struct(self):
        value = _NestedStruct(
            child=_ChildStruct(1),
            mapping={"child": _ChildStruct(2), "primitive": None},
            sequence=[_ChildStruct(3), {"deep": _ChildStruct(4)}],
            coordinates=(_ChildStruct(5), "origin"),
            labels={"fast", "cpu"},
            enabled=True,
        )

        result = msgspec_to_builtins(value)

        self.assertEqual(result["child"], {"value": 1})
        self.assertEqual(
            result["mapping"],
            {"child": {"value": 2}, "primitive": None},
        )
        self.assertEqual(
            result["sequence"],
            [{"value": 3}, {"deep": {"value": 4}}],
        )
        self.assertEqual(result["coordinates"], ({"value": 5}, "origin"))
        self.assertIsInstance(result["coordinates"], tuple)
        self.assertCountEqual(result["labels"], ["fast", "cpu"])
        self.assertIsInstance(result["labels"], list)
        self.assertIs(result["enabled"], True)

    def test_preserves_dict_keys_and_primitive_leaves(self):
        key = ("stable", 1)
        value = {key: _ChildStruct(7), "number": 42}

        result = msgspec_to_builtins(value)

        self.assertIn(key, result)
        self.assertEqual(result[key], {"value": 7})
        self.assertEqual(result["number"], 42)


class TestMsgspecStructPydanticCoreSchema(CustomTestCase):
    def setUp(self):
        self.adapter = TypeAdapter(_SchemaStruct)

    def test_required_field_rejects_missing_input(self):
        with self.assertRaisesRegex(ValidationError, "required_count"):
            self.adapter.validate_python({})

    def test_defaults_and_default_factory_are_applied_independently(self):
        first = self.adapter.validate_python({"required_count": 1})
        second = self.adapter.validate_python({"required_count": 2})

        self.assertEqual(first.name, "default-name")
        self.assertEqual(first.items, [])
        self.assertEqual(second.items, [])
        self.assertIsNot(first.items, second.items)

    def test_python_dict_and_json_build_structs(self):
        from_python = self.adapter.validate_python(
            {"required_count": 3, "name": "python", "items": [1]}
        )
        from_json = self.adapter.validate_json(
            b'{"required_count":4,"name":"json","items":[2,3]}'
        )

        self.assertEqual(
            from_python,
            _SchemaStruct(required_count=3, name="python", items=[1]),
        )
        self.assertEqual(
            from_json,
            _SchemaStruct(required_count=4, name="json", items=[2, 3]),
        )

    def test_existing_instance_is_preserved_by_python_path(self):
        value = _SchemaStruct(required_count=5)

        self.assertIs(self.adapter.validate_python(value), value)

    def test_wrong_field_type_fails_validation(self):
        with self.assertRaisesRegex(ValidationError, "required_count"):
            self.adapter.validate_python({"required_count": ["not-an-int"]})

    def test_extra_fields_are_ignored(self):
        value = self.adapter.validate_python(
            {"required_count": 6, "unexpected": "ignored"}
        )

        self.assertEqual(value, _SchemaStruct(required_count=6))
        self.assertFalse(hasattr(value, "unexpected"))

    def test_json_schema_marks_required_and_default_fields(self):
        schema = self.adapter.json_schema()
        definition = schema.get("$defs", {}).get("_SchemaStruct", schema)

        self.assertEqual(definition["required"], ["required_count"])
        self.assertEqual(definition["properties"]["name"]["default"], "default-name")
        self.assertNotIn("default", definition["properties"]["items"])


if __name__ == "__main__":
    unittest.main()
