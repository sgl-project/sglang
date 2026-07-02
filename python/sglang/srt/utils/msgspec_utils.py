from __future__ import annotations

import base64
import binascii
from typing import Any

import msgspec
from pydantic_core import core_schema


class Base64Bytes:
    """Pydantic marker for HTTP JSON base64-encoded bytes fields."""

    def __get_pydantic_core_schema__(self, source_type: Any, handler):
        return core_schema.no_info_before_validator_function(
            self._decode_value,
            handler(source_type),
        )

    @classmethod
    def _decode_value(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return base64.b64decode(value, validate=True)
            except binascii.Error as exc:
                raise ValueError("Expected base64-encoded bytes") from exc

        if isinstance(value, list):
            return [cls._decode_value(item) for item in value]

        if isinstance(value, tuple):
            return tuple(cls._decode_value(item) for item in value)

        return value


def msgspec_to_builtins(obj: Any) -> Any:
    """Recursively convert msgspec structs to dict/list Python builtins."""
    if isinstance(obj, msgspec.Struct):
        return {
            field.name: msgspec_to_builtins(getattr(obj, field.name))
            for field in msgspec.structs.fields(type(obj))
        }

    if isinstance(obj, dict):
        return {key: msgspec_to_builtins(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [msgspec_to_builtins(item) for item in obj]

    if isinstance(obj, tuple):
        return tuple(msgspec_to_builtins(item) for item in obj)

    if isinstance(obj, set):
        return [msgspec_to_builtins(item) for item in obj]

    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__qualname__}"

    return obj


def msgspec_struct_pydantic_core_schema(cls: type[msgspec.Struct], handler):
    fields = {}
    for struct_field in msgspec.structs.fields(cls):
        field_schema = handler.generate_schema(struct_field.type)
        required = (
            struct_field.default is msgspec.NODEFAULT
            and struct_field.default_factory is msgspec.NODEFAULT
        )

        if struct_field.default is not msgspec.NODEFAULT:
            field_schema = core_schema.with_default_schema(
                field_schema,
                default=struct_field.default,
            )
        elif struct_field.default_factory is not msgspec.NODEFAULT:
            field_schema = core_schema.with_default_schema(
                field_schema,
                default_factory=struct_field.default_factory,
            )

        fields[struct_field.name] = core_schema.typed_dict_field(
            field_schema,
            required=required,
        )

    typed_dict_schema = core_schema.typed_dict_schema(
        fields,
        cls_name=cls.__name__,
        extra_behavior="ignore",
        ref=cls.__name__,
    )

    def build_struct(value):
        return value if isinstance(value, cls) else cls(**value)

    dict_to_struct_schema = core_schema.no_info_after_validator_function(
        build_struct,
        typed_dict_schema,
    )
    return core_schema.json_or_python_schema(
        json_schema=dict_to_struct_schema,
        python_schema=core_schema.union_schema(
            [
                core_schema.is_instance_schema(cls),
                dict_to_struct_schema,
            ],
            mode="left_to_right",
        ),
    )
