"""Cache-neutral types and ingress validation for KV-hint metadata."""

from collections.abc import Mapping
from typing import Any, Optional, Union

import msgspec

from sglang.srt.utils.msgspec_utils import msgspec_struct_pydantic_core_schema

SUPPORTED_KV_HINTS_PROTOCOL_VERSION = "0.1"


class KvHintAction(msgspec.Struct, kw_only=True):
    action_id: str
    action_type: str
    action_version: str
    payload: dict[str, Any]

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class KvHints(msgspec.Struct, kw_only=True):
    protocol_version: str
    message_id: str
    actions: list[KvHintAction]

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


KvHintsInput = Optional[Union[KvHints, Mapping[str, Any]]]
KvHintsRequest = Union[KvHintsInput, list[KvHintsInput]]


def normalize_kv_hints(value: KvHintsInput) -> Optional[KvHints]:
    """Validate one envelope and return a canonical, independently owned copy."""
    if value is None:
        return None
    if isinstance(value, KvHints):
        builtins = msgspec.to_builtins(value)
    elif isinstance(value, Mapping):
        builtins = msgspec.to_builtins(dict(value))
    else:
        raise ValueError("kv_hints must be an object or None")

    try:
        hints = msgspec.convert(builtins, type=KvHints, strict=True)
    except (TypeError, msgspec.ValidationError) as exc:
        raise ValueError(f"Invalid kv_hints: {exc}") from exc

    if hints.protocol_version != SUPPORTED_KV_HINTS_PROTOCOL_VERSION:
        raise ValueError(
            "Unsupported kv_hints protocol_version "
            f"{hints.protocol_version!r}; expected "
            f"{SUPPORTED_KV_HINTS_PROTOCOL_VERSION!r}"
        )
    return hints
