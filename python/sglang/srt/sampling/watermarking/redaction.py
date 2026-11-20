from __future__ import annotations

import copy
import dataclasses
from typing import Any, Optional, Sequence

import msgspec

from .config import WatermarkRequestConfig, WatermarkServerConfig

WATERMARK_SECRET_FIELD_NAMES = frozenset(
    {"watermark_key", "watermark_key_b", "watermark_keys", "watermark_config"}
)
WATERMARK_CONTAINER_FIELD_NAMES = frozenset(
    {"watermark", "sampling_params", "preferred_sampling_params"}
)
_SCALAR_TYPES = frozenset({int, float, str, bool, bytes, type(None)})


def redact_watermark_secrets(value: Any, *, in_watermark_config: bool = False) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = copy.copy(value)
        for field in dataclasses.fields(value):
            item = getattr(value, field.name)
            if field.name in WATERMARK_SECRET_FIELD_NAMES:
                object.__setattr__(
                    result,
                    field.name,
                    "<redacted>" if item is not None else None,
                )
            elif field.name in WATERMARK_CONTAINER_FIELD_NAMES:
                object.__setattr__(
                    result,
                    field.name,
                    redact_watermark_secrets(
                        item, in_watermark_config=field.name == "watermark"
                    ),
                )
        return result
    if isinstance(value, WatermarkRequestConfig):
        return WatermarkRequestConfig(
            enabled=value.enabled,
            key="<redacted>" if value.key is not None else None,
            context_window=value.context_window,
        )
    if isinstance(value, WatermarkServerConfig):
        return msgspec.structs.replace(
            value,
            key="<redacted>" if value.key is not None else None,
            key_b="<redacted>" if value.key_b is not None else None,
            keys="<redacted>" if value.keys is not None else None,
        )
    if isinstance(value, msgspec.Struct):
        result = copy.copy(value)
        replacements = {}
        for field in msgspec.structs.fields(value):
            item = getattr(value, field.name)
            if field.name in WATERMARK_SECRET_FIELD_NAMES:
                replacements[field.name] = "<redacted>" if item is not None else None
            elif field.name in WATERMARK_CONTAINER_FIELD_NAMES:
                replacements[field.name] = redact_watermark_secrets(
                    item, in_watermark_config=field.name == "watermark"
                )
        for name, replacement in replacements.items():
            msgspec.Struct.__setattr__(result, name, replacement)
        return result if replacements else value
    if isinstance(value, dict):
        return {
            key: (
                "<redacted>"
                if key in WATERMARK_SECRET_FIELD_NAMES
                or (in_watermark_config and key in {"key", "key_b", "keys"})
                else redact_watermark_secrets(
                    item,
                    in_watermark_config=in_watermark_config or key == "watermark",
                )
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        # Secrets only sit under dict keys or struct/dataclass fields, so a
        # sequence of plain scalars can never contain one.
        if _SCALAR_TYPES.issuperset(map(type, value)):
            # The dump path pickles the result on a worker thread, so a
            # mutable list must be copied; tuples are safe to alias.
            return list(value) if isinstance(value, list) else value
        redacted = [
            item
            if type(item) in _SCALAR_TYPES
            else redact_watermark_secrets(item, in_watermark_config=in_watermark_config)
            for item in value
        ]
        return redacted if isinstance(value, list) else tuple(redacted)
    return value


def redact_watermark_command_line(
    argv: Sequence[str], watermark_config: Optional[str] = None
) -> str:
    result = []
    redact_next = False
    for argument in argv:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
        elif watermark_config is not None and argument == watermark_config:
            result.append("<redacted>")
        elif (
            watermark_config is not None
            and "=" in argument
            and argument.split("=", 1)[1] == watermark_config
        ):
            result.append(f"{argument.split('=', 1)[0]}=<redacted>")
        elif argument == "--watermark-config":
            result.append(argument)
            redact_next = True
        elif argument.startswith("--watermark-config="):
            result.append("--watermark-config=<redacted>")
        else:
            result.append(argument)
    return " ".join(result)
