from __future__ import annotations

import json
from typing import Optional

STRUCTURED_PREFIX_CACHE_KEY_PREFIX = "__sglang_prefix_cache_namespace__:"
ESCAPED_PREFIX_CACHE_KEY_PREFIX = "__sglang_prefix_cache_raw__:"
_RESERVED_PREFIX_CACHE_KEY_PREFIXES = (
    STRUCTURED_PREFIX_CACHE_KEY_PREFIX,
    ESCAPED_PREFIX_CACHE_KEY_PREFIX,
)


def _validate_prefix_cache_key_part(name: str, value: Optional[str]) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(
            f"Prefix-cache namespace part {name} must be a string, "
            f"but got {type(value).__name__}"
        )


def escape_prefix_cache_user_key(
    value: Optional[str],
    *,
    name: str = "extra_key",
) -> Optional[str]:
    """Escape user-controlled keys that overlap internal namespace prefixes."""
    _validate_prefix_cache_key_part(name, value)
    if value is None:
        return None
    if value.startswith(_RESERVED_PREFIX_CACHE_KEY_PREFIXES):
        return ESCAPED_PREFIX_CACHE_KEY_PREFIX + value
    return value


def encode_prefix_cache_key_parts(
    parts: list[tuple[str, Optional[str]]],
) -> Optional[str]:
    """Encode prefix-cache namespace parts without string concatenation ambiguity."""
    encoded_parts: list[list[str]] = []
    for name, value in parts:
        if value is None:
            continue
        _validate_prefix_cache_key_part(name, value)
        encoded_parts.append([name, value])

    if not encoded_parts:
        return None
    return STRUCTURED_PREFIX_CACHE_KEY_PREFIX + json.dumps(
        encoded_parts, separators=(",", ":"), ensure_ascii=True
    )


def build_prefix_cache_extra_key(
    cache_salt: Optional[str],
    extra_key: Optional[str],
    lora_id: Optional[str],
) -> Optional[str]:
    """Build the radix-cache extra key namespace for a request."""
    _validate_prefix_cache_key_part("cache_salt", cache_salt)
    _validate_prefix_cache_key_part("extra_key", extra_key)
    _validate_prefix_cache_key_part("lora_id", lora_id)
    if cache_salt is None and lora_id is None:
        return escape_prefix_cache_user_key(extra_key)
    if extra_key is None and lora_id is None:
        return escape_prefix_cache_user_key(cache_salt, name="cache_salt")

    return encode_prefix_cache_key_parts(
        [
            ("cache_salt", cache_salt or None),
            ("extra_key", extra_key or None),
            ("lora_id", lora_id or None),
        ]
    )
