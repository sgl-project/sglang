from __future__ import annotations

import json
from typing import Optional


def encode_prefix_cache_key_parts(
    parts: list[tuple[str, Optional[str]]],
) -> Optional[str]:
    """Encode prefix-cache namespace parts without string concatenation ambiguity."""
    encoded_parts: list[list[str]] = []
    for name, value in parts:
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError(
                f"Prefix-cache namespace part {name} must be a string, "
                f"but got {type(value).__name__}"
            )
        encoded_parts.append([name, value])

    if not encoded_parts:
        return None
    return json.dumps(encoded_parts, separators=(",", ":"), ensure_ascii=True)
