from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import msgspec


class WatermarkConfigError(ValueError):
    pass


class WatermarkServerConfig(msgspec.Struct, frozen=True, kw_only=True):
    key: str
    context_window: int

    def __repr__(self) -> str:
        return (
            "WatermarkServerConfig(key=<redacted>, "
            f"context_window={self.context_window!r})"
        )


def parse_watermark_key(value: Any) -> int:
    if not isinstance(value, str):
        raise ValueError("watermark key must be a hex string")
    digits = value[2:] if value.lower().startswith("0x") else value
    if not 1 <= len(digits) <= 16:
        raise ValueError("watermark key must contain 1 to 16 hex digits")
    if any(character not in "0123456789abcdefABCDEF" for character in digits):
        raise ValueError("watermark key must contain only hex digits")
    key = int(digits, 16)
    return key if key < (1 << 63) else key - (1 << 64)


def load_watermark_config(path: str) -> WatermarkServerConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise WatermarkConfigError("watermark config must be a readable regular file")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WatermarkConfigError("failed to read watermark config JSON") from error
    if not isinstance(raw, dict):
        raise WatermarkConfigError("watermark config must be a JSON object")
    unknown = sorted(set(raw) - {"key", "context_window"})
    if unknown:
        raise WatermarkConfigError("watermark config contains unknown fields")
    if set(raw) != {"key", "context_window"}:
        raise WatermarkConfigError("watermark config requires key and context_window")

    key = raw["key"]
    try:
        parse_watermark_key(key)
    except ValueError as error:
        raise WatermarkConfigError(str(error)) from error
    context_window = raw["context_window"]
    if (
        isinstance(context_window, bool)
        or not isinstance(context_window, int)
        or context_window < 1
    ):
        raise WatermarkConfigError(
            "watermark config context_window must be a positive integer"
        )
    return WatermarkServerConfig(key=key, context_window=context_window)
