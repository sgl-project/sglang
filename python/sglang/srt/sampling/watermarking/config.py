from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path
from typing import Any, Optional

import msgspec

logger = logging.getLogger(__name__)

_MAX_CONFIG_BYTES = 4096
MAX_WATERMARK_CONTEXT_WINDOW = 64
MAX_WATERMARKED_CONTEXTS_PER_REQUEST = 4096


class WatermarkConfigError(ValueError):
    pass


class WatermarkServerConfig(msgspec.Struct, frozen=True, kw_only=True):
    # None marks a field the file leaves unset, so server defaults still apply.
    key: str | None = None
    key_b: str | None = None
    context_window: int | None = None
    mixing_probability: float | None = None
    max_probability: float | None = None
    default_enabled: bool | None = None
    enforce_all: bool | None = None
    keys: list[int] | None = None
    ngram_len: int | None = None
    num_candidates: int | None = None
    context_history_size: int | None = None
    skip_first_ngram_calls: bool | None = None

    def __repr__(self) -> str:
        return (
            "WatermarkServerConfig(key=<redacted>, key_b=<redacted>, "
            f"context_window={self.context_window!r}, "
            f"mixing_probability={self.mixing_probability!r}, "
            f"max_probability={self.max_probability!r}, "
            f"default_enabled={self.default_enabled!r}, "
            f"enforce_all={self.enforce_all!r}, "
            f"keys={'<redacted>' if self.keys is not None else None}, "
            f"ngram_len={self.ngram_len!r}, num_candidates={self.num_candidates!r}, "
            f"context_history_size={self.context_history_size!r}, "
            f"skip_first_ngram_calls={self.skip_first_ngram_calls!r})"
        )


_SHARED_CONFIG_FIELDS = ("default_enabled", "enforce_all")
_GUMBEL_CONFIG_FIELDS = (
    "key",
    "key_b",
    "context_window",
    "mixing_probability",
    "max_probability",
)
SYNTHID_CONFIG_FIELDS = (
    "keys",
    "ngram_len",
    "num_candidates",
    "context_history_size",
    "skip_first_ngram_calls",
)
# The config fields of each --watermark-algorithm; its keys are the choices.
ALGORITHM_CONFIG_FIELDS = {
    "gumbel": frozenset((*_SHARED_CONFIG_FIELDS, *_GUMBEL_CONFIG_FIELDS)),
    "synthid": frozenset((*_SHARED_CONFIG_FIELDS, *SYNTHID_CONFIG_FIELDS)),
}


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


def _read_config_file(path: str) -> str:
    config_path = Path(path).expanduser()
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
        descriptor = os.open(config_path, flags)
        with os.fdopen(descriptor, encoding="utf-8") as config_file:
            file_stat = os.fstat(config_file.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise WatermarkConfigError(
                    "watermark config must be a readable regular file"
                )
            if stat.S_IMODE(file_stat.st_mode) & 0o077:
                logger.warning(
                    "watermark config is readable by group or other users; "
                    "restrict it to the server account"
                )
            return config_file.read(_MAX_CONFIG_BYTES + 1)
    except (OSError, UnicodeError):
        raise WatermarkConfigError("failed to read watermark config JSON") from None


def load_watermark_config(source: str, *, algorithm: str) -> WatermarkServerConfig:
    if source.lstrip().startswith("{"):
        payload = source
    else:
        payload = _read_config_file(source)
    if len(payload.encode("utf-8")) > _MAX_CONFIG_BYTES:
        raise WatermarkConfigError("watermark config exceeds 4096 bytes")
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        raise WatermarkConfigError("failed to read watermark config JSON") from None
    if not isinstance(raw, dict):
        raise WatermarkConfigError("watermark config must be a JSON object")
    if set(raw) - set(WatermarkServerConfig.__struct_fields__):
        raise WatermarkConfigError("watermark config contains unknown fields")
    # None for an unknown algorithm, which server-argument validation rejects.
    allowed = ALGORITHM_CONFIG_FIELDS.get(algorithm)
    set_fields = {name for name, value in raw.items() if value is not None}
    if allowed is not None and set_fields - allowed:
        raise WatermarkConfigError(
            f"watermark config fields {sorted(set_fields - allowed)} are not "
            f"supported with --watermark-algorithm {algorithm}"
        )

    try:
        for name in ("key", "key_b"):
            if raw.get(name) is not None:
                parse_watermark_key(raw[name])
    except ValueError as error:
        raise WatermarkConfigError(str(error)) from error
    context_window = raw.get("context_window")
    if context_window is not None and (
        isinstance(context_window, bool)
        or not isinstance(context_window, int)
        or not 1 <= context_window <= MAX_WATERMARK_CONTEXT_WINDOW
    ):
        raise WatermarkConfigError(
            "watermark config context_window must be an integer from 1 to 64"
        )
    for name in ("mixing_probability", "max_probability"):
        value = raw.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise WatermarkConfigError(f"watermark config {name} must be a number")
    for name in ("default_enabled", "enforce_all", "skip_first_ngram_calls"):
        if raw.get(name) is not None and not isinstance(raw[name], bool):
            raise WatermarkConfigError(f"watermark config {name} must be a boolean")
    keys = raw.get("keys")
    if keys is not None and (
        not isinstance(keys, list) or not all(_is_int(key) for key in keys)
    ):
        raise WatermarkConfigError("watermark config keys must be a list of integers")
    for name in ("ngram_len", "num_candidates", "context_history_size"):
        value = raw.get(name)
        if value is not None and not _is_int(value):
            raise WatermarkConfigError(f"watermark config {name} must be an integer")
    return WatermarkServerConfig(**raw)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class WatermarkRequestConfig(msgspec.Struct, frozen=True, kw_only=True):
    enabled: Optional[bool] = None
    key: Optional[str] = None
    context_window: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"WatermarkRequestConfig(enabled={self.enabled!r}, key=<redacted>, "
            f"context_window={self.context_window!r})"
        )


def normalize_watermark_request(value: Any) -> Optional[WatermarkRequestConfig]:
    if value is None:
        return None
    if isinstance(value, WatermarkRequestConfig):
        enabled = value.enabled
        key = value.key
        context_window = value.context_window
    else:
        if not isinstance(value, dict):
            raise ValueError("watermark must be an object")
        unknown = set(value) - {"enabled", "key", "context_window"}
        if unknown:
            raise ValueError("watermark contains unknown fields")
        enabled = value.get("enabled")
        key = value.get("key")
        context_window = value.get("context_window")
    if enabled is not None and type(enabled) is not bool:
        raise ValueError("watermark enabled must be a boolean")
    if key is not None:
        parse_watermark_key(key)
    if context_window is not None and (
        isinstance(context_window, bool)
        or not isinstance(context_window, int)
        or context_window < 1
    ):
        raise ValueError("watermark context_window must be a positive integer")
    return WatermarkRequestConfig(
        enabled=enabled,
        key=key,
        context_window=context_window,
    )
