"""Server-side algorithm selection and request watermark policy."""

from __future__ import annotations

import json
from typing import Any

from sglang.srt.arg_groups.overrides import ResolvedView

from .config import WatermarkConfigError, _read_config_file, load_watermark_config
from .core import resolve_watermark_request

_SYNTHID_FIELDS = {
    "ngram_len": "synthid_watermark_ngram_len",
    "top_k": "synthid_watermark_top_k",
    "context_history_size": "synthid_watermark_context_history_size",
    "skip_first_ngram_calls": "synthid_watermark_skip_first_ngram_calls",
}


def resolve_watermark_config(view: ResolvedView) -> dict:
    has_key_file = view.synthid_watermark_key_file is not None
    enabled = view.enable_watermark or has_key_file
    algorithm = view.watermark_algorithm or ("synthid" if has_key_file else "gumbel")
    if algorithm not in {"synthid", "gumbel"}:
        raise ValueError("watermark algorithm must be synthid or gumbel")
    if has_key_file and algorithm != "synthid":
        raise ValueError("SynthID key configuration cannot be combined with Gumbel")
    if not enabled:
        if view.watermark_config is not None or view.watermark_algorithm is not None:
            raise ValueError(
                "--watermark-config and --watermark-algorithm require --enable-watermark"
            )
        return {}
    resolved = {"enable_watermark": True, "watermark_algorithm": algorithm}
    if view.watermark_config is None:
        if algorithm == "synthid" and not has_key_file:
            raise ValueError(
                "SynthID requires a key file in --watermark-config or "
                "--synthid-watermark-key-file"
            )
        return resolved
    if algorithm == "gumbel":
        config = load_watermark_config(view.watermark_config)
        resolved.update(
            {
                f"watermark_{field}": value
                for field in config.__struct_fields__
                if (value := getattr(config, field)) is not None
            }
        )
        return resolved

    raw = _load_synthid_config(view.watermark_config)
    key_file = raw.get("key_file")
    if key_file is None and "keys" in raw:
        key_file = view.watermark_config
    if not isinstance(key_file, str) or not key_file:
        raise WatermarkConfigError(
            "SynthID config requires key_file or an integer keys list"
        )
    if has_key_file and key_file != view.synthid_watermark_key_file:
        raise ValueError("Conflicting SynthID key files")
    resolved["synthid_watermark_key_file"] = key_file
    for field, destination in _SYNTHID_FIELDS.items():
        if field in raw:
            resolved[destination] = raw[field]
    for name in ("default_enabled", "enforce_all"):
        if name in raw:
            resolved[f"watermark_{name}"] = raw[name]
    return resolved


def _load_synthid_config(source: str) -> dict:
    if source.lstrip().startswith("{"):
        raise WatermarkConfigError(
            "SynthID configuration must use a secret file, not inline JSON"
        )
    payload = _read_config_file(source)
    if len(payload.encode("utf-8")) > 4096:
        raise WatermarkConfigError("watermark config exceeds 4096 bytes")
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        raise WatermarkConfigError("failed to read watermark config JSON") from None
    if not isinstance(raw, dict) or set(raw) - {
        "key_file",
        "keys",
        "default_enabled",
        "enforce_all",
        *_SYNTHID_FIELDS,
    }:
        raise WatermarkConfigError("SynthID config must be an object with known fields")
    if "keys" in raw and "key_file" in raw:
        raise WatermarkConfigError("SynthID config cannot combine keys and key_file")
    for name in ("ngram_len", "top_k", "context_history_size"):
        if name in raw and type(raw[name]) is not int:
            raise WatermarkConfigError(f"SynthID config {name} must be an integer")
    for name in ("skip_first_ngram_calls", "default_enabled", "enforce_all"):
        if name in raw and type(raw[name]) is not bool:
            raise WatermarkConfigError(f"SynthID config {name} must be a boolean")
    return raw


def validate_watermark_request(sampling_params: Any, features: Any) -> bool:
    watermark = sampling_params.watermark
    if features.enable_watermark and features.watermark_algorithm == "synthid":
        if watermark is not None:
            if watermark.key is not None or watermark.context_window is not None:
                raise ValueError(
                    "SynthID keys and context are server-wide, not per-request"
                )
            if watermark.enabled is None:
                raise ValueError("SynthID request watermark must set enabled")
            if watermark.enabled is False and features.watermark_enforce_all:
                raise ValueError(
                    "request watermark enabled=false is not allowed when enforce_all is set"
                )
        return features.watermark_enforce_all or (
            watermark.enabled
            if watermark is not None
            else features.watermark_default_enabled
        )
    _, _, enabled = resolve_watermark_request(
        watermark,
        server_enabled=features.enable_watermark,
        default_key=features.watermark_key,
        default_context_window=features.watermark_context_window,
        default_enabled=features.watermark_default_enabled,
        enforce_all=features.watermark_enforce_all,
    )
    return enabled
