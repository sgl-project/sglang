# SPDX-License-Identifier: Apache-2.0
"""Stable, bounded model-namespace encoding for actual cache/store lookup keys."""

import hashlib

import msgspec

from sglang.srt.managers.kv_control.contracts import ControlError

SALT_ENCODING = 'sha256 of compact UTF-8 JSON ["sglang.kv.weight.v1", namespace, original_salt_or_null]'


def encode_cache_salt(namespace: str, original_salt: str | None) -> str:
    if original_salt is not None and not isinstance(original_salt, str):
        raise ControlError("invalid_payload", "Cache salt must be a string or null")
    try:
        return hashlib.sha256(
            msgspec.json.encode(
                ("sglang.kv.weight.v1", namespace, original_salt or None)
            )
        ).hexdigest()
    except (TypeError, UnicodeError, ValueError) as error:
        raise ControlError(
            "invalid_payload", "Cache salt is not valid UTF-8 JSON"
        ) from error
