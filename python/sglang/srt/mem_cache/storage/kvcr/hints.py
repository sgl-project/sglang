# SPDX-License-Identifier: Apache-2.0
"""Request-scoped hints; no hint is ever shared between unrelated requests."""

from __future__ import annotations

import msgspec

from sglang.srt.mem_cache.storage.kvcr.endpoints import rank_endpoint


def _storage_hashes(hashes: list[int], mapping: dict, keys) -> list[int]:
    """Authorize only requested full storage keys bound to advertised event IDs."""
    if not isinstance(mapping, dict) or not keys:
        return []
    advertised = set(hashes)
    translated = []
    for key in keys:
        if isinstance(key, bytes):
            try:
                key = key.decode("ascii")
            except UnicodeDecodeError:
                continue
        if not isinstance(key, str):
            continue
        page_key = key.split("#", 1)[0]
        if len(page_key) != 64 or any(c not in "0123456789abcdef" for c in page_key):
            continue
        event_hash = mapping.get(page_key)
        if type(event_hash) is int and event_hash in advertised:
            translated.append(int(page_key[:16], 16))
    return list(dict.fromkeys(translated))


def peer_hint(extra_info, offset: int, keys=None) -> dict | None:
    """Validate the supported action and select this rank's source endpoint.

    Dynamo advertises the TP/CP/PP-zero endpoint of the chosen DP replica.
    Neighbor ports represent the other shards in that same replica.
    Namespaced storage pages use a different hash seed from router events;
    translate only through the framework's request-local full-key binding.
    """
    if extra_info is None or not extra_info.extra_info:
        return None
    envelope = extra_info.extra_info.get("kv_hints")
    if isinstance(envelope, msgspec.Struct):
        envelope = msgspec.to_builtins(envelope)
    if not isinstance(envelope, dict) or envelope.get("protocol_version") != "0.1":
        return None
    actions = envelope.get("actions")
    if not isinstance(actions, list):
        return None
    for action in actions:
        if not isinstance(action, dict) or (
            action.get("action_type") != "kv.fetch"
            or action.get("action_version") != "1.0"
        ):
            continue
        payload = action.get("payload")
        if not isinstance(payload, dict) or payload.get("mode", "copy") != "copy":
            continue
        if "no_retain" in payload:
            continue
        endpoint = payload.get("source_control_endpoint")
        hashes = payload.get("block_hashes")
        if not isinstance(endpoint, str) or not isinstance(hashes, list) or not hashes:
            continue
        if any(type(h) is not int or not 0 <= h < 1 << 64 for h in hashes):
            continue
        selected_endpoint = rank_endpoint(endpoint, offset)
        if selected_endpoint is None:
            continue
        mapping = extra_info.extra_info.get("storage_key_to_event_hash")
        if mapping is not None:
            hashes = _storage_hashes(hashes, mapping, keys)
            if not hashes:
                continue
        return {
            "protocol_version": "0.1",
            "message_id": envelope.get("message_id", ""),
            "actions": [
                {
                    "action_id": action.get("action_id", ""),
                    "action_type": "kv.fetch",
                    "action_version": "1.0",
                    "payload": {
                        "source_control_endpoint": selected_endpoint,
                        "block_hashes": hashes,
                        "mode": "copy",
                    },
                }
            ],
        }
    return None
