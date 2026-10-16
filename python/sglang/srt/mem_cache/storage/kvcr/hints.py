# SPDX-License-Identifier: Apache-2.0
"""Request-scoped hints; no hint is ever shared between unrelated requests."""

from __future__ import annotations

from urllib.parse import urlsplit

import msgspec


def rank_offset(config) -> int:
    return config.tp_rank + config.tp_size * (
        config.attn_cp_rank + config.attn_cp_size * config.pp_rank
    )


def rank_endpoint(endpoint: str, offset: int) -> str | None:
    """Select a neighboring endpoint, rejecting malformed control addresses."""
    try:
        source = urlsplit(endpoint)
        port = source.port
        if (
            source.scheme != "tcp"
            or not source.hostname
            or port is None
            or not 0 < port + offset <= 65535
            or source.path
            or source.query
            or source.fragment
            or source.username is not None
        ):
            return None
    except ValueError:
        return None
    host = source.hostname
    if ":" in host:
        host = f"[{host}]"
    return f"tcp://{host}:{port + offset}"


def peer_hint(extra_info, offset: int) -> dict | None:
    """Validate the supported action and select this rank's source endpoint.

    Dynamo advertises the TP/CP/PP-zero endpoint of the chosen DP replica.
    Neighbor ports represent the other shards in that same replica.
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
