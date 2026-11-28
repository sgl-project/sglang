# SPDX-License-Identifier: Apache-2.0
"""Request-scoped hints; no hint is ever shared between unrelated requests."""

from __future__ import annotations

import msgspec

from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.mem_cache.storage.kvcr.endpoints import rank_endpoint


class FetchAction(msgspec.Struct, frozen=True):
    """One usable kv.fetch action, its endpoint already offset to a shard."""

    message_id: str
    action_id: str
    endpoint: str
    hashes: frozenset


def parse_router_hint(envelope, offset: int) -> tuple[FetchAction, ...]:
    """Validate the supported actions and select a shard's source endpoint.

    Dynamo advertises the TP/CP/PP-zero endpoint of the chosen DP replica.
    Neighbor ports represent the other shards in that same replica.
    """
    if isinstance(envelope, msgspec.Struct):
        envelope = msgspec.to_builtins(envelope)
    if not isinstance(envelope, dict) or envelope.get("protocol_version") != "0.1":
        return ()
    raw_actions = envelope.get("actions")
    if not isinstance(raw_actions, list):
        return ()
    actions = []
    for action in raw_actions:
        if not isinstance(action, dict) or (
            action.get("action_type") != "kv.fetch"
            or action.get("action_version") != "1.0"
        ):
            continue
        payload = action.get("payload")
        if not isinstance(payload, dict) or payload.get("mode", "copy") != "copy":
            continue
        no_retain = payload.get("no_retain")
        if no_retain is not None and no_retain is not False:
            # Retention marks are not forwarded; false/null mean no mark.
            continue
        hashes = payload.get("block_hashes")
        if not isinstance(hashes, list) or not hashes:
            continue
        if any(type(h) is not int or not 0 <= h < 1 << 64 for h in hashes):
            continue
        endpoint = rank_endpoint(payload.get("source_control_endpoint"), offset)
        if endpoint is None:
            continue
        actions.append(
            FetchAction(
                message_id=str(envelope.get("message_id", "")),
                action_id=str(action.get("action_id", "")),
                endpoint=endpoint,
                hashes=frozenset(hashes),
            )
        )
    return tuple(actions)


def fetch_hint(
    endpoint: str, hashes, *, message_id: str = "", action_id: str = ""
) -> KvHintsEnvelope:
    return KvHintsEnvelope(
        protocol_version="0.1",
        message_id=message_id,
        actions=[
            KvHintAction(
                action_id=action_id,
                action_type="kv.fetch",
                action_version="1.0",
                payload={
                    "source_control_endpoint": endpoint,
                    "block_hashes": list(hashes),
                    "mode": "copy",
                },
            )
        ],
    )


def router_hash(page_key: str) -> int:
    # The u64 prefix of a full storage-page hash, as router events carry it.
    return int(page_key[:16], 16)


def _bound(event_hashes: dict, key: str, advertised: frozenset) -> bool:
    event_hash = event_hashes.get(key)
    return type(event_hash) is int and event_hash in advertised


def peer_hint(actions, page_keys, event_hashes=None) -> KvHintsEnvelope | None:
    """The first action naming any of these pages, narrowed to them.

    Only requested pages bound to advertised event IDs are authorized. A
    namespaced request (cache_salt or extra_key) carries event_hashes, its
    storage page key -> KV event block ID mapping, since its page hashes
    differ from router events; an empty mapping binds nothing.
    """
    if not actions or not page_keys:
        return None
    if event_hashes is not None and not isinstance(event_hashes, dict):
        return None
    for action in actions:
        if event_hashes is None:
            candidates = map(router_hash, page_keys)
            selected = [h for h in dict.fromkeys(candidates) if h in action.hashes]
        else:
            selected = list(
                dict.fromkeys(
                    router_hash(key)
                    for key in page_keys
                    if _bound(event_hashes, key, action.hashes)
                )
            )
        if selected:
            return fetch_hint(
                action.endpoint,
                selected,
                message_id=action.message_id,
                action_id=action.action_id,
            )
    return None
