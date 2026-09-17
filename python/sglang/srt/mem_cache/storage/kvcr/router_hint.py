# SPDX-License-Identifier: Apache-2.0
"""Router-hint extraction and KVCR key translation for the KVCR linker.

A request may carry a versioned KV-hint envelope (``Req.kv_hints``). The KVCR
linker consumes exactly one action type, ``kv.fetch@1.0``, whose payload names
the peer's control endpoint and the router's 64-bit block hashes. Everything
here is torch-free so the contract can be tested on the CPU CI tier.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional

import msgspec

if TYPE_CHECKING:
    from kvcr.types import BlockKey
else:
    # kvcr's BlockKey is a NewType over bytes; aliasing keeps this module
    # importable without the wheel.
    BlockKey = bytes

# Action the KVCR core's own hint parser understands (kvcr/hint_parser.py) and
# the Dynamo router emits (lib/kv-router/src/kv_hints.rs).
KV_FETCH_ACTION_TYPE = "kv.fetch"
KV_FETCH_ACTION_VERSION = "1.0"
KV_HINT_PROTOCOL_VERSION = "0.1"

# Storage keys are ``<page-hash>#kvcr-linker-v1#<digest>#<pool>``. The router
# indexes the leading 16 hex chars of the page hash as an int64, so that prefix
# is the only identity the two sides share; the full key stays authoritative.
KEY_NAMESPACE = "kvcr-linker-v1"
_KEY_SEPARATOR = "#"
_EVENT_HASH_HEX_WIDTH = 16
_U64_MASK = (1 << 64) - 1
_CONTROL_SCHEME = "tcp://"
_UNDIALABLE_HOSTS = frozenset({"0.0.0.0", "::", "[::]", "*"})
MAX_TCP_PORT = 65535


class KVCRFetchHint(msgspec.Struct, frozen=True, kw_only=True):
    """The ``kv.fetch`` payload fields the linker forwards to KVCR."""

    source_control_endpoint: str
    # Unsigned 64-bit router block hashes, root-aligned.
    block_hashes: tuple[int, ...]

    def to_kvcr_hint(self) -> dict[str, Any]:
        """The envelope shape ``KVCR.submit_hint`` validates."""
        return {
            "protocol_version": KV_HINT_PROTOCOL_VERSION,
            "message_id": "sglang-kvcr-linker",
            "actions": [
                {
                    "action_id": "kv-fetch",
                    "action_type": KV_FETCH_ACTION_TYPE,
                    "action_version": KV_FETCH_ACTION_VERSION,
                    "payload": {
                        "source_control_endpoint": self.source_control_endpoint,
                        "block_hashes": list(self.block_hashes),
                        "mode": "copy",
                    },
                }
            ],
        }


def normalize_block_hash(value: object) -> Optional[int]:
    """Map a wire block hash onto the unsigned 64-bit value KVCR compares.

    The router serializes ``u64`` hashes as JSON numbers, which a signed
    decoder may hand back negative; SGLang's own event hash is a signed int64
    of the same bits. Both wrap back to the unsigned value. A hex digest is
    accepted for direct callers and truncated to the event prefix.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value & _U64_MASK
    if isinstance(value, str) and value:
        try:
            return int(value[:_EVENT_HASH_HEX_WIDTH], 16)
        except ValueError:
            return None
    return None


def _envelope_actions(envelope: object) -> Optional[list]:
    if envelope is None:
        return None
    if isinstance(envelope, Mapping):
        actions = envelope.get("actions")
        return list(actions) if isinstance(actions, (list, tuple)) else None
    # A typed KvHintsEnvelope (msgspec Struct) from the request transport.
    actions = getattr(envelope, "actions", None)
    return list(actions) if isinstance(actions, (list, tuple)) else None


def _action_field(action: object, name: str):
    if isinstance(action, Mapping):
        return action.get(name)
    return getattr(action, name, None)


def parse_fetch_hint(envelope: object) -> Optional[KVCRFetchHint]:
    """Extract the first well-formed ``kv.fetch@1.0`` action, or None.

    Never raises: a malformed hint degrades to local-only behavior. Actions of
    other types are skipped rather than rejected, and a hash that cannot be
    interpreted truncates the root-aligned list instead of shifting it.
    """
    actions = _envelope_actions(envelope)
    if not actions:
        return None
    for action in actions:
        if _action_field(action, "action_type") != KV_FETCH_ACTION_TYPE:
            continue
        if _action_field(action, "action_version") != KV_FETCH_ACTION_VERSION:
            continue
        payload = _action_field(action, "payload")
        if not isinstance(payload, Mapping):
            continue
        if payload.get("mode", "copy") != "copy" or "no_retain" in payload:
            continue
        endpoint = payload.get("source_control_endpoint")
        raw_hashes = payload.get("block_hashes")
        if not isinstance(endpoint, str) or not endpoint:
            continue
        if not isinstance(raw_hashes, (list, tuple)):
            continue
        hashes: list[int] = []
        for raw in raw_hashes:
            value = normalize_block_hash(raw)
            if value is None:
                break
            hashes.append(value)
        if not hashes:
            continue
        return KVCRFetchHint(
            source_control_endpoint=endpoint, block_hashes=tuple(hashes)
        )
    return None


def split_control_endpoint(endpoint: str) -> Optional[tuple[str, int]]:
    """``(host, port)`` of a dialable ``tcp://host:port`` endpoint, else None.

    Splits on the last colon so a bracketed IPv6 literal survives.
    """
    if not endpoint.startswith(_CONTROL_SCHEME):
        return None
    prefix, sep, port = endpoint.rpartition(":")
    if not sep or not port.isdigit():
        return None
    port_num = int(port)
    if not 1 <= port_num <= MAX_TCP_PORT:
        return None
    host = prefix[len(_CONTROL_SCHEME) :]
    if not host or host in _UNDIALABLE_HOSTS:
        return None
    return host, port_num


def offset_control_endpoint(endpoint: str, offset: int) -> Optional[str]:
    """The endpoint with ``offset`` added to its port, or None if undialable."""
    split = split_control_endpoint(endpoint)
    if split is None:
        return None
    host, port = split
    if port + offset > MAX_TCP_PORT or port + offset < 1:
        return None
    return f"{_CONTROL_SCHEME}{host}:{port + offset}"


def encode_object_key(page_hash: str, digest: str, pool: str) -> BlockKey:
    """One KVCR object per physical pool page."""
    return BlockKey(
        _KEY_SEPARATOR.join((page_hash, KEY_NAMESPACE, digest, pool)).encode("utf-8")
    )


def decode_object_key(key: bytes) -> tuple[str, str, str, str]:
    """``(page_hash, namespace, digest, pool)`` of an object key."""
    parts = key.decode("utf-8").split(_KEY_SEPARATOR, 3)
    if len(parts) != 4:
        raise ValueError(f"malformed KVCR linker key: {key!r}")
    return parts[0], parts[1], parts[2], parts[3]


def page_hash_event_int(page_hash: str) -> int:
    """Unsigned 64-bit event identity of a full page hash."""
    return int(page_hash[:_EVENT_HASH_HEX_WIDTH], 16)


def page_hash_to_int64(page_hash: str) -> int:
    """Signed int64 event hash, matching ``hash_str_to_int64``."""
    value = page_hash_event_int(page_hash)
    return value - (1 << 64) if value >= 1 << 63 else value


class KVCRLinkerKeyAdapter:
    """KVCR ``KeyAdapter`` over linker object keys.

    ``decode`` hands the core the unsigned router hash so hint membership works
    for every pool object of a hinted page.
    """

    def encode(self, framework_key: object) -> BlockKey:
        if isinstance(framework_key, bytes):
            return BlockKey(framework_key)
        if isinstance(framework_key, str):
            return BlockKey(framework_key.encode("utf-8"))
        raise TypeError(f"unsupported KVCR framework key: {type(framework_key)!r}")

    def decode(self, key: BlockKey) -> int:
        page_hash, _namespace, _digest, _pool = decode_object_key(key)
        return page_hash_event_int(page_hash)


def compatibility_digest(identity: Mapping[str, Any]) -> str:
    """Short digest over the byte-compatibility identity of a cache layout.

    Two ranks whose digests agree hold byte-identical KV for the same page
    hash. The caller supplies a JSON-serializable mapping; ordering is
    canonicalized here so field insertion order cannot change the digest.
    """
    canonical = json.dumps(dict(identity), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_EVENT_HASH_HEX_WIDTH]


def unique_page_hashes_from_keys(keys: Iterable[bytes]) -> list[str]:
    """Page hashes named by a sequence of object keys, deduplicated in order."""
    seen: dict[str, None] = {}
    for key in keys:
        try:
            page_hash = decode_object_key(key)[0]
        except (UnicodeDecodeError, ValueError):
            continue
        seen.setdefault(page_hash, None)
    return list(seen)
