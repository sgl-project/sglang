# SPDX-License-Identifier: Apache-2.0
"""Router-hint seam (Workstream B placeholder).

This module isolates the ONE thing that is genuinely undecided in the KVCR ->
SGLang integration: how a dynamo-router hint (which peer to fetch a prefix from)
travels from the scheduler, through the HiCache controller, into this storage
backend so the source-side P2P fetch can be issued.

HiCacheStorage is content-addressed (`get(hash)`), but a router hint is
per-request routing metadata (`source_control_endpoint` + which block hashes).
The carrier we plan to use is `HiCacheStorageExtraInfo.extra_info`, a free-form
dict already threaded from the controller into every v2 call.

The wire form is the versioned KV-hint envelope from dynamo PR #13134 ("define
typed KV hint contract"), which SGLang RFC #36224 proposes to make a first-class
request field:

    {"protocol_version": "0.1", "message_id": ..., "actions": [
        {"action_id": ..., "action_type": "kv.source_locations",
         "action_version": "1.0",
         "payload": {"source_control_endpoint": ..., "block_hashes": [...]}}]}

This backend reads exactly one action type, ``kv.source_locations``, and ignores
every other action in the envelope -- an envelope carrying actions nobody here
implements must still deliver the one that is implemented. The bare payload is
accepted unwrapped as well, which is what dynamo PR #11695 sends today; that
path retires once #13134 lands on both sides.

Until RFC #36224 lands a typed ``KvHints`` struct in SGLang core, the envelope
arrives as a plain dict and is parsed here. The migration is then a field type
change, not a wire change: this parser is deleted, not rewritten.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import msgspec

if TYPE_CHECKING:
    from kvcr.types import BlockKey
else:
    # kvcr's BlockKey is `NewType("BlockKey", bytes)` -- an identity function at
    # runtime. Aliasing it keeps this module importable without the wheel, which
    # is what lets the schema tests run on the CPU CI tier.
    BlockKey = bytes

# Key under which the controller stashes the envelope inside
# HiCacheStorageExtraInfo.extra_info. Matches the request field name proposed by
# SGLang RFC #36224, so the eventual typed field needs no rename here.
ROUTER_HINT_KEY = "kv_hints"

# The one action type this backend implements, and the envelope version it was
# specified against. Both are matched leniently: an unknown envelope version
# still has its actions read, since an action carries its own version.
SOURCE_LOCATIONS_ACTION_TYPE = "kv.source_locations"
SOURCE_LOCATIONS_ACTION_VERSION = "1.0"

# Width of the canonical block-hash key, in hex chars. See page_hash_key().
_BLOCK_HASH_HEX_WIDTH = 16
_U64_MASK = (1 << 64) - 1


def encode_key(key: str) -> BlockKey:
    """SGLang hicache keys are hex/hash strings; KVCR BlockKey is bytes."""
    return BlockKey(key.encode("utf-8"))


def page_hash_key(key: str) -> str:
    """Canonical form of an SGLang page hash for router-hint comparison.

    SGLang page keys are full SHA256 hex digests, but the KV events it
    publishes carry only ``hash_str_to_int64(digest)`` -- i.e. the leading 16
    hex chars reinterpreted as a signed int64 (``mem_cache/utils.py``). The
    router indexes and echoes back *that* value, so the widest representation
    both sides share is the 16-hex-char prefix. Truncation is what the event
    schema already committed us to; it is not a choice made here.
    """
    return key[:_BLOCK_HASH_HEX_WIDTH].lower()


def page_hash_int(key: str) -> int:
    """The u64 block hash a page (or segment) key carries.

    Inverse of :func:`normalize_block_hash`'s int branch, and the form KVCR's
    own hint parser holds: it validates every wire hash into ``0 <= h < 1<<64``
    and compares ``KeyAdapter.decode(key)`` against that set. SGLang's
    ``hash_str_to_int64`` produces a *signed* int64 from the same 16 hex chars,
    so decoding through the hex prefix rather than through that helper is what
    keeps both sides on the unsigned value the router indexed.
    """
    return int(page_hash_key(key.split("#", 1)[0]), 16)


def normalize_block_hash(value: Union[int, str]) -> Optional[str]:
    """Map one wire-form hint block hash onto :func:`page_hash_key` form.

    Two producers exist and both must land on the same string:

    - the **dynamo router** sends ``ExternalSequenceBlockHash(u64)``, which
      serializes as a bare JSON number. Rendering it as 16 zero-padded hex
      chars inverts ``hash_str_to_int64`` exactly (Python ints are unbounded,
      so a value that arrived as a negative i64 is wrapped back into u64
      first);
    - a **direct caller** (tests, or a future SGLang-native router) sends the
      page hash as a hex string, which only needs truncating.

    Returns None for anything else, so the caller can drop the hint rather
    than silently comparing against garbage.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"{value & _U64_MASK:0{_BLOCK_HASH_HEX_WIDTH}x}"
    if isinstance(value, str) and value:
        return page_hash_key(value)
    return None


class RouterHint(msgspec.Struct, kw_only=True):
    """Mirror of the dynamo RouterHint schema (oandreeva/router_hints).

    Fields intentionally match the *compact* 2-field wire schema on the dynamo
    branch (PR #11695, head ``b4c9823b81``, "Add compact router hints for remote
    KV reuse") so this parser stays a thin adapter:

    - source_control_endpoint: ZMQ control endpoint of the peer that holds the
      prefix (host:port). This is what KVCR's control channel connects to.
    - block_hashes: root-aligned block hashes (``block_hashes[i]`` is request
      block ``i``); the target decides which suffix to fetch.

    The earlier ``target_cached_prefix_blocks`` advisory int was dropped from
    the wire in that PR -- it moved into the router-internal
    ``RouterHintRootCandidates`` and is no longer sent to the backend.

    ``block_hashes`` is stored in the canonical :func:`page_hash_key` form, not
    as it arrived: the dynamo router sends bare u64 numbers while a direct
    caller sends hex digests, and everything downstream compares against SGLang
    page keys. Normalizing once at parse time keeps that conversion out of the
    membership test, which the core runs per block key.

    Not ``frozen``: msgspec forbids ``__post_init__`` writes on a frozen struct,
    and ``covered_pages`` has to be derived there. Treat it as immutable anyway
    -- rebind through ``msgspec.structs.replace``, which re-runs the hook.
    """

    source_control_endpoint: str
    block_hashes: Tuple[str, ...] = ()
    # Derived from block_hashes in __post_init__; never passed in. The core runs
    # covers() once per block key and one prefetch fans each page out into every
    # segment, so the set is built once here rather than per call. Keeping it on
    # the struct (rather than in a keyed cache) makes the lookup independent of
    # how many hashes the hint carries.
    covered_pages: frozenset = frozenset()

    def __post_init__(self) -> None:
        self.covered_pages = frozenset(self.block_hashes)

    @classmethod
    def maybe_from_payload(cls, payload) -> Optional[RouterHint]:
        """Build a hint from a raw wire dict, or None if it is not well-formed."""
        if not isinstance(payload, dict):
            return None
        endpoint = payload.get("source_control_endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            return None
        raw_hashes = payload.get("block_hashes")
        if not isinstance(raw_hashes, (list, tuple)):
            return None
        normalized: List[str] = []
        for raw_hash in raw_hashes:
            canonical = normalize_block_hash(raw_hash)
            if canonical is None:
                # A hint is root-aligned, so a value we cannot interpret breaks
                # the alignment of every block after it. Truncate rather than
                # drop the entry and silently shift the remaining hashes.
                break
            normalized.append(canonical)
        return cls(source_control_endpoint=endpoint, block_hashes=tuple(normalized))

    @classmethod
    def maybe_from_envelope(cls, envelope) -> Optional[RouterHint]:
        """Pull the ``kv.source_locations`` payload out of a v0.1 KV-hint envelope.

        A bare payload (no ``actions`` list) is accepted unwrapped, which is the
        pre-envelope shape dynamo PR #11695 sends. Actions of other types are
        skipped rather than rejected: an envelope is a list of independent
        actions, so one this backend does not implement must not suppress one it
        does. The first well-formed match wins.
        """
        if not isinstance(envelope, dict):
            return None
        actions = envelope.get("actions")
        if actions is None:
            return cls.maybe_from_payload(envelope)
        if not isinstance(actions, (list, tuple)):
            return None
        for action in actions:
            if not isinstance(action, dict):
                continue
            if action.get("action_type") != SOURCE_LOCATIONS_ACTION_TYPE:
                continue
            # A newer action version may reshape the payload, so parsing it
            # against this schema would silently misread it. Skip instead.
            if action.get("action_version") != SOURCE_LOCATIONS_ACTION_VERSION:
                continue
            hint = cls.maybe_from_payload(action.get("payload"))
            if hint is not None:
                return hint
        return None

    @classmethod
    def maybe_from_extra_info(cls, extra_info) -> Optional[RouterHint]:
        """Best-effort extraction from a HiCacheStorageExtraInfo.

        Returns None whenever no well-formed hint is present -- the backend then
        falls back to local-only behavior. This must never raise on malformed
        input: a bad hint should degrade to "no remote fetch", not crash a
        prefetch. (Fail-closed, matching the vLLM KVCR manager's hint handling.)
        """
        if extra_info is None:
            return None
        raw = extra_info.extra_info
        if not isinstance(raw, dict):
            return None
        return cls.maybe_from_envelope(raw.get(ROUTER_HINT_KEY))

    def covers(self, key: str) -> bool:
        """Is this SGLang page key (or one of its segment keys) in the hint?

        Accepts a segment key (``<page hash>#<seg>``) as well as a bare page
        key, because the KVCR core runs its membership test on the per-segment
        block identity that :meth:`KVCRStore._segment_key` produced, while the
        hint only ever names whole pages.
        """
        return page_hash_key(key.split("#", 1)[0]) in self.covered_pages

    def to_kvcr_hint(self) -> dict:
        """This hint in the shape KVCR's own parser accepts.

        Since kvcr#14 the core parses the hint and owns membership, so what
        crosses ``submit_hint`` is a plain dict rather than this struct. Block
        hashes go over as **unsigned** ints: the core validates them into
        ``0 <= h < 1<<64`` and compares them against ``KeyAdapter.decode``, so
        a signed value here would be rejected outright, and a signed decode on
        the other side would miss every block without erroring.
        """
        return {
            "source_control_endpoint": self.source_control_endpoint,
            "block_hashes": [int(h, 16) for h in self.block_hashes],
            "mode": "copy",
        }


class StrKeyAdapter:
    """KVCR ``KeyAdapter`` over SGLang string keys.

    The core owns hint membership since kvcr#14: it parses the hint itself into
    a ``frozenset[int]`` and tests ``decode(key) in hashes``, so this adapter
    only translates keys in both directions. ``encode`` maps a framework key
    (str or bytes) to a KVCR :class:`BlockKey`; ``decode`` maps a *segment* key
    back to the u64 block hash the router indexed.

    Kept torch-free here (alongside :class:`RouterHint`) so the KVCR<->SGLang
    hint contract can be exercised against the real core without importing the
    GPU/host-pool stack. ``KVCRStore`` imports this class directly.
    """

    def encode(self, framework_key: object) -> BlockKey:
        if isinstance(framework_key, bytes):
            return BlockKey(framework_key)
        if isinstance(framework_key, str):
            return encode_key(framework_key)
        raise TypeError(f"unsupported KVCR framework key: {type(framework_key)!r}")

    def decode(self, key: BlockKey) -> int:
        return page_hash_int(key.decode("utf-8"))
