# SPDX-License-Identifier: Apache-2.0
"""Router-hint seam between the SGLang request path and the KVCR core.

A request carries a :class:`~sglang.srt.managers.kv_hints.KvHintsEnvelope`, set
by a trusted orchestrator after worker selection. The scheduler forwards it to
the HiCache controller, which stashes it in ``HiCacheStorageExtraInfo.extra_info``
under :data:`ROUTER_HINT_KEY`; this module turns the one action this backend
implements into the hint KVCR's own parser accepts.

HiCacheStorage is content-addressed (``get(hash)``) while a hint is per-request
routing metadata, which is why the envelope rides the free-form ``extra_info``
dict rather than a storage-level argument.

Two representations meet here and they are not the same shape:

- **inbound** -- an envelope whose ``kv.fetch`` payload names a source endpoint
  and the source-side block hashes, as ``ExternalSequenceBlockHash`` u64s;
- **outbound** -- the envelope handed to ``KVCR.submit_hint``, which the core
  re-parses itself and whose ``source_control_endpoint`` this rank has realigned
  to its own within-DP offset (see ``KVCRStore._parse_hint``).

Everything in between is held in :class:`RouterHint`, in the canonical page-key
form SGLang compares against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import msgspec

from sglang.srt.managers.kv_hints import (
    KV_HINTS_PROTOCOL_VERSION,
    KvHintAction,
    KvHintsEnvelope,
)

if TYPE_CHECKING:
    from kvcr.types import BlockKey
else:
    # kvcr's BlockKey is `NewType("BlockKey", bytes)` -- an identity function at
    # runtime. Aliasing it keeps this module importable without the wheel, which
    # is what lets the schema tests run on the CPU CI tier.
    BlockKey = bytes

# Key under which the controller stashes the envelope inside
# HiCacheStorageExtraInfo.extra_info. Matches the request field name, so the
# envelope is findable under the same name at every layer it crosses.
ROUTER_HINT_KEY = "kv_hints"

# The action type this backend implements. `kvcr.hint_parser` hardcodes the same
# string, so it is the interoperable spelling rather than a name chosen here.
KV_FETCH_ACTION_TYPE = "kv.fetch"
KV_FETCH_ACTION_VERSION = "1.0"

# Pre-rename spelling of the same action, still emitted by a router that has not
# picked up the kv.fetch naming. Accepted on input only; never emitted.
_LEGACY_ACTION_TYPE = "kv.source_locations"

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
    """The one ``kv.fetch`` action this backend implements, parsed.

    - source_control_endpoint: ZMQ control endpoint of the peer that holds the
      prefix (host:port). This is what KVCR's control channel connects to.
    - block_hashes: root-aligned block hashes (``block_hashes[i]`` is request
      block ``i``); the target decides which suffix to fetch.
    - message_id / action_id: carried through from the inbound envelope so the
      hint KVCR logs is traceable back to the router's decision.

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
    message_id: str = ""
    action_id: str = ""
    # Derived from block_hashes in __post_init__; never passed in. The core runs
    # covers() once per block key and one prefetch fans each page out into every
    # segment, so the set is built once here rather than per call. Keeping it on
    # the struct (rather than in a keyed cache) makes the lookup independent of
    # how many hashes the hint carries.
    covered_pages: frozenset = frozenset()

    def __post_init__(self) -> None:
        self.covered_pages = frozenset(self.block_hashes)

    @classmethod
    def maybe_from_payload(
        cls,
        payload,
        *,
        message_id: str = "",
        action_id: str = "",
    ) -> Optional[RouterHint]:
        """Build a hint from a raw action payload, or None if not well-formed."""
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
        if not normalized:
            # KVCR rejects an empty hash set outright, so stop here rather than
            # let submit_hint raise on a hint that names nothing to fetch.
            return None
        return cls(
            source_control_endpoint=endpoint,
            block_hashes=tuple(normalized),
            message_id=message_id,
            action_id=action_id,
        )

    @classmethod
    def maybe_from_envelope(cls, envelope) -> Optional[RouterHint]:
        """Pull the ``kv.fetch`` payload out of a KV-hint envelope.

        Accepts the typed :class:`KvHintsEnvelope` the request path carries and
        the equivalent plain dict. Actions of other types are skipped rather
        than rejected: an envelope is a list of independent actions, so one this
        backend does not implement must not suppress one it does. The first
        well-formed match wins.
        """
        if isinstance(envelope, dict):
            try:
                envelope = msgspec.convert(envelope, KvHintsEnvelope)
            except (msgspec.ValidationError, TypeError):
                return None
        if not isinstance(envelope, KvHintsEnvelope):
            return None
        for action in envelope.actions:
            if action.action_type not in (KV_FETCH_ACTION_TYPE, _LEGACY_ACTION_TYPE):
                continue
            # A newer action version may reshape the payload, so parsing it
            # against this schema would silently misread it. Skip instead.
            if action.action_version != KV_FETCH_ACTION_VERSION:
                continue
            hint = cls.maybe_from_payload(
                action.payload,
                message_id=envelope.message_id,
                action_id=action.action_id,
            )
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
        """This hint as the envelope ``KVCR.submit_hint`` parses.

        The core owns hint parsing (kvcr#24), so what crosses ``submit_hint`` is
        a whole envelope rather than this struct or a bare payload -- it walks
        ``actions`` for its own ``kv.fetch`` and validates the payload itself.
        Re-emitting rather than forwarding the inbound envelope is what carries
        this rank's realigned ``source_control_endpoint``, which is the one field
        the router could not resolve (see ``KVCRStore._parse_hint``).

        Block hashes go over as **unsigned** ints: the core validates them into
        ``0 <= h < 1<<64`` and compares them against ``KeyAdapter.decode``, so
        a signed value here would be rejected outright, and a signed decode on
        the other side would miss every block without erroring.
        """
        return {
            "protocol_version": KV_HINTS_PROTOCOL_VERSION,
            "message_id": self.message_id,
            "actions": [
                {
                    "action_id": self.action_id,
                    "action_type": KV_FETCH_ACTION_TYPE,
                    "action_version": KV_FETCH_ACTION_VERSION,
                    "payload": {
                        "source_control_endpoint": self.source_control_endpoint,
                        "block_hashes": [int(h, 16) for h in self.block_hashes],
                        "mode": "copy",
                    },
                }
            ],
        }


def build_envelope(
    *,
    source_control_endpoint: str,
    block_hashes: List[int],
    message_id: str,
    action_id: str = "kv-fetch-0",
) -> KvHintsEnvelope:
    """A single-action ``kv.fetch`` envelope, for callers that produce hints.

    Used by the tests and by anything driving SGLang directly rather than
    through a router; the router builds the same shape on the wire.
    """
    return KvHintsEnvelope(
        protocol_version=KV_HINTS_PROTOCOL_VERSION,
        message_id=message_id,
        actions=[
            KvHintAction(
                action_id=action_id,
                action_type=KV_FETCH_ACTION_TYPE,
                action_version=KV_FETCH_ACTION_VERSION,
                payload={
                    "source_control_endpoint": source_control_endpoint,
                    "block_hashes": list(block_hashes),
                },
            )
        ],
    )


class StrKeyAdapter:
    """KVCR ``KeyAdapter`` over SGLang string keys.

    The core owns hint membership: it parses the hint itself into a
    ``frozenset[int]`` and tests ``decode(key) in hashes``, so this adapter only
    translates keys in both directions. ``encode`` maps a framework key (str or
    bytes) to a KVCR :class:`BlockKey`; ``decode`` maps a *segment* key back to
    the u64 block hash the router indexed.

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
