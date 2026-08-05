# SPDX-License-Identifier: Apache-2.0
"""Router-hint seam (Workstream B placeholder).

This module isolates the ONE thing that is genuinely undecided in the KVCC ->
SGLang integration: how a dynamo-router hint (which peer to fetch a prefix from)
travels from the scheduler, through the HiCache controller, into this storage
backend so the source-side P2P fetch can be issued.

HiCacheStorage is content-addressed (`get(hash)`), but a router hint is
per-request routing metadata (`source_control_endpoint` + which block hashes).
The carrier we plan to use is `HiCacheStorageExtraInfo.extra_info`, a free-form
dict already threaded from the controller into every v2 call.

It mirrors the wire shape from dynamo (`oandreeva/router_hints`, PR #11695 --
"Add compact router hints for remote KV reuse", still open at head
`b4c9823b81`) and a parser that tolerates its absence, so the store has a stable
seam to code against. The schema is now the compact 2-field form on that PR; if
the PR shifts before merge, this struct and `RFC_kvcc_hicache_backend.md` are
the two places to update in lockstep.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Tuple, Union

import msgspec
from kvcc.types import BlockKey

# Key under which the controller is expected to stash the hint inside
# HiCacheStorageExtraInfo.extra_info. Placeholder name.
ROUTER_HINT_KEY = "kvcc_router_hint"

# Width of the canonical block-hash key, in hex chars. See page_hash_key().
_BLOCK_HASH_HEX_WIDTH = 16
_U64_MASK = (1 << 64) - 1


def encode_key(key: str) -> BlockKey:
    """SGLang hicache keys are hex/hash strings; KVCC BlockKey is bytes."""
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


class RouterHint(msgspec.Struct, frozen=True, kw_only=True):
    """Mirror of the dynamo RouterHint schema (oandreeva/router_hints).

    Fields intentionally match the *compact* 2-field wire schema on the dynamo
    branch (PR #11695, head ``b4c9823b81``, "Add compact router hints for remote
    KV reuse") so this parser stays a thin adapter:

    - source_control_endpoint: ZMQ control endpoint of the peer that holds the
      prefix (host:port). This is what KVCC's control channel connects to.
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
    """

    source_control_endpoint: str
    # A tuple, not a list, so the struct stays hashable and the covered-page set
    # can be memoized across the per-block-key membership tests.
    block_hashes: Tuple[str, ...] = ()

    @classmethod
    def maybe_from_payload(cls, payload) -> Optional["RouterHint"]:
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
        return cls(
            source_control_endpoint=endpoint, block_hashes=tuple(normalized)
        )

    @classmethod
    def maybe_from_extra_info(cls, extra_info) -> Optional["RouterHint"]:
        """Best-effort extraction from a HiCacheStorageExtraInfo.

        Returns None whenever no well-formed hint is present -- the backend then
        falls back to local-only behavior. This must never raise on malformed
        input: a bad hint should degrade to "no remote fetch", not crash a
        prefetch. (Fail-closed, matching the vLLM KVCC manager's hint handling.)
        """
        if extra_info is None:
            return None
        raw = getattr(extra_info, "extra_info", None)
        if not isinstance(raw, dict):
            return None
        return cls.maybe_from_payload(raw.get(ROUTER_HINT_KEY))

    def covers(self, key: str) -> bool:
        """Is this SGLang page key (or one of its segment keys) in the hint?

        Accepts a segment key (``<page hash>#<seg>``) as well as a bare page
        key, because the KVCC core runs its membership test on the per-segment
        block identity that :meth:`KVCCStore._segment_key` produced, while the
        hint only ever names whole pages.
        """
        return page_hash_key(key.split("#", 1)[0]) in _covered_pages(
            tuple(self.block_hashes)
        )


@lru_cache(maxsize=64)
def _covered_pages(block_hashes: Tuple[str, ...]) -> frozenset:
    """Memoized page set for :meth:`RouterHint.covers`.

    The KVCC core runs the membership test once per block key, and one prefetch
    fans a page out into every segment, so rebuilding the set per call is the
    hot path. Keyed on the hashes rather than the hint, so a hint built directly
    with a list (rather than through ``maybe_from_payload``) still memoizes.
    """
    return frozenset(block_hashes)


class StrKeyHintAdapter:
    """KVCC ``KeyHintAdapter`` over SGLang string keys + our :class:`RouterHint`.

    The core calls ``matches(key, hint)`` to decide whether a queried block is
    covered by the current request's router hint. The key it passes is a
    *segment* key, so membership goes through :meth:`RouterHint.covers`, which
    strips the segment suffix and compares in canonical page-hash form.
    ``encode`` maps a framework key (str or bytes) to a KVCC :class:`BlockKey`.

    Kept torch-free here (alongside :class:`RouterHint`) so the KVCC<->SGLang
    hint contract can be exercised against the real core without importing the
    GPU/host-pool stack. ``KVCCStore`` imports this class directly.
    """

    def encode(self, framework_key: object) -> BlockKey:
        if isinstance(framework_key, bytes):
            return BlockKey(framework_key)
        if isinstance(framework_key, str):
            return encode_key(framework_key)
        raise TypeError(f"unsupported KVCC framework key: {type(framework_key)!r}")

    def matches(self, key: BlockKey, hint: object) -> bool:
        if not isinstance(hint, RouterHint):
            return False
        return hint.covers(key.decode("utf-8"))
