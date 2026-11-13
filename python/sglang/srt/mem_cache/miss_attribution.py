"""Why a request's prompt tokens missed the prefix cache.

With ``--enable-cache-miss-attribution`` the radix cache splits each new
request's uncached prompt tokens, at its first admission, into four parts
that tile ``(hit, prompt]``:

- ``new``: no earlier request computed these tokens;
- ``placement``: computed before, but past the deepest position where a state
  was ever saved on this prefix, so no cache of any size could resume there;
- ``dependency``: the KV up to a saved position is still cached, but the state
  there is not;
- ``capacity``: both the state and the KV were evicted.

A prefix counts as computed once a request publishes it to the tree
(``UnifiedRadixCache.checkpoint``), whether or not the tree keeps it; a
position counts as saved once an insert attaches a state there. For a model
without recurrent state every page boundary is a saved position, so its
dependency part is empty.

The index keeps 8-byte hashes of those prefixes and positions and the tokens
of at most one block past each block-aligned prefix, never KV. A state saved
only after the request's admission counts as ``placement``. It keeps at most
``max_entries`` continuations and as many saved positions, forgetting the
prefixes recorded least recently first, and at most 64 of each per prefix; a
forgotten prefix's tokens count as ``new`` or ``placement``.
"""

from __future__ import annotations

import hashlib
from array import array
from collections import OrderedDict
from typing import Dict, Optional, Sequence

_TOKEN_BYTES = 8
_DIGEST_BYTES = 8
# Continuations, or saved positions at one offset, kept per prefix at most.
_PER_PREFIX = 64

MISS_PARTS = ("new", "placement", "dependency", "capacity")


def _token_bytes(token_ids: Sequence[int], num_tokens: int) -> bytes:
    if isinstance(token_ids, array) and token_ids.typecode == "q":
        return token_ids[:num_tokens].tobytes()
    return array("q", token_ids[:num_tokens]).tobytes()


def _common_tokens(a: bytes, b: bytes) -> int:
    """The number of leading tokens two token byte strings share."""
    lo, hi = 0, min(len(a), len(b)) // _TOKEN_BYTES
    while lo < hi:
        mid = (lo + hi + 1) // 2
        end = mid * _TOKEN_BYTES
        if a[:end] == b[:end]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class CacheMissAttribution:
    """The computed prefixes and saved state positions seen so far, and the
    split of a new request's miss against them."""

    def __init__(
        self,
        page_size: int,
        records_states: bool,
        block: int = 256,
        max_entries: int = 1 << 15,
    ):
        self.page_size = page_size
        self.records_states = records_states
        self.block = block
        self.max_entries = max_entries
        # Well below the total, so the prefix just recorded is never forgotten.
        self._per_prefix = max(1, min(_PER_PREFIX, max_entries // 8))
        self._block_bytes = block * _TOKEN_BYTES
        # Hash of a block-aligned computed prefix -> the computed tokens that
        # followed it, at most one block each, none a prefix of another;
        # least recently recorded first.
        self._continuations: OrderedDict[bytes, list] = OrderedDict()
        self._num_continuations = 0
        # Hash of a block-aligned prefix -> {offset into the next block ->
        # hashes of the tokens up to that offset} where a state was saved.
        self._states: OrderedDict[bytes, Dict[int, dict]] = OrderedDict()
        self._num_states = 0
        # Prefix hashes of the requests still publishing, extended as they grow.
        self._request_chains: Dict[str, list] = {}

    def _chain(self, raw: bytes, num_blocks: int, start: Optional[list] = None):
        """chain[k] = the hash of the first k blocks of ``raw``."""
        chain = start if start is not None else [b""]
        step = self._block_bytes
        for k in range(len(chain) - 1, num_blocks):
            chain.append(
                hashlib.blake2b(
                    chain[k] + raw[k * step : (k + 1) * step],
                    digest_size=_DIGEST_BYTES,
                ).digest()
            )
        return chain

    def _tail_hash(self, block_hash: bytes, tail: bytes) -> bytes:
        return hashlib.blake2b(
            block_hash + tail, digest_size=_DIGEST_BYTES, person=b"tail"
        ).digest()

    def record_computed(
        self,
        token_ids: Sequence[int],
        num_tokens: int,
        request_id: Optional[str] = None,
        finished: bool = False,
    ) -> None:
        """Tokens ``[0, num_tokens)`` were computed; every prefix of them now
        counts as computed. ``request_id`` keeps the request's prefix hashes
        between calls until it finishes."""
        raw = _token_bytes(token_ids, num_tokens)
        num_blocks = num_tokens // self.block
        start = None
        if request_id is not None:
            start = self._request_chains.get(request_id)
        chain = self._chain(raw, num_blocks, start)
        if request_id is not None:
            if finished:
                self._request_chains.pop(request_id, None)
            else:
                self._request_chains[request_id] = chain
        step = self._block_bytes
        for k in range(num_blocks + 1):
            self._add_continuation(chain[k], raw[k * step : (k + 1) * step])

    def _add_continuation(self, key: bytes, tail: bytes) -> None:
        tails = self._continuations.pop(key, None) or []
        self._continuations[key] = tails
        for i, known in enumerate(tails):
            if known.startswith(tail):
                return
            if tail.startswith(known):
                tails[i] = tail
                return
        tails.append(tail)
        self._num_continuations += 1
        if len(tails) > self._per_prefix:
            del tails[0]
            self._num_continuations -= 1
        while self._num_continuations > self.max_entries:
            _, forgotten = self._continuations.popitem(last=False)
            self._num_continuations -= len(forgotten)

    def record_state(self, token_ids: Sequence[int], position: int) -> None:
        """A state was saved after ``token_ids[:position]``."""
        if position <= 0:
            return
        k, r = divmod(position, self.block)
        raw = _token_bytes(token_ids, position)
        chain = self._chain(raw, k)
        tail = b""
        if r:
            tail = self._tail_hash(chain[k], raw[k * self._block_bytes :])
        offsets = self._states.pop(chain[k], None) or {}
        self._states[chain[k]] = offsets
        tails = offsets.setdefault(r, {})
        if tail in tails:
            return
        tails[tail] = None
        self._num_states += 1
        if len(tails) > self._per_prefix:
            del tails[next(iter(tails))]
            self._num_states -= 1
        while self._num_states > self.max_entries:
            _, forgotten = self._states.popitem(last=False)
            self._num_states -= sum(map(len, forgotten.values()))

    def forget_request(self, request_id: str) -> None:
        self._request_chains.pop(request_id, None)

    def _longest_computed(self, raw: bytes, chain: list) -> int:
        step = self._block_bytes
        longest = 0
        for k, key in enumerate(chain):
            tails = self._continuations.get(key)
            if tails is None:
                break
            mine = raw[k * step : (k + 1) * step]
            longest = k * self.block + max(_common_tokens(mine, t) for t in tails)
        return longest

    def _deepest_saved(self, raw: bytes, chain: list, bound: int) -> int:
        """The deepest saved position at most ``bound`` on this prefix."""
        if not self.records_states:
            return bound // self.page_size * self.page_size
        step = self._block_bytes
        for k in range(min(bound // self.block, len(chain) - 1), -1, -1):
            offsets = self._states.get(chain[k])
            if not offsets:
                continue
            for r in sorted(offsets, reverse=True):
                position = k * self.block + r
                if position > bound:
                    continue
                if r == 0:
                    return position
                tail = raw[k * step : k * step + r * _TOKEN_BYTES]
                if self._tail_hash(chain[k], tail) in offsets[r]:
                    return position
        return 0

    def attribute(
        self,
        token_ids: Sequence[int],
        prompt_len: int,
        hit_len: int,
        kv_len: int,
    ) -> Dict[str, int]:
        """Split the ``prompt_len - hit_len`` uncached tokens of a request
        whose match served ``hit_len`` tokens and found ``kv_len`` tokens of
        cached KV (on any tier)."""
        raw = _token_bytes(token_ids, prompt_len)
        chain = self._chain(raw, prompt_len // self.block)
        # A request computes at least its last prompt token.
        computed = min(
            max(self._longest_computed(raw, chain), kv_len, hit_len),
            max(prompt_len - 1, hit_len),
        )
        kv_saved = max(self._deepest_saved(raw, chain, min(kv_len, computed)), hit_len)
        saved = max(self._deepest_saved(raw, chain, computed), kv_saved)
        return {
            "new": prompt_len - computed,
            "placement": computed - saved,
            "dependency": kv_saved - hit_len,
            "capacity": saved - kv_saved,
        }
