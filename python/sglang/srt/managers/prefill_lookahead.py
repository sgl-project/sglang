# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Starvation bookkeeping and head-prefix pinning for the opt-in NO_TOKEN
lookahead of prefill admission (SGLANG_PREFILL_NO_TOKEN_LOOKAHEAD).

No sglang imports: the tree cache is injected and used through
``inc_lock_ref`` / ``dec_lock_ref`` only.
"""

from typing import Any, Callable, Container, Optional, Tuple

# Outcomes of HeadPrefixLock.sync_head.
HEADLOCK_SYNC_IDLE = "idle"
HEADLOCK_SYNC_HELD = "held"
HEADLOCK_SYNC_PINNED = "pinned"
HEADLOCK_SYNC_RELEASED = "released"
HEADLOCK_SYNC_DENIED_CAPACITY = "denied_capacity"


def normalize_headlock_reserve_tokens(value: Optional[int], default: int) -> int:
    """Unset or 0 reserves ``default``; a negative value raises, since it would
    switch the capacity gate off while reading as configured."""
    if not value:
        return max(0, default)
    if value < 0:
        raise ValueError(
            "SGLANG_PREFILL_HEADLOCK_RESERVE_TOKENS must be a positive int "
            f"(unset or 0 uses max_prefill_tokens), got {value!r}"
        )
    return value


class PrefillLookaheadState:
    """Consecutive NO_TOKEN passes of the queue head, where the head is the
    first request an admission pass offers to the ``PrefillAdder``."""

    def __init__(self, max_candidates: int, aging_passes: int):
        self.max_candidates = max(0, max_candidates)
        self.aging_passes = max(0, aging_passes)
        self.head_rid: Optional[str] = None
        self.head_age: int = 0

    @property
    def enabled(self) -> bool:
        return self.max_candidates > 0

    def observe_head(self, head_rid: str, rejected_no_token: bool) -> int:
        """Fold one pass's head verdict into the age and return it. A head
        that was admitted, rejected for another reason, or replaced clears it."""
        if not rejected_no_token:
            self.head_rid = None
            self.head_age = 0
        elif self.head_rid == head_rid:
            self.head_age += 1
        else:
            self.head_rid = head_rid
            self.head_age = 1
        return self.head_age

    @property
    def aged_out(self) -> bool:
        """The head has waited past the bound: lookahead stays off until it is
        admitted or replaced."""
        return self.head_age > self.aging_passes


class HeadPrefixLock:
    """One tree lock pinning the matched prefix of the KV-blocked queue head.

    A lookahead candidate is admitted against free plus evictable KV, so it
    may evict; the pin keeps that eviction off the prefix the head already
    matched. One slot only: each held lock shrinks the evictable pool every
    other admission draws on.

    The pin is taken only while ``free + evictable - pin >= reserve_tokens``
    and is dropped by ``reconcile`` once headroom falls under the reserve, so
    work already in flight can still evict for its next allocation.

    The lock outlives a pass (an admitted candidate allocates when its batch
    runs), so ``reconcile`` runs at the start of every pass and drops a lock
    whose request left the waiting queue.

    ``inc_lock_ref`` is called directly rather than through ``lock()`` because
    the pin accounting reads its ``delta``; the caller must not offer a
    streaming session's virtual node, which only ``lock()`` skips.
    """

    def __init__(
        self,
        tree_cache: Any,
        logger: Any = None,
        headroom_fn: Optional[Callable[[], Tuple[int, int]]] = None,
        reserve_tokens: int = 0,
    ):
        self.tree_cache = tree_cache
        self.logger = logger
        # Returns live (allocator-free, cache-evictable) tokens. None, or a
        # non-positive reserve, leaves the capacity gate off.
        self.headroom_fn = headroom_fn
        self.reserve_tokens = max(0, reserve_tokens)
        self.rid: Optional[str] = None
        self.node: Any = None
        self._dec_params: Any = None
        self.tokens: int = 0
        self.last_outcome: str = HEADLOCK_SYNC_IDLE
        # Tokens the last pin moved from evictable to locked, and the part of
        # them the cache did not take out of its own evictable size.
        self.last_pin_tokens: int = 0
        self.last_pin_unaccounted: int = 0

    @property
    def held(self) -> bool:
        return self.rid is not None

    @property
    def locked_tokens(self) -> int:
        """Matched prefix tokens under the held lock; 0 when nothing is held."""
        return self.tokens if self.held else 0

    def _debug(self, event: str, rid: Optional[str], tokens: int, reason: str) -> None:
        if self.logger is None:
            return
        self.logger.debug(
            "prefill_lookahead head lock %s rid=%s tokens=%d reason=%s",
            event,
            rid,
            tokens,
            reason,
        )

    def _capacity_allows_pin(self, rid: str, pin_tokens: int) -> bool:
        # pin_tokens is the matched prefix length, an upper bound on what the
        # pin takes out of the evictable set; the error is on the side of
        # skipping the pin.
        if self.headroom_fn is None or self.reserve_tokens <= 0:
            return True
        available, evictable = self.headroom_fn()
        if available + evictable - pin_tokens >= self.reserve_tokens:
            return True
        if self.logger is not None:
            self.logger.debug(
                "prefill_lookahead head lock skip rid=%s tokens=%d "
                "reason=capacity free=%d evictable=%d reserve=%d",
                rid,
                pin_tokens,
                available,
                evictable,
                self.reserve_tokens,
            )
        return False

    def _headroom_under_reserve(self) -> bool:
        if self.headroom_fn is None or self.reserve_tokens <= 0:
            return False
        available, evictable = self.headroom_fn()
        return available + evictable < self.reserve_tokens

    def _record_pin(
        self, result: Any, tokens: int, evictable_before: Optional[int]
    ) -> None:
        # abs: RadixCache reports the evictable -> locked move as negative,
        # UnifiedRadixCache's FULL component as positive. A cache that reports
        # no delta is charged the whole prefix.
        delta = None if result is None else result.delta
        self.last_pin_tokens = tokens if delta is None else abs(delta)
        if evictable_before is None:
            self.last_pin_unaccounted = self.last_pin_tokens
            return
        moved = evictable_before - self.headroom_fn()[1]
        self.last_pin_unaccounted = max(0, self.last_pin_tokens - moved)

    def acquire(self, rid: str, node: Any, tokens: int = 0, reason: str = "") -> bool:
        """Pin ``node``'s prefix chain for ``rid``; True if a new lock was taken.
        The same rid and node keep the held lock; anything else replaces it."""
        # `==`, not `is`: a node handle is an int id for UnifiedRadixCache.
        if self.held and self.rid == rid and self.node == node:
            self.tokens = tokens
            self.last_outcome = HEADLOCK_SYNC_HELD
            return False
        if not self._capacity_allows_pin(rid, tokens):
            # A held lock is kept: a refused re-pin must not drop the
            # protection the head already has.
            self.last_outcome = HEADLOCK_SYNC_DENIED_CAPACITY
            return False
        if self.held:
            self.release(reason="refresh_match" if self.rid == rid else "head_changed")
        # Read after the release, so the measured move is this acquire's alone.
        evictable_before = None if self.headroom_fn is None else self.headroom_fn()[1]
        result = self.tree_cache.inc_lock_ref(node)
        self.rid = rid
        self.node = node
        self.tokens = tokens
        # The receipt replays at release exactly what this acquire locked.
        self._dec_params = None if result is None else result.to_dec_params()
        self._record_pin(result, tokens, evictable_before)
        self._debug("acquire", rid, tokens, reason or "head_no_token")
        self.last_outcome = HEADLOCK_SYNC_PINNED
        return True

    def release(self, reason: str = "") -> bool:
        """Drop the held lock, if any; True if one was dropped."""
        if not self.held:
            return False
        rid, node, params, tokens = self.rid, self.node, self._dec_params, self.tokens
        # Clear the slot first, so a raising dec is not retried on the next pass.
        self.rid = None
        self.node = None
        self._dec_params = None
        self.tokens = 0
        try:
            if params is not None:
                self.tree_cache.dec_lock_ref(node, params)
            else:
                self.tree_cache.dec_lock_ref(node)
        except Exception:
            # A lock keeps its node alive, so this is unexpected; losing one
            # lock ref is survivable and killing the scheduler loop is not.
            if self.logger is not None:
                self.logger.warning(
                    "prefill_lookahead head lock release failed rid=%s reason=%s; "
                    "slot dropped",
                    rid,
                    reason or "unspecified",
                    exc_info=True,
                )
            return True
        self._debug("release", rid, tokens, reason or "unspecified")
        return True

    def reconcile(self, alive_rids: Container[str]) -> bool:
        """Drop the lock if its request left the waiting queue or the pool no
        longer holds one reserve; True if it was dropped."""
        if not self.held:
            return False
        if self.rid not in alive_rids:
            return self.release(reason="rid_left_queue")
        if self._headroom_under_reserve():
            return self.release(reason="capacity_pressure")
        return False

    def sync_head(
        self,
        rid: str,
        node: Any,
        tokens: int,
        should_hold: bool,
        reason: str = "",
    ) -> str:
        """Fold one pass's head verdict into the slot and return a
        ``HEADLOCK_SYNC_*`` outcome. ``should_hold`` is True only for a head
        that is KV-blocked; an admitted head holds its own lock on the chain."""
        if not should_hold:
            if self.held:
                self.release(reason=reason or "head_unblocked")
                return HEADLOCK_SYNC_RELEASED
            return HEADLOCK_SYNC_IDLE
        self.acquire(rid, node, tokens, reason=reason or "head_no_token")
        return self.last_outcome
