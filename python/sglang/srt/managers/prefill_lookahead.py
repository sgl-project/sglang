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
"""Starvation bookkeeping and head-prefix pinning for the prefill admission
NO_TOKEN lookahead.

Kept free of any sglang import so the policy can be unit-tested without a
runtime (see test/registered/unit/managers/test_prefill_lookahead.py). The tree
cache is injected into :class:`HeadPrefixLock` and used through duck typing
(``inc_lock_ref`` / ``dec_lock_ref`` / ``is_tree_cache``) for the same reason.
"""

from typing import Any, Callable, Container, Optional, Tuple

# Outcome of :meth:`HeadPrefixLock.sync_head`. The boolean `acquire` returns
# only separates "locked" from "did not lock"; the scheduler additionally has to
# tell a no-op refresh apart from a pin the capacity gate refused, because the
# latter must also suppress the pass's lookahead.
HEADLOCK_SYNC_IDLE = "idle"
HEADLOCK_SYNC_HELD = "held"
HEADLOCK_SYNC_PINNED = "pinned"
HEADLOCK_SYNC_RELEASED = "released"
HEADLOCK_SYNC_DENIED_CAPACITY = "denied_capacity"


def normalize_headlock_reserve_tokens(value: Optional[int], default: int) -> int:
    """Validate SGLANG_PREFILL_HEADLOCK_RESERVE_TOKENS against ``default``.

    Unset or 0 keeps ``default`` (the scheduler's ``max_prefill_tokens``, the
    largest extend one batch can ask the allocator for). A negative override
    would switch the capacity gate off while still reading as configured, so it
    fails startup here: the env layer downgrades a parse error to a warning plus
    the default, which would silently run with the gate disabled.
    """
    if not value:
        return max(0, default)
    if value < 0:
        raise ValueError(
            "SGLANG_PREFILL_HEADLOCK_RESERVE_TOKENS must be a positive int "
            f"(unset or 0 uses max_prefill_tokens), got {value!r}"
        )
    return value


class PrefillLookaheadState:
    """Scheduler-lifetime state for the waiting-queue admission lookahead.

    Only tracks the *head* of the scan (the first request actually offered to
    the ``PrefillAdder`` in a pass). Its consecutive-NO_TOKEN age is the
    starvation signal: once it exceeds ``aging_passes`` the scheduler stops
    letting later requests jump ahead, which on a prefill-only node is
    equivalent to reserving KV for the head — admission is the only consumer of
    freshly released KV there, so refusing to admit anyone else *is* the
    reservation.
    """

    def __init__(self, max_candidates: int, aging_passes: int):
        self.max_candidates = max(0, max_candidates)
        self.aging_passes = max(0, aging_passes)
        self.head_rid: Optional[str] = None
        self.head_age: int = 0

    @property
    def enabled(self) -> bool:
        return self.max_candidates > 0

    @property
    def uses_head_lock(self) -> bool:
        """Whether an engaged lookahead pins the blocked head's prefix and lets
        candidates through the ordinary (free + evictable) budget gate. Always
        the case once lookahead is on; env=0 builds no lock at all."""
        return self.enabled

    def observe_head(self, head_rid: str, rejected_no_token: bool) -> int:
        """Fold this pass's head verdict into the age and return the new age.

        A head that got in, or that failed for a non-KV reason, clears the
        counter; so does a change of head, since the previous head is no longer
        the one being starved.
        """
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
        """True once the current head has been starved for more passes than
        allowed, i.e. lookahead must stay off for this pass and the ones after
        it until the head changes or is admitted."""
        return self.head_age > self.aging_passes


class HeadPrefixLock:
    """A single-slot tree lock pinning the KV-blocked queue head's prefix.

    Why it exists: a lookahead candidate is admitted through ``add_one_req``'s
    ordinary budget, whose headroom includes *evictable* KV. In a warm cache
    that is the only headroom there is, so a free-page-only gate (the first
    version of this lookahead) admitted nobody. Letting a candidate evict is fine for cold
    entries — the LRU was going to take them anyway — but not for the prefix the
    blocked head already matched (and possibly loaded back from host), because
    losing it turns the head's next pass into a longer prefill, which is the
    opposite of head-of-line relief. So while the head is blocked we hold a
    normal ``inc_lock_ref`` on its matched node, which the tree treats exactly
    like a running request's lock: the whole device-resident chain from that node
    up to root leaves the evictable leaf set and cannot be evicted or dropped.

    Why a single slot: every extra held lock shrinks the evictable pool that the
    *other* admissions draw on, so pinning more than the one request we are
    protecting would trade this head's stall for everyone else's.

    Why the capacity gate: an in-flight chunked request allocates its next chunk
    straight out of ``available + evictable`` when the batch runs and never
    passes through the admission budget, so it relies on eviction at alloc time.
    A pin big enough to swallow the evictable pool leaves it nothing to evict and
    turns that allocation into ``Prefill out of memory``, which kills the
    scheduler process. A pin is therefore taken only while ``available +
    evictable - pin >= reserve_tokens`` (one full ``max_prefill_tokens`` extend),
    and a lock already held is dropped by :meth:`reconcile` once headroom falls
    under the reserve. Losing the pin degrades this head to stock admission for
    as long as the pressure lasts; losing the allocation ends the engine.

    Lifetime is deliberately longer than a pass — an admitted candidate does not
    allocate its KV until the batch runs — so correctness rests on reconciliation
    rather than on unwinding every exit path: :meth:`reconcile` runs at the top of
    each admission pass and drops a lock whose request has left the waiting queue,
    and :meth:`sync_head` drops or moves it the moment the head changes, is
    admitted, or re-matches to a different node. Any leak therefore survives at
    most one pass.

    ``tree_cache`` is duck-typed (``inc_lock_ref`` / ``dec_lock_ref`` /
    ``is_tree_cache``) so this module keeps its zero-sglang-import property; every
    implementation in mem_cache/ takes ``(node, params)`` positionally, so no
    per-class adapter is needed.
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
        # Returns (allocator-free, tree-evictable) tokens of the pool an extend
        # allocates from, split so the skip trace can report both halves. Read
        # live, never cached: the whole point is what the allocator would see.
        # None, or a non-positive reserve, leaves the capacity gate off.
        self.headroom_fn = headroom_fn
        self.reserve_tokens = max(0, reserve_tokens)
        # Slot contents; all four move together.
        self.rid: Optional[str] = None
        self.node: Any = None
        self._dec_params: Any = None
        self.tokens: int = 0
        # Last acquire attempt, for the caller: the outcome drives this pass's
        # lookahead, and the two sizes drive the pass's admission budget.
        self.last_outcome: str = HEADLOCK_SYNC_IDLE
        self.last_pin_tokens: int = 0
        self.last_pin_unaccounted: int = 0

    @property
    def held(self) -> bool:
        return self.rid is not None

    @property
    def locked_tokens(self) -> int:
        """Prefix tokens the current lock covers; 0 when nothing is held.

        This is the matched prefix length, not the eviction-accounting delta:
        the delta is 0 whenever a running request already protects the same
        chain, which would make the trace field read as "no lock" while a lock
        is very much held.
        """
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
        """Whether ``pin_tokens`` can leave the evictable set and still leave one
        ``reserve_tokens`` extend behind for the allocator.

        ``pin_tokens`` is the matched prefix length, an OVER-estimate of what the
        pin actually takes out of the evictable set (a chain a running request
        already locks costs nothing), and a lock being refreshed is given no
        credit for the reference it is about to drop. Both errors point at "skip
        the pin", which is the side that cannot crash an allocation.
        """
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
        """Whether the pool can no longer serve one ``reserve_tokens`` extend,
        with the held lock still counted out of the evictable half."""
        if self.headroom_fn is None or self.reserve_tokens <= 0:
            return False
        available, evictable = self.headroom_fn()
        return available + evictable < self.reserve_tokens

    def _record_pin(self, result: Any, tokens: int, evictable_before: Any) -> None:
        """Size the pin just taken, twice, for the pass's admission budget.

        ``last_pin_tokens`` is the evictable -> locked transition: ``delta`` is
        that amount exactly (0 when a running request already protects the
        chain), and a cache that leaves it unset (swa_radix_cache,
        mamba_radix_cache, radix_cache_cpp) falls back to the conservative full
        prefix. ``abs``, because the sign is not a convention: RadixCache and
        HiRadixCache report the move as negative while UnifiedRadixCache's FULL
        component reports the same move as positive.

        ``last_pin_unaccounted`` is the part of it the tree did NOT already take
        out of its own evictable size — 0 for every cache in mem_cache/, because
        ``inc_lock_ref`` moves those tokens itself and the admission budget reads
        that live, so charging the pin on top would double-count it and deny
        every later candidate in the pass.
        """
        delta = getattr(result, "delta", None) if result is not None else None
        self.last_pin_tokens = tokens if delta is None else abs(delta)
        if evictable_before is None:
            self.last_pin_unaccounted = self.last_pin_tokens
            return
        moved = evictable_before - self.headroom_fn()[1]
        self.last_pin_unaccounted = max(0, self.last_pin_tokens - moved)

    def acquire(self, rid: str, node: Any, tokens: int = 0, reason: str = "") -> bool:
        """Pin ``node``'s prefix chain for ``rid``. Returns True if a new lock
        was taken, and records the attempt in ``last_outcome`` either way.

        Idempotent for the request/node pair already held, so a head that stays
        blocked over many passes never stacks references. A different rid or a
        different node releases the old slot first — the two calls are adjacent
        in the scheduler thread, so nothing can evict the shared ancestors in
        between.
        """
        # `==`, not `is`: a node handle is a NodeId (a plain int) for
        # UnifiedRadixCache and a TreeNode object elsewhere. No TreeNode defines
        # __eq__, so `==` is identity for objects, while `is` on an int id above
        # CPython's small-int cache would spuriously read as "changed" and churn
        # the lock every pass.
        if self.held and self.rid == rid and self.node == node:
            # Same head, same match: keep the reference we already hold and just
            # refresh the reported size. Not gated on capacity — no new tokens
            # leave the evictable set, and the held lock is reconcile's problem.
            self.tokens = tokens
            self.last_outcome = HEADLOCK_SYNC_HELD
            return False
        if not self._capacity_allows_pin(rid, tokens):
            # Slot deliberately untouched: a refresh we are not allowed to re-take
            # keeps the reference it already holds rather than dropping the head's
            # protection on the way to a pin that is not going to happen.
            self.last_outcome = HEADLOCK_SYNC_DENIED_CAPACITY
            return False
        if self.held:
            # Same head with a new match is a refresh; a different rid is the
            # head having changed under us.
            self.release(reason="refresh_match" if self.rid == rid else "head_changed")
        # Read after that release, so the move measured in _record_pin isolates
        # this acquire rather than netting the two operations against each other.
        evictable_before = None if self.headroom_fn is None else self.headroom_fn()[1]
        result = self.tree_cache.inc_lock_ref(node)
        self.rid = rid
        self.node = node
        self.tokens = tokens
        # Replay at release exactly what was skipped at acquire: a chain whose
        # bottom segment is an evicted tombstone locks only above it, and a
        # later load-back may turn that tombstone into a real value that some
        # other request now owns. Mirrors PrefillAdder._lock_node.
        self._dec_params = (
            result.to_dec_params()
            if result is not None and self.tree_cache.is_tree_cache()
            else None
        )
        self._record_pin(result, tokens, evictable_before)
        self._debug("acquire", rid, tokens, reason or "head_no_token")
        self.last_outcome = HEADLOCK_SYNC_PINNED
        return True

    def release(self, reason: str = "") -> bool:
        """Drop the held lock, if any. Returns True if one was dropped."""
        if not self.held:
            return False
        rid, node, params, tokens = self.rid, self.node, self._dec_params, self.tokens
        # Clear the slot first: a raising dec must not leave a slot pointing at
        # a reference we can no longer account for, or the next pass would try
        # to release it again.
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
            # A lock keeps its node alive, so this should be unreachable; if a
            # path we have not accounted for did drop the node anyway, losing a
            # lock ref on a dead node is survivable and killing the scheduler
            # loop is not. Loud, because it means the audit missed a path.
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
        """Pass-start check: drop the lock if its request left the waiting queue,
        or if the pool no longer holds one reserve-sized extend.

        The membership arm covers every way a blocked head can vanish without
        going through the adder — abort, timeout, cancellation, a flush — without
        hooking any of them. The capacity arm covers the pairing the pin-time
        gate cannot see: a pin taken while headroom was ample, and an in-flight
        chunked request that then grows chunk by chunk until its next chunk no
        longer fits. Returns True if the lock was dropped.
        """
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
        """Reconcile the slot against this pass's head verdict, and report the
        outcome as one of the ``HEADLOCK_SYNC_*`` constants.

        ``should_hold`` is True only while the head is the KV-blocked one worth
        protecting. When it is False the slot is released — that covers the head
        being admitted (``add_one_req`` took its own lock on the very same chain,
        so dropping ours takes the count 2 -> 1 and never through 0) as well as a
        head rejected for a non-KV reason, where pinning buys nothing.
        """
        if not should_hold:
            if self.held:
                self.release(reason=reason or "head_unblocked")
                return HEADLOCK_SYNC_RELEASED
            return HEADLOCK_SYNC_IDLE
        # acquire() itself handles the four cases: unchanged (no-op), same head
        # re-matched to a different node (release + re-lock), a new head, and a
        # pin the capacity gate refuses.
        self.acquire(rid, node, tokens, reason=reason or "head_no_token")
        return self.last_outcome
