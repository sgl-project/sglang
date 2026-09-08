"""Unit tests for the NO_TOKEN admission lookahead: the head-aging policy, the
head prefix lock's lifecycle and capacity gate, and the admission loop that
drives them."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.prefill_lookahead import (
    HEADLOCK_SYNC_DENIED_CAPACITY,
    HEADLOCK_SYNC_HELD,
    HEADLOCK_SYNC_PINNED,
    HEADLOCK_SYNC_RELEASED,
    HeadPrefixLock,
    PrefillLookaheadState,
    normalize_headlock_reserve_tokens,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
)
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

# The shape the capacity gate was written for: 16384-token extends against a
# 219008-token device pool.
RESERVE = 16384


class FakeIncResult:
    """Stand-in for IncLockRefResult; ``to_dec_params`` returns a sentinel so a
    test can assert the release replayed the acquire's receipt. ``delta`` is
    None by default, which is how a disabled UnifiedRadixCache reports."""

    def __init__(self, dec_params, delta=None):
        self._dec_params = dec_params
        self.delta = delta

    def to_dec_params(self):
        return self._dec_params


class FakeTreeCache:
    """Records inc/dec_lock_ref calls; ``net`` is the outstanding ref count,
    which every lifecycle test asserts back to 0."""

    def __init__(self, returns_result=True):
        self._returns_result = returns_result
        self.inc_calls = []
        self.dec_calls = []
        self.raise_on_dec = False

    def inc_lock_ref(self, node):
        self.inc_calls.append(node)
        if not self._returns_result:
            return None
        return FakeIncResult(dec_params=("dec_params_for", node))

    def dec_lock_ref(self, node, params=None):
        if self.raise_on_dec:
            raise KeyError(node)
        self.dec_calls.append((node, params))

    @property
    def net(self):
        return len(self.inc_calls) - len(self.dec_calls)


class FakePoolTreeCache(FakeTreeCache):
    """FakeTreeCache plus the KV accounting the capacity gate reads.

    ``headroom`` is the (allocator-free, tree-evictable) pair the scheduler
    feeds the lock, and ``inc_lock_ref`` / ``dec_lock_ref`` move a node's tokens
    between the evictable and locked halves exactly as every tree in mem_cache/
    does. ``delta_sign`` covers both live conventions: RadixCache reports the
    move as negative, UnifiedRadixCache's FULL component as positive. The two
    escape hatches model the caches that fit neither: ``report_delta`` off is
    a cache that reports no delta at all (a disabled UnifiedRadixCache), and
    ``account_locks`` off is a hypothetical cache whose lock accounting never
    reaches ``evictable_size()``, which is the only case where a pin still has to
    be charged to the adder.
    """

    def __init__(
        self,
        free,
        evictable,
        node_tokens=None,
        account_locks=True,
        report_delta=True,
        delta_sign=-1,
    ):
        super().__init__()
        self.free = free
        self.evictable = evictable
        self.node_tokens = dict(node_tokens or {})
        self.account_locks = account_locks
        self.report_delta = report_delta
        self.delta_sign = delta_sign

    def headroom(self):
        return (self.free, self.evictable)

    def inc_lock_ref(self, node):
        self.inc_calls.append(node)
        moved = self.node_tokens.get(node, 0)
        if self.account_locks:
            self.evictable -= moved
        return FakeIncResult(
            dec_params=("dec_params_for", node),
            delta=self.delta_sign * moved if self.report_delta else None,
        )

    def dec_lock_ref(self, node, params=None):
        super().dec_lock_ref(node, params)
        if self.account_locks:
            self.evictable += self.node_tokens.get(node, 0)


def build_gated_lock(cache, reserve=RESERVE, logger=None):
    return HeadPrefixLock(
        cache,
        logger=logger,
        headroom_fn=cache.headroom,
        reserve_tokens=reserve,
    )


class TestPrefillLookaheadState(CustomTestCase):
    def test_disabled_by_default_budget(self):
        state = PrefillLookaheadState(max_candidates=0, aging_passes=20)
        self.assertFalse(state.enabled)
        # Aging still tracks so the trace field stays meaningful when the
        # feature is off.
        self.assertEqual(state.observe_head("a", rejected_no_token=True), 1)

    def test_negative_config_is_clamped(self):
        state = PrefillLookaheadState(max_candidates=-5, aging_passes=-1)
        self.assertEqual(state.max_candidates, 0)
        self.assertEqual(state.aging_passes, 0)
        self.assertFalse(state.enabled)

    def test_consecutive_rejections_accumulate(self):
        state = PrefillLookaheadState(max_candidates=8, aging_passes=3)
        for expected in (1, 2, 3):
            self.assertEqual(
                state.observe_head("head", rejected_no_token=True), expected
            )
            self.assertFalse(state.aged_out)
        self.assertEqual(state.observe_head("head", rejected_no_token=True), 4)
        self.assertTrue(state.aged_out)

    def test_admission_resets(self):
        state = PrefillLookaheadState(max_candidates=8, aging_passes=1)
        state.observe_head("head", rejected_no_token=True)
        state.observe_head("head", rejected_no_token=True)
        self.assertTrue(state.aged_out)

        self.assertEqual(state.observe_head("head", rejected_no_token=False), 0)
        self.assertFalse(state.aged_out)
        self.assertIsNone(state.head_rid)

    def test_new_head_restarts_the_count(self):
        state = PrefillLookaheadState(max_candidates=8, aging_passes=1)
        state.observe_head("old", rejected_no_token=True)
        state.observe_head("old", rejected_no_token=True)
        self.assertTrue(state.aged_out)

        self.assertEqual(state.observe_head("new", rejected_no_token=True), 1)
        self.assertFalse(state.aged_out)
        self.assertEqual(state.head_rid, "new")

    def test_zero_aging_passes_disables_lookahead_on_first_rejection(self):
        state = PrefillLookaheadState(max_candidates=8, aging_passes=0)
        state.observe_head("head", rejected_no_token=True)
        self.assertTrue(state.aged_out)


class TestHeadPrefixLock(CustomTestCase):
    def setUp(self):
        self.cache = FakeTreeCache()
        self.lock = HeadPrefixLock(self.cache)

    def test_acquire_then_release(self):
        self.assertTrue(self.lock.acquire("head", node="n1", tokens=128))
        self.assertTrue(self.lock.held)
        self.assertEqual(self.lock.locked_tokens, 128)
        self.assertEqual(self.cache.inc_calls, ["n1"])

        self.assertTrue(self.lock.release(reason="admitted"))
        self.assertFalse(self.lock.held)
        self.assertEqual(self.lock.locked_tokens, 0)
        # The acquire's skip set is replayed at release, not dropped.
        self.assertEqual(self.cache.dec_calls, [("n1", ("dec_params_for", "n1"))])
        self.assertEqual(self.cache.net, 0)

    def test_repeat_acquire_is_idempotent(self):
        self.lock.acquire("head", node="n1", tokens=100)
        # A head blocked for many passes re-asserts the same lock every pass;
        # that must not stack references.
        self.assertFalse(self.lock.acquire("head", node="n1", tokens=140))
        self.assertEqual(len(self.cache.inc_calls), 1)
        self.assertEqual(self.cache.dec_calls, [])
        # ...but the reported size follows the latest match.
        self.assertEqual(self.lock.locked_tokens, 140)

        self.lock.release()
        self.assertEqual(self.cache.net, 0)

    def test_repeat_acquire_is_idempotent_for_large_int_node_ids(self):
        # UnifiedRadixCache hands out NodeId == int; two equal ints above the
        # small-int cache are not the same object, so identity comparison would
        # release and re-lock on every pass.
        self.lock.acquire("head", node=100_001, tokens=10)
        self.assertFalse(self.lock.acquire("head", node=100_000 + 1, tokens=10))
        self.assertEqual(len(self.cache.inc_calls), 1)
        self.assertEqual(self.cache.dec_calls, [])

    def test_release_without_lock_is_a_noop(self):
        self.assertFalse(self.lock.release())
        self.assertEqual(self.cache.dec_calls, [])

    def test_refresh_moves_the_lock_to_the_new_node(self):
        self.lock.acquire("head", node="n1", tokens=100)
        # Same head, re-matched deeper after a load-back.
        self.assertTrue(self.lock.acquire("head", node="n2", tokens=300))
        self.assertEqual(self.cache.inc_calls, ["n1", "n2"])
        self.assertEqual(self.cache.dec_calls, [("n1", ("dec_params_for", "n1"))])
        self.assertEqual(self.lock.node, "n2")
        self.assertEqual(self.lock.locked_tokens, 300)
        self.assertEqual(self.cache.net, 1)

        self.lock.release()
        self.assertEqual(self.cache.net, 0)

    def test_head_change_releases_the_old_lock(self):
        self.lock.acquire("old", node="n1", tokens=100)
        self.lock.acquire("new", node="n2", tokens=200)
        self.assertEqual(self.lock.rid, "new")
        self.assertEqual(self.cache.dec_calls, [("n1", ("dec_params_for", "n1"))])
        self.assertEqual(self.cache.net, 1)

    def test_reconcile_releases_when_rid_left_the_queue(self):
        self.lock.acquire("head", node="n1", tokens=100)
        # Still queued: kept.
        self.assertFalse(self.lock.reconcile({"head", "other"}))
        self.assertTrue(self.lock.held)
        # Aborted / timed out: dropped without any abort-path hook.
        self.assertTrue(self.lock.reconcile({"other"}))
        self.assertFalse(self.lock.held)
        self.assertEqual(self.cache.net, 0)

    def test_reconcile_without_lock_is_a_noop(self):
        self.assertFalse(self.lock.reconcile(set()))
        self.assertEqual(self.cache.dec_calls, [])

    def test_sync_head_holds_while_blocked(self):
        self.lock.sync_head(rid="head", node="n1", tokens=100, should_hold=True)
        self.assertTrue(self.lock.held)
        # Next pass, same head, still blocked, same match: no churn.
        self.lock.sync_head(rid="head", node="n1", tokens=100, should_hold=True)
        self.assertEqual(len(self.cache.inc_calls), 1)

    def test_sync_head_releases_when_head_is_admitted(self):
        self.lock.sync_head(rid="head", node="n1", tokens=100, should_hold=True)
        # add_one_req took its own lock on the same chain; ours must go or the
        # chain stays pinned forever.
        self.lock.sync_head(rid="head", node="n1", tokens=100, should_hold=False)
        self.assertFalse(self.lock.held)
        self.assertEqual(self.cache.net, 0)

    def test_sync_head_refreshes_on_a_new_match(self):
        self.lock.sync_head(rid="head", node="n1", tokens=100, should_hold=True)
        self.lock.sync_head(rid="head", node="n2", tokens=250, should_hold=True)
        self.assertEqual(self.cache.inc_calls, ["n1", "n2"])
        self.assertEqual(self.cache.dec_calls, [("n1", ("dec_params_for", "n1"))])
        self.assertEqual(self.lock.locked_tokens, 250)

    def test_sync_head_without_a_lock_and_unblocked_head_does_nothing(self):
        self.lock.sync_head(rid="head", node="n1", tokens=100, should_hold=False)
        self.assertFalse(self.lock.held)
        self.assertEqual(self.cache.inc_calls, [])
        self.assertEqual(self.cache.dec_calls, [])

    def test_acquire_without_a_result_releases_without_params(self):
        cache = FakeTreeCache(returns_result=False)
        lock = HeadPrefixLock(cache)
        lock.acquire("head", node="n1", tokens=10)
        lock.release()
        # No receipt to replay, so the release passes none.
        self.assertEqual(cache.dec_calls, [("n1", None)])

    def test_release_clears_the_slot_even_if_dec_raises(self):
        logger = _RecordingLogger()
        lock = HeadPrefixLock(self.cache, logger=logger)
        lock.acquire("head", node="n1", tokens=10)
        self.cache.raise_on_dec = True
        self.assertTrue(lock.release(reason="boom"))
        # Slot cleared, so the next pass cannot try to release it twice.
        self.assertFalse(lock.held)
        self.assertEqual(len(logger.warnings), 1)

    def test_debug_logging_covers_acquire_release_and_refresh(self):
        logger = _RecordingLogger()
        lock = HeadPrefixLock(self.cache, logger=logger)
        lock.acquire("head", node="n1", tokens=10, reason="head_no_token")
        lock.acquire("head", node="n2", tokens=20)
        lock.release(reason="admitted")
        reasons = [args[-1] for args in logger.debugs]
        self.assertEqual(
            reasons, ["head_no_token", "refresh_match", "head_no_token", "admitted"]
        )


class TestHeadlockReserveTokens(CustomTestCase):
    def test_unset_or_zero_uses_max_prefill_tokens(self):
        self.assertEqual(normalize_headlock_reserve_tokens(None, 16384), 16384)
        self.assertEqual(normalize_headlock_reserve_tokens(0, 16384), 16384)

    def test_positive_override_wins(self):
        self.assertEqual(normalize_headlock_reserve_tokens(32768, 16384), 32768)

    def test_negative_override_raises(self):
        # Silently turning the gate off would put the crash back without any
        # sign of it in the config, so this fails startup instead.
        with self.assertRaises(ValueError):
            normalize_headlock_reserve_tokens(-1, 16384)

    def test_negative_default_is_clamped(self):
        self.assertEqual(normalize_headlock_reserve_tokens(None, -5), 0)


class TestHeadPrefixLockCapacityGate(CustomTestCase):
    def test_pin_is_skipped_when_it_would_eat_the_reserve(self):
        cache = FakePoolTreeCache(
            free=1792, evictable=114_752, node_tokens={"n1": 114_752}
        )
        logger = _RecordingLogger()
        lock = build_gated_lock(cache, logger=logger)

        outcome = lock.sync_head(
            rid="head", node="n1", tokens=114_752, should_hold=True
        )

        self.assertEqual(outcome, HEADLOCK_SYNC_DENIED_CAPACITY)
        self.assertFalse(lock.held)
        self.assertEqual(cache.inc_calls, [])
        # The pool an in-flight chunked request evicts from is left as it was.
        self.assertEqual(cache.headroom(), (1792, 114_752))
        self.assertEqual(logger.debugs[-1], ("head", 114_752, 1792, 114_752, RESERVE))

    def test_pin_is_taken_when_the_reserve_still_fits_after_it(self):
        cache = FakePoolTreeCache(
            free=1792, evictable=114_752, node_tokens={"n1": 40_000}
        )
        lock = build_gated_lock(cache)

        outcome = lock.sync_head(rid="head", node="n1", tokens=40_000, should_hold=True)

        self.assertEqual(outcome, HEADLOCK_SYNC_PINNED)
        self.assertEqual(cache.inc_calls, ["n1"])
        self.assertGreaterEqual(sum(cache.headroom()), RESERVE)

    def test_exactly_the_reserve_is_still_allowed(self):
        # `>= reserve`, not `>`: one whole extend still fits.
        cache = FakePoolTreeCache(
            free=0, evictable=RESERVE + 1000, node_tokens={"n1": 1000}
        )
        lock = build_gated_lock(cache)
        self.assertEqual(
            lock.sync_head(rid="head", node="n1", tokens=1000, should_hold=True),
            HEADLOCK_SYNC_PINNED,
        )

    def test_gate_is_inert_without_a_headroom_source_or_reserve(self):
        # Mirrors env=0, which never constructs a lock at all.
        no_fn = FakePoolTreeCache(
            free=0, evictable=114_752, node_tokens={"n1": 114_752}
        )
        lock = HeadPrefixLock(no_fn, headroom_fn=None, reserve_tokens=RESERVE)
        self.assertEqual(
            lock.sync_head(rid="head", node="n1", tokens=114_752, should_hold=True),
            HEADLOCK_SYNC_PINNED,
        )

        no_reserve = FakePoolTreeCache(
            free=0, evictable=114_752, node_tokens={"n1": 114_752}
        )
        lock = build_gated_lock(no_reserve, reserve=0)
        self.assertEqual(
            lock.sync_head(rid="head", node="n1", tokens=114_752, should_hold=True),
            HEADLOCK_SYNC_PINNED,
        )

    def test_denied_refresh_keeps_the_pin_it_already_holds(self):
        cache = FakePoolTreeCache(
            free=100_000,
            evictable=120_000,
            node_tokens={"n1": 20_000, "n2": 119_000},
        )
        lock = build_gated_lock(cache)
        lock.sync_head(rid="head", node="n1", tokens=20_000, should_hold=True)
        # An in-flight chunked request has taken the free pages since, so the
        # deeper re-match no longer fits.
        cache.free = 0

        outcome = lock.sync_head(
            rid="head", node="n2", tokens=119_000, should_hold=True
        )

        self.assertEqual(outcome, HEADLOCK_SYNC_DENIED_CAPACITY)
        # Dropping n1 on the way to a pin that will not happen would strip the
        # head of protection it already had.
        self.assertEqual(lock.node, "n1")
        self.assertEqual(cache.inc_calls, ["n1"])
        self.assertEqual(cache.dec_calls, [])

    def test_reasserting_the_same_slot_is_never_re_gated(self):
        cache = FakePoolTreeCache(
            free=50_000, evictable=100_000, node_tokens={"n1": 90_000}
        )
        lock = build_gated_lock(cache)
        self.assertEqual(
            lock.sync_head(rid="head", node="n1", tokens=90_000, should_hold=True),
            HEADLOCK_SYNC_PINNED,
        )
        # Headroom is now under the reserve partly *because of* this pin; the
        # re-assertion takes nothing new, so gating it would only churn.
        cache.free = 0

        self.assertEqual(
            lock.sync_head(rid="head", node="n1", tokens=90_000, should_hold=True),
            HEADLOCK_SYNC_HELD,
        )
        self.assertEqual(cache.inc_calls, ["n1"])

    def test_unblocked_head_still_releases_under_pressure(self):
        cache = FakePoolTreeCache(
            free=50_000, evictable=100_000, node_tokens={"n1": 90_000}
        )
        lock = build_gated_lock(cache)
        lock.sync_head(rid="head", node="n1", tokens=90_000, should_hold=True)
        cache.free = 0

        outcome = lock.sync_head(
            rid="head", node="n1", tokens=90_000, should_hold=False
        )

        self.assertEqual(outcome, HEADLOCK_SYNC_RELEASED)
        self.assertFalse(lock.held)
        self.assertEqual(cache.net, 0)


class TestHeadPrefixLockPinAccounting(CustomTestCase):
    def test_exact_delta_needs_no_charge_to_the_pass_budget(self):
        cache = FakePoolTreeCache(
            free=50_000, evictable=100_000, node_tokens={"n1": 30_000}
        )
        lock = build_gated_lock(cache)

        lock.sync_head(rid="head", node="n1", tokens=30_000, should_hold=True)

        self.assertEqual(lock.last_pin_tokens, 30_000)
        # inc_lock_ref already took it out of evictable_size(), which
        # rem_total_tokens reads live; charging it again double-counts the pin.
        self.assertEqual(lock.last_pin_unaccounted, 0)

    def test_a_positive_delta_is_read_as_the_same_move(self):
        # UnifiedRadixCache's FULL component reports the
        # evictable -> protected move as +key_len, RadixCache as -len(key); a
        # sign-sensitive read would size the pin as a negative.
        cache = FakePoolTreeCache(
            free=50_000,
            evictable=100_000,
            node_tokens={"n1": 30_000},
            delta_sign=1,
        )
        lock = build_gated_lock(cache)

        lock.sync_head(rid="head", node="n1", tokens=30_000, should_hold=True)

        self.assertEqual(lock.last_pin_tokens, 30_000)
        self.assertEqual(lock.last_pin_unaccounted, 0)

    def test_a_chain_a_running_request_already_locks_costs_nothing(self):
        cache = FakePoolTreeCache(free=50_000, evictable=100_000, node_tokens={"n1": 0})
        lock = build_gated_lock(cache)

        lock.sync_head(rid="head", node="n1", tokens=30_000, should_hold=True)

        # delta == 0: the prefix length is not what left the evictable set.
        self.assertEqual(lock.last_pin_tokens, 0)
        self.assertEqual(lock.last_pin_unaccounted, 0)

    def test_a_cache_without_a_delta_falls_back_to_the_full_prefix(self):
        cache = FakePoolTreeCache(
            free=50_000,
            evictable=100_000,
            node_tokens={"n1": 30_000},
            report_delta=False,
        )
        lock = build_gated_lock(cache)

        lock.sync_head(rid="head", node="n1", tokens=30_000, should_hold=True)

        self.assertEqual(lock.last_pin_tokens, 30_000)
        self.assertEqual(lock.last_pin_unaccounted, 0)

    def test_a_pin_the_tree_does_not_account_for_is_charged_in_full(self):
        cache = FakePoolTreeCache(
            free=50_000,
            evictable=100_000,
            node_tokens={"n1": 30_000},
            account_locks=False,
        )
        lock = build_gated_lock(cache)

        lock.sync_head(rid="head", node="n1", tokens=30_000, should_hold=True)

        # Nothing moved out of evictable_size(), so the rest of the pass would
        # otherwise keep selling KV the head now owns.
        self.assertEqual(lock.last_pin_unaccounted, 30_000)

    def test_a_refresh_charges_the_new_lock_only(self):
        cache = FakePoolTreeCache(
            free=200_000,
            evictable=200_000,
            node_tokens={"n1": 30_000, "n2": 50_000},
        )
        lock = build_gated_lock(cache)
        lock.sync_head(rid="head", node="n1", tokens=30_000, should_hold=True)

        lock.sync_head(rid="head", node="n2", tokens=50_000, should_hold=True)

        # Measured across the inc alone: netting it against the release that
        # precedes it would make an accounted refresh look unaccounted.
        self.assertEqual(lock.last_pin_tokens, 50_000)
        self.assertEqual(lock.last_pin_unaccounted, 0)


class TestHeadPrefixLockCapacityRelease(CustomTestCase):
    def _pinned(self, logger=None):
        cache = FakePoolTreeCache(
            free=120_000, evictable=120_000, node_tokens={"n1": 100_000}
        )
        lock = build_gated_lock(cache, logger=logger)
        lock.sync_head(rid="head", node="n1", tokens=100_000, should_hold=True)
        return cache, lock

    def test_reconcile_keeps_the_lock_while_headroom_holds(self):
        cache, lock = self._pinned()
        self.assertFalse(lock.reconcile({"head"}))
        self.assertTrue(lock.held)
        self.assertEqual(cache.dec_calls, [])

    def test_reconcile_releases_a_lock_taken_before_the_pressure(self):
        logger = _RecordingLogger()
        cache, lock = self._pinned(logger=logger)
        # An in-flight chunked request grows into the rest of the pool over the
        # passes that follow: headroom the pin-time gate never got to see.
        cache.free = 0
        cache.evictable = 8_000

        self.assertTrue(lock.reconcile({"head"}))

        self.assertFalse(lock.held)
        self.assertEqual(cache.dec_calls, [("n1", ("dec_params_for", "n1"))])
        self.assertEqual(
            logger.debugs[-1], ("release", "head", 100_000, "capacity_pressure")
        )
        # The next chunk has something to evict again.
        self.assertGreaterEqual(sum(cache.headroom()), RESERVE)

    def test_a_released_pin_is_not_immediately_retaken(self):
        cache, lock = self._pinned()
        cache.free = 0
        cache.evictable = 8_000
        lock.reconcile({"head"})

        # Same pass, same blocked head: re-pinning would undo the release, so
        # the gate has to refuse it on the headroom the release just restored.
        outcome = lock.sync_head(
            rid="head", node="n1", tokens=100_000, should_hold=True
        )

        self.assertEqual(outcome, HEADLOCK_SYNC_DENIED_CAPACITY)
        self.assertFalse(lock.held)

    def test_a_departed_rid_is_still_reported_as_such(self):
        logger = _RecordingLogger()
        cache, lock = self._pinned(logger=logger)
        cache.free = 0
        cache.evictable = 8_000

        self.assertTrue(lock.reconcile({"other"}))

        self.assertEqual(logger.debugs[-1][-1], "rid_left_queue")

    def test_capacity_arm_is_off_without_a_reserve(self):
        cache = FakePoolTreeCache(free=0, evictable=0, node_tokens={"n1": 100_000})
        lock = build_gated_lock(cache, reserve=0)
        lock.sync_head(rid="head", node="n1", tokens=100_000, should_hold=True)
        self.assertFalse(lock.reconcile({"head"}))
        self.assertTrue(lock.held)


class TestHeadLockOutOfMemoryRegression(CustomTestCase):
    """The observed out-of-memory shape, at the point the pin is decided.

    A 219008-token device pool with an in-flight chunked prefill holding ~110k,
    a queue head whose matched prefix (114752) is essentially the whole evictable
    remainder, and a 4096-token next chunk that allocates through
    alloc_paged_token_slots_extend, i.e. outside the admission budget, relying
    on eviction at alloc time.
    """

    CHUNK = 4096
    FREE = 1792
    HEAD_PREFIX = 114_752

    def _crash_state(self):
        return FakePoolTreeCache(
            free=self.FREE,
            evictable=self.HEAD_PREFIX,
            node_tokens={"head_node": self.HEAD_PREFIX},
        )

    def test_pin_is_skipped_and_the_next_chunk_still_has_kv_to_evict(self):
        cache = self._crash_state()
        lock = build_gated_lock(cache)

        outcome = lock.sync_head(
            rid="0b67", node="head_node", tokens=self.HEAD_PREFIX, should_hold=True
        )

        self.assertEqual(outcome, HEADLOCK_SYNC_DENIED_CAPACITY)
        available, evictable = cache.headroom()
        self.assertEqual(evictable, self.HEAD_PREFIX)
        self.assertGreaterEqual(available + evictable, self.CHUNK)

    def test_the_ungated_pin_is_what_starved_the_chunk(self):
        # Same state with the gate off: evictable goes to 0 and 1792 free pages
        # cannot serve a 4096-token chunk, which is the RuntimeError that took
        # the scheduler process down.
        cache = self._crash_state()
        lock = HeadPrefixLock(cache, headroom_fn=None, reserve_tokens=0)

        lock.sync_head(
            rid="0b67", node="head_node", tokens=self.HEAD_PREFIX, should_hold=True
        )

        self.assertEqual(cache.headroom(), (self.FREE, 0))
        self.assertLess(sum(cache.headroom()), self.CHUNK)


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.kv = SimpleNamespace(holds_kv=False, holds_mamba=False)
        self.beam_group = None
        self.token_indices_to_pool = None
        self.last_node = f"node-{rid}"
        self.prefix_len = 64

    def init_next_round_input(self, tree_cache=None):
        pass


class _ScriptedAdder(PrefillAdder):
    """Answers each offered request from ``verdicts`` (default NO_TOKEN); a
    ``(result, True)`` verdict admits the request and still returns ``result``."""

    verdicts = {}
    offered = []
    rejection_hook_reqs = []
    chunked_req_waits_in_pass = False

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = 0
        self.chunked_req_waits = self.chunked_req_waits_in_pass
        self.charged = []

    def budget_state(self):
        return AddReqResult.CONTINUE

    def charge_head_prefix_pin(self, tokens):
        self.charged.append(tokens)

    def batch_is_full_after_rejection(self, req):
        type(self).rejection_hook_reqs.append(req.rid)
        return True

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        type(self).offered.append(req.rid)
        verdict = self.verdicts.get(req.rid, AddReqResult.NO_TOKEN)
        result, admit = verdict if isinstance(verdict, tuple) else (verdict, False)
        if admit or result == AddReqResult.CONTINUE:
            self.can_run_list.append(req)
        return result


class _BatchBuilt(Exception):
    pass


class TestAdmissionLoopLookahead(CustomTestCase):
    """`Scheduler._get_new_batch_prefill_raw` with a scripted adder."""

    def setUp(self):
        _ScriptedAdder.verdicts = {}
        _ScriptedAdder.offered = []
        _ScriptedAdder.rejection_hook_reqs = []
        _ScriptedAdder.chunked_req_waits_in_pass = False
        self.cache = FakeTreeCache()
        self.cache.buffer_pipeline = None
        self.cache.storage_prefetch_retries = None
        self.aborted = []

    def _scheduler(self, rids, *, max_candidates, aging_passes=20, headroom_fn=None):
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.waiting_queue = IndexedReqList(ListKind.WAITING)
        s.waiting_queue.join([_Req(rid) for rid in rids])
        s.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
        s.min_free_slots_delayer = None
        s.invariant_checker = Mock()
        s.policy = Mock(calc_priority=lambda waiting, *args, **kwargs: list(waiting))
        s.processed_tokens_counter = 0
        s.prefill_lookahead = PrefillLookaheadState(
            max_candidates=max_candidates, aging_passes=aging_passes
        )
        s.head_prefix_lock = (
            HeadPrefixLock(self.cache, headroom_fn=headroom_fn, reserve_tokens=RESERVE)
            if max_candidates > 0
            else None
        )
        s.metrics_reporter = SimpleNamespace(enable_metrics=False)
        s.chunked_prefill_size = 8192
        s.dynamic_chunk_sizer = None
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s.get_prefill_adder_class = lambda: _ScriptedAdder
        s.request_retirer = None
        s.page_size = 1
        s.tree_cache = self.cache
        s.token_to_kv_pool_allocator = SimpleNamespace(state_slots=None)
        s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
        s.max_prefill_tokens = 1 << 20
        s.is_mixed_chunk = False
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = 64
        s.dllm_config = None
        s.enable_lora = False
        s.disaggregation_mode = DisaggregationMode.NULL
        s.disagg_prefill_bootstrap_queue = None
        s.disagg_prefill_transfer_queue = []
        s.enable_hicache_storage = False
        s.enable_lmcache = False
        s.enable_unified_cache_external_linker = False
        s.truncation_align_size = None
        s.model_config = None
        s.enable_overlap = False
        s.spec_algorithm = None
        s.req_to_token_pool = SimpleNamespace(available_size=lambda: 64)
        s.beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)
        s.running_reqs = KeyedReqList(ListKind.DECODING)
        s._abort_unavailable = lambda req, message: self.aborted.append(req.rid)
        return s

    def _run_pass(self, s, verdicts):
        """Run one admission pass; returns (offered, admitted, batch_is_full)."""
        _ScriptedAdder.verdicts = verdicts
        _ScriptedAdder.offered = []
        running_batch = ScheduleBatch(reqs=[_Req("running")])

        def capture(can_run_list, *args, **kwargs):
            raise _BatchBuilt([req.rid for req in can_run_list])

        admitted = []
        with (
            get_context().override_server_args(pp_max_micro_batch_size=64),
            patch.object(ScheduleBatch, "init_new", side_effect=capture),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            try:
                s._get_new_batch_prefill_raw(
                    prefill_delayer_single_pass=None, running_batch=running_batch
                )
            except _BatchBuilt as built:
                admitted = built.args[0]
        return _ScriptedAdder.offered, admitted, running_batch.batch_is_full

    def test_disabled_lookahead_stops_at_the_first_no_token(self):
        s = self._scheduler(["head", "c1"], max_candidates=0)
        offered, admitted, batch_is_full = self._run_pass(
            s, {"c1": AddReqResult.CONTINUE}
        )
        self.assertEqual(offered, ["head"])
        self.assertEqual(admitted, [])
        self.assertTrue(batch_is_full)
        self.assertEqual(self.cache.inc_calls, [])

    def test_candidates_are_scanned_behind_a_pinned_head(self):
        s = self._scheduler(["head", "c1", "c2", "c3", "c4"], max_candidates=3)
        offered, admitted, batch_is_full = self._run_pass(
            s, {"c2": AddReqResult.CONTINUE}
        )
        # The head is pinned before c1 is offered; c4 is past the scan bound.
        self.assertEqual(offered, ["head", "c1", "c2", "c3"])
        self.assertEqual(admitted, ["c2"])
        self.assertEqual(self.cache.inc_calls, ["node-head"])
        self.assertEqual(s.head_prefix_lock.rid, "head")
        # The rejection hook runs once per pass, for the blocked head.
        self.assertEqual(_ScriptedAdder.rejection_hook_reqs, ["head"])
        self.assertTrue(batch_is_full)

    def test_scan_bound_counts_admitted_candidates(self):
        s = self._scheduler(["head", "c1", "c2", "c3"], max_candidates=2)
        offered, admitted, _ = self._run_pass(
            s,
            {
                "c1": AddReqResult.CONTINUE,
                "c2": AddReqResult.CONTINUE,
                "c3": AddReqResult.CONTINUE,
            },
        )
        self.assertEqual(offered, ["head", "c1", "c2"])
        self.assertEqual(admitted, ["c1", "c2"])
        self.assertEqual(_ScriptedAdder.rejection_hook_reqs, ["head"])

    def test_unschedulable_candidate_stays_queued(self):
        # The verdict was taken with the head's pin held, so it is not final.
        s = self._scheduler(["head", "c1", "c2"], max_candidates=4)
        offered, admitted, _ = self._run_pass(
            s, {"c1": AddReqResult.UNSCHEDULABLE, "c2": AddReqResult.CONTINUE}
        )
        self.assertEqual(offered, ["head", "c1", "c2"])
        self.assertEqual(admitted, ["c2"])
        self.assertEqual(self.aborted, [])
        self.assertEqual([r.rid for r in s.waiting_queue], ["head", "c1"])

    def test_unschedulable_head_is_still_aborted(self):
        s = self._scheduler(["head", "c1"], max_candidates=4)
        _, admitted, _ = self._run_pass(
            s, {"head": AddReqResult.UNSCHEDULABLE, "c1": AddReqResult.CONTINUE}
        )
        self.assertEqual(self.aborted, ["head"])
        self.assertEqual(admitted, ["c1"])
        self.assertEqual(self.cache.inc_calls, [])

    def test_aged_out_head_withholds_lookahead_but_keeps_its_pin(self):
        s = self._scheduler(["head", "c1"], max_candidates=4, aging_passes=1)
        offered, _, _ = self._run_pass(s, {})
        self.assertEqual(offered, ["head", "c1"])
        offered, admitted, batch_is_full = self._run_pass(
            s, {"c1": AddReqResult.CONTINUE}
        )
        self.assertEqual(offered, ["head"])
        self.assertEqual(admitted, [])
        self.assertTrue(batch_is_full)
        self.assertEqual(s.head_prefix_lock.rid, "head")
        self.assertEqual(self.cache.net, 1)

    def test_refused_pin_withholds_lookahead(self):
        s = self._scheduler(
            ["head", "c1"], max_candidates=4, headroom_fn=lambda: (0, RESERVE)
        )
        offered, admitted, _ = self._run_pass(s, {"c1": AddReqResult.CONTINUE})
        self.assertEqual(offered, ["head"])
        self.assertEqual(admitted, [])
        self.assertEqual(self.cache.inc_calls, [])

    def test_head_admitted_with_a_full_pool_ends_the_pass(self):
        # NO_TOKEN from an admitted request means the pool is spent: nothing
        # behind it can fit, so there is nothing to look ahead for.
        s = self._scheduler(["head", "c1"], max_candidates=4)
        offered, admitted, _ = self._run_pass(
            s, {"head": (AddReqResult.NO_TOKEN, True), "c1": AddReqResult.CONTINUE}
        )
        self.assertEqual(offered, ["head"])
        self.assertEqual(admitted, ["head"])
        self.assertEqual(self.cache.inc_calls, [])

    def test_waiting_chunked_request_gets_no_scan_and_no_pin(self):
        s = self._scheduler(["head", "c1"], max_candidates=4)
        self._run_pass(s, {})
        self.assertEqual(self.cache.net, 1)

        _ScriptedAdder.chunked_req_waits_in_pass = True
        offered, _, _ = self._run_pass(s, {"c1": AddReqResult.CONTINUE})
        self.assertEqual(offered, ["head"])
        # The pin from the earlier pass is dropped and the head does not age.
        self.assertEqual(self.cache.net, 0)
        self.assertEqual(s.prefill_lookahead.head_age, 0)

    def test_pin_is_released_when_the_head_is_admitted_or_leaves(self):
        s = self._scheduler(["head", "c1"], max_candidates=4)
        self._run_pass(s, {})
        self.assertEqual(self.cache.net, 1)
        _, admitted, _ = self._run_pass(s, {"head": AddReqResult.CONTINUE})
        self.assertEqual(admitted[0], "head")
        self.assertEqual(self.cache.net, 0)

        s = self._scheduler(["head"], max_candidates=4)
        self._run_pass(s, {})
        self.assertEqual(self.cache.net, 1)
        # An abort empties the queue without telling the lock, and the next
        # pass offers no request.
        s.waiting_queue.leave([s.waiting_queue[0]])
        self._run_pass(s, {})
        self.assertEqual(self.cache.net, 0)
        self.assertFalse(s.head_prefix_lock.held)


class _RecordingLogger:
    def __init__(self):
        self.debugs = []
        self.warnings = []

    def debug(self, _fmt, *args):
        self.debugs.append(args)

    def warning(self, _fmt, *args, **kwargs):
        self.warnings.append(args)


if __name__ == "__main__":
    unittest.main()
