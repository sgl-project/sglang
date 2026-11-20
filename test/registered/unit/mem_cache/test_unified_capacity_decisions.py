"""Capacity decisions over a unified pool ask a demand: a scalar read inside one is
a violation, and the prefill budget's token-count questions price demands instead."""

import logging
import unittest
from array import array
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
from test_prefill_memory_budget import (
    _cache,
    _shared_allocator,
)

from sglang.srt.environ import InvariantCheckLevel, envs
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.allocation_sizing import add_spec_decode_headroom_pages
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.capacity import (
    capacity_decision,
    create_prefill_budget,
    evict_to_free_tokens,
)
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand, WindowRetirement
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    build_swa_pool,
    build_tri_pool,
    largest_take,
    setup_allocator_context,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def setUpModule():
    patcher = patch(
        "sglang.srt.mem_cache.prefill_budget.window_retirement",
        return_value=WindowRetirement(0, 0, 1),
    )
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


@capacity_decision
def _decide(read):
    return read()


class TestEstimatesAreObserveOnly(CustomTestCase):
    def setUp(self):
        setup_allocator_context()
        self.addCleanup(reset_context)
        _, self.allocator = build_tri_pool(page_size=4)
        owner, (window,), state = self.allocator.sub_pool_roles()
        self.reads = {
            "available_size": self.allocator.available_size,
            "owner_available_size": self.allocator.owner_available_size,
            "full_available_size": self.allocator.full_available_size,
            "window_available_size": lambda: self.allocator.window_available_size(
                window
            ),
            "window_capacity_and_available": (
                lambda: self.allocator.window_capacity_and_available(window)
            ),
            "usage": self.allocator.usage,
            "state slots": self.allocator.state_slots.schedulable_available_size,
        }

    def test_strict_checks_refuse_an_estimate_inside_a_decision(self):
        with envs.SGLANG_INVARIANT_CHECK.override(InvariantCheckLevel.STRICT):
            for name, read in self.reads.items():
                with self.subTest(read=name):
                    read()  # an observer may read it
                    with self.assertRaisesRegex(AssertionError, "observe-only"):
                        _decide(read)

    def test_warn_logs_and_off_stays_silent(self):
        read = self.reads["full_available_size"]
        with envs.SGLANG_INVARIANT_CHECK.override(InvariantCheckLevel.WARN):
            with self.assertLogs("sglang.srt.mem_cache.capacity", logging.WARNING):
                self.assertEqual(_decide(read), read())
        with envs.SGLANG_INVARIANT_CHECK.override(InvariantCheckLevel.OFF):
            with self.assertNoLogs("sglang.srt.mem_cache.capacity", logging.WARNING):
                self.assertEqual(_decide(read), read())

    def test_demand_questions_pass_inside_a_decision(self):
        allocator = self.allocator
        with envs.SGLANG_INVARIANT_CHECK.override(InvariantCheckLevel.STRICT):
            self.assertTrue(
                _decide(
                    lambda: allocator.ensure_capacity(
                        KVDemand.same_tokens(allocator, 4, state_slots=1)
                    )
                )
            )
            self.assertTrue(_decide(lambda: allocator.state_slots.slots_fit(1)))

    def test_estimate_bounds_every_take_and_keeps_the_census_exact(self):
        allocator = self.allocator
        owner, (window,), state = allocator.sub_pool_roles()
        self.assertIsNotNone(allocator.alloc(12))
        self.assertIsNotNone(allocator.state_slots.alloc(2))
        for name in (owner, window, state):
            with self.subTest(sub_pool=name):
                # Bytes no live page holds bound what any take finds.
                self.assertGreaterEqual(
                    allocator.estimated_units_available(name),
                    largest_take(allocator, name),
                )
                capacity, free = allocator.usage()[name]
                self.assertEqual(capacity - free, allocator.held_units(name))
        self.assertEqual(allocator.held_units(owner), 12)
        self.assertEqual(allocator.held_units(state), 2)

    def test_state_slots_fit_is_the_take(self):
        slots = self.allocator.state_slots
        room = largest_take(self.allocator, self.allocator.sub_pool_roles()[2])
        self.assertTrue(slots.slots_fit(room))
        self.assertFalse(slots.slots_fit(room + 1))
        self.assertIsNotNone(slots.alloc(room))
        self.assertFalse(slots.slots_fit(1))


class TestPrefillBudgetPricesDemands(CustomTestCase):
    def setUp(self):
        self.allocator = _shared_allocator(pool_tokens=256)
        self.addCleanup(reset_context)
        self.budget = create_prefill_budget(self.allocator, _cache())
        self.request = dict(
            extend_input_len=12,
            max_new_tokens=4,
            chunk_limit=16,
            state_slots=0,
            cached_tokens=0,
        )

    def test_the_budget_has_no_remaining_token_count(self):
        self.assertTrue(self.budget.prices_by_demand)
        for name in ("remaining_total", "remaining_current", "remaining_window_tokens"):
            with self.subTest(read=name), self.assertRaises(TypeError):
                getattr(self.budget, name)

    def test_ignore_eos_admits_only_what_running_requests_leave(self):
        page = self.allocator.page_size
        # Running requests (tokens left, tokens held), fewest left first.
        states = [(4, 16), (8, 16), (20, 8)]
        self.assertTrue(
            self.budget.ignore_eos_decode_fits(
                states, paged_input=2 * page, reserve_tokens=1, **self.request
            )
        )
        self.assertFalse(
            self.budget.ignore_eos_decode_fits(
                states, paged_input=1 << 20, reserve_tokens=1, **self.request
            )
        )
        # A running request with more left to decode than the pool holds leaves
        # no room for even a small input.
        self.assertFalse(
            self.budget.ignore_eos_decode_fits(
                [(1 << 20, 0)], paged_input=2 * page, reserve_tokens=1, **self.request
            )
        )

    def test_largest_chunk_is_the_largest_admitted(self):
        def fits(tokens, now):
            return self.budget._chunk_fits(
                tokens,
                0,
                chunk_limit=None,
                host_hit_lengths=None,
                state_slots=0,
                now=now,
            )

        self.budget.reserve(12, 4, chunk_limit=16)
        for now in (False, True):
            with self.subTest(now=now):
                largest = self.budget.largest_chunk_tokens(4096, now=now)
                self.assertGreater(largest, 0)
                self.assertTrue(fits(largest, now))
                self.assertFalse(fits(largest + 1, now))


class TestDllmBlockWaitsWhenNoneFits(CustomTestCase):
    """When no dLLM block fits a full FULL + window pool, the block size is zero,
    so the request waits instead of being staged."""

    def test_a_full_pool_leaves_no_block(self):
        allocator = _shared_allocator()
        budget = create_prefill_budget(allocator, _cache())
        while allocator.alloc(4) is not None:
            pass
        self.assertEqual(budget.largest_chunk_tokens(64), 0)
        adder = SimpleNamespace(
            memory_budget=budget,
            dllm_config=SimpleNamespace(requires_separate_context_encoding=False),
            rem_dllm_tokens=64,
            dllm_block_size=64,
        )
        adder._get_dllm_remain_tokens_by_demand = MethodType(
            PrefillAdder._get_dllm_remain_tokens_by_demand, adder
        )
        self.assertEqual(PrefillAdder._get_dllm_remain_tokens(adder), 0)


class TestPlainTreeReclaimsADemand(CustomTestCase):
    """A prefix tree keeping no window sub-pools of its own (as FlexKV's radix
    cache does) reclaims a token allocation over a shared pool by its demand,
    reading no availability estimate inside the decision."""

    def test_token_rows_are_reclaimed_for_the_allocation(self):
        setup_allocator_context()
        self.addCleanup(reset_context)
        allocator = build_swa_pool(page_size=4)
        cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=ReqToTokenPool(
                    size=4, max_context_len=64, device="cpu", enable_memory_saver=False
                ),
                token_to_kv_pool_allocator=allocator,
                page_size=4,
            )
        )
        base = 0
        while (ids := allocator.alloc(4)) is not None:
            cache.insert(
                InsertParams(key=RadixKey(array("q", range(base, base + 4))), value=ids)
            )
            base += 4
        with envs.SGLANG_INVARIANT_CHECK.override(InvariantCheckLevel.STRICT):
            self.assertTrue(_decide(lambda: evict_to_free_tokens(cache, allocator, 16)))
        self.assertIsNotNone(allocator.alloc(16))


class TestSpeculativeHeadroomAsksADemand(CustomTestCase):
    """Speculative decode headroom over a shared pool is added only when its
    pages fit as a demand, not when the pool's availability estimate says so."""

    def setUp(self):
        super().setUp()
        setup_allocator_context()
        self.addCleanup(reset_context)

    def _headroom(self, allocator, need):
        page = allocator.page_size
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(kv=SimpleNamespace(kv_committed_len=need))],
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(1, 1024)),
            hisparse_coordinator=None,
            token_to_kv_pool_allocator=allocator,
        )
        aligned = -(-need // page) * page
        return add_spec_decode_headroom_pages(
            batch, [aligned], page, reserve=0, page_size=page
        )

    def test_headroom_only_where_its_pages_fit(self):
        allocator = build_swa_pool(page_size=4)
        page = allocator.page_size
        # The estimate counts every free byte, usable or not.
        overcount = patch.object(
            type(allocator), "available_size", return_value=1 << 30
        )
        with overcount:
            lens, needed = self._headroom(allocator, 8)
        self.assertGreater(lens[0], 8)
        # Its margin: the need, a page per request and four times the headroom.
        margin = page + page + 4 * 2 * page
        while largest_take(allocator) >= margin:
            self.assertIsNotNone(allocator.alloc(page))
        with overcount:
            lens, needed = self._headroom(allocator, 8)
        # The mandatory allocation only, though the estimate says it fits.
        self.assertEqual((lens, needed), ([8], page))


if __name__ == "__main__":
    unittest.main()
