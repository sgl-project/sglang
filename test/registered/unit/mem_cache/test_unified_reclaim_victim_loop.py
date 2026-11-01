"""Reclaim on unified memory: victims one at a time in the cache's order,
every walk stopping at the first victim after which the demand fits."""

import random
import unittest
from array import array
from unittest.mock import patch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.kv.alloc.pricing import first_plan
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    build_tri_cache,
    setup_allocator_context,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _evictable(cache):
    return (
        cache.full_evictable_size(),
        cache.swa_evictable_size(),
        cache.mamba_evictable_size(),
    )


class TestReclaimVictimLoop(CustomTestCase):
    def setUp(self):
        self.addCleanup(reset_context)
        setup_allocator_context()

    def _cache_prefixes(self, rng, allocator, cache, state_slots, count, lock):
        """``count`` cached prefixes, each with its state checkpoint, a share of
        them locked as running requests lock their prefixes."""
        receipts = []
        states = []
        while len(states) < count:
            state = state_slots.alloc(1)
            if state is None:
                break
            states.append(state.reshape(1))
        for i, state in enumerate(states):
            length = rng.randint(1, 8)
            tokens = allocator.alloc(length)
            if tokens is None:
                # Out of room: the states left stay with running requests.
                break
            key = RadixKey(array("q", range(i * 1000, i * 1000 + length)))
            cache.insert(InsertParams(key=key, value=tokens, mamba_value=state))
            if rng.random() < lock:
                node = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
                receipts.append(
                    (
                        node,
                        cache.inc_lock_ref(
                            node, skip_lock_components=(ComponentType.MAMBA,)
                        ),
                    )
                )
        return receipts

    def test_short_owner_is_walked_first(self):
        # The owner alone is short: its own cached entries are the first
        # victims, and the walk stops at the first one after which the demand
        # fits (rearranged if need be), long before its cache is gone.
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                rng = random.Random(7)
                bundle, allocator, cache = build_tri_cache(lazy=lazy)
                state_slots = bundle.token_to_kv_pool_allocator.state_slots
                self._cache_prefixes(rng, allocator, cache, state_slots, 6, 0.0)
                # Running requests fill the owner; their windows have retired.
                while allocator.ensure_capacity(4, 4):
                    allocator.free_swa(allocator.alloc(4))
                owner = allocator.full_attn_allocator.sub_pool_name
                demand = {owner: allocator.full_available_size() + 2}
                self.assertFalse(allocator._units_fit(demand))
                walks = []
                original = cache.evict_for_alloc

                def walk(params, **kwargs):
                    result = original(params, **kwargs)
                    walks.append((params, result))
                    return result

                with patch.object(cache, "evict_for_alloc", side_effect=walk):
                    self.assertTrue(allocator.reclaim(cache, demand))
                first = walks[0][0]
                self.assertGreater(first.num_tokens, 0)
                self.assertEqual(
                    (
                        first.swa_num_tokens,
                        sum(first.window_num_tokens.values()),
                        first.mamba_num,
                    ),
                    (0, 0, 0),
                )
                self.assertEqual(len(walks), 1)
                self.assertLess(walks[0][1].num_tokens_evicted, first.num_tokens)
                self.assertTrue(allocator._units_fit(demand))
                self.assertIsNotNone(
                    allocator.realize(first_plan(allocator.pricing_snapshot(), demand))
                )
                self.assertFalse(allocator.verify_byte_accounting())

    def _history(self, rng):
        bundle, allocator, cache = build_tri_cache(lazy=rng.random() < 0.5)
        state_slots = bundle.token_to_kv_pool_allocator.state_slots
        count = rng.randint(1, max(1, state_slots.available_size() - 3))
        receipts = self._cache_prefixes(rng, allocator, cache, state_slots, count, 0.3)
        # Running requests hold the rest of the token room.
        while rng.random() < 0.8 and allocator.ensure_capacity(4, 4):
            self.assertIsNotNone(allocator.alloc(4))
        return allocator, cache, receipts

    def test_admitted_demand_is_reclaimed_and_taken(self):
        # Admission credits every evictable entry; the reclaim the take runs
        # first realizes that credit; a demand admission refuses evicts
        # nothing, so the caller refuses or retracts with the cache intact.
        rng = random.Random(20261031)
        outcomes = set()
        for seed in range(200):
            with self.subTest(seed=seed):
                allocator, cache, receipts = self._history(rng)
                tokens = rng.randint(1, 48)
                slots = rng.randint(0, 2)
                admitted = allocator.admits(
                    tokens,
                    allocator._window_demand(tokens, tokens),
                    slots,
                    evictable=allocator.admission_evictable(cache),
                )
                before = _evictable(cache)
                ready = allocator.reclaim(
                    cache, allocator.page_demands(tokens, tokens, state_slots=slots)
                )
                self.assertEqual(ready, admitted)
                outcomes.add(admitted)
                if not admitted:
                    self.assertEqual(_evictable(cache), before)
                else:
                    if slots:
                        self.assertIsNotNone(allocator.alloc_state_slots(slots))
                    self.assertIsNotNone(allocator.alloc(tokens))
                self.assertFalse(allocator.verify_byte_accounting())
                for node, receipt in receipts:
                    cache.dec_lock_ref(node, receipt.to_dec_params())
        self.assertEqual(outcomes, {False, True})

    def test_component_shortfall_reclaims_the_demand(self):
        # Callers that make room for one component's take name its shortfall
        # against that component's capacity read (the state slot check before
        # binding a request, a host load's device rows). On unified memory the
        # shortfall is the component's demand less its read, so the reclaim
        # answers exactly as the demand's own admission and reclaim do.
        rng = random.Random(20261101)
        outcomes = set()
        for seed in range(150):
            with self.subTest(seed=seed):
                allocator, cache, receipts = self._history(rng)
                owner = allocator.full_attn_allocator.sub_pool_name
                window = allocator.swa_attn_allocator.sub_pool_name
                state = allocator.mamba_allocator.sub_pool_name
                component = rng.choice(("owner", "window", "state"))
                if component == "state":
                    need = rng.randint(1, 3)
                    read = allocator.state_slots_available()
                    params = EvictParams(mamba_num=need - read)
                    demand = {state: need}
                    admitted = allocator.admits(
                        0,
                        {},
                        need,
                        evictable=allocator.admission_evictable(cache),
                    )
                else:
                    page = allocator.page_size
                    need = rng.randint(1, 48)
                    name = owner if component == "owner" else window
                    read = allocator.units_available(name)
                    if component == "owner":
                        params = EvictParams(num_tokens=need - read)
                    else:
                        params = EvictParams(swa_num_tokens=need - read)
                    demand = {name: -(-need // page)}
                    admitted = allocator.admits(
                        need if component == "owner" else 0,
                        {} if component == "owner" else {window: need},
                        evictable=allocator.admission_evictable(cache),
                    )
                before = _evictable(cache)
                if need > read:
                    cache.evict_for_alloc(params)
                self.assertEqual(allocator._units_fit(demand), admitted)
                outcomes.add(admitted)
                if not admitted:
                    self.assertEqual(_evictable(cache), before)
                elif component == "state":
                    self.assertIsNotNone(allocator.alloc_state_slots(need))
                self.assertFalse(allocator.verify_byte_accounting())
                for node, receipt in receipts:
                    cache.dec_lock_ref(node, receipt.to_dec_params())
        self.assertEqual(outcomes, {False, True})


if __name__ == "__main__":
    unittest.main()
