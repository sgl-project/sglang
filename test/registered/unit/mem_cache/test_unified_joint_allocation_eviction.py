"""Real shared-pool allocation after allocation-driven cache eviction."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_memory_pool import (
    init_unified_mamba_swa_pools,
    init_unified_swa_pools,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.session.streaming_session import StreamingSession
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestUnifiedJointAllocationEviction(CustomTestCase):
    def setUp(self):
        reset_context()
        publish(ServerArgs(model_path="Qwen/Qwen3.5-0.8B"), role="tokenizer")

    def tearDown(self):
        reset_context()

    def build_cache(
        self, *, occupancy=96, lazy=False, page_size=1, ratio=(1, 1), total_bytes=None
    ):
        full_layers, swa_layers = ratio
        bundle = init_unified_swa_pools(
            device="cpu",
            kv_cache_dtype=torch.float16,
            head_num=1,
            head_dim=8,
            v_head_dim=8,
            swa_head_num=1,
            swa_head_dim=8,
            swa_v_head_dim=8,
            page_size=page_size,
            start_layer=0,
            end_layer=full_layers + swa_layers,
            swa_attention_layer_ids=list(range(full_layers, full_layers + swa_layers)),
            full_attention_layer_ids=list(range(full_layers)),
            full_max_total_num_tokens=100 * page_size,
            swa_max_total_num_tokens=100 * page_size,
            enable_memory_saver=False,
            need_sort=False,
            lazy_compaction=lazy,
            unified_total_bytes=total_bytes,
        )
        allocator = bundle.token_to_kv_pool_allocator
        req_pool = ReqToTokenPool(
            size=4,
            max_context_len=256 * page_size,
            device="cpu",
            enable_memory_saver=False,
        )
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
                sliding_window_size=128 * page_size,
                tree_components=(ComponentType.FULL, ComponentType.SWA),
            )
        )
        slots = allocator.alloc(occupancy * page_size)
        self.assertIsNotNone(slots)
        for i in range(occupancy):
            cache.insert(
                InsertParams(
                    key=RadixKey(array("q", range(i * 1000, i * 1000 + page_size))),
                    value=slots[i * page_size : (i + 1) * page_size],
                )
            )
        return allocator, cache

    def test_byte_shortfall_skips_recovery_until_first_sufficient_eviction(self):
        for lazy in (False, True):
            for page_size in (1, 4, 16):
                with self.subTest(lazy=lazy, page_size=page_size):
                    allocator, cache = self.build_cache(lazy=lazy, page_size=page_size)
                    with patch(
                        "sglang.srt.mem_cache.allocator.unified_hybrid_swa._relieve_for_alloc",
                        side_effect=AssertionError("recovery cannot create bytes"),
                    ):
                        evict_from_tree_cache(cache, 4 * page_size)
                    self.assertEqual(cache.full_evictable_size(), 95 * page_size)
                    self.assertIsNotNone(allocator.alloc(4 * page_size))
                    self.assertFalse(allocator.verify_byte_accounting())

    def test_byte_bound_allows_recovery_from_unequal_peer_holes(self):
        allocator, cache = self.build_cache(
            occupancy=0, lazy=True, ratio=(1, 4), total_bytes=256 * 32
        )
        slots = allocator.alloc(50)
        self.assertIsNotNone(slots)
        # Window retirement preserves live Full bindings. Larger SWA holes can
        # fund both members after compaction despite the immediate joint limit.
        allocator.free_swa(slots[:4])
        self.assertLess(allocator.available_size(), 3)
        self.assertGreaterEqual(allocator.full_available_size(), 3)
        self.assertGreaterEqual(allocator.swa_available_size(), 3)
        evict_from_tree_cache(cache, 3)
        self.assertEqual(allocator.full_attn_allocator.allocated_count(), 50)
        self.assertIsNotNone(allocator.alloc(3))
        self.assertFalse(allocator.verify_byte_accounting())

    def test_joint_shortfall_enters_eviction_when_individual_targets_fit(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                allocator, cache = self.build_cache(lazy=lazy)
                self.assertEqual(allocator.full_available_size(), 4)
                self.assertEqual(allocator.swa_available_size(), 4)
                self.assertEqual(allocator.available_size(), 3)
                evict_from_tree_cache(cache, 4)
                self.assertEqual(cache.full_evictable_size(), 95)
                self.assertIsNotNone(allocator.alloc(4))
                self.assertFalse(allocator.verify_byte_accounting())

    def test_joint_shortfall_outlives_individual_eviction_quotas(self):
        for lazy in (False, True):
            for page_size in (1, 4, 16):
                with self.subTest(lazy=lazy, page_size=page_size):
                    allocator, cache = self.build_cache(lazy=lazy, page_size=page_size)
                    evict_from_tree_cache(cache, 6 * page_size)
                    self.assertEqual(cache.full_evictable_size(), 93 * page_size)
                    self.assertIsNotNone(allocator.alloc(6 * page_size))
                    self.assertGreaterEqual(allocator.conserve_full_available_size(), 0)
                    self.assertGreaterEqual(allocator.conserve_swa_available_size(), 0)
                    self.assertFalse(allocator.verify_byte_accounting())

    def test_tri_pool_token_allocation_with_live_state(self):
        for temporal in ((0, 0, 0), (1, 4, 8)):
            cases = [(False, 96, 3), (True, 96, 3)]
            if temporal == (0, 0, 0):
                cases += [
                    (False, 80, 24),
                    (True, 80, 24),
                    (False, 96, 8),
                    (True, 96, 8),
                ]
            for lazy, occupancy, state_count in cases:
                with self.subTest(
                    temporal=temporal, lazy=lazy, state_count=state_count
                ):
                    state_config = SimpleNamespace(
                        shape=SimpleNamespace(conv=[(3, 8)], temporal=temporal),
                        dtype=SimpleNamespace(
                            conv=torch.bfloat16, temporal=torch.float32
                        ),
                        layers=[0],
                    )
                    bundle = init_unified_mamba_swa_pools(
                        device="cpu",
                        kv_cache_dtype=torch.float16,
                        head_num=1,
                        head_dim=8,
                        v_head_dim=8,
                        swa_head_num=1,
                        swa_head_dim=8,
                        swa_v_head_dim=8,
                        page_size=1,
                        start_layer=0,
                        end_layer=2,
                        swa_attention_layer_ids=[1],
                        full_attention_layer_ids=[0],
                        full_max_total_num_tokens=100,
                        swa_max_total_num_tokens=100,
                        enable_memory_saver=False,
                        need_sort=False,
                        lazy_compaction=lazy,
                        mamba_layer_ids=[0],
                        mamba2_cache_params=state_config,
                        max_mamba_cache_size=8,
                        model_context_len=256,
                        extra_max_context_len=1,
                        max_num_reqs=4,
                        enable_mamba_extra_buffer=False,
                        disable_overlap_schedule=True,
                        sliding_window_size=32,
                    )
                    allocator = bundle.token_to_kv_pool_allocator
                    req_pool = bundle.req_to_token_pool
                    cache = UnifiedRadixCache(
                        CacheInitParams(
                            disable=False,
                            req_to_token_pool=req_pool,
                            token_to_kv_pool_allocator=allocator,
                            page_size=1,
                            sliding_window_size=128,
                            tree_components=(
                                ComponentType.FULL,
                                ComponentType.SWA,
                                ComponentType.MAMBA,
                            ),
                        )
                    )
                    slots = allocator.alloc(occupancy)
                    states = req_pool.mamba_allocator.alloc(state_count)
                    self.assertIsNotNone(slots)
                    self.assertIsNotNone(states)
                    for i, (lo, hi) in enumerate(((0, 1), (1, 2), (2, occupancy))):
                        cache.insert(
                            InsertParams(
                                key=RadixKey(
                                    array("q", range(i * 1000, i * 1000 + hi - lo))
                                ),
                                value=slots[lo:hi],
                                mamba_value=states[i : i + 1],
                            )
                        )
                    demand = allocator.available_size() + 1
                    if state_count == 8:
                        demand += 1
                    if state_count == 24:
                        demand = min(
                            allocator.full_available_size(),
                            allocator.swa_available_size(),
                        )
                        with patch(
                            "sglang.srt.mem_cache.allocator.unified_hybrid_swa._relieve_for_alloc",
                            side_effect=AssertionError(
                                "live state consumes shared bytes"
                            ),
                        ):
                            self.assertFalse(allocator.prepare_token_allocation(demand))
                    evict_from_tree_cache(cache, demand)
                    if state_count == 8:
                        self.assertEqual(cache.full_evictable_size(), 95)
                    self.assertTrue(allocator.token_allocation_ready(demand))
                    self.assertIsNotNone(allocator.alloc(demand))
                    self.assertGreaterEqual(allocator.conserve_full_available_size(), 0)
                    self.assertGreaterEqual(allocator.conserve_swa_available_size(), 0)
                    self.assertFalse(allocator.verify_byte_accounting())

    def test_sufficient_capacity_preserves_all_prefixes(self):
        allocator, cache = self.build_cache()
        with patch.object(
            allocator,
            "prepare_token_allocation",
            side_effect=AssertionError("unexpected recovery"),
        ):
            evict_from_tree_cache(cache, 3)
        self.assertEqual(cache.full_evictable_size(), 96)
        self.assertIsNotNone(allocator.alloc(3))

    def test_impossible_target_exhausts_without_claiming_readiness(self):
        allocator, cache = self.build_cache()
        evict_from_tree_cache(cache, 101)
        self.assertEqual(cache.full_evictable_size(), 0)
        self.assertFalse(allocator.token_allocation_ready(101))
        self.assertIsNone(allocator.alloc(101))

    def test_queued_composite_free_avoids_cache_eviction(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                allocator, cache = self.build_cache(occupancy=94, lazy=lazy)
                temporary = allocator.alloc(2)
                self.assertIsNotNone(temporary)
                allocator.free_group_begin()
                allocator.free(temporary)
                self.assertEqual(allocator.available_size(), 3)
                evict_from_tree_cache(cache, 4)
                self.assertEqual(cache.full_evictable_size(), 94)
                self.assertIsNotNone(allocator.alloc(4))
                self.assertEqual(allocator.free_group, [])
                allocator.free_group_end()
                self.assertFalse(allocator.verify_byte_accounting())

    def test_decode_gate_preserves_component_limit_after_swa_release(self):
        allocator, cache = self.build_cache(occupancy=0)
        slots = allocator.alloc(99)
        allocator.free_swa(slots[:50])
        self.assertGreaterEqual(allocator.available_size(), 2)
        self.assertEqual(allocator.full_available_size(), 1)
        self.assertFalse(allocator.token_allocation_ready(2))
        self.assertFalse(
            allocator.check_decode_capacity(num_tokens=2, tree_cache=cache)
        )

    def test_locked_cache_cannot_satisfy_joint_shortfall(self):
        allocator, cache = self.build_cache()
        node_ids = [
            cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", [i * 1000])))
            ).last_device_node
            for i in range(96)
        ]
        locks = [(node_id, cache.inc_lock_ref(node_id)) for node_id in node_ids]
        try:
            self.assertEqual(cache.full_evictable_size(), 0)
            evict_from_tree_cache(cache, 4)
            self.assertFalse(allocator.token_allocation_ready(4))
            self.assertEqual(allocator.available_size(), 3)
            self.assertFalse(allocator.verify_byte_accounting())
        finally:
            for node_id, lock in locks:
                cache.dec_lock_ref(node_id, lock.to_dec_params())
        self.assertEqual(cache.full_evictable_size(), 96)

    def test_explicit_eviction_keeps_count_semantics(self):
        allocator, cache = self.build_cache()
        result = cache.evict(EvictParams(num_tokens=2))
        self.assertEqual(result.num_tokens_evicted, 2)
        self.assertEqual(cache.full_evictable_size(), 94)

    def test_streaming_session_forwards_allocation_readiness(self):
        session = object.__new__(StreamingSession)
        session.inner = MagicMock()

        def ready():
            return False

        params = EvictParams()
        session.evict_for_alloc(params, allocation_ready=ready)
        session.inner.evict_for_alloc.assert_called_once_with(
            params, allocation_ready=ready
        )


if __name__ == "__main__":
    unittest.main()
