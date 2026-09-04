"""Unit tests for hybrid HiCache pool assembly."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _evict_mamba_for_device_alloc,
    _evict_swa_for_device_alloc,
    _split_hicache_size,
    build_full_draft_pools,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls
from sglang.srt.mem_cache.pool_host.unified import UnifiedPageEnvelopeHostPool
from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Pool:
    def __init__(self, kv_bytes):
        self._kv_bytes = kv_bytes

    def get_kv_size_bytes(self):
        return self._kv_bytes


class TestDeviceAllocEviction(CustomTestCase):
    def test_swa_evicts_only_allocation_shortfall(self):
        cache = MagicMock()
        cache.token_to_kv_pool_allocator.swa_available_size.return_value = 8

        _evict_swa_for_device_alloc(cache, required_size=10)

        cache.evict_for_alloc.assert_called_once_with(EvictParams(swa_num_tokens=2))
        cache.evict.assert_not_called()

    def test_mamba_evicts_only_allocation_shortfall(self):
        cache = MagicMock()
        allocator = cache.req_to_token_pool.mamba_allocator
        allocator.schedulable_available_size.return_value = 8

        _evict_mamba_for_device_alloc(cache, required_size=10)

        cache.evict_for_alloc.assert_called_once_with(EvictParams(mamba_num=2))
        cache.evict.assert_not_called()

    def test_sufficient_capacity_skips_eviction(self):
        cache = MagicMock()
        cache.token_to_kv_pool_allocator.swa_available_size.return_value = 10
        cache.req_to_token_pool.mamba_allocator.schedulable_available_size.return_value = 10

        _evict_swa_for_device_alloc(cache, required_size=10)
        _evict_mamba_for_device_alloc(cache, required_size=10)

        cache.evict_for_alloc.assert_not_called()
        cache.evict.assert_not_called()


class TestSplitHicacheSize(CustomTestCase):
    def test_splits_total_budget_by_device_bytes(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(75 * 10**9), _Pool((15 * 10**9, 10 * 10**9)))
        )
        self.assertEqual(shares, (75.0, 25.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled

    def test_splits_total_budget_by_device_bytes_three_pools(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(55 * 10**9), _Pool((15 * 10**9, 10 * 10**9)), _Pool(20 * 10**9))
        )
        self.assertEqual(shares, (55.0, 25.0, 20.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled


class TestDraftSidecarPoolDispatch(CustomTestCase):
    def test_full_builder_unwraps_empty_hybrid_linear_pool(self):
        draft_kv_pool = object.__new__(HybridLinearKVPool)
        draft_kv_pool.full_kv_pool = SimpleNamespace(layer_num=0)

        specs, entries = build_full_draft_pools(
            draft_kv_pool=draft_kv_pool,
            tree_cache=None,
        )

        self.assertEqual(specs, [])
        self.assertEqual(entries, [])

    def test_full_builder_sizes_sidecar_for_anchor_logical_space(self):
        draft_kv_pool = SimpleNamespace(layer_num=1, size=800)
        draft_host_pool = SimpleNamespace(layer_num=1)
        tree_cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(size=100, logical_size=800),
                page_size=512,
            )
        )
        # The layout comes from the published configuration.
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(model_path="dummy", hicache_mem_layout="page_first")
        publish(server_args, role="scheduler")
        self.addCleanup(reset_context)

        with (
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "_build_mha_mla_host_pool",
                return_value=draft_host_pool,
            ) as build_host_pool,
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "_get_allocator_type",
                return_value="default",
            ),
        ):
            specs, entries = build_full_draft_pools(
                draft_kv_pool=draft_kv_pool,
                tree_cache=tree_cache,
            )

        self.assertEqual(build_host_pool.call_args.kwargs["host_to_device_ratio"], 1.0)
        self.assertEqual(len(specs), 1)
        self.assertIs(entries[0].host_pool, draft_host_pool)


def _build_unified_swa_pool(page_size: int = 4):
    return init_unified_swa_pools(
        device="cpu",
        kv_cache_dtype=torch.float16,
        head_num=2,
        head_dim=4,
        v_head_dim=4,
        swa_head_num=2,
        swa_head_dim=4,
        swa_v_head_dim=4,
        page_size=page_size,
        start_layer=0,
        end_layer=4,
        swa_attention_layer_ids=[3],
        full_attention_layer_ids=[0, 1, 2],
        total_bytes=4096,
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=True,
    )


def _build_unified_host_pair(bundle):
    pool = bundle.token_to_kv_pool
    host_pool_cls = get_mha_host_pool_cls(pool.full_kv_pool)
    assert host_pool_cls is UnifiedPageEnvelopeHostPool
    assert get_mha_host_pool_cls(pool.swa_kv_pool) is host_pool_cls
    return host_pool_cls.build_hybrid_swa_pool_pair(
        device_pools=(pool.full_kv_pool, pool.swa_kv_pool),
        host_to_device_ratio=1.0,
        host_size=0,
        page_size=pool.page_size,
        layout="layer_first",
        pin_memory=False,
        allocator_type="default",
    )


class TestUnifiedPageEnvelopeHostPool(CustomTestCase):
    def test_shared_arena_can_reuse_bytes_across_sides(self):
        page_size = 4
        full_pool, swa_pool = _build_unified_host_pair(
            _build_unified_swa_pool(page_size)
        )
        self.addCleanup(full_pool.destroy)
        self.addCleanup(swa_pool.destroy)

        full_indices = full_pool.alloc(full_pool.available_size())
        self.assertIsNotNone(full_indices)
        self.assertIsNotNone(swa_pool.alloc(swa_pool.available_size()))
        self.assertIsNone(swa_pool.alloc(page_size))

        full_pool.free(full_indices[page_size:])
        self.assertIsNotNone(swa_pool.alloc(page_size))


if __name__ == "__main__":
    unittest.main()
