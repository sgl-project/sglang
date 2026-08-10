"""Unit tests for hybrid HiCache pool assembly."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _get_mamba_hicache_ratio,
    _split_hicache_size,
    build_full_draft_pools,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Pool:
    def __init__(self, kv_bytes):
        self._kv_bytes = kv_bytes

    def get_kv_size_bytes(self):
        return self._kv_bytes


class TestSplitHicacheSize(CustomTestCase):
    def test_splits_total_budget_by_device_bytes(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(75 * 10**9), _Pool((15 * 10**9, 10 * 10**9)))
        )
        self.assertEqual(shares, (75.0, 25.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled

    def test_mamba_ratio_defaults_to_shared_hicache_ratio(self):
        server_args = SimpleNamespace(hicache_ratio=3.0, hicache_mamba_ratio=None)
        self.assertEqual(_get_mamba_hicache_ratio(server_args), 3.0)

    def test_mamba_ratio_can_be_overridden_independently(self):
        server_args = SimpleNamespace(hicache_ratio=3.0, hicache_mamba_ratio=1.0)
        self.assertEqual(_get_mamba_hicache_ratio(server_args), 1.0)

    def test_splits_total_budget_by_device_bytes_three_pools(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(55 * 10**9), _Pool((15 * 10**9, 10 * 10**9)), _Pool(20 * 10**9))
        )
        self.assertEqual(shares, (55.0, 25.0, 20.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled


class TestDraftSidecars(CustomTestCase):
    def test_hybrid_linear_draft_registers_full_attention_subpool(self):
        full_pool = MagicMock()
        full_pool.layer_num = 1
        full_pool.size = 128
        draft_pool = object.__new__(HybridLinearKVPool)
        draft_pool.full_kv_pool = full_pool

        draft_host_pool = MagicMock()
        draft_host_pool.layer_num = 1
        host_pool_group = MagicMock()
        host_pool_group.size = 256
        controller = MagicMock()
        controller.mem_pool_host = host_pool_group
        controller.page_size = 1
        tree_cache = MagicMock()
        tree_cache.cache_controller = controller
        server_args = MagicMock()
        server_args.hicache_mem_layout = "page_first"

        with (
            patch.object(
                hybrid_pool_assembler,
                "_build_mha_mla_host_pool",
                return_value=draft_host_pool,
            ) as build_host_pool,
            patch.object(
                hybrid_pool_assembler,
                "_get_allocator_type",
                return_value="kernel",
            ),
        ):
            specs, entries = build_full_draft_pools(
                draft_kv_pool=draft_pool,
                tree_cache=tree_cache,
                server_args=server_args,
            )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].pool_name, PoolName.DRAFT)
        self.assertEqual(len(entries), 1)
        self.assertIs(entries[0].device_pool, full_pool)
        self.assertIs(build_host_pool.call_args.kwargs["pool"], full_pool)


if __name__ == "__main__":
    unittest.main()
