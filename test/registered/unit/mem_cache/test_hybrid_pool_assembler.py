"""Unit tests for hybrid HiCache pool assembly."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _split_hicache_size,
    _validate_deepseek_v4_dcp_hicache_geometry,
    build_deepseek_v4_hicache_stack,
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
            server_args=None,
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
        server_args = SimpleNamespace(hicache_mem_layout="page_first")

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
                server_args=server_args,
            )

        self.assertEqual(build_host_pool.call_args.kwargs["host_to_device_ratio"], 1.0)
        self.assertEqual(len(specs), 1)
        self.assertIs(entries[0].host_pool, draft_host_pool)


class TestDeepSeekV4DCPHiCacheGeometry(CustomTestCase):
    @staticmethod
    def _make_geometry():
        allocator = SimpleNamespace(
            supports_dsv4_dcp=True,
            dcp_size=8,
            dcp_rank=3,
            physical_page_size=256,
            page_size=2048,
        )
        params = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            page_size=2048,
        )
        kvcache = SimpleNamespace(
            supports_dsv4_dcp=True,
            _unified_kv=True,
            dcp_size=8,
            dcp_rank=3,
            logical_page_size=2048,
        )
        return params, kvcache

    def test_accepts_matching_unified_kv_geometry(self):
        params, kvcache = self._make_geometry()

        self.assertTrue(
            _validate_deepseek_v4_dcp_hicache_geometry(
                params=params, kvcache=kvcache, requested_dcp_size=8
            )
        )

    def test_no_dcp_does_not_require_specialized_geometry(self):
        params = SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace())

        self.assertFalse(
            _validate_deepseek_v4_dcp_hicache_geometry(
                params=params,
                kvcache=SimpleNamespace(),
                requested_dcp_size=1,
            )
        )

    def test_rejects_missing_dcp_allocator(self):
        params = SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace())

        with self.assertRaisesRegex(NotImplementedError, "widened DSV4 DCP allocator"):
            _validate_deepseek_v4_dcp_hicache_geometry(
                params=params,
                kvcache=SimpleNamespace(),
                requested_dcp_size=8,
            )

    def test_rejects_requested_dcp_size_mismatch(self):
        params, kvcache = self._make_geometry()

        with self.assertRaisesRegex(ValueError, "requested 4, allocator 8"):
            _validate_deepseek_v4_dcp_hicache_geometry(
                params=params, kvcache=kvcache, requested_dcp_size=4
            )

    def test_rejects_non_unified_kv(self):
        params, kvcache = self._make_geometry()
        kvcache._unified_kv = False

        with self.assertRaisesRegex(NotImplementedError, "unified-KV"):
            _validate_deepseek_v4_dcp_hicache_geometry(
                params=params, kvcache=kvcache, requested_dcp_size=8
            )

    def test_rejects_unwidened_page(self):
        params, kvcache = self._make_geometry()
        params.page_size = 256

        with self.assertRaisesRegex(ValueError, "widened allocator page size"):
            _validate_deepseek_v4_dcp_hicache_geometry(
                params=params, kvcache=kvcache, requested_dcp_size=8
            )

    def test_rejects_kv_pool_geometry_mismatch(self):
        params, kvcache = self._make_geometry()
        kvcache.dcp_rank = 4

        with self.assertRaisesRegex(ValueError, "does not match its allocator"):
            _validate_deepseek_v4_dcp_hicache_geometry(
                params=params, kvcache=kvcache, requested_dcp_size=8
            )

    def test_enables_strict_transfers_for_all_compressed_sidecars(self):
        params, kvcache = self._make_geometry()
        params.token_to_kv_pool_allocator.size_full = 4096
        params.mtp_draft_device_pools = ()
        params.tp_cache_group = None
        params.attn_cp_cache_group = None
        params.attn_tp_cache_group = None
        params.pp_cache_group = None

        fake_index_buffer = SimpleNamespace(shape=(1, 8), element_size=lambda: 2)
        kvcache.size = 4096
        kvcache.swa_size = 2048
        kvcache.swa_page_size = 256
        kvcache.start_layer = 0
        kvcache.end_layer = 2
        kvcache.layer_mapping = [
            SimpleNamespace(compress_ratio=4, compress_layer_id=0),
            SimpleNamespace(compress_ratio=128, compress_layer_id=0),
        ]
        kvcache.c4_kv_pool = SimpleNamespace()
        kvcache.c128_kv_pool = SimpleNamespace()
        kvcache.c4_indexer_kv_pool = SimpleNamespace(
            index_k_with_scale_buffer=[fake_index_buffer]
        )
        kvcache.unified_region_buffers = lambda ratio: ([SimpleNamespace()], ratio)

        server_args = SimpleNamespace(
            dcp_size=8,
            hicache_size=0,
            hicache_mem_layout="page_first",
            hicache_write_policy="write_through",
            hicache_io_backend="kernel",
            hicache_host_memory_mode="cache",
        )
        fake_host_pool = SimpleNamespace(can_use_write_back_jit=False)

        with (
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "DeepSeekV4PagedHostPool",
                return_value=fake_host_pool,
            ) as build_paged_pool,
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "HybridCacheController"
            ),
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler." "get_memory",
                return_value=SimpleNamespace(hicache_ratio=1.0),
            ),
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "_get_allocator_type",
                return_value="default",
            ),
        ):
            host_pool_group, _ = build_deepseek_v4_hicache_stack(
                params=params,
                server_args=server_args,
                kvcache=kvcache,
                load_cache_event=None,
                storage_backend=None,
            )

        compressed_names = {
            PoolName.DEEPSEEK_V4_C4,
            PoolName.DEEPSEEK_V4_C4_INDEXER,
            PoolName.DEEPSEEK_V4_C128,
        }
        compressed_entries = {
            entry.name: entry
            for entry in host_pool_group.entries
            if entry.name in compressed_names
        }
        self.assertEqual(set(compressed_entries), compressed_names)
        self.assertEqual(len(build_paged_pool.call_args_list), 3)
        self.assertTrue(
            all(
                call.kwargs["require_full_page_transfers"]
                for call in build_paged_pool.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
