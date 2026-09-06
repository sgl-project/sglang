"""Unit tests for hybrid HiCache pool assembly."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _evict_mamba_for_device_alloc,
    _evict_swa_for_device_alloc,
    _split_hicache_size,
    build_full_draft_pools,
    build_hybrid_mamba_stack,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.pool_host.common import alloc_with_host_register
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
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


class _PackedRowGeometryFixtures:
    """Fake MLA device pools with nominal-width and packed (wider) rows.

    DSA device pools store packed rows wider than kv_lora_rank +
    qk_rope_head_dim, the width the MLA host pool assumes without an
    override.
    """

    KV_LORA_RANK = 8
    QK_ROPE_HEAD_DIM = 4
    NOMINAL_KV_CACHE_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
    PACKED_KV_CACHE_DIM = 16  # > NOMINAL_KV_CACHE_DIM
    PAGE_SIZE = 4
    LAYER_NUM = 2

    def _fake_device_pool(self, *, store_dtype, kv_cache_dim, layer_num=LAYER_NUM):
        return SimpleNamespace(
            size=64,
            store_dtype=store_dtype,
            kv_lora_rank=self.KV_LORA_RANK,
            qk_rope_head_dim=self.QK_ROPE_HEAD_DIM,
            kv_cache_dim=kv_cache_dim,
            layer_num=layer_num,
            start_layer=0,
            end_layer=layer_num,
            device="cpu",
            layers_to_capture=None,
            layer_shard_enabled=False,
            data_ptrs=torch.zeros(layer_num, dtype=torch.uint64),
            kv_buffer=[torch.empty(0, dtype=store_dtype) for _ in range(layer_num)],
        )

    def _fake_packed_device_pool(self, layer_num=LAYER_NUM):
        return self._fake_device_pool(
            store_dtype=torch.uint8,
            kv_cache_dim=self.PACKED_KV_CACHE_DIM,
            layer_num=layer_num,
        )

    def _fake_nominal_device_pool(self, layer_num=LAYER_NUM):
        return self._fake_device_pool(
            store_dtype=torch.bfloat16,
            kv_cache_dim=self.NOMINAL_KV_CACHE_DIM,
            layer_num=layer_num,
        )

    @staticmethod
    def _alloc_unpinned(dims, dtype, device, pin_memory, allocator, **kwargs):
        # Host rows are allocated without pinning so no CUDA context is needed.
        return alloc_with_host_register(dims, dtype, device, False, allocator, **kwargs)


class TestHybridMambaStackHostRowWidth(_PackedRowGeometryFixtures, CustomTestCase):
    """The hybrid Mamba stack's MLA host pool must mirror the device rows.

    Packed MTP draft layers share the target's host rows, so a draft pool
    with a different row geometry must be rejected.
    """

    def _build_stack(self, kv_pool, *, use_mla, draft_pools=(), extra_patches=()):
        """Build the hybrid Mamba stack with the real KV host pool class.

        Patched: the published memory/parallel config and allocator lookup,
        the MLA host pool's allocator (unpinned), plus the Mamba host pool,
        host pool group, and cache controller, which are not under test.
        Returns the HostPoolGroup constructor mock.
        """
        params = MagicMock()
        params.page_size = self.PAGE_SIZE
        params.mtp_draft_device_pools = tuple(
            SimpleNamespace(full_kv_pool=pool) for pool in draft_pools
        )
        memory = SimpleNamespace(
            hicache_size=0,
            hicache_ratio=2.0,
            hicache_mem_layout="layer_first",
            hicache_write_policy="write_through",
            hicache_io_backend="direct",
            hicache_host_memory_mode=None,
        )
        parallel = SimpleNamespace(dcp_enabled=False)
        prefix = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."

        with ExitStack() as stack:
            stack.enter_context(patch(prefix + "get_memory", return_value=memory))
            stack.enter_context(patch(prefix + "get_parallel", return_value=parallel))
            stack.enter_context(
                patch(prefix + "_get_allocator_type", return_value="default")
            )
            stack.enter_context(
                patch(
                    "sglang.srt.mem_cache.pool_host.mla.ALLOC_MEMORY_FUNCS",
                    {"cpu": self._alloc_unpinned},
                )
            )
            stack.enter_context(patch(prefix + "MambaPoolHost"))
            host_pool_group = stack.enter_context(patch(prefix + "HostPoolGroup"))
            stack.enter_context(patch(prefix + "HybridCacheController"))
            for extra_patch in extra_patches:
                stack.enter_context(extra_patch)
            build_hybrid_mamba_stack(
                params=params,
                kv_pool=kv_pool,
                mamba_pool=MagicMock(),
                full_layer_mapping={i: i for i in range(kv_pool.layer_num)},
                mamba_layer_mapping={kv_pool.layer_num: kv_pool.layer_num},
                load_cache_event=None,
                storage_backend=None,
                use_mla=use_mla,
            )
        return host_pool_group

    def _kv_host_pool(self, host_pool_group):
        entries = host_pool_group.call_args.args[0]
        return entries[0].host_pool

    def test_mla_host_pool_row_width_matches_packed_device_rows(self):
        kv_pool = self._fake_packed_device_pool()
        self.assertGreater(kv_pool.kv_cache_dim, self.NOMINAL_KV_CACHE_DIM)

        kv_host_pool = self._kv_host_pool(self._build_stack(kv_pool, use_mla=True))

        self.assertIsInstance(kv_host_pool, MLATokenToKVPoolHost)
        self.assertEqual(kv_host_pool.kv_cache_dim, kv_pool.kv_cache_dim)
        self.assertEqual(
            kv_host_pool.token_stride_size,
            kv_pool.kv_cache_dim * kv_pool.store_dtype.itemsize,
        )
        self.assertEqual(kv_host_pool.kv_buffer.shape[-1], kv_pool.kv_cache_dim)

    def test_non_mla_pool_exposing_kv_cache_dim_gets_no_override(self):
        # MHA host pool constructors take no override_kv_cache_dim; a non-MLA
        # pool that happens to expose kv_cache_dim must not trigger one.
        kv_pool = self._fake_packed_device_pool()
        mha_host_cls = MagicMock(name="MHAHostPool")
        prefix = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."

        self._build_stack(
            kv_pool,
            use_mla=False,
            extra_patches=(
                patch(prefix + "get_mha_host_pool_cls", return_value=mha_host_cls),
            ),
        )

        mha_host_cls.assert_called_once()
        self.assertNotIn("override_kv_cache_dim", mha_host_cls.call_args.kwargs)

    def test_packed_draft_pool_with_target_geometry_is_accepted(self):
        kv_pool = self._fake_packed_device_pool()
        draft_pool = self._fake_packed_device_pool(layer_num=1)

        kv_host_pool = self._kv_host_pool(
            self._build_stack(kv_pool, use_mla=True, draft_pools=(draft_pool,))
        )

        self.assertEqual(kv_host_pool.layer_num, kv_pool.layer_num + 1)
        self.assertEqual(kv_host_pool.kv_cache_dim, kv_pool.kv_cache_dim)
        self.assertEqual(kv_host_pool.kv_buffer.shape[0], kv_pool.layer_num + 1)

    def test_packed_draft_pool_with_narrower_rows_is_rejected(self):
        # fp8 packed target, bf16 nominal-width draft.
        kv_pool = self._fake_packed_device_pool()
        draft_pool = self._fake_nominal_device_pool(layer_num=1)

        with self.assertRaisesRegex(ValueError, "draft pool 0"):
            self._build_stack(kv_pool, use_mla=True, draft_pools=(draft_pool,))

    def test_packed_draft_pool_with_wider_rows_is_rejected(self):
        # bf16 nominal-width target, fp8 packed draft.
        kv_pool = self._fake_nominal_device_pool()
        draft_pool = self._fake_packed_device_pool(layer_num=1)

        with self.assertRaisesRegex(ValueError, "draft pool 0"):
            self._build_stack(kv_pool, use_mla=True, draft_pools=(draft_pool,))

    def test_same_dtype_draft_pool_with_different_width_is_rejected(self):
        # Same store dtype, so only the row width differs: fp8 packed target,
        # fp8 nominal-width draft.
        kv_pool = self._fake_packed_device_pool()
        draft_pool = self._fake_device_pool(
            store_dtype=torch.uint8,
            kv_cache_dim=self.NOMINAL_KV_CACHE_DIM,
            layer_num=1,
        )
        self.assertEqual(draft_pool.store_dtype, kv_pool.store_dtype)

        with self.assertRaisesRegex(ValueError, "draft pool 0"):
            self._build_stack(kv_pool, use_mla=True, draft_pools=(draft_pool,))


class TestMLAHostPoolRowGeometry(_PackedRowGeometryFixtures, CustomTestCase):
    """MLATokenToKVPoolHost itself must refuse rows narrower than the device's.

    Exercises the constructor directly so the check does not depend on a
    builder passing the override.
    """

    def _host_pool(self, kv_pool, **kwargs):
        with patch(
            "sglang.srt.mem_cache.pool_host.mla.ALLOC_MEMORY_FUNCS",
            {"cpu": self._alloc_unpinned},
        ):
            return MLATokenToKVPoolHost(
                kv_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=self.PAGE_SIZE,
                layout="layer_first",
                pin_memory=False,
                device="cpu",
                **kwargs,
            )

    def test_packed_device_pool_without_override_is_rejected(self):
        kv_pool = self._fake_packed_device_pool()

        with self.assertRaisesRegex(ValueError, "override_kv_cache_dim"):
            self._host_pool(kv_pool)

    def test_packed_device_pool_without_override_and_nominal_draft_is_rejected(self):
        # Without the override the assumed host width equals the nominal
        # width, so a same-dtype nominal-width draft matches the assumed
        # width while both differ from the packed device rows. The draft
        # must be compared against the device pool's kv_cache_dim, not the
        # assumed width.
        kv_pool = self._fake_packed_device_pool()
        draft_pool = self._fake_device_pool(
            store_dtype=kv_pool.store_dtype,
            kv_cache_dim=self.NOMINAL_KV_CACHE_DIM,
            layer_num=1,
        )

        with self.assertRaisesRegex(ValueError, "row geometry"):
            self._host_pool(kv_pool, mtp_draft_device_pools=(draft_pool,))

    def test_packed_device_pool_with_override_and_matching_draft(self):
        kv_pool = self._fake_packed_device_pool()
        draft_pool = self._fake_packed_device_pool(layer_num=1)

        host_pool = self._host_pool(
            kv_pool,
            override_kv_cache_dim=kv_pool.kv_cache_dim,
            mtp_draft_device_pools=(draft_pool,),
        )

        self.assertEqual(host_pool.kv_cache_dim, kv_pool.kv_cache_dim)
        self.assertEqual(host_pool.layer_num, kv_pool.layer_num + 1)
        self.assertEqual(host_pool.kv_buffer.shape[-1], kv_pool.kv_cache_dim)

    def test_nominal_device_pool_needs_no_override(self):
        # Flat MLA callers pass no override; a nominal-width pool must still
        # construct because its kv_cache_dim equals the assumed width.
        kv_pool = self._fake_nominal_device_pool()

        host_pool = self._host_pool(kv_pool)

        self.assertEqual(host_pool.kv_cache_dim, self.NOMINAL_KV_CACHE_DIM)


if __name__ == "__main__":
    unittest.main()
