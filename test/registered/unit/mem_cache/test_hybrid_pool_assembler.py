"""Unit tests for hybrid HiCache pool assembly."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _build_mha_mla_host_pool,
    _evict_mamba_for_device_alloc,
    _evict_swa_for_device_alloc,
    _MambaStrategy,
    _MambaSwaStrategy,
    _require_single_row_dsv4_swa_pages,
    _split_hicache_size,
    _SwaStrategy,
    build_full_draft_pools,
    build_hybrid_swa_group,
    build_hybrid_mamba_stack,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
)
from sglang.srt.mem_cache.pool_host.unified import UnifiedPageEnvelopeHostPool
from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools
from sglang.srt.mem_cache.pool_host.common import alloc_with_host_register
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDeepSeekV4SWAPageLayout(CustomTestCase):
    def test_split_physical_rows_are_rejected_for_hicache_consumers(self):
        with self.assertRaisesRegex(ValueError, "direct SWA KV layout"):
            _require_single_row_dsv4_swa_pages(
                logical_page_size=256,
                physical_page_size=64,
                consumer="test consumer",
            )

    def test_matching_page_geometry_is_supported(self):
        _require_single_row_dsv4_swa_pages(
            logical_page_size=256,
            physical_page_size=256,
            consumer="test consumer",
        )


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


class TestHybridStageLayerMappings(CustomTestCase):
    def test_strategies_pass_stage_local_maps_without_changing_device_maps(self):
        """Later pipeline stages must transfer every layer before signaling completion."""
        cases = [
            (
                _MambaStrategy,
                "build_hybrid_mamba_stack",
                {"full": {3: 0}, "mamba": {0: 2, 1: 0, 2: 1}},
            ),
            (
                _SwaStrategy,
                "build_hybrid_swa_stack",
                {"full": {3: 0}, "swa": {0: 2, 1: 0, 2: 1}},
            ),
            (
                _MambaSwaStrategy,
                "build_hybrid_mamba_swa_stack",
                {"full": {3: 0}, "swa": {0: 2, 1: 0, 2: 1}, "mamba": {0: 1, 3: 0}},
            ),
        ]
        for strategy_cls, builder_name, local_maps in cases:
            for start_layer in (0, 4):
                with self.subTest(strategy=strategy_cls.__name__, start=start_layer):
                    global_maps = {
                        name: {
                            layer + start_layer: index
                            for layer, index in mapping.items()
                        }
                        for name, mapping in local_maps.items()
                    }
                    layers_mapping = {
                        layer: (index, name == "swa")
                        for name in ("full", "swa")
                        for layer, index in global_maps.get(name, {}).items()
                    }
                    kvcache = SimpleNamespace(
                        start_layer=start_layer,
                        full_attention_layer_id_mapping=global_maps["full"].copy(),
                        layers_mapping=layers_mapping.copy(),
                        full_kv_pool=object(),
                        swa_kv_pool=object(),
                        use_mla=False,
                    )
                    req_pool = SimpleNamespace(
                        mamba_map=global_maps.get("mamba", {}).copy(),
                        mamba_pool=object(),
                    )
                    params = SimpleNamespace(
                        req_to_token_pool=req_pool,
                        tp_cache_group=None,
                        pp_cache_group=None,
                    )
                    with patch.object(
                        hybrid_pool_assembler,
                        builder_name,
                        return_value=(
                            MagicMock(),
                            SimpleNamespace(transfer_layer_id_max=4),
                        ),
                    ) as build_stack:
                        result = strategy_cls().build(
                            cache=SimpleNamespace(page_size=1),
                            kvcache=kvcache,
                            params=params,
                            server_args=None,
                            load_cache_event=None,
                        )

                    build_stack.assert_called_once()
                    for name, mapping in local_maps.items():
                        self.assertEqual(
                            build_stack.call_args.kwargs[f"{name}_layer_mapping"],
                            mapping,
                        )
                    self.assertEqual(result.cache_controller.transfer_layer_id_max, 4)
                    self.assertEqual(
                        kvcache.full_attention_layer_id_mapping, global_maps["full"]
                    )
                    self.assertEqual(kvcache.layers_mapping, layers_mapping)
                    self.assertEqual(req_pool.mamba_map, global_maps.get("mamba", {}))


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


_ASSEMBLER = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."


class TestTransferLayerSpan(CustomTestCase):
    """``transfer_layer_id_max`` must span global layer ids, not count the mapped ones.

    A hybrid model with an uncached layer type keys its mappings non-contiguously,
    and the per-layer transfer loop then never reaches the high layer ids.
    """

    def test_pool_entries_span_the_highest_global_layer_id(self):
        # Global ids 0/2/4/6 with holes between them, the shape NemotronH's
        # cache-ineligible MLP layers produce: 4 mapped layers spanning 7 ids.
        full_layer_mapping = {0: 0, 6: 1}
        swa_layer_mapping = {2: 0, 4: 1}

        with (
            patch(_ASSEMBLER + "build_kv_host_pool"),
            patch(_ASSEMBLER + "HostPoolGroup"),
            patch(_ASSEMBLER + "build_pool_entry") as build_pool_entry,
        ):
            build_hybrid_swa_group(
                page_size=64,
                full_kv_pool=MagicMock(),
                swa_kv_pool=MagicMock(),
                full_layer_mapping=full_layer_mapping,
                swa_layer_mapping=swa_layer_mapping,
                use_mla=False,
            )

        self.assertEqual(
            [
                c.kwargs["transfer_layer_id_max"]
                for c in build_pool_entry.call_args_list
            ],
            [7, 7],
        )


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
    return UnifiedPageEnvelopeHostPool.build_hybrid_swa_pool_pair(
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

    def _build_stack(
        self,
        kv_pool,
        *,
        use_mla,
        draft_pools=(),
        extra_patches=(),
        layout="layer_first",
    ):
        """Build the hybrid Mamba stack with the real KV host pool class.

        Patched: the published memory/parallel config and allocator lookup,
        the MLA host pool's allocator (unpinned), plus the Mamba host pool,
        host pool group, and cache controller, which are not under test.
        Returns the HostPoolGroup constructor mock.
        """
        params = MagicMock()
        params.page_size = self.PAGE_SIZE
        wrapped_draft_pools = []
        for pool in draft_pools:
            wrapper = object.__new__(HybridLinearKVPool)
            wrapper.full_kv_pool = pool
            wrapped_draft_pools.append(wrapper)
        params.mtp_draft_device_pools = tuple(wrapped_draft_pools)
        memory = SimpleNamespace(
            hicache_size=0,
            hicache_ratio=2.0,
            hicache_mem_layout=layout,
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

    def test_split_rows_are_sized_by_both_device_components_in_builders(self):
        for packed in (False, True):
            with self.subTest(packed_fp8=packed):
                kv_pool = (
                    self._fake_packed_device_pool()
                    if packed
                    else self._fake_device_pool(
                        store_dtype=torch.bfloat16, kv_cache_dim=self.KV_LORA_RANK
                    )
                )
                kv_pool.kr_cache_dim = 0 if packed else self.QK_ROPE_HEAD_DIM
                kv_pool.dsa_kv_cache_store_fp8 = packed
                kv_pool.index_head_dim = None
                expected_width = kv_pool.kv_cache_dim + kv_pool.kr_cache_dim
                host = self._kv_host_pool(
                    self._build_stack(
                        kv_pool, use_mla=True, layout="page_first_kv_split"
                    )
                )
                self.assertEqual(host.kv_cache_dim, expected_width)
                self.assertEqual(host.k_buffer.shape[-1], kv_pool.kv_cache_dim)
                self.assertEqual(
                    host.size_per_token,
                    expected_width * kv_pool.store_dtype.itemsize * self.LAYER_NUM,
                )
                with patch(
                    "sglang.srt.mem_cache.pool_host.mla.ALLOC_MEMORY_FUNCS",
                    {"cpu": self._alloc_unpinned},
                ):
                    draft_host = _build_mha_mla_host_pool(
                        pool=kv_pool,
                        host_to_device_ratio=2,
                        page_size=self.PAGE_SIZE,
                        layout="page_first_kv_split",
                        allocator_type="default",
                        pool_label="draft",
                    )
                self.assertEqual(draft_host.kv_cache_dim, expected_width)
                self.assertEqual(draft_host.size_per_token, host.size_per_token)
                if not packed:
                    self.assertEqual(host.v_buffer.shape[-1], kv_pool.kr_cache_dim)
                    self.assertEqual(
                        host.size_per_token * host.size,
                        host.k_buffer.nbytes + host.v_buffer.nbytes,
                    )

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
        layout = kwargs.pop("layout", "layer_first")
        with patch(
            "sglang.srt.mem_cache.pool_host.mla.ALLOC_MEMORY_FUNCS",
            {"cpu": self._alloc_unpinned},
        ):
            return MLATokenToKVPoolHost(
                kv_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=self.PAGE_SIZE,
                layout=layout,
                pin_memory=False,
                device="cpu",
                **kwargs,
            )

    def test_decode_offload_constructs_the_stored_mla_row_width(self):
        from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
            DecodeKVCacheOffloadManager,
        )
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        publish(
            ServerArgs(
                model_path="dummy",
                page_size=self.PAGE_SIZE,
                hicache_ratio=2,
                hicache_mem_layout="layer_first",
            ),
            role="scheduler",
        )
        self.addCleanup(reset_context)
        pool = object.__new__(MLATokenToKVPool)
        pool.__dict__.update(self._fake_packed_device_pool().__dict__)
        prefix = "sglang.srt.disaggregation.decode_kvcache_offload_manager."
        with (
            patch(
                "sglang.srt.mem_cache.pool_host.mla.ALLOC_MEMORY_FUNCS",
                {"cpu": self._alloc_unpinned},
            ),
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler._get_allocator_type",
                return_value="default",
            ),
            patch(prefix + "torch.distributed.get_world_size", return_value=1),
            patch(prefix + "HiCacheController"),
        ):
            manager = DecodeKVCacheOffloadManager(
                None, SimpleNamespace(get_kvcache=lambda: pool), None, None
            )
        self.addCleanup(manager.release_host_resources)
        self.assertEqual(
            manager.decode_host_mem_pool.kv_cache_dim, self.PACKED_KV_CACHE_DIM
        )
        self.assertEqual(
            manager.decode_host_mem_pool.token_stride_size, self.PACKED_KV_CACHE_DIM
        )

    def test_fp4_rows_are_rejected_before_host_allocation(self):
        fp4 = object.__new__(MLATokenToKVPoolFP4)
        fp4.__dict__.update(self._fake_packed_device_pool(layer_num=1).__dict__)
        # The logical dimension agrees, but the physical values use half a
        # byte per element and scales live in a different buffer.
        fp4.kv_buffer = [torch.zeros((68, 1, 8), dtype=torch.uint8)]
        fp4.kv_scale_buffer = [torch.zeros((68, 1, 1), dtype=torch.uint8)]
        ordinary = self._fake_packed_device_pool(layer_num=1)
        for target, drafts in ((fp4, ()), (ordinary, (fp4,))):
            with self.subTest(target_fp4=target is fp4):
                allocate = MagicMock(side_effect=self._alloc_unpinned)
                with (
                    patch(
                        "sglang.srt.mem_cache.pool_host.mla.ALLOC_MEMORY_FUNCS",
                        {"cpu": allocate},
                    ),
                    self.assertRaisesRegex(NotImplementedError, "FP4 MLA KV"),
                ):
                    MLATokenToKVPoolHost(
                        target,
                        host_to_device_ratio=2,
                        host_size=0,
                        page_size=self.PAGE_SIZE,
                        layout="layer_first",
                        pin_memory=False,
                        device="cpu",
                        override_kv_cache_dim=target.kv_cache_dim,
                        mtp_draft_device_pools=drafts,
                    )
                allocate.assert_not_called()

    def test_packed_device_pool_without_override_is_rejected(self):
        kv_pool = self._fake_packed_device_pool()

        with self.assertRaisesRegex(ValueError, "override_kv_cache_dim"):
            self._host_pool(kv_pool)

    def test_nominal_device_pool_needs_no_override(self):
        # Flat MLA callers pass no override; a nominal-width pool must still
        # construct because its kv_cache_dim equals the assumed width.
        kv_pool = self._fake_nominal_device_pool()

        host_pool = self._host_pool(kv_pool)

        self.assertEqual(host_pool.kv_cache_dim, self.NOMINAL_KV_CACHE_DIM)

    def test_split_device_rows_keep_both_components_in_host_budget(self):
        # NPU's kv_cache_dim is the latent K width. RoPE occupies a separate
        # device/host buffer and must still count towards the host allocation.
        kv_pool = self._fake_device_pool(
            store_dtype=torch.bfloat16, kv_cache_dim=self.KV_LORA_RANK
        )
        kv_pool.kr_cache_dim = self.QK_ROPE_HEAD_DIM
        kv_pool.index_head_dim = None
        host_pool = self._host_pool(kv_pool, layout="page_first_kv_split")
        self.assertEqual(host_pool.k_buffer.shape[-1], self.KV_LORA_RANK)
        self.assertEqual(host_pool.v_buffer.shape[-1], self.QK_ROPE_HEAD_DIM)
        self.assertEqual(host_pool.kv_cache_dim, self.NOMINAL_KV_CACHE_DIM)
        self.assertEqual(
            host_pool.size_per_token,
            self.NOMINAL_KV_CACHE_DIM * torch.bfloat16.itemsize * self.LAYER_NUM,
        )
        with self.assertRaisesRegex(ValueError, "row geometry"):
            self._host_pool(
                kv_pool,
                layout="page_first_kv_split",
                override_kv_cache_dim=self.KV_LORA_RANK,
            )
        for dtype, width in (
            (torch.uint8, self.KV_LORA_RANK),
            (torch.bfloat16, self.NOMINAL_KV_CACHE_DIM),
        ):
            with self.subTest(draft_dtype=dtype, draft_width=width):
                draft = self._fake_device_pool(
                    store_dtype=dtype, kv_cache_dim=width, layer_num=1
                )
                with self.assertRaisesRegex(ValueError, "draft pool 0"):
                    self._host_pool(
                        kv_pool,
                        layout="page_first_kv_split",
                        mtp_draft_device_pools=(draft,),
                    )

    def test_split_packed_fp8_rows_still_require_matching_width(self):
        kv_pool = self._fake_packed_device_pool()
        kv_pool.dsa_kv_cache_store_fp8 = True
        kv_pool.kr_cache_dim = 0
        kv_pool.index_head_dim = None
        with self.assertRaisesRegex(ValueError, "override_kv_cache_dim"):
            self._host_pool(kv_pool, layout="page_first_kv_split")
        host_pool = self._host_pool(
            kv_pool,
            layout="page_first_kv_split",
            override_kv_cache_dim=kv_pool.kv_cache_dim,
        )
        self.assertEqual(host_pool.k_buffer.shape[-1], kv_pool.kv_cache_dim)


# These CPU geometry tests allocate tiny, unpinned pools. Host-budget policy is
# tested separately; a co-resident serving process must not affect their oracle.
def setUpModule():
    global _host_budget_patch
    _host_budget_patch = patch(
        "sglang.srt.mem_cache.pool_host.base.available_host_memory_bytes",
        return_value=1 << 40,
    )
    _host_budget_patch.start()


def tearDownModule():
    _host_budget_patch.stop()


if __name__ == "__main__":
    unittest.main()
