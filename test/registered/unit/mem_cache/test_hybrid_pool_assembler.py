"""Unit tests for hybrid HiCache pool assembly."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    StackBuildResult,
    _DsaStrategy,
    _evict_mamba_for_device_alloc,
    _evict_swa_for_device_alloc,
    _legacy_build_anchor_sidecar_stack,
    _MambaStrategy,
    _MambaSwaStrategy,
    _require_single_row_dsv4_swa_pages,
    _split_hicache_size,
    _SwaStrategy,
    _verify_declared_pools,
    assemble_declared_stack,
    build_full_draft_pools,
    build_hybrid_swa_group,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool
from sglang.srt.mem_cache.pool_host import dsa as pool_host_dsa
from sglang.srt.mem_cache.pool_host.host_pool_decl import HostPoolDecl
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

    def test_full_builder_registers_separate_dsa_draft_indexer(self):
        """The separate-draft DSA branch must build its indexer mirror from a
        DRAFT_INDEXER desc; a constructor change that skips this call site
        breaks only here, not on the target path."""
        draft_kv_pool = object.__new__(DSATokenToKVPool)
        draft_kv_pool.layer_num = 1
        draft_kv_pool.size = 800
        draft_kv_pool.index_head_dim = 128
        draft_kv_pool.index_key_cache = SimpleNamespace(buffer=[object()])
        draft_host_pool = SimpleNamespace(layer_num=1)
        tree_cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(size=100, logical_size=800),
                page_size=512,
            )
        )
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        publish(
            ServerArgs(model_path="dummy", hicache_mem_layout="page_first"),
            role="scheduler",
        )
        self.addCleanup(reset_context)

        seen = {}

        def fake_indexer_host(desc, device_pool, anchor_host, *, allocator_type):
            self.assertIsInstance(desc, HostPoolDecl)
            self.assertIs(device_pool, draft_kv_pool)
            self.assertIs(anchor_host, draft_host_pool)
            seen["desc"] = desc
            return SimpleNamespace(layer_num=1)

        with (
            patch.object(
                hybrid_pool_assembler,
                "_build_mha_mla_host_pool",
                return_value=draft_host_pool,
            ),
            patch.object(
                hybrid_pool_assembler, "_get_allocator_type", return_value="default"
            ),
            patch.object(
                hybrid_pool_assembler, "DSAIndexerPoolHost", fake_indexer_host
            ),
        ):
            specs, entries = build_full_draft_pools(
                draft_kv_pool=draft_kv_pool,
                tree_cache=tree_cache,
            )

        self.assertEqual(seen["desc"].name, PoolName.DRAFT_INDEXER)
        self.assertEqual(seen["desc"].index_source, PoolName.KV)
        self.assertEqual(specs[1], seen["desc"].sidecar_spec())
        self.assertEqual(entries[1].name, PoolName.DRAFT_INDEXER)


def _dsa_pool_stub(*, layer_num: int, size: int = 4096, shard: tuple | None = None):
    """DSATokenToKVPool shape without CUDA. ``shard=(rank, size)`` marks the pool
    layer-sharded with rank owning a contiguous local layer range."""
    pool = object.__new__(DSATokenToKVPool)
    pool.layer_num = layer_num
    pool.size = size
    pool.start_layer = 0
    pool.end_layer = layer_num - 1
    pool.store_dtype = torch.bfloat16
    pool.kv_lora_rank = 512
    pool.qk_rope_head_dim = 64
    pool.kv_cache_dim = 576
    pool.index_head_dim = 128
    pool.index_key_cache = SimpleNamespace(buffer=[object()] * layer_num)
    pool.layer_shard_enabled = shard is not None
    if shard is not None:
        from sglang.srt.layers.cp.utils import get_layer_shard_range

        rank, shard_size = shard
        pool.layer_shard_size = shard_size
        pool._owned_local_layer_range = lambda: get_layer_shard_range(
            rank, shard_size, layer_num
        )
    return pool


def _mirror_shape(host):
    shape = {
        "size": host.size,
        "page_num": host.page_num,
        "layer_num": host.layer_num,
        "size_per_token": host.size_per_token,
        "layout": host.layout,
    }
    if isinstance(host, pool_host_dsa.DSAIndexerPoolHost):
        shape["indexer_page_stride_size"] = host.indexer_page_stride_size
        shape["indexer_layout_dim"] = host.indexer_layout_dim
    return shape


def _entry_shape(group, transfer_layer_num):
    """Everything the controller reads from a HostPoolGroup, in comparable form."""
    return [
        (
            entry.name,
            entry.is_primary_index_anchor,
            id(entry.device_pool),
            tuple(id(p) for p in entry.packed_draft_device_pools),
            tuple(entry.layer_mapper(i) for i in range(-1, transfer_layer_num + 2)),
            _mirror_shape(entry.host_pool),
        )
        for entry in group.entries
    ]


class TestDeclaredStackParity(CustomTestCase):
    """assemble_declared_stack must build the same mirrors (real dummy host
    pools: size, page_num, layers, byte stride), entries, layer mapping,
    sidecars and controller arguments as the pre-declaration DSA assembly."""

    def _run(self, builder, *, pool, params, full_layer_mapping, **kw):
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        real_indexer_host = pool_host_dsa.DSAIndexerPoolHost

        def dummy_kv_host(**kwargs):
            return MLATokenToKVPoolHost(
                kwargs["kv_pool"],
                host_to_device_ratio=2,
                host_size=0,
                page_size=kwargs["page_size"],
                layout="page_first",
                pin_memory=False,
                is_dummy=True,
                override_kv_cache_dim=kwargs["override_kv_cache_dim"],
                mtp_draft_device_pools=kwargs["mtp_draft_device_pools"],
            )

        def dummy_indexer_host(decl, device_pool, anchor_host, *, allocator_type):
            return real_indexer_host(
                decl,
                device_pool,
                anchor_host,
                allocator_type=allocator_type,
                pin_memory=False,
                is_dummy=True,
            )

        controller = MagicMock()
        with (
            patch.object(hybrid_pool_assembler, "build_kv_host_pool", dummy_kv_host),
            patch.object(
                hybrid_pool_assembler, "DSAIndexerPoolHost", dummy_indexer_host
            ),
            patch.object(pool_host_dsa, "DSAIndexerPoolHost", dummy_indexer_host),
            patch.object(hybrid_pool_assembler, "HybridCacheController", controller),
            patch.object(
                hybrid_pool_assembler, "_get_allocator_type", return_value="default"
            ),
            patch.object(
                hybrid_pool_assembler,
                "get_memory",
                return_value=SimpleNamespace(
                    hicache_write_policy="write_through",
                    hicache_io_backend="kernel",
                    hicache_host_memory_mode="cache",
                ),
            ),
        ):
            out = builder(
                params=params,
                kv_pool=pool,
                full_layer_mapping=dict(full_layer_mapping),
                load_cache_event=None,
                storage_backend=None,
                use_mla=True,
                override_kv_cache_dim=pool.kv_cache_dim,
                **kw,
            )
        (call,) = controller.call_args_list
        controller_args = (call.args[2], dict(call.kwargs))  # page_size, kwargs
        return out, controller_args

    def test_matches_legacy_assembly(self):
        from sglang.srt.mem_cache.pool_host.dsa import dsa_indexer_pool_decl

        cases = {
            "identity": dict(shard=None, mapping={0: 0, 1: 1, 2: 2}, drafts=0),
            "packed_draft": dict(shard=None, mapping={0: 0, 1: 1, 2: 2}, drafts=1),
            "sharded_permuted": dict(
                shard=(0, 2), mapping={0: 1, 1: 2, 2: 0}, drafts=0
            ),
        }
        for name, case in cases.items():
            with self.subTest(case=name):
                pool = _dsa_pool_stub(layer_num=3, shard=case["shard"])
                drafts = tuple(
                    SimpleNamespace(index_k_with_scale_buffer=[object()])
                    for _ in range(case["drafts"])
                )
                params = SimpleNamespace(
                    page_size=64,
                    mtp_draft_device_pools=drafts,
                    token_to_kv_pool_allocator=None,
                    tp_cache_group=None,
                    attn_cp_cache_group=None,
                    attn_tp_cache_group=None,
                    pp_cache_group=None,
                )
                (legacy_group, _), legacy_ctrl = self._run(
                    _legacy_build_anchor_sidecar_stack,
                    pool=pool,
                    params=params,
                    full_layer_mapping=case["mapping"],
                    indexer_decl=dsa_indexer_pool_decl(pool),
                )
                stack, new_ctrl = self._run(
                    assemble_declared_stack,
                    pool=pool,
                    params=params,
                    full_layer_mapping=case["mapping"],
                    decls=pool.host_pool_decls(),
                )
                transfer_layer_num = len(case["mapping"]) + case["drafts"]
                self.assertEqual(
                    _entry_shape(stack.host_pool_group, transfer_layer_num),
                    _entry_shape(legacy_group, transfer_layer_num),
                )
                self.assertEqual(new_ctrl, legacy_ctrl)
                # target transfer layers exclude packed tail layers
                self.assertEqual(
                    new_ctrl[1]["transfer_layer_id_max"], len(case["mapping"])
                )
                self.assertEqual(
                    stack.sidecars, [dsa_indexer_pool_decl(pool).sidecar_spec()]
                )


class TestDeclaredPoolPlanning(CustomTestCase):
    """Sidecar indices resolve from one primary source in HostPoolGroup, so the
    planner must reject self-references and sidecar chains up front."""

    def _plan(self, decls):
        return hybrid_pool_assembler._plan_declared_pools(
            decls=decls,
            device_pool=object(),
            full_layer_mapping={0: 0},
            transfer_layer_id_max=1,
            packed_draft_device_pools=(),
        )

    def test_rejects_self_referencing_index_source(self):
        import msgspec

        kv, indexer = _dsa_pool_stub(layer_num=1).host_pool_decls()
        bad = msgspec.structs.replace(indexer, index_source=PoolName.INDEXER)
        with self.assertRaisesRegex(ValueError, "index_source"):
            self._plan((kv, bad))

    def test_rejects_self_referencing_layout_source(self):
        import msgspec

        kv, indexer = _dsa_pool_stub(layer_num=1).host_pool_decls()
        bad = msgspec.structs.replace(indexer, layout_source=PoolName.INDEXER)
        with self.assertRaisesRegex(ValueError, "layout_source"):
            self._plan((kv, bad))

    def test_accepts_dsa_declaration(self):
        plans = self._plan(_dsa_pool_stub(layer_num=1).host_pool_decls())
        self.assertEqual([p.decl.name for p in plans], [PoolName.KV, PoolName.INDEXER])


class TestHiRadixExtraPoolsFromDeclaration(CustomTestCase):
    """HiRadixCache's per-request transfers must come from the assembled
    sidecar specs, not from a model-type table beside them."""

    def test_extra_pools_follow_sidecar_specs(self):
        from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, SidecarPoolSpec
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        cache = object.__new__(HiRadixCache)
        cache.cache_controller = MagicMock(spec=HybridCacheController)
        cache.kv_cache = _dsa_pool_stub(layer_num=1)
        cache.sidecar_pool_specs = [
            SidecarPoolSpec(pool_name=PoolName.INDEXER, indices_from_pool=PoolName.KV)
        ]
        (transfer,) = cache._get_extra_pools()["extra_pools"]
        self.assertEqual(transfer.name, PoolName.INDEXER)
        self.assertEqual(transfer.indices_from_pool, PoolName.KV)
        self.assertEqual(transfer.hit_policy, PoolHitPolicy.ALL_PAGES)


class TestDeclaredPoolVerification(CustomTestCase):
    def _result_with(self, *names):
        group = SimpleNamespace(entry_map={n: object() for n in names})
        return StackBuildResult(
            host_pool_group=group, cache_controller=None, component_host_pools={}
        )

    def test_unmigrated_strategy_logs_missing_indexer(self):
        pool = _dsa_pool_stub(layer_num=2)
        with self.assertLogs(hybrid_pool_assembler.logger, level="ERROR") as logs:
            _verify_declared_pools(
                pool, self._result_with(PoolName.KV), _MambaStrategy()
            )
        self.assertIn("indexer", logs.output[0])

    def test_migrated_strategy_raises_on_missing_indexer(self):
        pool = _dsa_pool_stub(layer_num=2)
        with self.assertRaisesRegex(ValueError, "indexer"):
            _verify_declared_pools(pool, self._result_with(PoolName.KV), _DsaStrategy())

    def test_complete_stack_passes_silently(self):
        pool = _dsa_pool_stub(layer_num=2)
        _verify_declared_pools(
            pool, self._result_with(PoolName.KV, PoolName.INDEXER), _DsaStrategy()
        )


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


if __name__ == "__main__":
    unittest.main()
