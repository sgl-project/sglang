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
    _split_hicache_size,
    _SwaStrategy,
    _verify_declared_states,
    assemble_declared_stack,
    build_full_draft_pools,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool
from sglang.srt.mem_cache.pool_host import dsa as pool_host_dsa
from sglang.srt.mem_cache.pool_host.state_spec import HostStateDecl
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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
                        return_value=(MagicMock(), object()),
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
                    self.assertEqual(result.transfer_layer_num, 4)
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
            self.assertIsInstance(desc, HostStateDecl)
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


def _dsa_pool_stub(*, layer_num: int, size: int = 4096):
    pool = object.__new__(DSATokenToKVPool)
    pool.layer_num = layer_num
    pool.size = size
    pool.start_layer = 0
    pool.end_layer = layer_num - 1
    pool.layer_shard_enabled = False
    pool.store_dtype = torch.bfloat16
    pool.kv_cache_dim = 576
    pool.index_head_dim = 128
    pool.index_key_cache = SimpleNamespace(buffer=[object()] * layer_num)
    return pool


def _fake_mirror(layer_num: int):
    return SimpleNamespace(
        layer_num=layer_num,
        layout="page_first",
        page_size=64,
        device="cpu",
        size=8192,
        logical_size=8192,
        page_num=128,
        mtp_draft_device_pools=(),
        can_use_write_back_jit=False,
    )


def _entry_shape(group, transfer_layer_num):
    """Everything the controller reads from a HostPoolGroup, in comparable form."""
    return [
        (
            entry.name,
            entry.is_primary_index_anchor,
            id(entry.device_pool),
            tuple(id(p) for p in entry.packed_draft_device_pools),
            tuple(entry.layer_mapper(i) for i in range(-1, transfer_layer_num + 2)),
        )
        for entry in group.entries
    ]


class TestDeclaredStackParity(CustomTestCase):
    """assemble_declared_stack must produce the same entries, layer mapping and
    sidecars as the pre-declaration DSA assembly it replaces, with and without
    packed MTP drafts."""

    def _run(self, builder, *, pool, params, **kw):
        anchors = []

        def fake_kv_host(**kwargs):
            anchors.append(kwargs)
            return _fake_mirror(pool.layer_num + len(params.mtp_draft_device_pools))

        def fake_indexer_host(decl, device_pool, anchor_host, *, allocator_type):
            return _fake_mirror(anchor_host.layer_num)

        with (
            patch.object(hybrid_pool_assembler, "build_kv_host_pool", fake_kv_host),
            patch.object(
                hybrid_pool_assembler, "DSAIndexerPoolHost", fake_indexer_host
            ),
            patch.object(pool_host_dsa, "DSAIndexerPoolHost", fake_indexer_host),
            patch.object(hybrid_pool_assembler, "HybridCacheController", MagicMock()),
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
                full_layer_mapping={i: i for i in range(pool.layer_num)},
                load_cache_event=None,
                storage_backend=None,
                use_mla=True,
                override_kv_cache_dim=pool.kv_cache_dim,
                **kw,
            )
        return out, anchors

    def test_matches_legacy_assembly(self):
        for draft_layers in (0, 1):
            with self.subTest(draft_layers=draft_layers):
                pool = _dsa_pool_stub(layer_num=3)
                drafts = tuple(
                    SimpleNamespace(index_k_with_scale_buffer=[object()])
                    for _ in range(draft_layers)
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
                from sglang.srt.mem_cache.pool_host.dsa import dsa_indexer_state_decl

                (legacy_group, _), legacy_anchor = self._run(
                    _legacy_build_anchor_sidecar_stack,
                    pool=pool,
                    params=params,
                    indexer_decl=dsa_indexer_state_decl(pool),
                )
                stack, new_anchor = self._run(
                    assemble_declared_stack,
                    pool=pool,
                    params=params,
                    decls=pool.host_states(),
                )
                transfer_layer_num = pool.layer_num + draft_layers
                self.assertEqual(legacy_anchor, new_anchor)
                self.assertEqual(
                    _entry_shape(stack.host_pool_group, transfer_layer_num),
                    _entry_shape(legacy_group, transfer_layer_num),
                )
                self.assertEqual(
                    stack.sidecars, [dsa_indexer_state_decl(pool).sidecar_spec()]
                )


class TestDeclaredStateVerification(CustomTestCase):
    def _result_with(self, *names):
        group = SimpleNamespace(entry_map={n: object() for n in names})
        return StackBuildResult(
            host_pool_group=group, cache_controller=None, component_host_pools={}
        )

    def test_unmigrated_strategy_logs_missing_indexer(self):
        pool = _dsa_pool_stub(layer_num=2)
        with self.assertLogs(hybrid_pool_assembler.logger, level="ERROR") as logs:
            _verify_declared_states(
                pool, self._result_with(PoolName.KV), _MambaStrategy()
            )
        self.assertIn("indexer", logs.output[0])

    def test_migrated_strategy_raises_on_missing_indexer(self):
        pool = _dsa_pool_stub(layer_num=2)
        with self.assertRaisesRegex(ValueError, "indexer"):
            _verify_declared_states(
                pool, self._result_with(PoolName.KV), _DsaStrategy()
            )

    def test_complete_stack_passes_silently(self):
        pool = _dsa_pool_stub(layer_num=2)
        _verify_declared_states(
            pool, self._result_with(PoolName.KV, PoolName.INDEXER), _DsaStrategy()
        )


if __name__ == "__main__":
    unittest.main()
