"""Unit tests for hybrid HiCache pool assembly."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.hicache_storage import PoolName, SidecarPoolSpec
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    StackBuildResult,
    _DsaStrategy,
    _evict_mamba_for_device_alloc,
    _evict_swa_for_device_alloc,
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
from sglang.srt.mem_cache.pool_host import qsa as pool_host_qsa
from sglang.srt.mem_cache.pool_host.host_pool_decl import (
    HostPoolDecl,
    draft_sidecar_decls,
    kv_pool_decl,
    packable_draft_pools,
    plan_host_pools,
)
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
                        host_pool_decls=lambda: (),
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
                    controller = SimpleNamespace(transfer_layer_id_max=4)
                    built = (
                        SimpleNamespace(
                            host_pool_group=MagicMock(),
                            cache_controller=controller,
                            plans=(),
                            sidecars=[],
                        )
                        if strategy_cls is _MambaStrategy
                        else (MagicMock(), controller)
                    )
                    with patch.object(
                        hybrid_pool_assembler, builder_name, return_value=built
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
        full = SimpleNamespace(layer_num=0)
        full.host_pool_decls = lambda: (kv_pool_decl(full),)
        draft_kv_pool.full_kv_pool = full

        specs, entries = build_full_draft_pools(
            draft_kv_pool=draft_kv_pool,
            tree_cache=None,
        )

        self.assertEqual(specs, [])
        self.assertEqual(entries, [])

    def test_full_builder_sizes_sidecar_for_anchor_logical_space(self):
        draft_kv_pool = SimpleNamespace(layer_num=1, size=800)
        draft_kv_pool.host_pool_decls = lambda: (kv_pool_decl(draft_kv_pool),)
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
        draft_kv_pool.skip_topk_layers = [False]
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

        def fake_indexer_host(
            decl, anchor_host, *, allocator_type, packed_draft_device_pools=()
        ):
            self.assertIsInstance(decl, HostPoolDecl)
            self.assertIs(decl.device_pool, draft_kv_pool)
            self.assertIs(anchor_host, draft_host_pool)
            self.assertEqual(packed_draft_device_pools, ())
            seen["desc"] = decl
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
            patch.object(pool_host_dsa, "DSAIndexerPoolHost", fake_indexer_host),
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
    pool.skip_topk_layers = [False] * layer_num
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


def _kv_pool_stub(*, layer_num: int, size: int = 4096):
    """A plain full-attention KV pool: declares KV only."""
    pool = SimpleNamespace(
        layer_num=layer_num,
        size=size,
        start_layer=0,
        end_layer=layer_num - 1,
        store_dtype=torch.bfloat16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        kv_cache_dim=576,
        layer_shard_enabled=False,
    )
    pool.host_pool_decls = lambda: (kv_pool_decl(pool),)
    return pool


def _qsa_pool_stub(*, layer_num: int, size: int = 4096, ratio: int = 4):
    """QSATokenToKVPool shape without CUDA: KV on the full sub-pool, compressed
    keys on the hybrid pool."""
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    pool = object.__new__(QSATokenToKVPool)
    pool.full_kv_pool = _kv_pool_stub(layer_num=layer_num, size=size)
    pool.qsa_index_kv_heads = 1
    pool.qsa_index_head_dim = 128
    pool.qsa_compress_ratio = ratio
    pool.qsa_compressed_k_buffer_pool = [
        torch.zeros(
            (size + 64) // ratio, 1, 128, dtype=QSATokenToKVPool.index_state_dtype
        )
        for _ in range(layer_num)
    ]
    return pool


def _recording_qsa_mirror(seen: list):
    """Stand-in for QSAIndexerPoolHost: records constructor arguments, since the
    real page-row mirror pins host memory."""

    def build(*, decl, anchor_host, allocator_type, packed_draft_device_pools=()):
        seen.append(
            dict(
                decl=decl,
                anchor_host=anchor_host,
                packed_draft_device_pools=packed_draft_device_pools,
            )
        )
        return SimpleNamespace(
            layer_num=len(decl.device_pool.qsa_compressed_k_buffer_pool)
            + sum(
                len(p.qsa_compressed_k_buffer_pool) for p in packed_draft_device_pools
            ),
            can_use_write_back_jit=False,
        )

    return build


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


def _legacy_build_anchor_sidecar_stack(
    *,
    params,
    kv_pool,
    indexer_decl,
    full_layer_mapping,
    load_cache_event,
    storage_backend,
    use_mla,
    override_kv_cache_dim=None,
    prefetch_threshold=256,
    model_name=None,
    storage_backend_extra_config=None,
    enable_storage_metrics=False,
):
    """Pre-declaration DSA assembly (main before 2a), kept here only as the
    parity oracle for assemble_declared_stack."""
    transfer_layer_id_max = len(full_layer_mapping)
    mtp_draft_device_pools = tuple(
        pool for pool in params.mtp_draft_device_pools if pool.index_k_with_scale_buffer
    )
    kv_host_pool = hybrid_pool_assembler.build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=params.page_size,
        use_mla=use_mla,
        override_kv_cache_dim=override_kv_cache_dim,
        mtp_draft_device_pools=mtp_draft_device_pools,
    )
    sidecar_host_pool = pool_host_dsa.DSAIndexerPoolHost(
        indexer_decl,
        kv_host_pool,
        packed_draft_device_pools=mtp_draft_device_pools,
        allocator_type=hybrid_pool_assembler._get_allocator_type(),
    )
    if mtp_draft_device_pools:
        full_layer_mapping = hybrid_pool_assembler._with_mtp_layer_mapping(
            full_layer_mapping,
            transfer_layer_start=transfer_layer_id_max,
            target_device_layer_num=kv_pool.layer_num,
            draft_layer_num=len(mtp_draft_device_pools),
        )
    entries = [
        hybrid_pool_assembler.build_pool_entry(
            name=PoolName.KV,
            host_pool=kv_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_id_max=transfer_layer_id_max + len(mtp_draft_device_pools),
            is_anchor=True,
            packed_draft_device_pools=mtp_draft_device_pools,
        ),
        hybrid_pool_assembler.build_pool_entry(
            name=indexer_decl.name,
            host_pool=sidecar_host_pool,
            device_pool=kv_pool,
            layer_mapping=full_layer_mapping,
            transfer_layer_id_max=transfer_layer_id_max + len(mtp_draft_device_pools),
            packed_draft_device_pools=mtp_draft_device_pools,
        ),
    ]
    host_pool_group = hybrid_pool_assembler.HostPoolGroup(entries)
    cache_controller = hybrid_pool_assembler.HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        params.page_size,
        params.tp_cache_group,
        load_cache_event=load_cache_event,
        attn_cp_group=params.attn_cp_cache_group,
        attn_tp_group=params.attn_tp_cache_group,
        pp_group=params.pp_cache_group,
        write_policy=hybrid_pool_assembler.get_memory().hicache_write_policy,
        io_backend=hybrid_pool_assembler.get_memory().hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_id_max=transfer_layer_id_max,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=hybrid_pool_assembler.get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller


def _legacy_build_kv_only_stack(
    *,
    params,
    kv_pool,
    full_layer_mapping,
    load_cache_event,
    storage_backend,
    use_mla,
    override_kv_cache_dim=None,
    prefetch_threshold=256,
    model_name=None,
    storage_backend_extra_config=None,
    enable_storage_metrics=False,
):
    """Pre-declaration plain KV assembly (_PlainKvStrategy before the
    consolidation), kept here only as the parity oracle."""
    transfer_layer_id_max = len(full_layer_mapping)
    mtp_draft_device_pools = params.mtp_draft_device_pools
    kv_host_pool = hybrid_pool_assembler.build_kv_host_pool(
        kv_pool=kv_pool,
        page_size=params.page_size,
        use_mla=use_mla,
        override_kv_cache_dim=override_kv_cache_dim,
        mtp_draft_device_pools=mtp_draft_device_pools,
    )
    if mtp_draft_device_pools:
        full_layer_mapping = hybrid_pool_assembler._with_mtp_layer_mapping(
            full_layer_mapping,
            transfer_layer_start=transfer_layer_id_max,
            target_device_layer_num=kv_pool.layer_num,
            draft_layer_num=len(mtp_draft_device_pools),
        )
    host_pool_group = hybrid_pool_assembler.HostPoolGroup(
        [
            hybrid_pool_assembler.build_pool_entry(
                name=PoolName.KV,
                host_pool=kv_host_pool,
                device_pool=kv_pool,
                layer_mapping=full_layer_mapping,
                transfer_layer_id_max=transfer_layer_id_max
                + len(mtp_draft_device_pools),
                is_anchor=True,
                packed_draft_device_pools=mtp_draft_device_pools,
            )
        ]
    )
    cache_controller = hybrid_pool_assembler.HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_pool_group,
        params.page_size,
        params.tp_cache_group,
        load_cache_event=load_cache_event,
        attn_cp_group=params.attn_cp_cache_group,
        attn_tp_group=params.attn_tp_cache_group,
        pp_group=params.pp_cache_group,
        write_policy=hybrid_pool_assembler.get_memory().hicache_write_policy,
        io_backend=hybrid_pool_assembler.get_memory().hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_id_max=transfer_layer_id_max,
        enable_storage_metrics=enable_storage_metrics,
        host_memory_mode=hybrid_pool_assembler.get_memory().hicache_host_memory_mode,
    )
    return host_pool_group, cache_controller


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

        def dummy_indexer_host(
            decl, anchor_host, *, allocator_type, packed_draft_device_pools=()
        ):
            return real_indexer_host(
                decl,
                anchor_host,
                packed_draft_device_pools=packed_draft_device_pools,
                allocator_type=allocator_type,
                pin_memory=False,
                is_dummy=True,
            )

        controller = MagicMock()
        with (
            patch.object(hybrid_pool_assembler, "build_kv_host_pool", dummy_kv_host),
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
                    _dsa_pool_stub(layer_num=1) for _ in range(case["drafts"])
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
                    kv_pool=pool,
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

    def test_matches_legacy_kv_only_assembly(self):
        # A plain MLA pool declares KV only; the declared assembler must build
        # the same anchor-only group the removed build_kv_only_stack built.
        for name, drafts in (("no_draft", 0), ("packed_draft", 1)):
            with self.subTest(case=name):
                pool = _kv_pool_stub(layer_num=3)
                params = SimpleNamespace(
                    page_size=64,
                    mtp_draft_device_pools=tuple(
                        _kv_pool_stub(layer_num=1) for _ in range(drafts)
                    ),
                    token_to_kv_pool_allocator=None,
                    tp_cache_group=None,
                    attn_cp_cache_group=None,
                    attn_tp_cache_group=None,
                    pp_cache_group=None,
                )
                mapping = {0: 0, 1: 1, 2: 2}
                (legacy_group, _), legacy_ctrl = self._run(
                    _legacy_build_kv_only_stack,
                    pool=pool,
                    params=params,
                    full_layer_mapping=mapping,
                    kv_pool=pool,
                )
                stack, new_ctrl = self._run(
                    assemble_declared_stack,
                    pool=pool,
                    params=params,
                    full_layer_mapping=mapping,
                    decls=pool.host_pool_decls(),
                )
                self.assertEqual(
                    _entry_shape(stack.host_pool_group, 3 + drafts),
                    _entry_shape(legacy_group, 3 + drafts),
                )
                self.assertEqual(new_ctrl, legacy_ctrl)
                self.assertEqual(stack.sidecars, [])


class TestDraftSidecarDeclarations(CustomTestCase):
    """Separate drafts reuse the target's declarations under DRAFT_* names."""

    def test_dsa_draft_maps_to_draft_and_draft_indexer(self):
        decls = draft_sidecar_decls(_dsa_pool_stub(layer_num=1).host_pool_decls())
        self.assertEqual(
            [(d.name, d.index_source, d.layout_source) for d in decls],
            [
                (PoolName.DRAFT, PoolName.KV, None),
                (PoolName.DRAFT_INDEXER, PoolName.KV, PoolName.DRAFT),
            ],
        )

    def test_sidecar_group_plans_with_external_index_primary(self):
        pool = _dsa_pool_stub(layer_num=2)
        plans = plan_host_pools(
            decls=draft_sidecar_decls(pool.host_pool_decls()),
            full_layer_mapping={0: 0, 1: 1},
            transfer_layer_id_max=2,
            index_primary=PoolName.KV,
        )
        self.assertEqual(
            [p.decl.name for p in plans], [PoolName.DRAFT, PoolName.DRAFT_INDEXER]
        )
        with self.assertRaisesRegex(ValueError, "every index from the target"):
            plan_host_pools(
                decls=pool.host_pool_decls(),
                full_layer_mapping={0: 0, 1: 1},
                transfer_layer_id_max=2,
                index_primary=PoolName.KV,
            )


class TestPackedDraftPairing(CustomTestCase):
    """Packing appends draft layers to the target mirrors, so a draft must
    declare every target pool with an identical per-layer layout."""

    def test_draft_without_indexer_is_not_packed(self):
        # Pre-declaration code filtered on ``pool.index_k_with_scale_buffer``.
        target = _dsa_pool_stub(layer_num=2)
        no_index = _dsa_pool_stub(layer_num=1)
        no_index.index_key_cache = SimpleNamespace(buffer=[])
        full = _dsa_pool_stub(layer_num=1)
        self.assertEqual(
            packable_draft_pools(target.host_pool_decls(), (no_index, full)), (full,)
        )

    def test_layout_mismatch_is_rejected(self):
        target = _dsa_pool_stub(layer_num=2)
        wide = _dsa_pool_stub(layer_num=1)
        wide.index_head_dim = 256
        with self.assertRaisesRegex(ValueError, "layout"):
            packable_draft_pools(target.host_pool_decls(), (wide,))


def _legacy_build_full_draft_pools(
    *,
    draft_kv_pool,
    tree_cache,
):
    """Pre-2b separate-draft assembly (isinstance DSA branch), kept only as the
    parity oracle for the declaration-driven build_full_draft_pools."""

    pool = draft_kv_pool
    if isinstance(pool, HybridLinearKVPool):
        # Hybrid draft runners keep their sole attention layer in this sub-pool.
        pool = pool.full_kv_pool
    if pool.layer_num == 0:
        return [], []

    controller = tree_cache.cache_controller
    host_pool_group = controller.mem_pool_host

    # Note(kpham-sgl): DCP x DSpark draft KV is replicated and spans the virtual
    # loc space, so match the target host's logical_size instead of physical size.
    draft_host_pool = hybrid_pool_assembler._build_mha_mla_host_pool(
        pool=pool,
        host_to_device_ratio=host_pool_group.logical_size / pool.size,
        page_size=controller.page_size,
        layout=hybrid_pool_assembler.get_memory().hicache_mem_layout,
        allocator_type=hybrid_pool_assembler._get_allocator_type(),
        pool_label="draft",
    )
    draft_layer_mapping = {i: i for i in range(pool.layer_num)}

    specs = [
        SidecarPoolSpec(
            pool_name=PoolName.DRAFT,
            indices_from_pool=PoolName.KV,
        )
    ]
    entries = [
        hybrid_pool_assembler.build_pool_entry(
            name=PoolName.DRAFT,
            host_pool=draft_host_pool,
            device_pool=pool,
            layer_mapping=draft_layer_mapping,
            transfer_layer_id_max=draft_host_pool.layer_num,
        )
    ]

    if isinstance(pool, DSATokenToKVPool) and pool.index_k_with_scale_buffer:
        # Separate draft indexer: its own host mirror laid out on the draft KV
        # mirror, but transfer indices still follow the target KV anchor.
        indexer_decl = pool_host_dsa.dsa_indexer_pool_decl(
            pool, name=PoolName.DRAFT_INDEXER
        )
        indexer_host_pool = pool_host_dsa.DSAIndexerPoolHost(
            decl=indexer_decl,
            anchor_host=draft_host_pool,
            allocator_type=hybrid_pool_assembler._get_allocator_type(),
        )
        specs.append(indexer_decl.sidecar_spec())
        entries.append(
            hybrid_pool_assembler.build_pool_entry(
                name=indexer_decl.name,
                host_pool=indexer_host_pool,
                device_pool=pool,
                layer_mapping=draft_layer_mapping,
                transfer_layer_id_max=indexer_host_pool.layer_num,
            )
        )

    return specs, entries


class TestSeparateDraftParity(CustomTestCase):
    """Declaration-driven build_full_draft_pools must match the isinstance-based
    assembly it replaces, including skipping the indexer for a draft pool that
    owns no index buffers."""

    def _run(self, builder, pool):
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        real_indexer_host = pool_host_dsa.DSAIndexerPoolHost

        def dummy_draft_host(
            *, pool, host_to_device_ratio, page_size, layout, allocator_type, pool_label
        ):
            return MLATokenToKVPoolHost(
                pool,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=page_size,
                layout=layout,
                pin_memory=False,
                is_dummy=True,
                override_kv_cache_dim=pool.kv_cache_dim,
                pool_label=pool_label,
            )

        def dummy_indexer_host(
            decl, anchor_host, *, allocator_type, packed_draft_device_pools=()
        ):
            return real_indexer_host(
                decl=decl,
                anchor_host=anchor_host,
                packed_draft_device_pools=packed_draft_device_pools,
                allocator_type=allocator_type,
                pin_memory=False,
                is_dummy=True,
            )

        tree_cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(size=8192, logical_size=8192),
                page_size=64,
            )
        )
        with (
            patch.object(
                hybrid_pool_assembler, "_build_mha_mla_host_pool", dummy_draft_host
            ),
            patch.object(pool_host_dsa, "DSAIndexerPoolHost", dummy_indexer_host),
            patch.object(
                hybrid_pool_assembler, "_get_allocator_type", return_value="default"
            ),
            patch.object(
                hybrid_pool_assembler,
                "get_memory",
                return_value=SimpleNamespace(hicache_mem_layout="page_first"),
            ),
        ):
            return builder(draft_kv_pool=pool, tree_cache=tree_cache)

    def test_matches_legacy_separate_draft_assembly(self):
        for name, with_index in (
            ("dsa_with_indexer", True),
            ("dsa_no_index_buffers", False),
        ):
            with self.subTest(case=name):
                pool = _dsa_pool_stub(layer_num=2, size=4096)
                if not with_index:
                    pool.index_key_cache = SimpleNamespace(buffer=[])
                legacy_specs, legacy_entries = self._run(
                    _legacy_build_full_draft_pools, pool
                )
                specs, entries = self._run(build_full_draft_pools, pool)
                self.assertEqual(specs, legacy_specs)
                self.assertEqual(
                    _entry_shape(SimpleNamespace(entries=entries), 2),
                    _entry_shape(SimpleNamespace(entries=legacy_entries), 2),
                )
                self.assertEqual(len(entries), 2 if with_index else 1)


class TestHybridMambaDeclaredIndexer(CustomTestCase):
    """hybrid Mamba + DSA reuses the target's indexer declaration: the KV/Mamba
    stack gains an INDEXER entry whose mirror and layer mapping cover only the
    DSA layers that own index buffers."""

    def test_stack_declares_indexer_and_skips_empty_layers(self):
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        kv_pool = _dsa_pool_stub(layer_num=3)
        kv_pool.skip_topk_layers = [False, True, False]  # device layer 1: no buffer
        mamba_pool = SimpleNamespace(layer_num=2, size=8)
        # transfer layers 0,2,4 are DSA (device 0,1,2); 1,3 are Mamba
        full_mapping = {0: 0, 2: 1, 4: 2}
        mamba_mapping = {1: 0, 3: 1}
        params = SimpleNamespace(
            page_size=64,
            mtp_draft_device_pools=(),
            token_to_kv_pool_allocator=None,
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
            req_to_token_pool=SimpleNamespace(
                mamba_allocator=SimpleNamespace(
                    alloc=lambda n: None, free=lambda x: None
                )
            ),
        )
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
                override_kv_cache_dim=None,
                mtp_draft_device_pools=kwargs["mtp_draft_device_pools"],
            )

        def dummy_indexer_host(
            decl, anchor_host, *, allocator_type, packed_draft_device_pools=()
        ):
            return real_indexer_host(
                decl=decl,
                anchor_host=anchor_host,
                packed_draft_device_pools=packed_draft_device_pools,
                allocator_type=allocator_type,
                pin_memory=False,
                is_dummy=True,
            )

        with (
            patch.object(hybrid_pool_assembler, "build_kv_host_pool", dummy_kv_host),
            patch.object(pool_host_dsa, "DSAIndexerPoolHost", dummy_indexer_host),
            patch.object(
                hybrid_pool_assembler,
                "MambaPoolHost",
                return_value=SimpleNamespace(layer_num=2, can_use_write_back_jit=False),
            ),
            patch.object(hybrid_pool_assembler, "HybridCacheController", MagicMock()),
            patch.object(
                hybrid_pool_assembler, "_get_allocator_type", return_value="default"
            ),
            patch.object(
                hybrid_pool_assembler,
                "get_memory",
                return_value=SimpleNamespace(
                    hicache_size=0,
                    hicache_ratio=2,
                    hicache_mem_layout="page_first",
                    hicache_write_policy="write_through",
                    hicache_io_backend="kernel",
                    hicache_host_memory_mode="cache",
                ),
            ),
        ):
            stack = hybrid_pool_assembler.build_hybrid_mamba_stack(
                params=params,
                decls=kv_pool.host_pool_decls(),
                mamba_pool=mamba_pool,
                full_layer_mapping=full_mapping,
                mamba_layer_mapping=mamba_mapping,
                load_cache_event=None,
                storage_backend=None,
                use_mla=True,
            )

        names = [e.name for e in stack.host_pool_group.entries]
        self.assertEqual(names, [PoolName.KV, PoolName.INDEXER, PoolName.MAMBA])
        indexer = stack.host_pool_group.entry_map[PoolName.INDEXER]
        kv = stack.host_pool_group.entry_map[PoolName.KV]
        # KV still maps every DSA transfer layer; the indexer drops device layer 1.
        self.assertEqual([kv.layer_mapper(t) for t in range(5)], [0, None, 1, None, 2])
        self.assertEqual(
            [indexer.layer_mapper(t) for t in range(5)], [0, None, None, None, 2]
        )
        self.assertEqual(indexer.host_pool.layer_num, 2)
        self.assertEqual(indexer.host_pool._host_layer_index(2), 1)
        self.assertEqual(stack.sidecars, [kv_pool.host_pool_decls()[1].sidecar_spec()])


class TestHybridMambaDeclaredQsaIndexer(CustomTestCase):
    """QSA joins the KV/Mamba stack through its declaration alone: the KV
    mirror is built from the full sub-pool, the compressed-key mirror from the
    hybrid pool, and a packed MTP draft contributes its own owner per pool."""

    def _params(self, drafts):
        return SimpleNamespace(
            page_size=64,
            mtp_draft_device_pools=drafts,
            token_to_kv_pool_allocator=None,
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
            req_to_token_pool=SimpleNamespace(
                mamba_allocator=SimpleNamespace(
                    alloc=lambda n: None, free=lambda x: None
                )
            ),
        )

    def _build(self, kv_pool, *, drafts=(), full_mapping, mamba_mapping):
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        seen = []

        def dummy_kv_host(**kwargs):
            return MLATokenToKVPoolHost(
                kwargs["kv_pool"],
                host_to_device_ratio=2,
                host_size=0,
                page_size=kwargs["page_size"],
                layout="page_first",
                pin_memory=False,
                is_dummy=True,
                mtp_draft_device_pools=kwargs["mtp_draft_device_pools"],
            )

        with (
            patch.object(hybrid_pool_assembler, "build_kv_host_pool", dummy_kv_host),
            patch.object(
                pool_host_qsa, "QSAIndexerPoolHost", _recording_qsa_mirror(seen)
            ),
            patch.object(
                hybrid_pool_assembler,
                "MambaPoolHost",
                return_value=SimpleNamespace(layer_num=2, can_use_write_back_jit=False),
            ),
            patch.object(hybrid_pool_assembler, "HybridCacheController", MagicMock()),
            patch.object(
                hybrid_pool_assembler, "_get_allocator_type", return_value="default"
            ),
            patch.object(
                hybrid_pool_assembler,
                "get_memory",
                return_value=SimpleNamespace(
                    hicache_size=0,
                    hicache_ratio=2,
                    hicache_mem_layout="page_first",
                    hicache_write_policy="write_through",
                    hicache_io_backend="kernel",
                    hicache_host_memory_mode="cache",
                ),
            ),
        ):
            stack = hybrid_pool_assembler.build_hybrid_mamba_stack(
                params=self._params(drafts),
                decls=kv_pool.host_pool_decls(),
                mamba_pool=SimpleNamespace(layer_num=2, size=8),
                full_layer_mapping=full_mapping,
                mamba_layer_mapping=mamba_mapping,
                load_cache_event=None,
                storage_backend=None,
                use_mla=False,
            )
        return stack, seen

    def test_kv_from_sub_pool_and_compressed_keys_from_hybrid(self):
        pool = _qsa_pool_stub(layer_num=2)
        stack, seen = self._build(
            pool, full_mapping={1: 0, 3: 1}, mamba_mapping={0: 0, 2: 1}
        )

        entries = stack.host_pool_group.entries
        self.assertEqual(
            [e.name for e in entries], [PoolName.KV, PoolName.INDEXER, PoolName.MAMBA]
        )
        self.assertIs(entries[0].device_pool, pool.full_kv_pool)
        self.assertIs(entries[1].device_pool, pool)
        (build,) = seen
        self.assertIs(build["decl"].device_pool, pool)
        self.assertIs(build["anchor_host"], entries[0].host_pool)
        self.assertEqual(
            [entries[1].layer_mapper(t) for t in range(4)], [None, 0, None, 1]
        )
        self.assertEqual(stack.sidecars, [pool.host_pool_decls()[1].sidecar_spec()])

    def test_packed_draft_owner_follows_each_declaration(self):
        pool = _qsa_pool_stub(layer_num=2)
        draft = _qsa_pool_stub(layer_num=1)
        stack, seen = self._build(
            pool, drafts=(draft,), full_mapping={1: 0, 3: 1}, mamba_mapping={0: 0, 2: 1}
        )

        kv, indexer, _ = stack.host_pool_group.entries
        # The KV mirror packs the draft's full sub-pool; the compressed-key
        # mirror packs the draft hybrid pool that owns its keys.
        self.assertEqual(kv.packed_draft_device_pools, (draft.full_kv_pool,))
        self.assertEqual(indexer.packed_draft_device_pools, (draft,))
        self.assertEqual(seen[0]["packed_draft_device_pools"], (draft,))
        self.assertEqual(indexer.host_pool.layer_num, 3)
        # packed tail: transfer layer 4 -> device layer 2 on both mirrors
        self.assertEqual((kv.layer_mapper(4), indexer.layer_mapper(4)), (2, 2))

    def test_separate_draft_declares_draft_indexer_on_the_hybrid(self):
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        draft = _qsa_pool_stub(layer_num=1, size=1024)
        seen = []

        def dummy_draft_host(*, pool, host_to_device_ratio, page_size, layout, **_):
            return MLATokenToKVPoolHost(
                pool,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=page_size,
                layout=layout,
                pin_memory=False,
                is_dummy=True,
            )

        tree_cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(size=8192, logical_size=8192),
                page_size=64,
            )
        )
        with (
            patch.object(
                hybrid_pool_assembler, "_build_mha_mla_host_pool", dummy_draft_host
            ),
            patch.object(
                pool_host_qsa, "QSAIndexerPoolHost", _recording_qsa_mirror(seen)
            ),
            patch.object(
                hybrid_pool_assembler, "_get_allocator_type", return_value="default"
            ),
            patch.object(
                hybrid_pool_assembler,
                "get_memory",
                return_value=SimpleNamespace(hicache_mem_layout="page_first"),
            ),
        ):
            specs, entries = build_full_draft_pools(
                draft_kv_pool=draft, tree_cache=tree_cache
            )

        self.assertEqual(
            [(s.pool_name, s.indices_from_pool) for s in specs],
            [(PoolName.DRAFT, PoolName.KV), (PoolName.DRAFT_INDEXER, PoolName.KV)],
        )
        self.assertEqual(
            [(e.name, e.device_pool) for e in entries],
            [(PoolName.DRAFT, draft.full_kv_pool), (PoolName.DRAFT_INDEXER, draft)],
        )
        self.assertIs(seen[0]["anchor_host"], entries[0].host_pool)


class TestDeclaredPoolPlanning(CustomTestCase):
    """Sidecar indices resolve from one primary source in HostPoolGroup, so the
    planner must reject self-references and sidecar chains up front."""

    def _plan(self, decls):
        return plan_host_pools(
            decls=decls,
            full_layer_mapping={0: 0},
            transfer_layer_id_max=1,
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

    def test_rejects_primary_that_is_not_kv(self):
        import msgspec

        kv, indexer = _dsa_pool_stub(layer_num=1).host_pool_decls()
        swa_primary = msgspec.structs.replace(kv, name=PoolName.SWA)
        follower = msgspec.structs.replace(
            indexer, index_source=PoolName.SWA, layout_source=PoolName.SWA
        )
        with self.assertRaisesRegex(ValueError, "primary KV pool"):
            self._plan((swa_primary, follower))

    def test_accepts_dsa_declaration(self):
        plans = self._plan(_dsa_pool_stub(layer_num=1).host_pool_decls())
        self.assertEqual([p.decl.name for p in plans], [PoolName.KV, PoolName.INDEXER])


class TestHiRadixExtraPoolsFromDeclaration(CustomTestCase):
    """HiRadixCache's per-request transfers must come from the assembled
    sidecar specs, not from a model-type table beside them."""

    def _cache(self, specs):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        cache = object.__new__(HiRadixCache)
        cache.cache_controller = MagicMock(spec=HybridCacheController)
        cache.kv_cache = _dsa_pool_stub(layer_num=1)
        cache.sidecar_pool_specs = specs
        return cache

    def test_extra_pools_follow_sidecar_specs(self):
        from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, SidecarPoolSpec

        specs = [
            SidecarPoolSpec(pool_name=PoolName.INDEXER, indices_from_pool=PoolName.KV),
            SidecarPoolSpec(
                pool_name=PoolName.DRAFT_INDEXER,
                indices_from_pool=PoolName.KV,
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]
        transfers = self._cache(specs)._get_extra_pools()["extra_pools"]
        self.assertEqual(
            [(t.name, t.indices_from_pool, t.hit_policy) for t in transfers],
            [(s.pool_name, s.indices_from_pool, s.hit_policy) for s in specs],
        )

    def test_dsa_without_declared_sidecars_transfers_nothing_extra(self):
        # The pre-declaration table returned INDEXER for any DSATokenToKVPool.
        self.assertEqual(self._cache([])._get_extra_pools(), {})


class TestDeclaredPoolVerification(CustomTestCase):
    def _result_with(self, *names):
        group = SimpleNamespace(entry_map={n: object() for n in names})
        return StackBuildResult(
            host_pool_group=group, cache_controller=None, component_host_pools={}
        )

    def test_unmigrated_strategy_logs_missing_indexer(self):
        pool = _dsa_pool_stub(layer_num=2)
        with self.assertLogs(hybrid_pool_assembler.logger, level="ERROR") as logs:
            _verify_declared_pools(pool, self._result_with(PoolName.KV), _SwaStrategy())
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

    def test_hybrid_pool_declares_through_its_sub_pool(self):
        # HybridLinearKVPool once exposed host_pool_decls as a property, so the
        # polymorphic call every strategy makes raised on hybrid models.
        pool = object.__new__(HybridLinearKVPool)
        pool.full_kv_pool = _dsa_pool_stub(layer_num=2)
        self.assertEqual(
            [(d.name, d.device_pool) for d in pool.host_pool_decls()],
            [(PoolName.KV, pool.full_kv_pool), (PoolName.INDEXER, pool.full_kv_pool)],
        )

    def test_qsa_stack_without_compressed_keys_is_rejected(self):
        # The KV + MAMBA stack that restored QSA models before the declaration
        # existed: valid KV, stale block selection after a host hit.
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _MambaStrategy,
        )

        pool = _qsa_pool_stub(layer_num=2)
        with self.assertRaisesRegex(ValueError, "indexer"):
            _verify_declared_pools(
                pool, self._result_with(PoolName.KV, PoolName.MAMBA), _MambaStrategy()
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
