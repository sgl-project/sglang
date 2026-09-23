"""Unit tests for hybrid HiCache pool assembly."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    StackBuildResult,
    _check_declared_pools_present,
    _DsaStrategy,
    _evict_mamba_for_device_alloc,
    _evict_swa_for_device_alloc,
    _MambaStrategy,
    _MambaSwaStrategy,
    _require_single_row_dsv4_swa_pages,
    _split_hicache_size,
    _SwaStrategy,
    assemble_host_pools_from_decls,
    build_full_draft_pools,
    build_hybrid_swa_group,
    prepare_host_pool_configs,
    validate_packed_draft_pools,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool
from sglang.srt.mem_cache.pool_host import dsa as pool_host_dsa
from sglang.srt.mem_cache.pool_host import qsa as pool_host_qsa
from sglang.srt.mem_cache.pool_host.host_pool_decl import (
    HostPoolDecl,
    make_draft_sidecar_decls,
    make_kv_pool_decl,
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
                            configs=(),
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
        full.host_pool_decls = lambda: (make_kv_pool_decl(full),)
        draft_kv_pool.full_kv_pool = full

        specs, entries = build_full_draft_pools(
            draft_kv_pool=draft_kv_pool,
            tree_cache=None,
        )

        self.assertEqual(specs, [])
        self.assertEqual(entries, [])

    def test_full_builder_sizes_sidecar_for_anchor_logical_space(self):
        draft_kv_pool = SimpleNamespace(layer_num=1, size=800)
        draft_kv_pool.host_pool_decls = lambda: (make_kv_pool_decl(draft_kv_pool),)
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
        DRAFT_INDEXER decl; a constructor change that skips this call site
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
            seen["decl"] = decl
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

        self.assertEqual(seen["decl"].pool_name, PoolName.DRAFT_INDEXER)
        self.assertEqual(seen["decl"].indices_from_pool, PoolName.KV)
        self.assertEqual(specs[1], seen["decl"].sidecar_spec())
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
    """A plain MLA KV pool without CUDA: declares KV only."""
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    pool = object.__new__(MLATokenToKVPool)
    pool.layer_num = layer_num
    pool.size = size
    pool.start_layer = 0
    pool.end_layer = layer_num - 1
    pool.store_dtype = torch.bfloat16
    pool.kv_lora_rank = 512
    pool.qk_rope_head_dim = 64
    pool.kv_cache_dim = 576
    pool.layer_shard_enabled = False
    return pool


def _qsa_pool_stub(*, layer_num: int, size: int = 4096, ratio: int = 4):
    """QSATokenToKVPool shape without CUDA: KV on the full sub-pool, compressed
    keys on the hybrid pool."""
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    pool = object.__new__(QSATokenToKVPool)
    pool.full_kv_pool = _kv_pool_stub(layer_num=layer_num, size=size)
    pool.full_kv_pool.head_num = 1
    pool.full_kv_pool.head_dim = 128
    pool.full_kv_pool.v_head_dim = 128
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


def _target_params(drafts=(), page_size=64):
    return SimpleNamespace(
        page_size=page_size,
        mtp_draft_device_pools=tuple(drafts),
        token_to_kv_pool_allocator=None,
        tp_cache_group=None,
        attn_cp_cache_group=None,
        attn_tp_cache_group=None,
        pp_cache_group=None,
    )


# Mirror geometry for the 4096-token stubs at page 64 and host ratio 2:
# 8256 host tokens (2 x 4096 plus one reserve page) = 129 pages; the MLA row is
# kv_cache_dim 576 x bf16 = 1152 B per layer; the DSA index row is 132 B per
# token, 8448 B per page, per layer.
_HOST_SIZE, _HOST_PAGES, _KV_ROW, _INDEX_ROW, _INDEX_PAGE = 8256, 129, 1152, 132, 8448


def _kv_shape(layers):
    return {
        "size": _HOST_SIZE,
        "page_num": _HOST_PAGES,
        "layer_num": layers,
        "size_per_token": _KV_ROW * layers,
        "layout": "page_first",
    }


def _indexer_shape(layers):
    return {
        **_kv_shape(layers),
        "size_per_token": _INDEX_ROW * layers,
        "indexer_page_stride_size": _INDEX_PAGE,
        "indexer_layout_dim": _INDEX_PAGE * layers,
    }


class TestDeclaredStackStructure(CustomTestCase):
    """Everything the controller reads from a declared target stack, pinned
    against the pre-declaration DSA and plain-KV assembly it replaced: entry
    order, anchor, device owners, layer mapper over every transfer layer,
    packed drafts, mirror geometry, controller arguments, sidecars."""

    def _run(self, *, pool, drafts, full_layer_mapping):
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
            stack = assemble_host_pools_from_decls(
                params=_target_params(drafts),
                decls=pool.host_pool_decls(),
                full_layer_mapping=dict(full_layer_mapping),
                load_cache_event=None,
                storage_backend=None,
                use_mla=True,
                override_kv_cache_dim=pool.kv_cache_dim,
            )
        (call,) = controller.call_args_list
        # target transfer layers exclude packed tail layers
        self.assertEqual(call.args[2], 64)
        self.assertEqual(call.kwargs["transfer_layer_id_max"], len(full_layer_mapping))
        return stack

    def test_dsa_target(self):
        pool = _dsa_pool_stub(layer_num=3)
        stack = self._run(pool=pool, drafts=(), full_layer_mapping={0: 0, 1: 1, 2: 2})
        mapper = (None, 0, 1, 2, None, None)
        self.assertEqual(
            _entry_shape(stack.host_pool_group, 3),
            [
                (PoolName.KV, True, id(pool), (), mapper, _kv_shape(3)),
                (PoolName.INDEXER, False, id(pool), (), mapper, _indexer_shape(3)),
            ],
        )
        self.assertEqual(
            stack.sidecars,
            [pool_host_dsa.make_dsa_indexer_pool_decl(pool).sidecar_spec()],
        )

    def test_dsa_target_with_packed_draft(self):
        pool = _dsa_pool_stub(layer_num=3)
        draft = _dsa_pool_stub(layer_num=1)
        stack = self._run(
            pool=pool, drafts=(draft,), full_layer_mapping={0: 0, 1: 1, 2: 2}
        )
        # transfer layer 3 is the draft's tail layer, device layer 3 = target_layer_num + depth
        mapper = (None, 0, 1, 2, 3, None, None)
        self.assertEqual(
            _entry_shape(stack.host_pool_group, 4),
            [
                (PoolName.KV, True, id(pool), (id(draft),), mapper, _kv_shape(4)),
                (
                    PoolName.INDEXER,
                    False,
                    id(pool),
                    (id(draft),),
                    mapper,
                    _indexer_shape(4),
                ),
            ],
        )

    def test_dsa_target_layer_sharded(self):
        # rank 0 of 2 owns local layers 0 and 1 of 3; the permuted stage mapping
        # is passed through untouched and the host_pools hold only the owned layers.
        pool = _dsa_pool_stub(layer_num=3, shard=(0, 2))
        stack = self._run(pool=pool, drafts=(), full_layer_mapping={0: 1, 1: 2, 2: 0})
        mapper = (None, 1, 2, 0, None, None)
        self.assertEqual(
            _entry_shape(stack.host_pool_group, 3),
            [
                (PoolName.KV, True, id(pool), (), mapper, _kv_shape(2)),
                (PoolName.INDEXER, False, id(pool), (), mapper, _indexer_shape(2)),
            ],
        )

    def test_plain_mla_target(self):
        # A pool that declares KV only builds the anchor-only group the removed
        # build_kv_only_stack built, packed draft included.
        for drafts in (0, 1):
            with self.subTest(drafts=drafts):
                pool = _kv_pool_stub(layer_num=3)
                ds = tuple(_kv_pool_stub(layer_num=1) for _ in range(drafts))
                stack = self._run(
                    pool=pool, drafts=ds, full_layer_mapping={0: 0, 1: 1, 2: 2}
                )
                mapper = (None, 0, 1, 2, *([3] if drafts else []), None, None)
                self.assertEqual(
                    _entry_shape(stack.host_pool_group, 3 + drafts),
                    [
                        (
                            PoolName.KV,
                            True,
                            id(pool),
                            tuple(id(d) for d in ds),
                            mapper,
                            _kv_shape(3 + drafts),
                        )
                    ],
                )
                self.assertEqual(stack.sidecars, [])


class TestDraftSidecarDeclarations(CustomTestCase):
    """Separate drafts reuse the target's declarations under DRAFT_* names."""

    def test_dsa_draft_maps_to_draft_and_draft_indexer(self):
        decls = make_draft_sidecar_decls(_dsa_pool_stub(layer_num=1).host_pool_decls())
        self.assertEqual(
            [(d.pool_name, d.indices_from_pool, d.layout_source) for d in decls],
            [
                (PoolName.DRAFT, PoolName.KV, None),
                (PoolName.DRAFT_INDEXER, PoolName.KV, PoolName.DRAFT),
            ],
        )

    def test_sidecar_group_plans_with_external_index_primary(self):
        pool = _dsa_pool_stub(layer_num=2)
        configs = prepare_host_pool_configs(
            decls=make_draft_sidecar_decls(pool.host_pool_decls()),
            full_layer_mapping={0: 0, 1: 1},
            transfer_layer_id_max=2,
            index_primary=PoolName.KV,
        )
        self.assertEqual(
            [c.decl.pool_name for c in configs],
            [PoolName.DRAFT, PoolName.DRAFT_INDEXER],
        )
        with self.assertRaisesRegex(ValueError, "every index from the target"):
            prepare_host_pool_configs(
                decls=pool.host_pool_decls(),
                full_layer_mapping={0: 0, 1: 1},
                transfer_layer_id_max=2,
                index_primary=PoolName.KV,
            )


class TestPackedDraftPairing(CustomTestCase):
    """Packing appends draft layers to the target host_pools, so a draft must
    declare every target pool with an identical per-layer layout."""

    def test_draft_without_indexer_is_rejected(self):
        # The pre-declaration filter on ``index_k_with_scale_buffer`` dropped
        # such a draft silently, leaving its state without any host mirror.
        target = _dsa_pool_stub(layer_num=2)
        no_index = _dsa_pool_stub(layer_num=1)
        no_index.index_key_cache = SimpleNamespace(buffer=[])
        with self.assertRaisesRegex(ValueError, "draft counterpart"):
            validate_packed_draft_pools(
                target_decls=target.host_pool_decls(), draft_pools=(no_index,)
            )

    def test_draft_with_empty_index_layers_is_rejected(self):
        # A 0-row placeholder layer would be packed as a null device pointer.
        target = _dsa_pool_stub(layer_num=2)
        shared = _dsa_pool_stub(layer_num=1)
        shared.skip_topk_layers = [True]
        shared.index_key_cache = SimpleNamespace(buffer=[object()])
        with self.assertRaisesRegex(ValueError, "draft counterpart"):
            validate_packed_draft_pools(
                target_decls=target.host_pool_decls(), draft_pools=(shared,)
            )
        partial = _dsa_pool_stub(layer_num=2)
        partial.skip_topk_layers = [False, True]
        with self.assertRaisesRegex(ValueError, "owns buffers on 1 of 2"):
            validate_packed_draft_pools(
                target_decls=target.host_pool_decls(), draft_pools=(partial,)
            )

    def test_storage_info_mismatch_is_rejected(self):
        target = _dsa_pool_stub(layer_num=2)
        wide = _dsa_pool_stub(layer_num=1)
        wide.index_head_dim = 256
        with self.assertRaisesRegex(ValueError, "storage"):
            validate_packed_draft_pools(
                target_decls=target.host_pool_decls(), draft_pools=(wide,)
            )

    def test_pool_with_only_shared_topk_layers_declares_kv_only(self):
        pool = _dsa_pool_stub(layer_num=2)
        pool.skip_topk_layers = [True, True]
        self.assertEqual([d.pool_name for d in pool.host_pool_decls()], [PoolName.KV])


class TestKvHostPoolRow(CustomTestCase):
    """The KV mirror takes its row width from the MLA device pool (an fp8 DSA
    store is wider than kv_lora_rank + qk_rope_head_dim) and refuses packed
    drafts whose KV rows differ from the target's."""

    def _build(self, pool, drafts=()):
        seen = {}

        def fake_host(kv_pool, ratio, size, page_size, layout, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(layer_num=kv_pool.layer_num)

        with (
            patch.object(hybrid_pool_assembler, "MLATokenToKVPoolHost", fake_host),
            patch.object(
                hybrid_pool_assembler,
                "get_parallel",
                return_value=SimpleNamespace(dcp_enabled=False),
            ),
            patch.object(
                hybrid_pool_assembler,
                "get_memory",
                return_value=SimpleNamespace(
                    hicache_ratio=2, hicache_size=0, hicache_mem_layout="page_first"
                ),
            ),
            patch.object(
                hybrid_pool_assembler, "_get_allocator_type", return_value="default"
            ),
        ):
            hybrid_pool_assembler.build_kv_host_pool(
                kv_pool=pool,
                page_size=64,
                use_mla=True,
                mtp_draft_device_pools=drafts,
            )
        return seen

    def test_mla_row_width_comes_from_the_device_pool(self):
        pool = _dsa_pool_stub(layer_num=2)
        pool.kv_cache_dim = 656  # fp8 DSA store: 576 + fp32 scales
        self.assertEqual(self._build(pool)["override_kv_cache_dim"], 656)

    def test_packed_draft_with_a_different_kv_row_is_rejected(self):
        pool = _dsa_pool_stub(layer_num=2)
        draft = _dsa_pool_stub(layer_num=1)
        draft.store_dtype = torch.float8_e4m3fn
        with self.assertRaisesRegex(ValueError, "KV row"):
            self._build(pool, (draft,))
        self._build(pool, (_dsa_pool_stub(layer_num=1),))


class TestSeparateDraftStructure(CustomTestCase):
    """Separate draft sidecars: DRAFT sized on the target's logical space, its
    indexer laid out on DRAFT, both taking transfer indices from target KV; a
    draft without index buffers gets no indexer sidecar."""

    def _run(self, pool):
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
            return build_full_draft_pools(draft_kv_pool=pool, tree_cache=tree_cache)

    def test_dsa_draft_sidecars(self):
        pool = _dsa_pool_stub(layer_num=2, size=4096)
        specs, entries = self._run(pool)
        self.assertEqual(
            [(s.pool_name, s.indices_from_pool) for s in specs],
            [(PoolName.DRAFT, PoolName.KV), (PoolName.DRAFT_INDEXER, PoolName.KV)],
        )
        # 8192 logical target tokens / 4096 draft tokens -> ratio 2, same geometry
        mapper = (None, 0, 1, None, None)
        self.assertEqual(
            _entry_shape(SimpleNamespace(entries=entries), 2),
            [
                (PoolName.DRAFT, False, id(pool), (), mapper, _kv_shape(2)),
                (
                    PoolName.DRAFT_INDEXER,
                    False,
                    id(pool),
                    (),
                    mapper,
                    _indexer_shape(2),
                ),
            ],
        )

    def test_draft_without_index_buffers_has_no_indexer_sidecar(self):
        pool = _dsa_pool_stub(layer_num=2, size=4096)
        pool.index_key_cache = SimpleNamespace(buffer=[])
        specs, entries = self._run(pool)
        self.assertEqual([s.pool_name for s in specs], [PoolName.DRAFT])
        self.assertEqual([e.name for e in entries], [PoolName.DRAFT])


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
        # packed tail: transfer layer 4 -> device layer 2 on both host_pools
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

    def _prepare(self, decls):
        return prepare_host_pool_configs(
            decls=decls,
            full_layer_mapping={0: 0},
            transfer_layer_id_max=1,
        )

    def test_rejects_self_referencing_index_source(self):
        import msgspec

        kv, indexer = _dsa_pool_stub(layer_num=1).host_pool_decls()
        bad = msgspec.structs.replace(indexer, indices_from_pool=PoolName.INDEXER)
        with self.assertRaisesRegex(ValueError, "indices_from_pool"):
            self._prepare((kv, bad))

    def test_rejects_self_referencing_layout_source(self):
        import msgspec

        kv, indexer = _dsa_pool_stub(layer_num=1).host_pool_decls()
        bad = msgspec.structs.replace(indexer, layout_source=PoolName.INDEXER)
        with self.assertRaisesRegex(ValueError, "layout_source"):
            self._prepare((kv, bad))

    def test_rejects_primary_that_is_not_kv(self):
        import msgspec

        kv, indexer = _dsa_pool_stub(layer_num=1).host_pool_decls()
        swa_primary = msgspec.structs.replace(kv, pool_name=PoolName.SWA)
        follower = msgspec.structs.replace(
            indexer, indices_from_pool=PoolName.SWA, layout_source=PoolName.SWA
        )
        with self.assertRaisesRegex(ValueError, "primary KV pool"):
            self._prepare((swa_primary, follower))

    def test_accepts_dsa_declaration(self):
        configs = self._prepare(_dsa_pool_stub(layer_num=1).host_pool_decls())
        self.assertEqual(
            [c.decl.pool_name for c in configs], [PoolName.KV, PoolName.INDEXER]
        )


class TestHostPoolPreflight(CustomTestCase):
    """Malformed declarations must fail before reserving host memory."""

    def test_invalid_dependencies_do_not_allocate(self):
        kv, indexer = _dsa_pool_stub(layer_num=2).host_pool_decls()
        cyclic_indexer = msgspec.structs.replace(indexer, layout_source=PoolName.SWA)
        cyclic_swa = msgspec.structs.replace(
            indexer, pool_name=PoolName.SWA, layout_source=PoolName.INDEXER
        )
        cases = [
            ((kv, msgspec.structs.replace(indexer, host_pool_builder=None)), "builder"),
            ((kv, msgspec.structs.replace(indexer, storage_info=None)), "storage_info"),
            ((kv, cyclic_indexer, cyclic_swa), "cyclic"),
            (
                (kv, msgspec.structs.replace(indexer, owned_device_layers=(0, 0))),
                "owned_device_layers",
            ),
        ]
        for decls, error in cases:
            with (
                self.subTest(error=error),
                patch.object(hybrid_pool_assembler, "build_kv_host_pool") as allocate,
            ):
                with self.assertRaisesRegex(ValueError, error):
                    assemble_host_pools_from_decls(
                        params=_target_params(),
                        decls=decls,
                        full_layer_mapping={0: 0, 1: 1},
                        load_cache_event=None,
                        storage_backend=None,
                        use_mla=True,
                    )
                allocate.assert_not_called()

    def test_draft_declarations_are_collected_once(self):
        target = _dsa_pool_stub(layer_num=2)
        draft = _dsa_pool_stub(layer_num=1)
        draft.host_pool_decls = MagicMock(wraps=draft.host_pool_decls)
        decls = target.host_pool_decls()
        drafts = validate_packed_draft_pools(target_decls=decls, draft_pools=(draft,))
        configs = prepare_host_pool_configs(
            decls=decls,
            full_layer_mapping={0: 0, 4: 1, 5: 2},
            transfer_layer_id_max=6,
            packed_draft_decls=drafts,
        )
        draft.host_pool_decls.assert_called_once()
        self.assertIs(configs[1].packed_draft_device_pools[0], draft)
        self.assertEqual(
            configs[1].layer_binding.transfer_to_device, {0: 0, 4: 1, 5: 2}
        )

    def test_packed_draft_rejects_duplicate_pool_names(self):
        target = _dsa_pool_stub(layer_num=2)
        draft = _dsa_pool_stub(layer_num=1)
        declarations = draft.host_pool_decls()
        draft.host_pool_decls = lambda: (*declarations, declarations[1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_packed_draft_pools(
                target_decls=target.host_pool_decls(), draft_pools=(draft,)
            )

    def test_transfer_range_is_not_the_number_of_entries(self):
        decls = _dsa_pool_stub(layer_num=2).host_pool_decls()
        with self.assertRaisesRegex(ValueError, "transfer layer 4"):
            prepare_host_pool_configs(
                decls=decls, full_layer_mapping={0: 0, 4: 1}, transfer_layer_id_max=2
            )

    def test_build_order_resolves_dependencies_before_allocation(self):
        kv, indexer = _dsa_pool_stub(layer_num=1).host_pool_decls()
        configs = prepare_host_pool_configs(
            decls=(indexer, kv), full_layer_mapping={0: 0}, transfer_layer_id_max=1
        )
        self.assertEqual(
            [c.decl.pool_name for c in configs], [PoolName.KV, PoolName.INDEXER]
        )

    def test_qsa_page_mismatch_does_not_allocate_anchor(self):
        pool = _qsa_pool_stub(layer_num=1)
        with patch.object(hybrid_pool_assembler, "build_kv_host_pool") as allocate:
            with self.assertRaisesRegex(ValueError, "multiple"):
                assemble_host_pools_from_decls(
                    params=_target_params(page_size=63),
                    decls=pool.host_pool_decls(),
                    full_layer_mapping={0: 0},
                    load_cache_event=None,
                    storage_backend=None,
                    use_mla=True,
                )
            allocate.assert_not_called()

    def test_same_qsa_bytes_do_not_allow_a_different_dtype(self):
        pool = _qsa_pool_stub(layer_num=1)
        decl = pool.host_pool_decls()[1]
        pool.qsa_compressed_k_buffer_pool[0] = pool.qsa_compressed_k_buffer_pool[
            0
        ].view(torch.int16)
        with self.assertRaisesRegex(ValueError, "dtype"):
            decl.host_pool_builder.validate(
                decl=decl, page_size=64, packed_draft_device_pools=()
            )


class TestDeclaredPoolVerification(CustomTestCase):
    def _result_with(self, *names):
        group = SimpleNamespace(entry_map={n: object() for n in names})
        return StackBuildResult(
            host_pool_group=group, cache_controller=None, component_host_pools={}
        )

    def test_unmigrated_strategy_logs_missing_indexer(self):
        pool = _dsa_pool_stub(layer_num=2)
        with self.assertLogs(hybrid_pool_assembler.logger, level="ERROR") as logs:
            _check_declared_pools_present(
                kvcache=pool,
                result=self._result_with(PoolName.KV),
                strategy=_SwaStrategy(),
            )
        self.assertIn("indexer", logs.output[0])

    def test_migrated_strategy_raises_on_missing_indexer(self):
        pool = _dsa_pool_stub(layer_num=2)
        with self.assertRaisesRegex(ValueError, "indexer"):
            _check_declared_pools_present(
                kvcache=pool,
                result=self._result_with(PoolName.KV),
                strategy=_DsaStrategy(),
            )

    def test_complete_stack_passes_silently(self):
        pool = _dsa_pool_stub(layer_num=2)
        _check_declared_pools_present(
            kvcache=pool,
            result=self._result_with(PoolName.KV, PoolName.INDEXER),
            strategy=_DsaStrategy(),
        )

    def test_hybrid_pool_declares_through_its_sub_pool(self):
        # HybridLinearKVPool once exposed host_pool_decls as a property, so the
        # polymorphic call every strategy makes raised on hybrid models.
        pool = object.__new__(HybridLinearKVPool)
        pool.full_kv_pool = _dsa_pool_stub(layer_num=2)
        self.assertEqual(
            [(d.pool_name, d.device_pool) for d in pool.host_pool_decls()],
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
            _check_declared_pools_present(
                kvcache=pool,
                result=self._result_with(PoolName.KV, PoolName.MAMBA),
                strategy=_MambaStrategy(),
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
