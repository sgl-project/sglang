"""Unit tests for hybrid HiCache pool assembly."""

import unittest
from functools import partial
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import torch

from sglang.srt.configs.inkling import (
    InklingConvCacheParams,
    InklingConvStateShape,
    InklingStateDType,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo, MambaStateBufferInfo
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.host_pool_config import (
    prepare_host_pool_config,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    StackBuildResult,
    _check_declared_pools_present,
    _DsaStrategy,
    _MambaStrategy,
    _MambaSwaStrategy,
    _require_single_row_dsv4_swa_pages,
    _SwaStrategy,
    build_full_draft_pools,
    build_host_pool_group,
    build_hybrid_swa_group,
    build_unified_host_stack,
    validate_fused_hicache_draft,
)
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv.layout.mha import DraftMHAView, MHAHostLayout
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    FusedDraftPlacement,
)
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    HybridLinearKVPool,
    MambaPool,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host import dsa as pool_host_dsa
from sglang.srt.mem_cache.pool_host.host_pool_decl import (
    HostPoolDecl,
    make_kv_pool_decl,
)
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.unified import UnifiedPageEnvelopeHostPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import build_test_unified_pools

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


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
                    state = SimpleNamespace(pool=object())
                    params = SimpleNamespace(
                        req_to_token_pool=SimpleNamespace(),
                        token_to_kv_pool_allocator=SimpleNamespace(state=state),
                        tp_cache_group=None,
                        pp_cache_group=None,
                        attn_cp_cache_group=None,
                        attn_tp_cache_group=None,
                    )
                    controller = SimpleNamespace(transfer_layer_id_max=4)
                    built = (
                        (MagicMock(), controller, SimpleNamespace(pools=()))
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
                    if "mamba" in local_maps:
                        self.assertEqual(
                            build_stack.call_args.kwargs["start_layer"], start_layer
                        )
                    for name, mapping in local_maps.items():
                        if name == "mamba":
                            continue
                        self.assertEqual(
                            build_stack.call_args.kwargs[f"{name}_layer_mapping"],
                            mapping,
                        )
                    self.assertEqual(result.cache_controller.transfer_layer_id_max, 4)
                    self.assertEqual(
                        kvcache.full_attention_layer_id_mapping, global_maps["full"]
                    )
                    self.assertEqual(kvcache.layers_mapping, layers_mapping)


class TestMambaInfoRouting(CustomTestCase):
    def test_checkpoint_axis_routes_stage_layers_for_both_stacks(self):
        for tri_pool in (False, True):
            for start_layer in (0, 8):
                for local_ids in ((4, 2, 3), ()):
                    with self.subTest(
                        tri_pool=tri_pool, start=start_layer, layers=local_ids
                    ):

                        def make_pool(ids):
                            return MambaPool(
                                size=8,
                                spec_state_size=0,
                                cache_params=InklingConvCacheParams(
                                    shape=InklingConvStateShape(
                                        conv=[(2,)], temporal=(2,)
                                    ),
                                    layers=list(ids),
                                    dtype=InklingStateDType(
                                        conv=torch.float32, temporal=torch.float32
                                    ),
                                ),
                                mamba_layer_ids=list(ids),
                                device="cpu",
                            )

                        mamba_pool = make_pool(
                            tuple(layer + start_layer for layer in local_ids)
                        )
                        temporal = mamba_pool.mamba_cache.temporal
                        temporal.copy_(
                            torch.arange(temporal.numel()).reshape(temporal.shape)
                        )
                        # Buffer-only capacity may exceed the physical checkpoint slots.
                        mamba_pool.host_capacity_tokens = 16 if start_layer else None
                        drafts = (
                            tuple(
                                make_pool([start_layer + 5 + depth])
                                for depth in range(2)
                            )
                            if local_ids
                            else ()
                        )
                        captured = []
                        original_provider = mamba_pool.get_device_pool_infos

                        def capture():
                            result = original_provider()
                            captured.append(result[0])
                            return result

                        provider = MagicMock(side_effect=capture)
                        kv = SimpleNamespace(layer_num=1, size=8)
                        allocator = SimpleNamespace(alloc=MagicMock(), free=MagicMock())
                        params = SimpleNamespace(
                            page_size=1,
                            mtp_draft_device_pools=(),
                            mtp_draft_mamba_pools=drafts,
                            token_to_kv_pool_allocator=SimpleNamespace(
                                swa_attn_allocator=allocator, state_slots=allocator
                            ),
                            tp_cache_group=None,
                            pp_cache_group=None,
                            attn_cp_cache_group=None,
                            attn_tp_cache_group=None,
                        )
                        memory = SimpleNamespace(
                            hicache_size=0,
                            hicache_ratio=2,
                            hicache_mem_layout="page_first_direct",
                            hicache_write_policy="write_through",
                            hicache_io_backend="direct",
                            hicache_host_memory_mode="cache",
                            enable_unified_memory=False,
                        )
                        with (
                            patch.object(mamba_pool, "get_device_pool_infos", provider),
                            patch.object(
                                hybrid_pool_assembler, "get_memory", return_value=memory
                            ),
                            patch.object(
                                hybrid_pool_assembler,
                                "_get_allocator_type",
                                return_value="default",
                            ),
                            patch.object(
                                hybrid_pool_assembler,
                                "build_kv_host_pool",
                                return_value=SimpleNamespace(
                                    layer_num=1,
                                    layout="page_first_direct",
                                    page_size=1,
                                    device="cpu",
                                    size=17,
                                    logical_size=17,
                                    can_use_write_back_jit=False,
                                ),
                            ),
                            patch.object(
                                hybrid_pool_assembler,
                                "MambaPoolHost",
                                partial(MambaPoolHost, pin_memory=False),
                            ),
                            patch.object(
                                hybrid_pool_assembler, "HybridCacheController"
                            ),
                        ):
                            common = dict(
                                params=params,
                                mamba_pool=mamba_pool,
                                full_layer_mapping={0: 0},
                                start_layer=start_layer,
                                load_cache_event=None,
                                storage_backend=None,
                            )
                            if tri_pool:
                                group, _ = (
                                    hybrid_pool_assembler.build_hybrid_mamba_swa_stack(
                                        **common,
                                        full_kv_pool=kv,
                                        swa_kv_pool=kv,
                                        swa_layer_mapping={1: 0},
                                        page_size=1,
                                        tp_group=None,
                                    )
                                )
                            else:
                                group, _, _ = (
                                    hybrid_pool_assembler.build_hybrid_mamba_stack(
                                        **common,
                                        decls=(make_kv_pool_decl(kv),),
                                        use_mla=False,
                                    )
                                )
                        entry = group.entry_map[PoolName.MAMBA]
                        host = entry.host_pool
                        try:
                            info = captured[0]
                            self.assertIs(host.buffer_info, info.buffer_info)
                            self.assertEqual(
                                host.size, (16 if start_layer else 8) * 2 + 1
                            )
                            self.assertEqual(
                                len(host.mtp_draft_buffer_infos), len(drafts)
                            )
                            for depth, draft in enumerate(drafts):
                                self.assertIs(
                                    host.mtp_draft_buffer_infos[depth].temporal,
                                    draft.mamba_cache.temporal,
                                )
                                self.assertEqual(
                                    entry.layer_mapper(5 + depth),
                                    len(local_ids) + depth,
                                )
                                self.assertEqual(
                                    host.temporal_device_layers[
                                        len(local_ids) + depth
                                    ].data_ptr(),
                                    draft.mamba_cache.temporal[0].data_ptr(),
                                )
                            self.assertEqual(provider.call_count, 1)
                            self.assertEqual(
                                info.layer_ids,
                                tuple(layer + start_layer for layer in local_ids),
                            )
                            self.assertIs(entry.device_alloc_fn, allocator.alloc)
                            self.assertIs(entry.device_free_fn, allocator.free)
                            for layer in range(-1, 5):
                                slot = entry.layer_mapper(layer)
                                if layer not in local_ids:
                                    self.assertIsNone(slot)
                                else:
                                    torch.testing.assert_close(
                                        host.buffer_info.temporal[slot],
                                        temporal[local_ids.index(layer)],
                                    )
                        finally:
                            host.destroy()


class TestDraftSidecarPoolDispatch(CustomTestCase):
    def test_full_builder_unwraps_empty_hybrid_linear_pool(self):
        draft_kv_pool = object.__new__(HybridLinearKVPool)
        full = SimpleNamespace(layer_num=0)
        full.host_pool_decls = lambda: (make_kv_pool_decl(full),)
        draft_kv_pool.full_kv_pool = full
        draft_kv_pool.full_attention_layer_id_mapping = {}

        specs, entries = build_full_draft_pools(
            draft_kv_pool=draft_kv_pool,
            tree_cache=None,
        )

        self.assertEqual(specs, [])
        self.assertEqual(entries, [])

    def test_full_builder_sizes_sidecar_for_anchor_logical_space(self):
        draft_kv_pool = SimpleNamespace(layer_num=1, size=800, start_layer=0)
        draft_kv_pool.layer_router = PlainPoolRouter(draft_kv_pool, IdSpaceKind.FULL)
        draft_kv_pool.host_pool_decls = lambda: (make_kv_pool_decl(draft_kv_pool),)
        draft_host_pool = SimpleNamespace(layer_num=1)
        tree_cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(size=100, logical_size=800),
                page_size=512,
                transfer_layer_id_max=1,
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
        DRAFT_INDEXER decl. A constructor change that skips this call site
        breaks only here, not on the target path."""
        draft_kv_pool = object.__new__(DSATokenToKVPool)
        draft_kv_pool.layer_num = 1
        draft_kv_pool.start_layer = 0
        draft_kv_pool.size = 800
        draft_kv_pool.index_head_dim = 128
        draft_kv_pool.page_size = 64
        draft_kv_pool.index_page_size = 64
        draft_kv_pool.skip_topk_layers = [False]
        draft_kv_pool.index_key_cache = SimpleNamespace(buffer=[object()])
        draft_host_pool = SimpleNamespace(layer_num=1)
        tree_cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(size=100, logical_size=800),
                page_size=512,
                transfer_layer_id_max=1,
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
    pool.page_size = 64
    pool.index_page_size = 64
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
# 8256 host tokens (2 x 4096 plus one reserve page) = 129 pages. The MLA row is
# kv_cache_dim 576 x bf16 = 1152 B per layer. The DSA index row is 132 B per
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

    def _run(self, *, pool, drafts, full_layer_mapping, page_size=64):
        from sglang.srt.mem_cache.pool_host.mla import (
            MLAHostAllocation,
            MLATokenToKVPoolHost,
        )

        real_indexer_host = pool_host_dsa.DSAIndexerPoolHost

        def dummy_kv_host(**kwargs):
            pool = kwargs["kv_pool"]
            drafts = kwargs["mtp_draft_device_pools"]
            width = kwargs["kv_pool"].kv_cache_dim or (
                pool.kv_lora_rank + pool.qk_rope_head_dim
            )
            target_layer_num = (
                (pool.layer_num + pool.layer_shard_size - 1) // pool.layer_shard_size
                if pool.layer_shard_enabled
                else pool.layer_num
            )
            return MLATokenToKVPoolHost(
                MLAHostAllocation(
                    dtype=pool.store_dtype,
                    kv_lora_rank=pool.kv_lora_rank,
                    qk_rope_head_dim=pool.qk_rope_head_dim,
                    kv_cache_dim=width,
                    size_per_token=width
                    * pool.store_dtype.itemsize
                    * (target_layer_num + len(drafts)),
                    draft_layer_num=len(drafts),
                ),
                device_capacity_tokens=pool.size,
                start_layer=pool.start_layer,
                end_layer=pool.end_layer,
                target_layer_num=target_layer_num,
                host_to_device_ratio=2,
                host_size=0,
                page_size=kwargs["page_size"],
                layout="page_first",
                pin_memory=False,
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

        with (
            patch.object(hybrid_pool_assembler, "build_kv_host_pool", dummy_kv_host),
            patch.object(pool_host_dsa, "DSAIndexerPoolHost", dummy_indexer_host),
            patch.object(
                hybrid_pool_assembler, "_get_allocator_type", return_value="default"
            ),
        ):
            config = prepare_host_pool_config(
                decls=pool.host_pool_decls(),
                full_layer_mapping=dict(full_layer_mapping),
                transfer_layer_id_max=max(full_layer_mapping) + 1,
                transfer_page_size=page_size,
                packed_draft_device_pools=drafts,
            )
            group = build_host_pool_group(config=config)
        return SimpleNamespace(
            host_pool_group=group,
            sidecars=[
                c.decl.sidecar_spec() for c in config.pools if not c.decl.is_primary
            ],
        )

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

    def test_group_uses_prepared_page_size(self):
        pool = _dsa_pool_stub(layer_num=1)
        stack = self._run(
            pool=pool, drafts=(), full_layer_mapping={0: 0}, page_size=128
        )
        for entry in stack.host_pool_group.entries:
            self.assertEqual(entry.host_pool.page_size, 128)
        indexer = stack.host_pool_group.get_pool(PoolName.INDEXER)
        self.assertEqual(indexer.indexer_page_stride_size, 128 * _INDEX_ROW)

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

    def test_strategy_keeps_target_controller_range_with_packed_draft(self):
        pool = _dsa_pool_stub(layer_num=3)
        draft = _dsa_pool_stub(layer_num=1)
        stack = self._run(
            pool=pool, drafts=(draft,), full_layer_mapping={0: 0, 1: 1, 2: 2}
        )
        with (
            patch.object(
                hybrid_pool_assembler,
                "build_host_pool_group",
                return_value=stack.host_pool_group,
            ) as build,
            patch.object(hybrid_pool_assembler, "HybridCacheController") as controller,
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
            result = _DsaStrategy().build(
                cache=None,
                kvcache=pool,
                params=_target_params((draft,)),
                server_args=None,
                load_cache_event=None,
            )
        self.assertEqual(controller.call_args.kwargs["transfer_layer_id_max"], 3)
        self.assertEqual(controller.call_args.args[2], 64)
        for config in build.call_args.kwargs["config"].pools:
            self.assertEqual(config.transfer_layer_id_max, 4)
            self.assertEqual(config.layer_mapping, {0: 0, 1: 1, 2: 2, 3: 3})
            self.assertEqual(config.packed_draft_device_pools, (draft,))
        self.assertIs(result.host_pool_group, stack.host_pool_group)
        self.assertEqual(result.sidecars, stack.sidecars)

    def test_dsa_target_layer_sharded(self):
        # rank 0 of 2 owns local layers 0 and 1 of 3. The permuted stage mapping
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
            prepare_host_pool_config(
                decls=target.host_pool_decls(),
                full_layer_mapping={i: i for i in range(target.layer_num)},
                transfer_layer_id_max=target.layer_num,
                transfer_page_size=64,
                packed_draft_device_pools=(no_index,),
            )

    def test_draft_with_empty_index_layers_is_rejected(self):
        # A 0-row placeholder layer would be packed as a null device pointer.
        target = _dsa_pool_stub(layer_num=2)
        shared = _dsa_pool_stub(layer_num=1)
        shared.skip_topk_layers = [True]
        shared.index_key_cache = SimpleNamespace(buffer=[object()])
        with self.assertRaisesRegex(ValueError, "draft counterpart"):
            prepare_host_pool_config(
                decls=target.host_pool_decls(),
                full_layer_mapping={i: i for i in range(target.layer_num)},
                transfer_layer_id_max=target.layer_num,
                transfer_page_size=64,
                packed_draft_device_pools=(shared,),
            )
        partial = _dsa_pool_stub(layer_num=2)
        partial.skip_topk_layers = [False, True]
        with self.assertRaisesRegex(ValueError, "owns buffers on 1 of 2"):
            prepare_host_pool_config(
                decls=target.host_pool_decls(),
                full_layer_mapping={i: i for i in range(target.layer_num)},
                transfer_layer_id_max=target.layer_num,
                transfer_page_size=64,
                packed_draft_device_pools=(partial,),
            )

    def test_storage_info_mismatch_is_rejected(self):
        target = _dsa_pool_stub(layer_num=2)
        wide = _dsa_pool_stub(layer_num=1)
        wide.index_head_dim = 256
        with self.assertRaisesRegex(ValueError, "storage"):
            prepare_host_pool_config(
                decls=target.host_pool_decls(),
                full_layer_mapping={i: i for i in range(target.layer_num)},
                transfer_layer_id_max=target.layer_num,
                transfer_page_size=64,
                packed_draft_device_pools=(wide,),
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
        for device_pool in (pool, *drafts):
            device_pool.model_layer_ids = tuple(range(device_pool.layer_num))
            device_pool.kv_buffer = [
                torch.empty(
                    (
                        device_pool.size + device_pool.page_size,
                        1,
                        device_pool.kv_cache_dim,
                    ),
                    dtype=device_pool.store_dtype,
                )
                for _ in range(device_pool.layer_num)
            ]
            device_pool.data_ptrs = torch.tensor(
                [buffer.data_ptr() for buffer in device_pool.kv_buffer],
                dtype=torch.uint64,
            )
            device_pool.kv_scale_buffer = None
            device_pool.dsa_kv_cache_store_fp8 = False
            device_pool.host_capacity_tokens = None

        def fake_host(buffer_info, ratio, size, page_size, layout, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(layer_num=kwargs["target_layer_num"])

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


class TestHybridMambaDeclaredIndexer(CustomTestCase):
    """hybrid Mamba + DSA reuses the target's indexer declaration: the KV/Mamba
    stack gains an INDEXER entry whose mirror and layer mapping cover only the
    DSA layers that own index buffers."""

    def test_stack_declares_indexer_and_skips_empty_layers(self):
        from sglang.srt.mem_cache.pool_host.mla import (
            MLAHostAllocation,
            MLATokenToKVPoolHost,
        )

        kv_pool = _dsa_pool_stub(layer_num=3)
        kv_pool.skip_topk_layers = [False, True, False]  # device layer 1: no buffer
        mamba_info = DevicePoolInfo(
            pool_name=PoolName.MAMBA,
            indices_from_pool=PoolName.MAMBA,
            layer_ids=(1, 3),
            buffer_info=MambaStateBufferInfo(
                temporal=torch.zeros(2, 9, 4), conv=(torch.zeros(2, 9, 4),)
            ),
        )
        mamba_pool = SimpleNamespace(get_device_pool_infos=lambda: (mamba_info,))
        # transfer layers 0,2,4 are DSA (device 0,1,2), 1,3 are Mamba
        full_mapping = {0: 0, 2: 1, 4: 2}
        params = SimpleNamespace(
            page_size=64,
            mtp_draft_device_pools=(),
            mtp_draft_mamba_pools=(),
            token_to_kv_pool_allocator=SimpleNamespace(
                state_slots=SimpleNamespace(alloc=lambda n: None, free=lambda x: None)
            ),
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
        )
        real_indexer_host = pool_host_dsa.DSAIndexerPoolHost

        def dummy_kv_host(**kwargs):
            pool = kwargs["kv_pool"]
            drafts = kwargs["mtp_draft_device_pools"]
            width = None or (pool.kv_lora_rank + pool.qk_rope_head_dim)
            return MLATokenToKVPoolHost(
                MLAHostAllocation(
                    dtype=pool.store_dtype,
                    kv_lora_rank=pool.kv_lora_rank,
                    qk_rope_head_dim=pool.qk_rope_head_dim,
                    kv_cache_dim=width,
                    size_per_token=width
                    * pool.store_dtype.itemsize
                    * (pool.layer_num + len(drafts)),
                    draft_layer_num=len(drafts),
                ),
                device_capacity_tokens=pool.size,
                start_layer=pool.start_layer,
                end_layer=pool.end_layer,
                target_layer_num=pool.layer_num,
                host_to_device_ratio=2,
                host_size=0,
                page_size=kwargs["page_size"],
                layout="page_first",
                pin_memory=False,
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
                "build_mamba_host_pool",
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
            group, _, config = hybrid_pool_assembler.build_hybrid_mamba_stack(
                params=params,
                decls=kv_pool.host_pool_decls(),
                mamba_pool=mamba_pool,
                full_layer_mapping=full_mapping,
                start_layer=0,
                load_cache_event=None,
                storage_backend=None,
                use_mla=True,
            )

        names = [e.name for e in group.entries]
        self.assertEqual(names, [PoolName.KV, PoolName.INDEXER, PoolName.MAMBA])
        indexer = group.entry_map[PoolName.INDEXER]
        kv = group.entry_map[PoolName.KV]
        # KV still maps every DSA transfer layer. The indexer drops device layer 1.
        self.assertEqual([kv.layer_mapper(t) for t in range(5)], [0, None, 1, None, 2])
        self.assertEqual(
            [indexer.layer_mapper(t) for t in range(5)], [0, None, None, None, 2]
        )
        self.assertEqual(indexer.host_pool.layer_num, 2)
        self.assertEqual(indexer.host_pool._host_layer_index(2), 1)
        self.assertEqual(
            [c.decl.pool_name for c in config.pools], [PoolName.KV, PoolName.INDEXER]
        )


class TestDeclaredPoolPlanning(CustomTestCase):
    """Sidecar indices resolve from one primary source in HostPoolGroup, so the
    planner must reject self-references and sidecar chains up front."""

    def _prepare(self, decls):
        return prepare_host_pool_config(
            decls=decls,
            full_layer_mapping={0: 0},
            transfer_layer_id_max=1,
            transfer_page_size=64,
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
        config = self._prepare(_dsa_pool_stub(layer_num=1).host_pool_decls())
        self.assertEqual(
            [c.decl.pool_name for c in config.pools], [PoolName.KV, PoolName.INDEXER]
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
                    config = prepare_host_pool_config(
                        decls=decls,
                        full_layer_mapping={0: 0, 1: 1},
                        transfer_layer_id_max=2,
                        transfer_page_size=64,
                    )
                    build_host_pool_group(config=config)
                allocate.assert_not_called()

    def test_incompatible_packed_buffers_fail_before_allocation(self):
        target = _dsa_pool_stub(layer_num=2)
        bad_kv = _dsa_pool_stub(layer_num=1)
        bad_kv.store_dtype = torch.float8_e4m3fn
        bad_index = _dsa_pool_stub(layer_num=1)
        # Same 132 bytes per row, different key/scale format.
        bad_index.index_head_dim = 132
        bad_index.quant_block_size = 256
        for draft, error in ((bad_kv, "KV row"), (bad_index, "index key format")):
            with (
                self.subTest(error=error),
                patch.object(hybrid_pool_assembler, "build_kv_host_pool") as allocate,
            ):
                with self.assertRaisesRegex(ValueError, error):
                    _DsaStrategy().build(
                        cache=None,
                        kvcache=target,
                        params=_target_params((draft,)),
                        server_args=None,
                        load_cache_event=None,
                    )
                allocate.assert_not_called()

    def test_draft_declarations_are_collected_once(self):
        target = _dsa_pool_stub(layer_num=2)
        draft = _dsa_pool_stub(layer_num=1)
        draft.host_pool_decls = MagicMock(wraps=draft.host_pool_decls)
        decls = target.host_pool_decls()
        target_mapping = {0: 0, 4: 1}
        config = prepare_host_pool_config(
            decls=decls,
            full_layer_mapping=target_mapping,
            transfer_layer_id_max=5,
            transfer_page_size=64,
            packed_draft_device_pools=(draft,),
        )
        self.assertEqual(target_mapping, {0: 0, 4: 1})
        draft.host_pool_decls.assert_called_once()
        self.assertIs(config.pools[1].packed_draft_device_pools[0], draft)
        self.assertEqual(config.pools[1].layer_mapping, {0: 0, 4: 1, 5: 2})

    def test_packed_draft_rejects_duplicate_pool_names(self):
        target = _dsa_pool_stub(layer_num=2)
        draft = _dsa_pool_stub(layer_num=1)
        declarations = draft.host_pool_decls()
        draft.host_pool_decls = lambda: (*declarations, declarations[1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare_host_pool_config(
                decls=target.host_pool_decls(),
                full_layer_mapping={i: i for i in range(target.layer_num)},
                transfer_layer_id_max=target.layer_num,
                transfer_page_size=64,
                packed_draft_device_pools=(draft,),
            )

    def test_transfer_range_is_not_the_number_of_entries(self):
        decls = _dsa_pool_stub(layer_num=2).host_pool_decls()
        with self.assertRaisesRegex(ValueError, "transfer layer 4"):
            prepare_host_pool_config(
                decls=decls,
                full_layer_mapping={0: 0, 4: 1},
                transfer_layer_id_max=2,
                transfer_page_size=64,
            )

    def test_build_order_resolves_dependencies_before_allocation(self):
        kv, indexer = _dsa_pool_stub(layer_num=1).host_pool_decls()
        config = prepare_host_pool_config(
            decls=(indexer, kv),
            full_layer_mapping={0: 0},
            transfer_layer_id_max=1,
            transfer_page_size=64,
        )
        self.assertEqual(
            [c.decl.pool_name for c in config.pools], [PoolName.KV, PoolName.INDEXER]
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


def _build_unified_swa_pool(page_size: int = 4, *, fused_draft=None):
    return build_test_unified_pools(
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
        fused_draft=fused_draft,
    )


def _build_unified_host_pair(bundle):
    pool = bundle.token_to_kv_pool
    return UnifiedPageEnvelopeHostPool.build_hybrid_swa_pool_pair(
        device_pools=tuple(map(pool.layer_router.view, ("full", "swa"))),
        host_to_device_ratio=1.0,
        host_size=0,
        page_size=pool.page_size,
        layout="layer_first",
        pin_memory=False,
        allocator_type="default",
    )


class TestFusedDraftHiCacheValidation(CustomTestCase):
    def test_fused_draft_binding_uses_its_host_sub_pool(self):
        """A bound full draft passes HiCache admission; a window draft is refused."""
        region = DenseDraftRegion(
            lane_num=1, head_num=2, head_dim=4, store_dtype=torch.float16
        )
        for host in ("full", "swa"):
            with self.subTest(host=host):
                placement = FusedDraftPlacement.from_counts(
                    counts={host: [1]}, regions={host: region}
                )
                bundle = _build_unified_swa_pool(fused_draft=placement)
                draft = DraftMHAView(
                    arena=bundle.arena,
                    host_spec=bundle.storage.plan.sub_pool(host).spec,
                    layer_lanes={0: 0},
                    page_size=bundle.token_to_kv_pool.page_size,
                )
                if host == "full":
                    validate_fused_hicache_draft(draft_device_pools=(draft,))
                else:
                    with self.assertRaisesRegex(NotImplementedError, "full-attention"):
                        validate_fused_hicache_draft(draft_device_pools=(draft,))


class TestUnifiedPageEnvelopeHostPool(CustomTestCase):
    def test_packed_drafts_are_not_silently_dropped_by_shared_arena(self):
        storage = _build_unified_swa_pool().token_to_kv_pool_allocator.storage
        draft = _build_unified_swa_pool().token_to_kv_pool
        for full_indices in (False, True):
            draft.swa_uses_full_indices = full_indices
            with (
                self.subTest(full_indices=full_indices),
                patch(_ASSEMBLER + "get_memory", return_value=_unified_memory(None)),
                patch(
                    _ASSEMBLER + "get_parallel",
                    return_value=SimpleNamespace(dcp_enabled=False),
                ),
                patch(_ASSEMBLER + "_get_allocator_type", return_value="default"),
                patch.object(
                    MHAHostLayout,
                    "build",
                    side_effect=NotImplementedError("packed pool reached host factory"),
                ),
                patch.object(
                    UnifiedPageEnvelopeHostPool, "build_hybrid_swa_pool_pair"
                ) as build_pair,
            ):
                with self.assertRaises(Exception):
                    build_unified_host_stack(
                        storage=storage,
                        params=_unified_params(storage, drafts=(draft,)),
                        cache=None,
                        load_cache_event=None,
                        storage_backend=None,
                    )
                build_pair.assert_not_called()

    def test_sidecar_read_error_is_a_cache_miss_with_unified_memory(self):
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        seen_prefixes = []

        class FailingStorage:
            def batch_get_v2(self, transfers, *, extra_info=None):
                seen_prefixes.append(extra_info.prefix_keys)
                raise ValueError("sidecar read failed")

        self.addCleanup(reset_context)
        reset_context()
        publish(
            ServerArgs(model_path="dummy", enable_unified_memory=True),
            role="tokenizer",
        )
        cc = HybridCacheController.__new__(HybridCacheController)
        cc.storage_backend = FailingStorage()
        cc.prefetch_sync_queue = Queue()
        operation = PrefetchOperation(
            CacheRequestHandle("r", 0),
            [1],
            pool_transfers=[PoolTransfer(name=PoolName.SWA)],
        )
        operation.hash_value = ["h0"]
        operation.prefix_keys = ["prefix"]
        with self.assertLogs(level="ERROR"):
            cc._page_transfer_sidecar(operation, kv_completed_pages=1)
        ack = cc.prefetch_sync_queue.get_nowait()
        self.assertIs(ack.operation, operation)
        self.assertEqual(ack.pool_hits, {})
        self.assertTrue(cc.prefetch_sync_queue.empty())
        self.assertEqual(seen_prefixes, [["prefix", "h0"]])

    def test_shorter_prefetch_reserves_full_and_swa_without_mutating_probe_keys(self):
        page_size = 4
        full_pool, swa_pool = _build_unified_host_pair(
            _build_unified_swa_pool(page_size)
        )
        self.addCleanup(full_pool.destroy)
        self.addCleanup(swa_pool.destroy)
        cc = HybridCacheController.__new__(HybridCacheController)
        cc.page_size = page_size
        cc.host_memory_mode = "cache"
        cc.attn_cp_group = cc.attn_tp_group = cc.tp_group = None
        cc.mem_pool_host = HostPoolGroup(
            [
                PoolEntry(PoolName.KV, full_pool, None, None),
                PoolEntry(PoolName.SWA, swa_pool, None, None),
            ]
        )
        hit_tokens = full_pool.available_size()
        hashes = [str(i) for i in range(hit_tokens // page_size)]
        transfer = PoolTransfer(
            name=PoolName.SWA,
            keys=hashes[-2:],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        operation = PrefetchOperation(
            CacheRequestHandle("r", 0),
            list(range(hit_tokens)),
            pool_transfers=[transfer],
        )
        operation.hash_value = hashes
        self.assertIsNone(cc.alloc_prefetch_host_buffers(operation, hit_tokens))
        fitting = [
            n
            for n in range(page_size, hit_tokens + 1, page_size)
            if cc.can_fit_prefetch_host_buffers(operation, n, empty=False)
        ]
        self.assertTrue(fitting)
        self.assertEqual(transfer.keys, hashes[-2:])
        length = max(fitting)
        self.assertLess(length, hit_tokens)
        host_indices = cc.alloc_prefetch_host_buffers(operation, length)
        self.assertEqual(host_indices.numel(), length)
        self.assertEqual(transfer.host_indices.numel(), 2 * page_size)
        self.assertEqual(
            transfer.keys, hashes[length // page_size - 2 : length // page_size]
        )
        cc.free_prefetch_host_buffers(operation, host_indices)

        # A hit-time rematch can trim some or all FULL pages while the entire
        # trailing SWA window still needs staging from the same shared arena.
        for full_tokens in (page_size, 0):
            with self.subTest(full_tokens=full_tokens):
                transfer.keys = hashes[-2:]
                operation.hash_value = list(hashes)
                operation.storage_hit_count = hit_tokens
                operation.sidecar_hash_values = None
                operation.prefetch_kv_page_offset = 0
                cc.trim_prefetch_full_head(operation, hit_tokens - full_tokens)
                self.assertEqual(
                    operation.prefetch_kv_page_offset,
                    (hit_tokens - full_tokens) // page_size,
                )
                self.assertTrue(
                    cc.can_fit_prefetch_host_buffers(operation, full_tokens)
                )
                host_indices = cc.alloc_prefetch_host_buffers(operation, full_tokens)
                self.assertEqual(host_indices.numel(), full_tokens)
                self.assertEqual(transfer.host_indices.numel(), 2 * page_size)
                self.assertEqual(transfer.keys, hashes[-2:])
                cc.free_prefetch_host_buffers(operation, host_indices)

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


class TestUnifiedPageEnvelopeSelection(CustomTestCase):
    """Only no backend and Mori store the complete shared page, so only they
    keep the page-envelope host arena."""

    def test_only_no_backend_and_mori_select_the_envelope(self):
        storage = _build_unified_swa_pool().token_to_kv_pool_allocator.storage
        for backend, expected in (
            (None, "pair"),
            ("mori", "pair"),
            ("mooncake", "kv host"),
            ("dynamic", "kv host"),
        ):
            with (
                self.subTest(backend=backend),
                patch(_ASSEMBLER + "get_memory", return_value=_unified_memory(backend)),
                patch(
                    _ASSEMBLER + "get_parallel",
                    return_value=SimpleNamespace(dcp_enabled=False),
                ),
                patch(_ASSEMBLER + "_get_allocator_type", return_value="default"),
                patch.object(
                    MHAHostLayout,
                    "build",
                    side_effect=NotImplementedError("kv host"),
                ),
                patch.object(
                    UnifiedPageEnvelopeHostPool,
                    "build_hybrid_swa_pool_pair",
                    side_effect=NotImplementedError("pair"),
                ),
            ):
                with self.assertRaises(NotImplementedError) as raised:
                    build_unified_host_stack(
                        storage=storage,
                        params=_unified_params(storage),
                        cache=None,
                        load_cache_event=None,
                        storage_backend=backend,
                    )
                self.assertEqual(str(raised.exception), expected)


def _unified_memory(backend):
    return SimpleNamespace(
        hicache_storage_backend=backend,
        hicache_size=0,
        hicache_ratio=1.0,
        hicache_mem_layout="layer_first",
        hicache_host_memory_mode="auto",
    )


def _unified_params(storage, drafts=()):
    return SimpleNamespace(
        page_size=4, mtp_draft_device_pools=drafts, mtp_draft_mamba_pools=()
    )


class TestMtpMhaPhysicalConstruction(CustomTestCase):
    def test_full_token_draft_swa_preserves_capacity_and_physical_rows(self):
        """MTP sidecars keep their capacity override and both draft allocations."""
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

        original_init = MHATokenToKVPoolHost.__init__
        allocated = []

        def unpinned(host, *args, **kwargs):
            kwargs["pin_memory"] = False
            original_init(host, *args, **kwargs)
            allocated.append(host)

        def pool(size, layers, start, heads):
            return MHATokenToKVPool(
                size=size,
                page_size=4,
                dtype=torch.float16,
                head_num=heads,
                head_dim=8,
                layer_num=layers,
                device="cpu",
                enable_memory_saver=False,
                start_layer=start,
                end_layer=start + layers,
            )

        for layout in ("layer_first", "page_first"):
            with self.subTest(layout=layout):
                full = pool(32, 2, 20, 2)
                swa = pool(16, 1, 20, 1)
                first, second = pool(16, 1, 23, 1), pool(16, 1, 24, 1)
                first.host_capacity_tokens = 32
                try:
                    with (
                        patch.object(MHATokenToKVPoolHost, "__init__", unpinned),
                        patch.object(
                            hybrid_pool_assembler,
                            "get_memory",
                            return_value=SimpleNamespace(
                                hicache_ratio=1.0,
                                hicache_size=0,
                                hicache_mem_layout=layout,
                            ),
                        ),
                        patch.object(
                            hybrid_pool_assembler,
                            "get_parallel",
                            return_value=SimpleNamespace(dcp_enabled=False),
                        ),
                        patch.object(
                            hybrid_pool_assembler,
                            "_get_allocator_type",
                            return_value="default",
                        ),
                    ):
                        group = build_hybrid_swa_group(
                            page_size=4,
                            full_kv_pool=full,
                            swa_kv_pool=swa,
                            full_layer_mapping={0: 0, 2: 1},
                            swa_layer_mapping={1: 0},
                            use_mla=False,
                            mtp_swa_full_device_pools=(first, second),
                            transfer_layer_id_max=3,
                        )
                    host = group.get_pool(PoolName.DRAFT_SWA)
                    self.assertEqual(group.get_pool(PoolName.KV).size, 36)
                    self.assertEqual(host.device_capacity_tokens, 32)
                    self.assertEqual(host.size, 76)
                    self.assertEqual((host.start_layer, host.end_layer), (23, 24))
                    self.assertEqual((host.target_layer_num, host.layer_num), (1, 2))
                    self.assertIs(host.device_k_buffers[0], first.k_buffer[0])
                    self.assertIs(host.device_k_buffers[1], second.k_buffer[0])
                    self.assertIs(host.device_v_buffers[0], first.v_buffer[0])
                    self.assertIs(host.device_v_buffers[1], second.v_buffer[0])
                    self.assertEqual(
                        [
                            group.get_entry(PoolName.DRAFT_SWA).layer_mapper(i)
                            for i in (0, 3)
                        ],
                        [0, 1],
                    )
                    self.assertEqual(
                        group.get_entry(PoolName.DRAFT_SWA).packed_draft_device_pools,
                        (second,),
                    )
                finally:
                    for host in allocated:
                        host.destroy()
                    allocated.clear()


class TestMambaSwaStackMatchesOtherStacks(CustomTestCase):
    """The full + window + state stack builds its controller and state host the
    way the full + window and full + state stacks do."""

    def test_passes_the_attention_groups(self):
        params = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                state=SimpleNamespace(pool=object())
            ),
            tp_cache_group=object(),
            pp_cache_group=object(),
            attn_cp_cache_group=object(),
            attn_tp_cache_group=object(),
        )
        kvcache = SimpleNamespace(
            layers_mapping={0: (0, False), 1: (0, True)},
            start_layer=0,
            full_kv_pool=object(),
            swa_kv_pool=object(),
        )
        with patch.object(
            hybrid_pool_assembler,
            "build_hybrid_mamba_swa_stack",
            return_value=(MagicMock(), MagicMock()),
        ) as build:
            _MambaSwaStrategy().build(
                cache=SimpleNamespace(page_size=1),
                kvcache=kvcache,
                params=params,
                server_args=None,
                load_cache_event=None,
            )
        kwargs = build.call_args.kwargs
        self.assertIs(kwargs["attn_cp_group"], params.attn_cp_cache_group)
        self.assertIs(kwargs["attn_tp_group"], params.attn_tp_cache_group)

    def test_state_host_has_no_kv_split(self):
        for layout, expected in (
            ("page_first_kv_split", "page_first_direct"),
            ("page_first", "page_first"),
        ):
            with (
                self.subTest(layout=layout),
                patch.object(
                    hybrid_pool_assembler,
                    "get_memory",
                    return_value=SimpleNamespace(hicache_mem_layout=layout),
                ),
            ):
                self.assertEqual(hybrid_pool_assembler._state_host_layout(), expected)


if __name__ == "__main__":
    unittest.main()
