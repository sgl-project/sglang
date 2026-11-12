"""The ``page_unified`` page layout and the unified L3 key scheme that names it.

Two halves of one thing: the layout fixes where a chunk's bytes sit inside a
page, and the key scheme names that same chunk as a layer-range x head-range
coordinate. They share a file because they have to agree on the grid.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.deepseek_v4_buffer_info import (
    DSV4CheckpointBufferInfo,
    DSV4PagedBufferInfo,
    DSV4StateBufferInfo,
)
from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo
from sglang.srt.mem_cache.hicache_key_scheme import (
    UnifiedPoolPlan,
    derive_namespace,
    namespace_digest,
    normalize_dtype,
    plan_unified_kv,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _mha(**overrides):
    kwargs = dict(
        page_size=4,
        layer_num=6,
        head_num=4,
        head_group_num=2,
        head_dim=8,
        itemsize=2,
    )
    kwargs.update(overrides)
    return PageUnifiedLayout(**kwargs)


def _dsv4_plan_fixture(
    *,
    layer_ids=(3, 7),
    capacity=4,
    name=PoolName.DEEPSEEK_V4_C4,
    layout="v41",
    native_shape=(16,),
    dtype=torch.uint8,
    page_slots=64,
    host_page_size=256,
    item_bytes=16,
    state_dtypes=(),
):
    if state_dtypes:
        region = DSV4CheckpointBufferInfo(
            states=tuple(
                DSV4StateBufferInfo(
                    pages=torch.zeros((capacity, item_bytes), dtype=torch.uint8),
                    ring_size=2,
                    dtype=state_dtype,
                )
                for state_dtype in state_dtypes
            )
        )
    else:
        region = DSV4PagedBufferInfo.from_buffers(
            [torch.zeros((capacity, *native_shape), dtype=dtype) for _ in layer_ids],
            page_slots=page_slots,
            layout=layout,
        )
    info = DevicePoolInfo(
        pool_name=name,
        indices_from_pool=PoolName.KV,
        layer_ids=layer_ids,
        buffer_info=region,
    )
    root = SimpleNamespace(
        get_device_pool_infos=lambda: (info,),
        index_pools={ratio: SimpleNamespace() for ratio in (1, 2, 4, 128)},
    )
    host = SimpleNamespace(
        layout="page_unified",
        page_size=host_page_size,
        layer_num=len(layer_ids),
        item_bytes=item_bytes,
        state_page_bytes=item_bytes,
    )
    group = SimpleNamespace(
        entries=[
            SimpleNamespace(
                name=PoolName.KV,
                packed_draft_device_pools=(),
                host_pool=SimpleNamespace(layout="page_unified", page_size=256),
            ),
            SimpleNamespace(name=name, packed_draft_device_pools=(), host_pool=host),
        ]
    )
    return root, group


class TestDSV4UnifiedPoolPlans(CustomTestCase):
    @staticmethod
    def _plans(root, group, *, page_size=256, **partitions):
        with patch.object(
            hybrid_pool_assembler,
            "get_memory",
            return_value=SimpleNamespace(**partitions),
        ):
            return hybrid_pool_assembler.build_dsv4_unified_pool_plans(
                "deepseek/test", root, group, page_size
            )

    def test_native_layer_objects_survive_tp_pp_and_capacity_changes(self):
        """A PP subset and a different TP width must find identical owner bytes."""
        for name in (
            PoolName.SWA,
            PoolName.DEEPSEEK_V4_C1,
            PoolName.DEEPSEEK_V4_C2,
            PoolName.DEEPSEEK_V4_C4,
            PoolName.DEEPSEEK_V4_C128,
        ):
            with self.subTest(pool=name):
                root, group = _dsv4_plan_fixture(name=name)
                root.tp_size, root.pp_size = 2, 1
                complete = self._plans(root, group)
                root, group = _dsv4_plan_fixture(name=name, layer_ids=(7,), capacity=13)
                root.tp_size, root.pp_size = 8, 3
                subset = self._plans(root, group)
                self.assertEqual(complete[name].suffixes[1:], subset[name].suffixes)
                self.assertEqual(complete[name].chunk_spans, ((0, 16), (16, 16)))
                self.assertEqual(subset[name].chunk_spans, ((0, 16),))
                self.assertEqual(complete[PoolName.KV], subset[PoolName.KV])
                self.assertEqual(subset[PoolName.KV].page_bytes, 0)

    def test_equal_sized_native_formats_do_not_share_objects(self):
        """Equal page bytes can still encode different payload/scale positions."""
        root, group = _dsv4_plan_fixture()
        baseline = self._plans(root, group)[PoolName.DEEPSEEK_V4_C4].suffixes
        for override in (
            {"layout": "v4"},
            {"native_shape": (2, 8)},
            {"native_shape": (8,), "dtype": torch.float16},
            {"page_slots": 128},
            {"host_page_size": 128},
        ):
            with self.subTest(override=override):
                root, group = _dsv4_plan_fixture(**override)
                self.assertNotEqual(
                    baseline, self._plans(root, group)[PoolName.DEEPSEEK_V4_C4].suffixes
                )
        root, group = _dsv4_plan_fixture()
        self.assertNotEqual(
            baseline,
            self._plans(root, group, page_size=128)[PoolName.DEEPSEEK_V4_C4].suffixes,
        )

    def test_indexer_packing_flags_are_object_identity(self):
        name = PoolName.DEEPSEEK_V4_C4_INDEXER
        root, group = _dsv4_plan_fixture(name=name, layout="index_fused")
        baseline = self._plans(root, group)[name].suffixes
        for flag in (
            "use_fp4_indexer",
            "index_k_page8",
            "index_k_rne",
            "uses_aiter_fp4_layout",
        ):
            with self.subTest(flag=flag):
                root, group = _dsv4_plan_fixture(name=name, layout="index_fused")
                setattr(root.index_pools[4], flag, True)
                self.assertNotEqual(baseline, self._plans(root, group)[name].suffixes)

    def test_uint8_pages_retain_native_fp8_encoding_identity(self):
        """FN and FNUZ bytes must not collide when both stores expose uint8."""
        for name, layout in (
            (PoolName.SWA, "v4"),
            (PoolName.DEEPSEEK_V4_C4, "v4"),
            (PoolName.DEEPSEEK_V4_C4_INDEXER, "index_fused"),
        ):
            with self.subTest(pool=name):
                root, group = _dsv4_plan_fixture(name=name, layout=layout)
                with patch(
                    "sglang.kernels.ops.quantization.fp8_kernel.fp8_dtype",
                    torch.float8_e4m3fn,
                ):
                    fn = self._plans(root, group)
                with patch(
                    "sglang.kernels.ops.quantization.fp8_kernel.fp8_dtype",
                    torch.float8_e4m3fnuz,
                ):
                    fnuz = self._plans(root, group)
                self.assertNotEqual(fn[name].suffixes, fnuz[name].suffixes)
                self.assertEqual(fn[name].chunk_spans, fnuz[name].chunk_spans)
                self.assertEqual(fn[PoolName.KV], fnuz[PoolName.KV])

    def test_fp8_indexer_preshuffle_is_native_page_identity(self):
        """Equal-sized page64 stores require distinct keys for each swizzle."""
        name = PoolName.DEEPSEEK_V4_C4_INDEXER
        root, group = _dsv4_plan_fixture(name=name, layout="index_fused")
        plans = []
        for enabled, tile in ((False, 16), (True, 16), (True, 32)):
            with (
                patch(
                    "sglang.srt.layers.attention.dsa.utils.aiter_can_use_preshuffle_paged_mqa",
                    return_value=enabled,
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.INDEXER_K_CACHE_PRESHUFFLE_TILE",
                    tile,
                ),
            ):
                plans.append(self._plans(root, group)[name])
        self.assertEqual(len({plan.suffixes for plan in plans}), 3)
        self.assertEqual(len({plan.chunk_spans for plan in plans}), 1)

        # FP4 has its own packing flags and does not use the FP8 preshuffle.
        root.index_pools[4].use_fp4_indexer = True
        with patch(
            "sglang.srt.layers.attention.dsa.utils.aiter_can_use_preshuffle_paged_mqa",
            return_value=False,
        ):
            unshuffled = self._plans(root, group)[name]
        with patch(
            "sglang.srt.layers.attention.dsa.utils.aiter_can_use_preshuffle_paged_mqa",
            return_value=True,
        ):
            self.assertEqual(unshuffled, self._plans(root, group)[name])

    def test_required_native_side_pools_cannot_be_omitted(self):
        """Missing scales or RoPE must fail before storage can report a hit."""
        for name in (
            PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE,
            PoolName.DEEPSEEK_V4_C4_ROPE,
        ):
            with self.subTest(pool=name):
                root, group = _dsv4_plan_fixture()
                (kv_info,) = root.get_device_pool_infos()
                side_info = DevicePoolInfo(
                    pool_name=name,
                    indices_from_pool=PoolName.KV,
                    layer_ids=kv_info.layer_ids,
                    buffer_info=kv_info.buffer_info,
                )
                root.get_device_pool_infos = lambda: (kv_info, side_info)
                with self.assertRaisesRegex(ValueError, name.value):
                    self._plans(root, group)

    def test_checkpoint_dtype_is_per_layer_identity(self):
        """One layer's state dtype must not rename other PP stages' objects."""
        name = PoolName.DEEPSEEK_V4_C4_STATE
        root, group = _dsv4_plan_fixture(
            name=name, state_dtypes=(torch.float16, torch.float32)
        )
        complete = self._plans(root, group)[name]
        root, group = _dsv4_plan_fixture(
            name=name, layer_ids=(7,), capacity=9, state_dtypes=(torch.float32,)
        )
        self.assertEqual(complete.suffixes[1:], self._plans(root, group)[name].suffixes)
        root, group = _dsv4_plan_fixture(
            name=name, state_dtypes=(torch.float32, torch.float32)
        )
        changed = self._plans(root, group)[name]
        self.assertNotEqual(complete.suffixes[0], changed.suffixes[0])
        self.assertEqual(complete.suffixes[1], changed.suffixes[1])

    def test_drafts_and_unsupported_partitions_are_rejected(self):
        root, group = _dsv4_plan_fixture()
        for field in (
            "hicache_storage_head_partition",
            "hicache_storage_layer_partition",
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(NotImplementedError, "unset or 1"),
            ):
                self._plans(root, group, **{field: 2})
        group.entries[1].packed_draft_device_pools = (object(),)
        with self.assertRaisesRegex(NotImplementedError, "drafts"):
            self._plans(root, group)

    def test_invalid_chunk_coverage_is_rejected(self):
        for spans in (((0, 8), (9, 8)), ((0, 8), (7, 8)), ((0, 0), (0, 8))):
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                UnifiedPoolPlan(("a", "b"), spans)


class TestDSAUnifiedPoolPlans(CustomTestCase):
    def setUp(self):
        context = get_parallel().override(attn_cp_size=1, attn_dcp_size=1)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    @staticmethod
    def _fixture(
        start=0, end=78, *, capacity=64, dtype=torch.float8_e4m3fn, packed=True
    ):
        # GLM-5.2 shares top-k on all but 21 owner layers, including layer 39
        # at the PP2 boundary. Use the real provider to preserve that packing.
        owners = {0, 1, 2, *range(6, 78, 4)}
        root = DSATokenToKVPool(
            size=capacity,
            page_size=64,
            kv_lora_rank=512,
            dtype=dtype,
            qk_rope_head_dim=64,
            layer_num=end - start,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=656 if packed else 576,
            start_layer=start,
            end_layer=end,
            skip_topk_layers=[layer not in owners for layer in range(start, end)],
        )
        entries = []
        for info in root.get_device_pool_infos():
            host = SimpleNamespace(
                layout="page_unified", page_size=64, layer_num=len(info.layer_ids)
            )
            if info.pool_name == PoolName.KV:
                host.kv_cache_dim, host.dtype = root.kv_cache_dim, root.store_dtype
                host.page_unified_layout = PageUnifiedLayout(
                    page_size=64,
                    layer_num=root.layer_num,
                    head_num=1,
                    head_group_num=1,
                    head_dim=root.kv_cache_dim,
                    itemsize=root.store_dtype.itemsize,
                    is_mla=True,
                )
                host.storage_format_tag = None
            else:
                host.indexer_page_stride_size = info.buffer_info.page_bytes
            entries.append(
                SimpleNamespace(
                    name=info.pool_name, host_pool=host, packed_draft_device_pools=()
                )
            )
        return root, SimpleNamespace(entries=entries, layout="page_unified")

    @staticmethod
    def _plans(root, group, **partitions):
        with patch.object(
            hybrid_pool_assembler,
            "get_memory",
            return_value=SimpleNamespace(**partitions),
        ):
            return hybrid_pool_assembler.build_dsa_unified_pool_plans(
                "glm/test", root, group, 64
            )

    def test_sparse_global_owners_cover_pp_split_without_duplicate_consumer(self):
        """PP39 consumes owner38's top-k but owns no second copy of its index KV."""
        root, group = self._fixture()
        with get_parallel().override(tp_size=8, pp_size=1):
            full = self._plans(root, group)
        parts = []
        for start, end in ((0, 39), (39, 78)):
            root, group = self._fixture(start, end, capacity=128)
            with get_parallel().override(tp_size=4, pp_size=2):
                parts.append(self._plans(root, group))
        for name, count, layer_bytes in (
            (PoolName.KV, 78, 64 * 656),
            (PoolName.INDEXER, 21, 64 * 132),
        ):
            self.assertEqual(len(full[name].suffixes), count)
            self.assertEqual(
                full[name].suffixes, parts[0][name].suffixes + parts[1][name].suffixes
            )
            self.assertEqual(full[name].page_bytes, count * layer_bytes)
        self.assertTrue(parts[0][PoolName.INDEXER].suffixes[-1].endswith("_L38-39"))
        self.assertTrue(parts[1][PoolName.INDEXER].suffixes[0].endswith("_L42-43"))
        root, group = self._fixture(39, 42)
        self.assertEqual(set(self._plans(root, group)), {PoolName.KV})

    def test_native_formats_partition_equal_dtype_keyspaces(self):
        """Packed FN rows and raw FN rows require different keys at the same layer."""
        root, group = self._fixture(0, 3)
        packed = self._plans(root, group)
        root, group = self._fixture(0, 3, packed=False)
        raw = self._plans(root, group)
        self.assertNotEqual(packed[PoolName.KV].suffixes, raw[PoolName.KV].suffixes)
        self.assertEqual(packed[PoolName.KV].chunk_spans[0], (0, 64 * 656))
        self.assertEqual(raw[PoolName.KV].chunk_spans[0], (0, 64 * 576))
        root, group = self._fixture(0, 3, dtype=torch.float8_e5m2, packed=False)
        e5m2 = self._plans(root, group)[PoolName.KV]
        self.assertEqual(raw[PoolName.KV].chunk_spans, e5m2.chunk_spans)
        self.assertNotEqual(raw[PoolName.KV].suffixes, e5m2.suffixes)
        root, group = self._fixture(0, 3, dtype=torch.bfloat16, packed=False)
        self.assertEqual(
            self._plans(root, group)[PoolName.KV].chunk_spans[0], (0, 64 * 576 * 2)
        )

        variants = []
        for dtype, shuffled in (
            (torch.float8_e4m3fn, False),
            (torch.float8_e4m3fnuz, False),
            (torch.float8_e4m3fn, True),
        ):
            with (
                patch("sglang.kernels.ops.quantization.fp8_kernel.fp8_dtype", dtype),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.aiter_can_use_preshuffle_paged_mqa",
                    return_value=shuffled,
                ),
            ):
                variants.append(self._plans(root, group))
        self.assertEqual(len({p[PoolName.INDEXER].suffixes for p in variants}), 3)
        self.assertEqual(len({p[PoolName.KV].suffixes for p in variants}), 1)

    def test_unsupported_variants_cannot_silently_publish_partial_pages(self):
        root, group = self._fixture(0, 3)

        class UnsupportedDSAVariant(DSATokenToKVPool):
            pass

        with self.assertRaisesRegex(NotImplementedError, "ordinary DSATokenToKVPool"):
            self._plans(object.__new__(UnsupportedDSAVariant), group)
        for name in (
            "hicache_storage_head_partition",
            "hicache_storage_layer_partition",
        ):
            with (
                self.subTest(partition=name),
                self.assertRaisesRegex(NotImplementedError, "unset or 1"),
            ):
                self._plans(root, group, **{name: 2})
        for name, value in (
            ("index_kpool", 2),
            ("index_kpool_compress", True),
            ("layer_shard_enabled", True),
            ("dcp_replicated", True),
        ):
            old = getattr(root, name)
            with self.subTest(variant=name), self.assertRaises(NotImplementedError):
                setattr(root, name, value)
                self._plans(root, group)
            setattr(root, name, old)
        for name in ("attn_cp_size", "attn_dcp_size"):
            with (
                get_parallel().override(**{name: 2}),
                self.assertRaisesRegex(NotImplementedError, "CP or DCP"),
            ):
                self._plans(root, group)
        group.entries[1].packed_draft_device_pools = (object(),)
        with self.assertRaisesRegex(NotImplementedError, "drafts"):
            self._plans(root, group)
        group.entries.pop()
        with self.assertRaisesRegex(ValueError, "every native host pool"):
            self._plans(root, group)

    def test_controller_uses_native_plans_with_real_anchor_and_one_tp_writer(self):
        """The controller must bypass the dense single-pool gate for DSA only."""
        from sglang.srt.managers.cache_controller import HiCacheController

        root, group = self._fixture(39, 78)
        controller = object.__new__(HiCacheController)
        controller.mem_pool_device, controller.mem_pool_host = root, group
        controller.storage_host_pool = group.entries[0].host_pool
        controller.page_size, controller.enable_storage_metrics = 64, False
        parallel = SimpleNamespace(
            tp_rank=1,
            tp_size=4,
            pp_rank=1,
            pp_size=2,
            attn_cp_size=1,
            attn_dcp_size=1,
            attn_dcp_rank=0,
        )
        with (
            patch(
                "sglang.srt.managers.cache_controller.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.managers.cache_controller.get_memory",
                return_value=SimpleNamespace(hicache_storage_key_scheme="unified"),
            ),
            patch.object(
                hybrid_pool_assembler, "get_memory", return_value=SimpleNamespace()
            ),
            patch(
                "sglang.srt.managers.cache_controller.is_dp_attention_enabled",
                return_value=False,
            ),
            patch.object(controller, "get_attn_cp_rank_and_size", return_value=(0, 1)),
        ):
            config = controller._generate_storage_config("glm/test")
        self.assertFalse(config.is_storage_writer)
        self.assertIsNone(config.unified_suffixes)
        self.assertEqual(len(config.unified_pool_plans[PoolName.INDEXER].suffixes), 9)
        self.assertGreater(config.unified_pool_plans[PoolName.KV].page_bytes, 0)


class TestPageUnifiedLayout(CustomTestCase):
    def test_chunks_tile_the_page_exactly_once(self):
        """The grid's chunks must partition a page: no gap, no overlap.

        A gap is KV that is never stored; an overlap is two keys naming the
        same bytes, which diverge as soon as one is evicted.
        """
        layout = _mha(layer_num=6, head_num=3, head_group_num=3)
        covered = bytearray(layout.bytes_per_page)
        for head_group in range(layout.head_group_num):
            for start in range(0, layout.layer_num, 2):
                offset, length = layout.chunk_span((start, start + 2), head_group)
                for byte in range(offset, offset + length):
                    covered[byte] += 1
        self.assertEqual(set(covered), {1})

    def test_ragged_layer_window_is_still_one_run(self):
        """A short final window is one run, which is the point of the axis order.

        DeepSeek-V3's 61 layers at layer_partition=30 leave a width-1 tail. If
        that tail were several runs the transport would have to switch
        descriptor forms mid-plan.
        """
        layout = _mha(layer_num=61, head_num=4, head_group_num=1)
        spans = [layout.chunk_span(r) for r in [(0, 30), (30, 60), (60, 61)]]
        self.assertEqual(
            spans,
            [
                (0, 30 * layout.layer_bytes),
                (30 * layout.layer_bytes, 30 * layout.layer_bytes),
                (60 * layout.layer_bytes, layout.layer_bytes),
            ],
        )
        self.assertEqual(sum(length for _, length in spans), layout.bytes_per_page)

    def test_unaligned_group_row_is_rejected(self):
        """The transfer kernels move 16-byte vectors and cannot express a tail."""
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            _mha(head_num=2, head_group_num=2, head_dim=7, itemsize=1)

    def test_head_group_must_divide_the_pools_heads(self):
        with self.assertRaisesRegex(ValueError, "must divide"):
            PageUnifiedLayout(
                page_size=4,
                layer_num=2,
                head_num=6,
                head_group_num=4,
                head_dim=16,
                itemsize=2,
            )

    def test_chunk_span_rejects_out_of_range_coordinates(self):
        layout = _mha()
        with self.assertRaisesRegex(ValueError, "layer range"):
            layout.chunk_span((0, layout.layer_num + 1))
        with self.assertRaisesRegex(ValueError, "layer range"):
            layout.chunk_span((3, 3))
        with self.assertRaisesRegex(ValueError, "head group"):
            layout.chunk_span((0, 1), layout.head_group_num)


# GQA-70B shape: 8 kv heads, 80 layers.
BASE = dict(
    model_id="meta-llama/Llama-3.1-70B",
    dtype="bfloat16",
    page_size=64,
    rank_replicated=False,
    attn_cp_size=1,
    start_layer=0,
    end_layer=80,
    is_final_stage=True,
)


def _plan(**overrides):
    kwargs = dict(BASE)
    kwargs.update(overrides)
    return plan_unified_kv(**kwargs)


class TestUnifiedKeyGrid(CustomTestCase):
    def test_tp_shards_of_one_grid_cover_each_other(self):
        """TP2's objects are exactly the union of TP4's, in order.

        This is the whole point of the scheme: a cache written by one topology
        must be readable by another. Nothing else in the stack checks it, and
        the failure mode is a silent 0% hit rate.
        """
        tp2 = [
            _plan(local_kv_heads=4, attn_tp_rank=r, attn_tp_size=2, head_group=2)
            for r in range(2)
        ]
        tp4 = [
            _plan(local_kv_heads=2, attn_tp_rank=r, attn_tp_size=4, head_group=2)
            for r in range(4)
        ]
        self.assertEqual(
            [s for p in tp2 for s in p.suffixes],
            [s for p in tp4 for s in p.suffixes],
        )

    def test_pp_stages_cover_the_unsplit_range(self):
        """Same property along the layer axis, including a ragged tail."""
        whole = _plan(
            local_kv_heads=8,
            attn_tp_rank=0,
            attn_tp_size=1,
            head_group=8,
            layer_partition=30,
            end_layer=61,
        )
        stage0 = _plan(
            local_kv_heads=8,
            attn_tp_rank=0,
            attn_tp_size=1,
            head_group=8,
            layer_partition=30,
            start_layer=0,
            end_layer=30,
            is_final_stage=False,
        )
        stage1 = _plan(
            local_kv_heads=8,
            attn_tp_rank=0,
            attn_tp_size=1,
            head_group=8,
            layer_partition=30,
            start_layer=30,
            end_layer=61,
        )
        self.assertEqual(whole.suffixes, stage0.suffixes + stage1.suffixes)
        # The 61st layer forms a short trailing chunk rather than being dropped.
        self.assertTrue(whole.suffixes[-1].endswith("_L60-61_H0"))

    def test_suffix_order_matches_the_range_cross_product(self):
        """Layer-major / head-minor, and the ranges pair with it positionally.

        The storage layer zips suffixes against layer_ranges x head_ranges to
        find each object's bytes; a reordering here writes every chunk under
        its neighbour's key.
        """
        plan = _plan(
            local_kv_heads=4,
            attn_tp_rank=0,
            attn_tp_size=2,
            head_group=2,
            layer_partition=40,
        )
        self.assertEqual(len(plan.layer_ranges), 2)
        self.assertEqual(plan.head_ranges, [(0, 2), (2, 4)])
        digest = namespace_digest(plan.namespace)
        self.assertEqual(
            plan.suffixes,
            [
                f"{digest}_L0-40_H0",
                f"{digest}_L0-40_H1",
                f"{digest}_L40-80_H0",
                f"{digest}_L40-80_H1",
            ],
        )

    def test_head_index_is_global_not_rank_local(self):
        """Rank 1's groups must not collide with rank 0's."""
        plans = [
            _plan(local_kv_heads=2, attn_tp_rank=r, attn_tp_size=4, head_group=2)
            for r in range(4)
        ]
        self.assertEqual(
            [p.suffixes[0].rsplit("_", 1)[1] for p in plans],
            ["H0", "H1", "H2", "H3"],
        )

    def test_local_ranges_are_pool_relative(self):
        """A PP stage indexes its own pool, so its ranges start at 0."""
        stage1 = _plan(
            local_kv_heads=8,
            attn_tp_rank=0,
            attn_tp_size=1,
            head_group=8,
            layer_partition=20,
            start_layer=40,
            end_layer=80,
        )
        self.assertEqual(stage1.layer_ranges, [(0, 20), (20, 40)])
        self.assertTrue(stage1.suffixes[0].endswith("_L40-60_H0"))

    def test_attention_cp_preserves_head_shards_and_namespace(self):
        for rank_replicated in (False, True):
            for tp_rank in range(2):
                with self.subTest(rank_replicated=rank_replicated, tp_rank=tp_rank):
                    shard = dict(
                        rank_replicated=rank_replicated,
                        local_kv_heads=4,
                        attn_tp_rank=tp_rank,
                        attn_tp_size=2,
                        head_group=4,
                    )
                    self.assertEqual(
                        _plan(**shard, attn_cp_size=4),
                        _plan(**shard, attn_cp_size=1),
                    )


class TestNamespaceIdentity(CustomTestCase):
    def test_identity_fields_partition_the_keyspace(self):
        """Every identity field must change the digest.

        A field that does not reach the digest lets two deployments with
        different bytes collide on one key, which is corruption rather than a
        miss.
        """
        base = dict(
            model_id="m",
            dtype="bfloat16",
            page_size=64,
            rank_replicated=False,
            total_kv_heads=8,
            head_group=2,
        )
        digests = {namespace_digest(derive_namespace(**base))}
        for field, value in [
            ("model_id", "other"),
            ("dtype", "float16"),
            ("page_size", 32),
            ("total_kv_heads", 16),
            ("head_group", 4),
            ("layer_partition", 30),
            ("object_layout", "something-else"),
        ]:
            digests.add(namespace_digest(derive_namespace(**{**base, field: value})))
        self.assertEqual(len(digests), 8)

    def test_fp8_variants_do_not_share_a_keyspace(self):
        """Both store as uint8, so only the logical dtype separates them."""
        self.assertNotEqual(
            namespace_digest(
                _plan(
                    local_kv_heads=8,
                    attn_tp_rank=0,
                    attn_tp_size=1,
                    head_group=8,
                    dtype="float8_e4m3fn",
                ).namespace
            ),
            namespace_digest(
                _plan(
                    local_kv_heads=8,
                    attn_tp_rank=0,
                    attn_tp_size=1,
                    head_group=8,
                    dtype="float8_e5m2",
                ).namespace
            ),
        )

    def test_rank_replicated_drops_the_head_axis(self):
        plan = _plan(
            rank_replicated=True,
            local_kv_heads=0,
            attn_tp_rank=3,
            attn_tp_size=8,
        )
        self.assertEqual(plan.head_ranges, [(0, 1)])
        self.assertNotIn("_H", plan.suffixes[0])
        # Every rank derives the same keys, which is what makes one writer enough.
        other = _plan(
            rank_replicated=True, local_kv_heads=0, attn_tp_rank=0, attn_tp_size=8
        )
        self.assertEqual(plan.suffixes, other.suffixes)

    def test_normalize_dtype_strips_the_torch_prefix(self):
        self.assertEqual(normalize_dtype(torch.bfloat16), "bfloat16")


class TestUnifiedKeySchemeRejections(CustomTestCase):
    def test_head_group_must_tile_this_ranks_heads(self):
        with self.assertRaisesRegex(ValueError, "must divide this rank's"):
            _plan(local_kv_heads=3, attn_tp_rank=0, attn_tp_size=2, head_group=2)

    def test_misaligned_stage_start_is_rejected(self):
        """A stage starting off-grid would name chunks no reader can assemble."""
        with self.assertRaisesRegex(ValueError, "does not start on a multiple"):
            _plan(
                local_kv_heads=8,
                attn_tp_rank=0,
                attn_tp_size=1,
                head_group=8,
                layer_partition=30,
                start_layer=20,
                end_layer=50,
            )

    def test_non_final_stage_may_not_end_short(self):
        with self.assertRaisesRegex(ValueError, "only the FINAL pipeline stage"):
            _plan(
                local_kv_heads=8,
                attn_tp_rank=0,
                attn_tp_size=1,
                head_group=8,
                layer_partition=30,
                start_layer=0,
                end_layer=50,
                is_final_stage=False,
            )

    def test_one_kv_head_per_rank_needs_an_explicit_head_group(self):
        """1 head per rank is ambiguous: sharded, or replicated across ranks?

        Replication would have several ranks racing to write one key, so the
        operator has to attest which it is.
        """
        with self.assertRaises(NotImplementedError):
            _plan(local_kv_heads=1, attn_tp_rank=0, attn_tp_size=8)

    def test_empty_model_id_is_refused(self):
        with self.assertRaisesRegex(ValueError, "non-empty model_id"):
            _plan(
                model_id="",
                local_kv_heads=8,
                attn_tp_rank=0,
                attn_tp_size=1,
                head_group=8,
            )


if __name__ == "__main__":
    unittest.main()
