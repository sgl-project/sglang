"""
Unit tests for the DeepSeek-V4.1 low-ratio (NPU) path in
sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend and the row layout
helpers in ...hardware_backend.npu.dsv4.dsv4_memory_pool.

Everything here is pure device-agnostic torch arithmetic plus a stub for the
``ds41`` operator module, so the suite runs on CPU without an Ascend device.
"""

import sys
import unittest
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=8, suite="base-a-test-npu")

for mod in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.attention",
    "sgl_kernel_npu.attention.sinks_attention",
    "sgl_kernel_npu.norm",
    "sgl_kernel_npu.norm.add_rmsnorm_bias",
    "sglang.srt.speculative",
    "sglang.srt.speculative.decoupled_spec_io",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.speculative.eagle_info",
    "sglang.srt.speculative.base_spec_worker",
):
    sys.modules.setdefault(mod, MagicMock())

# Stub deepseek_v2._is_hip to avoid importing the heavy model.
_ds2_stub = ModuleType("sglang.srt.models.deepseek_v2")
_ds2_stub._is_hip = False
sys.modules.setdefault("sglang.srt.models.deepseek_v2", _ds2_stub)

import sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend as dsv4_backend
from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
    DeepseekV4AscendAttnBackend,
    LowRatioCompressAscendBackendMixin,
)
from sglang.srt.hardware_backend.npu.dsv4 import dsv4_memory_pool
from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
    DSV4NPUTokenToKVPool,
    DSV41NPUSingleKVPool,
    dsv41_quant,
    dsv41_row_layout,
)
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DSV41_INDEX_PAGE_SIZE,
    get_dsv4_indexer_bytes_per_token,
)
from sglang.srt.mem_cache.registry import create_unified_radix_cache


class _Ds41Stub(ModuleType):
    """Stand-in for ``cann_ops_transformer.ops.ds41``.

    Only the two metadata entry points are exercised by these tests; both return
    a real int32 tensor so the graph-buffer ``torch.zeros_like`` probe works.
    """

    def __init__(self):
        super().__init__("ds41_stub")
        self.mixed_quant_sparse_flash_mla = MagicMock(return_value=(None, None))
        self.quant_lightning_indexer = MagicMock(return_value=(None, None, None, None))
        self.compressor = MagicMock()
        self.indexer_prologue_k = MagicMock()

    @staticmethod
    def mixed_quant_sparse_flash_mla_metadata(*args, **kwargs):
        # The operator returns (2 * cube_core_num,) int32.
        return torch.zeros(48, dtype=torch.int32)

    @staticmethod
    def quant_lightning_indexer_metadata(*args, **kwargs):
        # QLI metadata is a fixed 1024 int32 (QLI README).
        return torch.zeros(1024, dtype=torch.int32)


DS41 = _Ds41Stub()
# ``_ds41_ops`` is a module-level cached accessor; replacing the module global is
# enough because every call site resolves it at call time.
dsv4_backend._ds41_ops = lambda: DS41


def _low_ratio_backend(*, ratios=(1, 2), window=128):
    backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
    backend._dsv4_compress_ratios = (0,) + tuple(ratios)
    backend._dsv4_unique_compress_ratios = list(
        dict.fromkeys(backend._dsv4_compress_ratios)
    )
    backend._dsv4_has_low_ratio = any(
        r in (1, 2) for r in backend._dsv4_compress_ratios
    )
    backend._dsv4_sliding_window_size = window
    backend._dsv4_q_head_num = 64
    backend._dsv4_head_dim = 512
    backend._dsv4_index_topk = 512
    backend._dsv4_index_n_heads = 32
    backend._dsv4_index_head_dim = 128
    backend.page_size = 128
    backend.forward_metadata = SimpleNamespace()
    backend.graph_metadata = {}
    # The real pool (DSV4NPUTokenToKVPool) exposes qk_nope_head_dim /
    # qk_rope_head_dim but *no* ``head_dim``: the attention width is the
    # backend's own cfg head_dim.  Keep the stub bare so a regression to
    # ``pool.head_dim`` fails here (AttributeError) instead of on the A5 box.
    backend.token_to_kv_pool = SimpleNamespace()
    return backend


def _batch(*, positions, out_cache_loc, seq_lens, batch_size):
    return SimpleNamespace(
        positions=torch.tensor(positions, dtype=torch.int64),
        out_cache_loc=torch.tensor(out_cache_loc, dtype=torch.int64),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        batch_size=batch_size,
    )


class TestRowLayout(unittest.TestCase):
    """Row widths must match the operators' documented byte layouts.

    The format follows the *side* of the attention, not the layer's compress
    ratio: a ratio-1 aggregate is still read as ``cmp_kv``.
    """

    def test_window_row_is_the_origin_544b_layout(self):
        # mixed_quant_sparse_flash_mla_dsl/README.md L188: ori_kv is
        # nope[448] + rope[64] = 512 bytes of FP8 E4M3 plus 16 BF16 scales.
        row_bytes, dtype = dsv41_row_layout(512, compressed=False)
        self.assertEqual(row_bytes, 544)
        self.assertEqual(dtype, torch.float8_e4m3fn)

    def test_compressed_row_is_the_320b_layout_for_both_ratios(self):
        # README L189: cmp_kv is 256 bytes of packed FP4 E2M1 plus 32 BF16
        # scales, whatever the layer's ratio (recipes
        # common_modules.get_kv_cache_dim(head_dim, is_compressed=True) == 320).
        row_bytes, dtype = dsv41_row_layout(512, compressed=True)
        self.assertEqual(row_bytes, 320)
        self.assertEqual(dtype, torch.uint8)

    def test_quant_modes_match_the_two_sides(self):
        self.assertEqual(dsv41_quant(False), ("mxfp8_bf16", 32))
        self.assertEqual(dsv41_quant(True), ("mxfp4_bf16", 16))

    def test_index_row_is_fp4_packed_pairs(self):
        # index_head_dim // 2 + index_head_dim // 32 == 64 + 4 == 68, the same
        # number the CUDA pool class reports.
        self.assertEqual(get_dsv4_indexer_bytes_per_token(128, True), 68)

    def test_empty_head_dim_is_rejected(self):
        with self.assertRaises(AssertionError):
            dsv41_row_layout(0, compressed=True)


class TestPoolFactory(unittest.TestCase):
    """The V4.1 factory picks the row format from the pool's role.

    The window pool and the ratio-1 compressed pool arrive with the same page
    geometry, so keying the format off the ratio would silently give the
    ratio-1 compressed pool the window's 544-byte fp8 rows while the ds41
    attention op reads ``cmp_kv`` as 320-byte fp4.
    """

    def _make(self, *, compression_ratios, page_size, global_page_size, compressed):
        import sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool as pool_mod

        calls = {}

        def record(name):
            def factory(*args, **kwargs):
                calls[name] = kwargs
                return name

            return factory

        fake_self = SimpleNamespace(
            compression_ratios=compression_ratios,
            c128_page_size=128,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
        )
        with (
            patch.object(
                pool_mod, "DSV41NPUSingleKVPool", record("DSV41NPUSingleKVPool")
            ),
            patch.object(
                pool_mod,
                "NPUDeepSeekV4SingleKVPool",
                record("NPUDeepSeekV4SingleKVPool"),
            ),
        ):
            out = pool_mod.DSV4NPUTokenToKVPool._make_kv_pool(
                fake_self,
                size=1024,
                page_size=page_size,
                dtype=torch.bfloat16,
                layer_num=4,
                device="cpu",
                enable_memory_saver=False,
                global_page_size=global_page_size,
                compressed=compressed,
            )
        return out, calls

    def test_window_pool_is_the_fp8_side(self):
        out, calls = self._make(
            compression_ratios=(0, 1, 2),
            page_size=128,
            global_page_size=128,
            compressed=False,
        )
        self.assertEqual(out, "DSV41NPUSingleKVPool")
        self.assertIs(calls["DSV41NPUSingleKVPool"]["is_compressed"], False)
        # One row per slot: the window page is the global page.
        self.assertEqual(calls["DSV41NPUSingleKVPool"]["kernel_page_size"], 128)

    def test_ratio1_compressed_pool_is_fp4_despite_the_global_page(self):
        out, calls = self._make(
            compression_ratios=(0, 1, 2),
            page_size=128,
            global_page_size=128,
            compressed=True,
        )
        self.assertEqual(out, "DSV41NPUSingleKVPool")
        self.assertIs(calls["DSV41NPUSingleKVPool"]["is_compressed"], True)

    def test_ratio2_compressed_pool_is_fp4(self):
        out, calls = self._make(
            compression_ratios=(0, 1, 2),
            page_size=64,
            global_page_size=128,
            compressed=True,
        )
        self.assertEqual(out, "DSV41NPUSingleKVPool")
        self.assertIs(calls["DSV41NPUSingleKVPool"]["is_compressed"], True)

    def test_v4_model_keeps_the_packed_fp8_pool(self):
        out, calls = self._make(
            compression_ratios=(0, 4, 128),
            page_size=32,
            global_page_size=128,
            compressed=True,
        )
        self.assertEqual(out, "NPUDeepSeekV4SingleKVPool")
        self.assertNotIn("DSV41NPUSingleKVPool", calls)


class TestPairStatePool(unittest.TestCase):
    """The ratio-2 pending-pair state must expose the physical bank the fused
    ``ds41.compressor`` reads as ``state_cache``.

    The community factory allocates the same ring but keeps the flat
    ``page_size=1`` default, on which ``state_cache_3d`` raises; the A5 CYCLE
    ABI addresses one request bank per ``state_block_table`` row, so the NPU
    override has to turn the second-axis page size on.
    """

    def _make(self):
        import sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool as pool_mod

        fake_self = SimpleNamespace(
            num_req_slots=5,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            device="cpu",
            get_ring_size=lambda ratio: 2,
        )
        return pool_mod.DSV4NPUTokenToKVPool._make_pair_state_pool(
            fake_self, enable_memory_saver=False
        )

    def test_pair_state_is_paged_for_the_state_cache_view(self):
        pool = self._make()
        self.assertEqual(pool.ratio, 2)
        self.assertEqual(pool.page_size, 2)
        cache = pool.state_cache_3d  # would raise on the flat default
        self.assertEqual(cache.shape[1], 2)
        self.assertEqual(cache.shape[2], 2 * (448 + 64))
        # One request bank per row: every req_pool_idx is addressable.
        self.assertGreaterEqual(cache.shape[0], 5)

    def test_pair_state_row_is_the_kv_score_pair(self):
        pool = self._make()
        # score (second half) is -inf-initialized; kv is zeroed.
        cache = pool.state_cache_3d
        self.assertTrue(torch.isneginf(cache[0, :, -1]).all())
        self.assertTrue((cache[0, :, 0] == 0).all())


class TestIndexerBufferAccessors(unittest.TestCase):
    """Index K and its scale live on the per-ratio index pool.

    ``get_index_k``/``get_index_scale`` are defined by ``DSV41NPUIndexerPool``;
    the token-to-KV pool has to resolve the layer's *ratio* first and then pass
    the per-ratio ``compress_layer_id``. Reading them straight off the
    token-to-KV pool (or passing a ``compress_layer_id`` where a ``layer_id`` is
    expected) is an AttributeError / wrong-buffer bug.
    """

    def _fake(self, mapping, pools):
        return SimpleNamespace(
            layer_mapping=mapping,
            wait_layer_transfer=lambda _lid: None,
            _indexer_pool=lambda ratio: pools[ratio],
        )

    def test_token_to_kv_pool_exposes_both_accessors(self):
        import sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool as pool_mod

        for name in ("get_index_k", "get_index_scale"):
            self.assertTrue(hasattr(pool_mod.DSV4NPUTokenToKVPool, name), name)

    def test_routes_by_ratio_and_passes_the_per_ratio_index(self):
        import sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool as pool_mod

        seen = {}
        k_buf = torch.zeros((2, 64, 1, 64), dtype=torch.uint8)
        s_buf = torch.zeros((2, 64, 1, 4), dtype=torch.uint8)

        def serve(tag, buf):
            def fn(compress_layer_id):
                seen[tag] = compress_layer_id
                return buf

            return fn

        idx_pool = SimpleNamespace(
            get_index_k=serve("k_cid", k_buf),
            get_index_scale=serve("s_cid", s_buf),
        )
        mapping = [
            SimpleNamespace(compress_ratio=1, compress_layer_id=0),
            # Ratio 2's source is also index 0 of ITS ratio's pool.
            SimpleNamespace(compress_ratio=2, compress_layer_id=0),
        ]
        fake = self._fake(mapping, {1: idx_pool, 2: idx_pool})
        cls = pool_mod.DSV4NPUTokenToKVPool

        self.assertIs(cls.get_index_k(fake, 1), k_buf)
        self.assertIs(cls.get_index_scale(fake, 0), s_buf)
        self.assertEqual(seen["k_cid"], 0)
        self.assertEqual(seen["s_cid"], 0)


class TestPdBufferEnumeration(unittest.TestCase):
    """``get_contiguous_buf_infos`` must describe the V4.1 pools, not nothing.

    V4.1 has no C4 pool, so keying the description off ``c4_kv_pool`` returned
    an empty main component: a PD peer would register no compressed KV and
    transfer nothing. V4.1's main component is the per-ratio compressed KV
    followed by that ratio's index K then index scale, one item per full page
    (the index pools page at ``DSV41_INDEX_PAGE_SIZE``, so their item is the run
    of index pages tiling a full page).
    """

    def _pool(self, *, ratios, page_size, kv_layers=2, index_layers=1):
        import sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool as pool_mod

        kv_pools = {}
        index_pools = {}
        for ratio in ratios:
            kv_pools[ratio] = SimpleNamespace(
                kv_buffer=[
                    torch.zeros((4, page_size // ratio, 1, 320), dtype=torch.uint8)
                    for _ in range(kv_layers)
                ]
            )
            index_pools[ratio] = SimpleNamespace(
                index_k_buffer=[
                    torch.zeros((8, DSV41_INDEX_PAGE_SIZE, 1, 64), dtype=torch.uint8)
                    for _ in range(index_layers)
                ],
                index_scale_buffer=[
                    torch.zeros((8, DSV41_INDEX_PAGE_SIZE, 1, 4), dtype=torch.uint8)
                    for _ in range(index_layers)
                ],
                page_size=DSV41_INDEX_PAGE_SIZE,
            )
        fake = SimpleNamespace(
            c4_kv_pool=None,
            c128_kv_pool=None,
            kv_pools=kv_pools,
            page_size=page_size,
            _indexer_pool=lambda ratio: index_pools[ratio],
        )
        return (
            pool_mod.DSV4NPUTokenToKVPool.get_contiguous_buf_infos(fake),
            kv_pools,
            index_pools,
        )

    def test_c128_scan_of_a_v41_pool_is_empty_instead_of_raising(self):
        import sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool as pool_mod

        fake = SimpleNamespace(c128_kv_pool=None)
        self.assertEqual(
            pool_mod.DSV4NPUTokenToKVPool.get_c128_kv_buf_infos(fake),
            ([], [], []),
        )

    def test_each_ratio_registers_its_kv_then_index_k_then_index_scale(self):
        page_size = 256
        (ptrs, lens, items), kv_pools, index_pools = self._pool(
            ratios=(1,), page_size=page_size, kv_layers=2, index_layers=1
        )
        index_pages = page_size // DSV41_INDEX_PAGE_SIZE
        self.assertEqual(len(ptrs), 4)
        # Ratio-1 rows are one full page wide, so its compressed KV item is one
        # physical page; the index pool pages at DSV41_INDEX_PAGE_SIZE, so its
        # item is the run of index pages tiling a full page (R10).
        self.assertEqual(
            items,
            [
                page_size * 320,
                page_size * 320,
                DSV41_INDEX_PAGE_SIZE * 64 * index_pages,
                DSV41_INDEX_PAGE_SIZE * 4 * index_pages,
            ],
        )
        for buf, ptr in zip(
            (
                *kv_pools[1].kv_buffer,
                *index_pools[1].index_k_buffer,
                *index_pools[1].index_scale_buffer,
            ),
            ptrs,
        ):
            self.assertEqual(ptr, buf.data_ptr())
        self.assertEqual(lens[0], kv_pools[1].kv_buffer[0].nbytes)

    def test_ratio_two_pages_at_half_the_full_page(self):
        page_size = 256
        (_, _, items), _, _ = self._pool(
            ratios=(2,), page_size=page_size, kv_layers=1, index_layers=1
        )
        index_pages = (page_size // 2) // DSV41_INDEX_PAGE_SIZE
        self.assertEqual(
            items,
            [
                (page_size // 2) * 320,
                DSV41_INDEX_PAGE_SIZE * 64 * index_pages,
                DSV41_INDEX_PAGE_SIZE * 4 * index_pages,
            ],
        )

    def test_both_ratios_are_walked_in_the_fixed_low_ratio_order(self):
        page_size = 256
        (ptrs, _, items), _, _ = self._pool(
            ratios=(1, 2), page_size=page_size, kv_layers=1, index_layers=1
        )
        self.assertEqual(len(ptrs), 6)
        # Ratio 1 first (KV, K, scale), then ratio 2 -- the order both PD peers
        # derive from the shared compression config, whatever order the model
        # declares the layers in.
        one = page_size // DSV41_INDEX_PAGE_SIZE
        two = (page_size // 2) // DSV41_INDEX_PAGE_SIZE
        self.assertEqual(
            items[:3],
            [
                page_size * 320,
                DSV41_INDEX_PAGE_SIZE * 64 * one,
                DSV41_INDEX_PAGE_SIZE * 4 * one,
            ],
        )
        self.assertEqual(
            items[3:],
            [
                (page_size // 2) * 320,
                DSV41_INDEX_PAGE_SIZE * 64 * two,
                DSV41_INDEX_PAGE_SIZE * 4 * two,
            ],
        )

    def test_a_stage_without_low_ratios_describes_nothing(self):
        (ptrs, lens, items), _, _ = self._pool(ratios=(), page_size=128)
        self.assertEqual((ptrs, lens, items), ([], [], []))


class TestCompact1d(unittest.TestCase):
    def test_dense_front_with_custom_fill(self):
        dst = torch.full((5,), 99, dtype=torch.int32)
        values = torch.tensor([10, 20, 30], dtype=torch.int64)
        keep = torch.tensor([True, False, True], dtype=torch.bool)
        LowRatioCompressAscendBackendMixin._dsv41_compact_1d(dst, values, keep, fill=-1)
        self.assertEqual(dst.tolist(), [10, 30, -1, -1, -1])

    def test_overflow_rows_are_dropped_instead_of_raised(self):
        dst = torch.full((2,), 7, dtype=torch.int32)
        values = torch.tensor([1, 2, 3], dtype=torch.int64)
        keep = torch.tensor([True, True, True])
        LowRatioCompressAscendBackendMixin._dsv41_compact_1d(dst, values, keep, fill=0)
        self.assertEqual(dst.tolist(), [1, 2])

    def test_size_mismatch_is_rejected(self):
        dst = torch.zeros(3, dtype=torch.int32)
        with self.assertRaises(ValueError):
            LowRatioCompressAscendBackendMixin._dsv41_compact_1d(
                dst, torch.tensor([1, 2]), torch.tensor([True]), fill=0
            )


class TestWindowIndices(unittest.TestCase):
    def test_causal_window_is_left_clamped_and_none_padded(self):
        backend = _low_ratio_backend(window=4)
        batch = _batch(
            positions=[0, 1, 5],
            out_cache_loc=[0, 0, 0],
            seq_lens=[1, 2, 6],
            batch_size=3,
        )
        indices, lengths = backend._dsv41_window_indices(batch)
        self.assertEqual(indices.shape, (3, 1, 4))
        self.assertEqual(indices.dtype, torch.int32)
        # position 0 -> [0, -1, -1, -1]; position 1 -> [0, 1, -1, -1];
        # position 5 -> [2, 3, 4, 5].
        self.assertEqual(
            indices[:, 0, :].tolist(), [[0, -1, -1, -1], [0, 1, -1, -1], [2, 3, 4, 5]]
        )
        self.assertEqual(lengths[:, 0].tolist(), [1, 2, 4])

    def test_buffers_are_reused_when_the_row_count_matches(self):
        backend = _low_ratio_backend(window=4)
        batch = _batch(
            positions=[0, 1], out_cache_loc=[0, 0], seq_lens=[1, 2], batch_size=2
        )
        first, lengths_first = backend._dsv41_window_indices(batch)
        second, lengths_second = backend._dsv41_window_indices(batch)
        self.assertIs(first, second)
        self.assertIs(lengths_first, lengths_second)


class TestCompressedTopkLength(unittest.TestCase):
    def test_lengths_follow_the_compressed_position_and_are_capped(self):
        backend = _low_ratio_backend()
        backend._dsv4_index_topk = 3
        batch = _batch(
            positions=[1, 5, 20],
            out_cache_loc=[0, 0, 0],
            seq_lens=[2, 6, 21],
            batch_size=3,
        )
        self.assertEqual(
            backend._dsv41_cmp_topk_length(batch, 2)[:, 0].tolist(), [1, 3, 3]
        )
        self.assertEqual(
            backend._dsv41_cmp_topk_length(batch, 1)[:, 0].tolist(), [2, 3, 3]
        )


class TestRatioMetadata(unittest.TestCase):
    """``_build_dsv41_ratio_metadata`` derives slots as ``full_loc // ratio``."""

    def _build(self, ratio, batch, block_tables=None):
        backend = _low_ratio_backend()
        backend.forward_metadata.block_tables = (
            torch.zeros((batch.batch_size, 2), dtype=torch.int32)
            if block_tables is None
            else block_tables
        )
        backend._build_dsv41_ratio_metadata(batch, ratio)
        return backend.forward_metadata

    def test_ratio2_prefill_compacts_group_closers(self):
        # positions 0..5 all live (one request), out slots >= one page so the
        # reserved page-0 filter keeps them.
        batch = _batch(
            positions=[0, 1, 2, 3, 4, 5],
            out_cache_loc=[200, 201, 202, 203, 204, 205],
            seq_lens=[6],
            batch_size=1,
        )
        fm = self._build(2, batch)
        # Rows exist for the group-closing tokens (odd positions).
        self.assertEqual(fm.c2_loc.tolist(), [100, 101, 102, -1])
        self.assertEqual(fm.positions_cmp_padding_c2.tolist(), [0, 2, 4, 0])
        # seqused_k / cmp_residual_k are derived from seq_lens where they are
        # consumed (eager and graph replay alike), so no per-step metadata copy
        # of the compressed length exists any more.
        self.assertFalse(hasattr(fm, "dsv41_cmp_seq_lens_c2"))
        self.assertFalse(hasattr(fm, "dsv41_cmp_seq_remainders_c2"))

    def test_ratio2_decode_keeps_a_per_request_row(self):
        # Two requests decoding one token each; only request 0 closes a pair.
        batch = _batch(
            positions=[3, 4], out_cache_loc=[205, 208], seq_lens=[4, 5], batch_size=2
        )
        fm = self._build(2, batch)
        self.assertEqual(fm.c2_loc.tolist(), [102, -1])
        self.assertEqual(fm.positions_cmp_padding_c2.tolist(), [2, 0])

    def test_ratio1_keeps_per_token_rows(self):
        # ratio 1 is a plain projection (no fused compressor), so rows must stay
        # per-token: slots are the full slots themselves.
        batch = _batch(
            positions=[0, 1, 2],
            out_cache_loc=[200, 201, 202],
            seq_lens=[3],
            batch_size=1,
        )
        fm = self._build(1, batch)
        self.assertEqual(fm.c1_loc.tolist(), [200, 201, 202])
        self.assertEqual(fm.positions_cmp_padding_c1.tolist(), [0, 1, 2])

    def test_padding_rows_are_skipped(self):
        # out_cache_loc 0 marks a DP padding row; it must never become a slot.
        batch = _batch(
            positions=[0, 0], out_cache_loc=[200, 0], seq_lens=[1, 0], batch_size=2
        )
        fm = self._build(1, batch)
        self.assertEqual(fm.c1_loc.tolist(), [200, -1])

    def test_ratio1_keeps_token_order_when_a_row_is_dropped(self):
        # ratio 1's latent is the plain wkv projection, so row i is token i:
        # the slot row must stay in token order (a gap becomes -1 in place).
        # Compacting would shift the following rows onto the wrong latent row.
        batch = _batch(
            positions=[0, 1, 2],
            out_cache_loc=[200, 0, 202],
            seq_lens=[1, 0, 3],
            batch_size=3,
        )
        fm = self._build(1, batch)
        self.assertEqual(fm.c1_loc.tolist(), [200, -1, 202])
        self.assertEqual(fm.positions_cmp_padding_c1.tolist(), [0, 0, 2])

    def test_compressed_page_table_aliases_the_full_one(self):
        block_tables = torch.tensor([[7, 8]], dtype=torch.int32)
        batch = _batch(positions=[1], out_cache_loc=[201], seq_lens=[2], batch_size=1)
        fm = self._build(2, batch, block_tables=block_tables)
        self.assertIs(fm.c2_page_table, block_tables)
        # (page_size // ratio) // DSV41_INDEX_PAGE_SIZE index pages per full page.
        pages_per_full = (128 // 2) // DSV41_INDEX_PAGE_SIZE
        self.assertEqual(pages_per_full, 1)
        self.assertIs(fm.c2_index_page_table, block_tables)

    def test_index_page_table_expands_to_the_index_pool_page(self):
        # The indexer-K pool pages at DSV41_INDEX_PAGE_SIZE slots while a full
        # page holds page_size // ratio, so full page p owns the CONSECUTIVE
        # index pages [k*p, k*p + k) -- the same mapping CUDA builds with
        # expand_index_page_table. repeating the full id would point every index
        # page at the same physical page and score the wrong K.
        block_tables = torch.tensor([[7, 8]], dtype=torch.int32)
        batch = _batch(positions=[1], out_cache_loc=[201], seq_lens=[2], batch_size=1)
        fm = self._build(1, batch, block_tables=block_tables)
        blocks_per_page = (128 // 1) // DSV41_INDEX_PAGE_SIZE
        self.assertEqual(blocks_per_page, 2)
        self.assertEqual(fm.c1_index_page_table.tolist(), [[14, 15, 16, 17]])


class TestGraphBuffers(unittest.TestCase):
    def test_init_allocates_stable_buffers_and_dispatch_tables(self):
        backend = _low_ratio_backend()
        metadata = SimpleNamespace(block_tables=torch.zeros((4, 2), dtype=torch.int32))
        metadata.kernel_metadata = {}
        backend._init_dsv41_graph_metadata(
            metadata, bs=4, tokens_per_req=1, device="cpu"
        )

        # Decode bucket: one row per request, tail filled with the skip sentinel.
        self.assertEqual(metadata.c1_loc.shape, (4,))
        self.assertTrue(torch.all(metadata.c1_loc == -1))
        self.assertEqual(metadata.positions_cmp_padding_c1.shape, (4,))
        self.assertEqual(metadata.dsv41_win_sparse_indices.shape, (4, 1, 128))
        self.assertEqual(metadata.dsv41_win_topk_length.shape, (4, 1))
        self.assertEqual(metadata.dsv41_cmp_topk_length_c2.shape, (4, 1))
        # MQSMLA metadata is (2 * cube cores,), not the V4 fixed 1024.
        self.assertEqual(metadata.kernel_metadata["win_metadata"].shape, (48,))
        self.assertEqual(metadata.kernel_metadata["c1a_metadata"].shape, (48,))
        self.assertEqual(metadata.kernel_metadata["c2a_metadata"].shape, (48,))
        # QLI metadata is the fixed 1024.
        self.assertEqual(metadata.kernel_metadata["c1a_qli_metadata"].shape, (1024,))
        self.assertEqual(metadata.kernel_metadata["c2a_qli_metadata"].shape, (1024,))

    def test_graph_index_page_table_is_a_dedicated_expanded_buffer(self):
        backend = _low_ratio_backend()
        block_tables = torch.tensor([[7, 8]], dtype=torch.int32)
        metadata = SimpleNamespace(block_tables=block_tables, kernel_metadata={})
        backend._init_dsv41_graph_metadata(
            metadata, bs=1, tokens_per_req=1, device="cpu"
        )
        # ratio 2 pages at the full granule: its own 1:1 copy buffer, never the
        # shared block-table alias.
        self.assertIsNot(metadata.c2_index_page_table, block_tables)
        self.assertEqual(metadata.c2_index_page_table.shape, (1, 2))
        # ratio 1 needs two index pages per full page -> widened row [2p, 2p+1].
        self.assertEqual(metadata.c1_index_page_table.shape, (1, 4))

    def test_replay_refresh_expands_the_index_page_table(self):
        backend = _low_ratio_backend()
        backend.page_size = 128
        metadata = SimpleNamespace(
            block_tables=torch.tensor([[7, 8]], dtype=torch.int32),
            kernel_metadata={},
        )
        backend._init_dsv41_graph_metadata(
            metadata, bs=1, tokens_per_req=1, device="cpu"
        )
        c1_pt_before = metadata.c1_index_page_table
        batch = _batch(positions=[1], out_cache_loc=[201], seq_lens=[2], batch_size=1)
        backend.forward_metadata = metadata
        ctx = SimpleNamespace(
            fm=metadata,
            bs=1,
            live_seq_lens=torch.tensor([2], dtype=torch.int32),
            forward_batch=batch,
        )
        backend._refresh_graph_dsv41_1d(ctx)
        # Address is stable (the captured graph holds it) and the content is the
        # [k*p, k*p + k) expansion of the refreshed block table.
        self.assertIs(metadata.c1_index_page_table, c1_pt_before)
        self.assertEqual(metadata.c1_index_page_table.tolist(), [[14, 15, 16, 17]])
        self.assertEqual(metadata.c2_index_page_table.tolist(), [[7, 8]])

    def test_compressed_side_reuses_the_full_page_table(self):
        """``cmp_block_table`` is the FULL table: page id == compressed page id.

        The compressed pools page at ``page_size // ratio`` rows but keep the FULL
        page numbering, so the shared ``block_tables`` is the right table for both
        the MQSMLA compressed side and the index-page expansion. Anything that
        rebinds ``block_tables`` instead of refreshing it in place would leave
        these aliases pointing at capture-time pages.
        """
        backend = _low_ratio_backend()
        block_tables = torch.tensor([[7, 8]], dtype=torch.int32)
        metadata = SimpleNamespace(block_tables=block_tables, kernel_metadata={})
        backend._init_dsv41_graph_metadata(
            metadata, bs=1, tokens_per_req=1, device="cpu"
        )
        self.assertIs(metadata.c1_page_table, block_tables)
        self.assertIs(metadata.c2_page_table, block_tables)

    def test_replay_refresh_expands_the_refreshed_table_not_a_capture_snapshot(self):
        backend = _low_ratio_backend()
        backend.page_size = 128
        block_tables = torch.tensor([[7, 8]], dtype=torch.int32)
        metadata = SimpleNamespace(block_tables=block_tables, kernel_metadata={})
        backend._init_dsv41_graph_metadata(
            metadata, bs=1, tokens_per_req=1, device="cpu"
        )
        # The parent's replay refresh rewrites the shared table in place.
        block_tables[0, 0] = 9
        backend.forward_metadata = metadata
        batch = _batch(positions=[1], out_cache_loc=[201], seq_lens=[2], batch_size=1)
        ctx = SimpleNamespace(
            fm=metadata,
            bs=1,
            live_seq_lens=torch.tensor([2], dtype=torch.int32),
            forward_batch=batch,
        )
        backend._refresh_graph_dsv41_1d(ctx)
        # ratio 2 pages 1:1, ratio 1 expands each full page into two index pages.
        self.assertEqual(metadata.c2_index_page_table.tolist(), [[9, 8]])
        self.assertEqual(metadata.c1_index_page_table.tolist(), [[18, 19, 16, 17]])

    def test_replay_refresh_writes_into_the_same_buffers(self):
        backend = _low_ratio_backend()
        metadata = SimpleNamespace(block_tables=torch.zeros((2, 2), dtype=torch.int32))
        metadata.kernel_metadata = {}
        backend._init_dsv41_graph_metadata(
            metadata, bs=2, tokens_per_req=1, device="cpu"
        )
        metadata.actual_seq_lengths_q_pa = torch.arange(0, 3, dtype=torch.int32)
        loc_before = metadata.c1_loc
        win_before = metadata.dsv41_win_sparse_indices

        backend.forward_metadata = metadata
        batch = _batch(
            positions=[3, 4], out_cache_loc=[205, 208], seq_lens=[4, 5], batch_size=2
        )
        ctx = SimpleNamespace(
            fm=metadata,
            bs=2,
            live_seq_lens=torch.tensor([4, 5], dtype=torch.int32),
            forward_batch=batch,
        )
        backend._refresh_graph_dsv41_1d(ctx)
        # The dispatch-table pass refreshes the window index / length buffers.
        backend._dsv41_attention_kernel_metadata(batch)

        # Addresses are stable (the captured graph holds them) and the contents
        # match the eager derivation.
        self.assertIs(metadata.c1_loc, loc_before)
        self.assertIs(metadata.dsv41_win_sparse_indices, win_before)
        self.assertEqual(metadata.c1_loc.tolist(), [205, 208])
        self.assertEqual(metadata.c2_loc.tolist(), [102, -1])
        self.assertEqual(metadata.dsv41_win_topk_length.tolist(), [[4], [5]])
        # Window = [max(0, p - window + 1), p]; with window=128 both rows start
        # at 0 and the tail is the -1 pad.
        self.assertEqual(
            metadata.dsv41_win_sparse_indices[0, 0, :5].tolist(), [0, 1, 2, 3, -1]
        )
        self.assertEqual(
            metadata.dsv41_win_sparse_indices[1, 0, :6].tolist(), [0, 1, 2, 3, 4, -1]
        )
        self.assertEqual(metadata.dsv41_cmp_topk_length_c2.tolist(), [[2], [2]])


class TestSinks(unittest.TestCase):
    def test_missing_sink_becomes_an_explicit_zero_tensor(self):
        backend = _low_ratio_backend()
        sinks = backend._dsv41_attention_sinks(None, torch.device("cpu"), 64)
        self.assertEqual(sinks.shape, (64,))
        self.assertEqual(sinks.dtype, torch.float32)
        self.assertEqual(sinks.abs().sum().item(), 0.0)
        # Cached, so the graph sees one stable tensor.
        self.assertIs(
            sinks, backend._dsv41_attention_sinks(None, torch.device("cpu"), 64)
        )

    def test_provided_sink_is_flattened_and_cast(self):
        backend = _low_ratio_backend()
        sink = torch.arange(64, dtype=torch.float16).unsqueeze(0)
        out = backend._dsv41_attention_sinks(sink, torch.device("cpu"), 64)
        self.assertEqual(out.shape, (64,))
        self.assertEqual(out.dtype, torch.float32)


class TestForwardDispatch(unittest.TestCase):
    def _batch_for_dispatch(self):
        return SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))

    def test_ratio_outside_the_generation_is_rejected(self):
        backend = _low_ratio_backend()
        backend._forward_low_ratio_attention = MagicMock()
        with self.assertRaises(ValueError):
            backend.forward(
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                SimpleNamespace(layer_id=0),
                self._batch_for_dispatch(),
                compress_ratio=4,
                save_kv_cache=False,
            )

    def test_low_ratios_go_through_the_mqsmla_path(self):
        backend = _low_ratio_backend()
        backend._forward_low_ratio_attention = MagicMock(return_value="OUT")
        q = torch.zeros(1)
        layer = SimpleNamespace(layer_id=0)
        batch = self._batch_for_dispatch()
        for ratio in (0, 1, 2):
            self.assertEqual(
                backend.forward(
                    q,
                    torch.zeros(1),
                    torch.zeros(1),
                    layer,
                    batch,
                    compress_ratio=ratio,
                    save_kv_cache=False,
                ),
                "OUT",
            )
        self.assertEqual(backend._forward_low_ratio_attention.call_count, 3)

    def test_v4_backend_still_rejects_v41_ratios(self):
        backend = _low_ratio_backend(ratios=())
        with self.assertRaises(ValueError):
            backend.forward(
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                SimpleNamespace(layer_id=0),
                self._batch_for_dispatch(),
                compress_ratio=1,
                save_kv_cache=False,
            )


class TestIndexerRopeSource(unittest.TestCase):
    """``_dsv41_index_full_topk`` gathers Q's RoPE from the LAYER, not the indexer.

    ``DeepseekV41Indexer`` is built without ``freqs_cis``/``rotary_emb``
    (deepseek_v4.py L1366-1372) -- only ``DeepseekV41Compressor``/``C4Indexer``
    get them -- so reading them off the indexer raises AttributeError on the
    first index source layer.  The reference model uses the layer's
    ``compress_rotary_emb`` for this RoPE (recipes indexer.py L234).
    """

    def _fake_indexer(self, n_heads, index_head_dim):
        return SimpleNamespace(
            n_heads=n_heads,
            index_head_dim=index_head_dim,
            rope_head_dim=64,
            wq_b=lambda ql: (torch.zeros(ql.shape[0], n_heads * index_head_dim), None),
            head_weights=lambda x: torch.zeros(x.shape[0], n_heads),
        )

    def _run(
        self, *, ratio, seq_lens, n_heads=4, index_head_dim=128, packed=64, topk=3
    ):
        """Run ``_dsv41_index_full_topk`` against stubbed ops; return what QLI saw."""
        import sglang.srt.hardware_backend.npu.dsv4.dsv4_rope as rope_mod

        seen = {}
        sentinel = torch.zeros(8, 2)

        class _RopeStub:
            def __init__(self, freqs_cis, rotary_emb=None):
                seen["freqs_cis"] = freqs_cis

            # The backend reaches the table through ``for_freqs`` (a classmethod
            # on the real Dsv4NpuRoPE), so the stub must expose it too.
            @staticmethod
            def for_freqs(freqs_cis, rotary_emb=None):
                return _RopeStub(freqs_cis, rotary_emb)

            def get_cos_sin(self, *args, **kwargs):
                return (
                    torch.zeros(1, 1, 1, 64),
                    torch.zeros(1, 1, 1, 64),
                )

            @staticmethod
            def apply_rotary_mul_inplace(*args, **kwargs):
                return None

        class _QLI:
            def __call__(self, **kwargs):
                seen["qli_kwargs"] = kwargs
                return torch.zeros(1, topk, dtype=torch.int32), None, None, None

        backend = _low_ratio_backend()
        backend._dsv4_index_topk = topk
        backend.forward_metadata = SimpleNamespace(
            actual_seq_lengths_q_pa=torch.arange(0, 2, dtype=torch.int32),
            **{
                f"c{ratio}_index_page_table": torch.zeros((1, 2), dtype=torch.int32),
                "kernel_metadata": {
                    f"c{ratio}a_qli_metadata": torch.zeros(1024, dtype=torch.int32)
                },
            },
        )
        backend.token_to_kv_pool = SimpleNamespace(
            # The K and scale buffers are per-ratio, so the pool resolves them
            # from a layer_id -- never from a per-ratio compress_layer_id.
            get_index_k=lambda lid: (
                seen.setdefault("k_layer_id", lid),
                torch.zeros((1, 64, 1, packed), dtype=torch.uint8),
            )[1],
            get_index_scale=lambda lid: (
                seen.setdefault("scale_layer_id", lid),
                torch.zeros((1, 64, 1, 4), dtype=torch.uint8),
            )[1],
        )
        layer = SimpleNamespace(
            layer_id=1,
            compress_ratio=ratio,
            freqs_cis=sentinel,
            rotary_emb=None,
            indexer=self._fake_indexer(n_heads, index_head_dim),
        )
        with (
            patch.object(
                sys.modules["torch_npu"],
                "npu_dynamic_mx_quant",
                lambda q, dst_type=None: (
                    torch.zeros(q.shape[0], n_heads, packed, dtype=torch.uint8),
                    torch.zeros(q.shape[0], n_heads, 4, dtype=torch.uint8),
                ),
            ),
            patch.object(rope_mod, "Dsv4NpuRoPE", _RopeStub),
            patch.object(DS41, "quant_lightning_indexer", _QLI()),
        ):
            out = backend._dsv41_index_full_topk(
                layer,
                torch.zeros(1, 8),
                torch.zeros(1, 8),
                torch.tensor([0]),
                # seqused_k / cmp_residual_k come from this batch's seq_lens, not
                # from a metadata tensor: seq_lens is a graph input and is
                # refreshed every replay, while a captured buffer would keep the
                # capture-time lengths for all later steps.
                SimpleNamespace(seq_lens=torch.tensor(seq_lens, dtype=torch.int32)),
            )
        return seen, sentinel, layer, out

    def test_query_rope_gathers_the_layer_table(self):
        seen, sentinel, layer, out = self._run(ratio=1, seq_lens=[3])
        self.assertIs(seen["freqs_cis"], sentinel)
        self.assertEqual(seen["k_layer_id"], layer.layer_id)
        self.assertEqual(seen["scale_layer_id"], layer.layer_id)
        self.assertEqual(out.tolist(), [[0, 0, 0]])
        # The layer's compressed table drives Q's rope, so the partial slice must
        # be the index head's trailing rope_head_dim (recipes partial_slice).
        self.assertEqual(seen["qli_kwargs"]["cmp_ratio"], 1)
        self.assertEqual(seen["qli_kwargs"]["seqused_k"].tolist(), [3])
        # Ratio 1 has no unfinished group, so the residual must stay None.
        self.assertIsNone(seen["qli_kwargs"]["cmp_residual_k"])

    def test_lengths_come_from_seq_lens_not_from_captured_metadata(self):
        # A ratio-2 layer: seqused_k is the compressed length and cmp_residual_k
        # the live remainder (recipes indexer.py L359-L366). 7 tokens -> 3 full
        # pairs consumed by the indexer and 1 token left over, and neither value
        # may be read from a metadata field (there is none).
        seen, _, _, _ = self._run(ratio=2, seq_lens=[7])
        self.assertEqual(seen["qli_kwargs"]["seqused_k"].tolist(), [3])
        self.assertEqual(seen["qli_kwargs"]["cmp_residual_k"].tolist(), [1])
        self.assertEqual(seen["qli_kwargs"]["cmp_ratio"], 2)


class TestLowRatioAttentionBody(unittest.TestCase):
    """``_forward_low_ratio_attention`` wires both KV sides into MQSMLA.

    The window side is always present (explicit indices + the SWA page table);
    the compressed side is added for ratio 1/2 with its own indices, page table
    and per-query top-k lengths.
    """

    def _run(self, *, ratio, seq_lens, topk_indices=True, window=4):
        seen = {}

        class _MQSMLA:
            def __call__(self, q, **kwargs):
                seen["q"] = q
                seen["kwargs"] = kwargs
                return torch.zeros(q.shape[0], 64, 512), None

        backend = _low_ratio_backend(window=window)
        # Ratio 0 is dispatched under the "win" attention type, the low ratios
        # under "c{r}a" (the same key the metadata op is probed with).
        attention_type = "win" if ratio == 0 else f"c{ratio}a"
        fm = SimpleNamespace(
            actual_seq_lengths_q_pa=torch.arange(0, 2, dtype=torch.int32),
            actual_seq_lengths_kv=torch.tensor(seq_lens, dtype=torch.int32),
            swa_page_table=torch.zeros((1, 2), dtype=torch.int32),
            **{
                f"c{ratio}_page_table": torch.zeros((1, 2), dtype=torch.int32),
                "kernel_metadata": {
                    f"{attention_type}_metadata": torch.zeros(48, dtype=torch.int32)
                },
            },
        )
        if topk_indices:
            setattr(
                fm,
                f"c{ratio}_topk_indices",
                torch.zeros((1, backend._dsv4_index_topk), dtype=torch.int32),
            )
        backend.forward_metadata = fm
        backend.token_to_kv_pool = SimpleNamespace(
            head_dim=512,
            get_swa_buffer=lambda lid: torch.zeros(
                (1, window, 1, 544), dtype=torch.uint8
            ),
            get_compress_buffer=lambda lid, compressed: torch.zeros(
                (1, 2, 1, 320), dtype=torch.uint8
            ),
        )
        layer = SimpleNamespace(layer_id=3, scaling=0.1)
        batch = SimpleNamespace(
            positions=torch.tensor([seq_lens[0] - 1], dtype=torch.int64),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            batch_size=1,
        )
        with patch.object(DS41, "mixed_quant_sparse_flash_mla", _MQSMLA()):
            out = backend._forward_low_ratio_attention(
                torch.zeros(1, 64, 512), layer, batch, None, ratio
            )
        return seen, fm, layer, out

    def test_both_sides_are_wired_with_their_own_block_tables(self):
        seen, fm, layer, out = self._run(ratio=2, seq_lens=[7])
        kw = seen["kwargs"]
        self.assertEqual(out.shape, (1, 64, 512))
        self.assertEqual(seen["q"].shape, (1, 64, 512))
        # Window side: explicit ascending indices + lengths, addressed by the
        # SWA page table (the window pool holds every token).
        self.assertIs(kw["ori_sparse_indices"], fm.dsv41_win_sparse_indices)
        self.assertIs(kw["ori_topk_length"], fm.dsv41_win_topk_length)
        self.assertIs(kw["ori_block_table"], fm.swa_page_table)
        self.assertEqual(kw["ori_topk_length"].tolist(), [[4]])
        # Compressed side.
        self.assertEqual(kw["cmp_kv"].shape, (1, 2, 1, 320))
        self.assertIs(kw["cmp_block_table"], fm.c2_page_table)
        self.assertEqual(kw["cmp_sparse_indices"].shape, (1, 1, 512))
        self.assertIs(kw["metadata"], fm.kernel_metadata["c2a_metadata"])
        # Per-query lengths are derived from the live positions: p = 6 -> the
        # 7th token closes the 3rd pair, capped by index_topk.
        self.assertEqual(kw["cmp_topk_length"].tolist(), [[3]])
        # Reserved by the operator, but derived from seq_lens rather than read
        # from a capture-time metadata field.
        self.assertEqual(kw["seqused_cmp_kv"].tolist(), [3])
        # One float32 sink per Q head, shared by every row.
        self.assertEqual(kw["sinks"].shape, (64,))
        self.assertEqual(kw["sinks"].dtype, torch.float32)
        self.assertEqual(kw["quant_mode"], 1)
        self.assertEqual(kw["softmax_scale"], layer.scaling)
        self.assertEqual((kw["layout_q"], kw["layout_kv"]), ("TND", "PA_BBND"))

    def test_window_only_ratio_drops_the_compressed_side(self):
        seen, fm, _, _ = self._run(ratio=0, seq_lens=[7])
        self.assertNotIn("cmp_kv", seen["kwargs"])
        self.assertIs(seen["kwargs"]["ori_block_table"], fm.swa_page_table)

    def test_missing_topk_indices_fails_loudly(self):
        # The indexer source layer must run first; a missing topk is a wiring
        # bug, so it must raise rather than attend to a stale buffer.
        with self.assertRaises(RuntimeError):
            self._run(ratio=2, seq_lens=[7], topk_indices=False)


class TestCompressedRowIdentity(unittest.TestCase):
    """One flat row addresses the compressed KV pool and the indexer-K pool alike.

    ``fm.c{r}_loc = full_loc // ratio`` is written into the compressed pool by
    ``kv_compress_epilog_v2`` (slot_mapping) and into the index pool by
    ``indexer_prologue_k`` (cache_index); the operators derive that same physical
    row again from their page tables.  That only holds while each FULL page owns
    the consecutive index pages [k*p, k*p + k) -- the convention the shared
    ``expand_index_page_table`` implements (it pages at DSV41_INDEX_PAGE_SIZE
    while the compressed pools page at page_size // ratio).
    """

    def test_index_rows_equal_compressed_rows_across_a_full_page(self):
        from sglang.srt.layers.attention.dsv4.metadata import (
            expand_index_page_table,
        )
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DSV41_INDEX_PAGE_SIZE,
        )

        page_size = 256
        table = torch.tensor([[7, 8]], dtype=torch.int32)
        for ratio in (1, 2):
            slots_per_page = page_size // ratio
            self.assertEqual(slots_per_page % DSV41_INDEX_PAGE_SIZE, 0)
            blocks_per_page = slots_per_page // DSV41_INDEX_PAGE_SIZE
            index_table = expand_index_page_table(
                table,
                full_page_size=page_size,
                compress_ratio=ratio,
                index_page_size=DSV41_INDEX_PAGE_SIZE,
            )
            for row_in_page in range(page_size):
                full_loc = 7 * page_size + row_in_page
                # What the epilog / prologue write.
                written = full_loc // ratio
                # What the operators read: compressed slot q of FULL page 7 sits
                # at column 0 * blocks_per_page + q // 64 of the index table.
                q = row_in_page // ratio
                column = q // DSV41_INDEX_PAGE_SIZE
                index_page = int(index_table[0, column])
                read = index_page * DSV41_INDEX_PAGE_SIZE + q % DSV41_INDEX_PAGE_SIZE
                self.assertEqual(read, written, (ratio, row_in_page))

    def test_compressed_pool_slot_matches_the_full_page_table_lookup(self):
        """`c{r}_loc` is a flat *physical* row; the operator re-derives that same
        row from `fm.c{r}_page_table`.

        The compressed pool pages at `page_size // ratio`, so compressed page j
        is FULL page j and the full page table doubles as the compressed one
        (`_build_dsv41_ratio_metadata`).  The writer takes its index from the
        allocator's physical slot (`out_cache_loc // ratio`) while the reader
        resolves the *logical* compressed row through that table -- with a
        permuted table the two only agree because both sides carry the same
        physical page id.
        """
        table = torch.tensor([[5, 1, 9, 2]], dtype=torch.int32)
        for ratio in (1, 2):
            backend = _low_ratio_backend()
            page_size = backend.page_size
            slots_per_page = page_size // ratio
            positions, out_loc = [], []
            for full_page, phys in enumerate(table[0].tolist()):
                for row_in_page in range(page_size):
                    positions.append(full_page * page_size + row_in_page)
                    out_loc.append((phys * page_size + row_in_page))
            backend._build_dsv41_ratio_metadata(
                _batch(
                    positions=positions,
                    out_cache_loc=out_loc,
                    seq_lens=[len(positions)],
                    batch_size=1,
                ),
                ratio,
            )
            loc = getattr(backend.forward_metadata, f"c{ratio}_loc")
            completed = [
                (positions[i], out_loc[i])
                for i in range(len(positions))
                if positions[i] % ratio == ratio - 1 and out_loc[i] >= page_size
            ]
            # ratio 1 masks in place, ratio >= 2 compacts; dropping the -1 skips
            # (the documented "skip" sentinel) leaves both in position order.
            self.assertEqual(
                [int(v) for v in loc.tolist() if v != -1],
                [full_loc // ratio for _, full_loc in completed],
            )
            for position, full_loc in completed:
                written = full_loc // ratio
                # What the operator reads: the LOGICAL compressed row is what the
                # sparse indices carry; its page comes from the full page table.
                logical = position // ratio
                page = int(table[0, logical // slots_per_page])
                read = page * slots_per_page + logical % slots_per_page
                self.assertEqual(read, written, (ratio, position))


class TestSeamBGuards(unittest.TestCase):
    def test_candidate_configuration_fails_loudly(self):
        backend = _low_ratio_backend()
        layer = SimpleNamespace(
            layer_id=24,
            compress_ratio=1,
            indexer=SimpleNamespace(
                owns_k=False, is_candidate_source=False, uses_candidates=True
            ),
        )
        with self.assertRaises(NotImplementedError):
            backend._low_ratio_index_topk(
                layer, None, torch.zeros(1), torch.zeros(1), torch.zeros(1), None
            )

    def test_index_k_store_requires_the_latent(self):
        backend = _low_ratio_backend()
        layer = SimpleNamespace(
            layer_id=2, compress_ratio=2, indexer=SimpleNamespace(owns_k=True)
        )
        with self.assertRaises(RuntimeError):
            backend._dsv41_index_k_store(layer, None)


class TestNPUBlock32Dispatch(unittest.TestCase):
    def test_native_dispatch_matches_the_load_time_mxfp8_gate(self):
        native_linear = object()
        backend = SimpleNamespace(is_auto=lambda: True, is_deep_gemm=lambda: False)
        for arch35 in (False, True):
            for ue8m0 in (False, True):
                for requant in (False, True):
                    with (
                        self.subTest(arch35=arch35, ue8m0=ue8m0, requant=requant),
                        patch.object(
                            fp8_utils,
                            "get_fp8_gemm_runner_backend",
                            return_value=backend,
                        ),
                        patch.object(
                            fp8_utils.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", False
                        ),
                        patch.object(fp8_utils, "is_npu_arch35", return_value=arch35),
                        patch.object(
                            fp8_utils.envs.SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8,
                            "get",
                            return_value=requant,
                        ),
                        patch.object(
                            fp8_utils,
                            "npu_w8a8_mxfp8_linear",
                            native_linear,
                            create=True,
                        ),
                    ):
                        linear = fp8_utils.dispatch_w8a8_block_fp8_linear(
                            weight_block_size=[32, 32], act_scale_ue8m0=ue8m0
                        )
                    if arch35 and (ue8m0 or requant):
                        self.assertIs(linear, native_linear)
                    else:
                        self.assertIs(
                            linear.func, fp8_utils.triton_w8a8_block_fp8_linear
                        )
                        self.assertEqual(linear.keywords, {"act_scale_ue8m0": ue8m0})


class TestC128ComponentGating(unittest.TestCase):
    """The C128 sidecar component must only attach where a C128 pool exists.

    DeepSeek-V4.1 keeps the NPU req pool (so ``req_to_c128_sidecar`` exists, all
    zeros) but runs on the generic SWA allocator, which has no
    ``c128_attn_allocator``. Attaching the component there makes its per-insert
    hooks dereference that missing attribute, so registration is gated on the
    model actually carrying ratio-128 layers.
    """

    def _components_for(self, compress_ratios):
        captured = {}

        class _StubCache:
            def __init__(self, params):
                captured["components"] = tuple(
                    slot.component for slot in params.tree_slots
                )

        params = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_c128_sidecar=torch.zeros(1, 1, dtype=torch.int32)
            ),
            token_to_kv_pool_allocator=SimpleNamespace(storage=None, size=1024),
            sliding_window_size=128,
            draft_swa_window=0,
            component_registry_override=None,
            tree_slots=(),
        )
        ctx = SimpleNamespace(
            server_args=None,
            params=params,
            is_hybrid_swa=True,
            is_hybrid_ssm=False,
            full_tokens_per_layer=1024,
            enable_hierarchical_cache=False,
            model_config=SimpleNamespace(compress_ratios=compress_ratios),
            tp_worker=None,
        )
        with (
            patch(
                "sglang.srt.mem_cache.registry.get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_retraction_backup=None
                ),
            ),
            patch(
                "sglang.srt.mem_cache.registry.get_memory",
                return_value=SimpleNamespace(
                    enable_unified_cache_external_linker=False
                ),
            ),
        ):
            create_unified_radix_cache(ctx, cache_class=_StubCache)
        return captured["components"]

    def test_dsv4_keeps_the_c128_component(self):
        components = self._components_for((0, 0) + (4,) * 2 + (128,) * 2)
        self.assertIn("C128", components)

    def test_dsv41_drops_the_c128_component(self):
        components = self._components_for((0, 0) + (2,) * 2 + (1,) * 2)
        self.assertNotIn("C128", components)
        self.assertIn("SWA", components)

    def test_missing_sidecar_pool_never_registers_the_component(self):
        params = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(),
            token_to_kv_pool_allocator=SimpleNamespace(storage=None, size=1024),
            sliding_window_size=128,
            draft_swa_window=0,
            component_registry_override=None,
            tree_slots=(),
        )
        ctx = SimpleNamespace(
            server_args=None,
            params=params,
            is_hybrid_swa=True,
            is_hybrid_ssm=False,
            full_tokens_per_layer=1024,
            enable_hierarchical_cache=False,
            model_config=SimpleNamespace(compress_ratios=(128,)),
            tp_worker=None,
        )

        class _StubCache:
            def __init__(self, params):
                self.tree_slots = tuple(params.tree_slots)

        with (
            patch(
                "sglang.srt.mem_cache.registry.get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_retraction_backup=None
                ),
            ),
            patch(
                "sglang.srt.mem_cache.registry.get_memory",
                return_value=SimpleNamespace(
                    enable_unified_cache_external_linker=False
                ),
            ),
        ):
            create_unified_radix_cache(ctx, cache_class=_StubCache)
        self.assertNotIn("C128", tuple(slot.component for slot in params.tree_slots))


class TestLowRatioCompressGuards(unittest.TestCase):
    def test_ratio2_rejects_a_fused_compressor_module(self):
        """The op call reads `wkv`/`wgate`; a fused `wkv_gate` module must fail loud.

        `fused_low_ratio_compress_supported()` is CUDA-only, so NPU always builds
        the unfused module; if that ever changed, the kernel call would die with an
        AttributeError on `compressor.wkv` instead of saying what is wrong.
        """
        backend = _low_ratio_backend()
        layer = SimpleNamespace(
            compress_ratio=2,
            layer_id=2,
            compressor=SimpleNamespace(use_fused_gate=True),
        )
        with patch.object(dsv4_backend, "is_npu_arch35", return_value=True):
            with self.assertRaises(AssertionError):
                backend._low_ratio_compress(layer, torch.zeros(1, 8), None, None)


class TestEagerGraphMetadataParity(unittest.TestCase):
    """Eager prefill and aclgraph replay must derive the same rows.

    They are two independent implementations of one contract: the eager builder
    reads ``forward_batch.positions`` and compacts, the replay path derives
    positions from ``live_seq_lens`` and compacts (ratio 2) or writes them in
    place (ratio 1, where every real token closes its group). A divergence here
    would make prefill and decode disagree about which slot each compressed row
    belongs to.
    """

    def _graph_side(self, backend, batch, live):
        metadata = SimpleNamespace(
            block_tables=torch.zeros((len(live), 1), dtype=torch.int32),
            kernel_metadata={},
        )
        backend._init_dsv41_graph_metadata(
            metadata, bs=len(live), tokens_per_req=1, device="cpu"
        )
        backend.forward_metadata = metadata
        ctx = SimpleNamespace(
            fm=metadata,
            bs=len(live),
            live_seq_lens=torch.tensor(live, dtype=torch.int32),
            forward_batch=batch,
        )
        backend._refresh_graph_dsv41_1d(ctx)
        return metadata

    def test_ratio_rows_match_between_eager_and_replay(self):
        batch = _batch(
            positions=[3, 4], out_cache_loc=[205, 208], seq_lens=[4, 5], batch_size=2
        )
        eager_backend = _low_ratio_backend()
        eager_backend.forward_metadata = SimpleNamespace(
            block_tables=torch.zeros((2, 1), dtype=torch.int32)
        )
        for ratio in (1, 2):
            eager_backend._build_dsv41_ratio_metadata(batch, ratio)
        eager = eager_backend.forward_metadata

        graph = self._graph_side(_low_ratio_backend(), batch, live=[4, 5])

        # ratio 1: every real token closes its group -> per-token rows.
        self.assertEqual(eager.c1_loc.tolist(), graph.c1_loc.tolist())
        self.assertEqual(eager.c1_loc.tolist(), [205, 208])
        self.assertEqual(
            eager.positions_cmp_padding_c1.tolist(),
            graph.positions_cmp_padding_c1.tolist(),
        )
        # ratio 2: only the second token of a pair closes it, so row 1 is the
        # group start (0) and the unfinished pair leaves the skip sentinel.
        self.assertEqual(eager.c2_loc.tolist(), graph.c2_loc.tolist())
        self.assertEqual(eager.c2_loc.tolist(), [102, -1])
        self.assertEqual(eager.positions_cmp_padding_c2.tolist(), [2, 0])
        self.assertEqual(
            eager.positions_cmp_padding_c2.tolist(),
            graph.positions_cmp_padding_c2.tolist(),
        )

    def test_mid_batch_dropped_row_matches_between_eager_and_replay(self):
        # A dropped row *between* two live ones: ratio 2 skips it (it closes no
        # group) and ratio 1 must leave the sentinel in place rather than
        # compacting, so the row index still matches the latent's token order.
        batch = _batch(
            positions=[3, 4, 5],
            out_cache_loc=[205, 0, 208],
            seq_lens=[4, 0, 6],
            batch_size=3,
        )
        eager_backend = _low_ratio_backend()
        eager_backend.forward_metadata = SimpleNamespace(
            block_tables=torch.zeros((3, 1), dtype=torch.int32)
        )
        for ratio in (1, 2):
            eager_backend._build_dsv41_ratio_metadata(batch, ratio)
        eager = eager_backend.forward_metadata

        graph = self._graph_side(_low_ratio_backend(), batch, live=[4, 0, 6])

        self.assertEqual(eager.c1_loc.tolist(), [205, -1, 208])
        self.assertEqual(
            eager.positions_cmp_padding_c1.tolist(),
            [3, 0, 5],
        )
        self.assertEqual(eager.c1_loc.tolist(), graph.c1_loc.tolist())
        self.assertEqual(
            eager.positions_cmp_padding_c1.tolist(),
            graph.positions_cmp_padding_c1.tolist(),
        )
        self.assertEqual(eager.c2_loc.tolist(), [102, 104, -1])
        self.assertEqual(eager.c2_loc.tolist(), graph.c2_loc.tolist())
        self.assertEqual(
            eager.positions_cmp_padding_c2.tolist(),
            graph.positions_cmp_padding_c2.tolist(),
        )

    def test_padding_rows_are_dropped_identically(self):
        # Row 1 is a graph-padding row: no live tokens and the dummy-page slot.
        batch = _batch(
            positions=[3, 0], out_cache_loc=[205, 0], seq_lens=[4, 4], batch_size=2
        )
        graph = self._graph_side(_low_ratio_backend(), batch, live=[4, 0])
        self.assertEqual(graph.c1_loc.tolist(), [205, -1])
        # The eager path never sees padding rows (it slices to the real tokens),
        # so its per-ratio rows are the same dense prefix.
        eager_backend = _low_ratio_backend()
        eager_backend.forward_metadata = SimpleNamespace(
            block_tables=torch.zeros((1, 1), dtype=torch.int32)
        )
        real = _batch(positions=[3], out_cache_loc=[205], seq_lens=[4], batch_size=1)
        eager_backend._build_dsv41_ratio_metadata(real, 1)
        self.assertEqual(eager_backend.forward_metadata.c1_loc.tolist(), [205])
        self.assertEqual(
            graph.c1_loc[:1].tolist(), eager_backend.forward_metadata.c1_loc.tolist()
        )


class TestVerifyGraphGuard(unittest.TestCase):
    """A DeepSeek-V4.1 ``target_verify`` *replay* must fail loudly (R39).

    The verify arm of ``_apply_dsv4_graph_metadata`` refreshes the c4/c128
    buffers only, so a V4.1 verify graph would scatter this step's compressed
    rows to the previous step's slots and silently corrupt the compressed /
    index pools. The guard has to fire for V4.1 and leave V4 alone.
    """

    class _GraphMode:
        def __init__(self, target_verify):
            self._target_verify = target_verify

        def is_decode(self):
            return not self._target_verify

        def is_target_verify(self):
            return self._target_verify

    _REFRESHES = (
        "_refresh_graph_seq_metadata",
        "_refresh_graph_compress_page_tables_direct",
        "_refresh_graph_decode_compress_1d_direct",
        "_refresh_graph_explicit_state_block_tables",
        "_refresh_graph_swa_metadata_direct",
        "_refresh_graph_dspark_sparse_metadata",
        "_refresh_graph_kernel_metadata",
    )

    def _apply(self, *, has_low_ratio, target_verify=True, is_idle_replay=False):
        backend = _low_ratio_backend()
        backend._dsv4_has_low_ratio = has_low_ratio
        ctx = SimpleNamespace(
            fm=SimpleNamespace(),
            bs=1,
            forward_batch=SimpleNamespace(),
            graph_mode=self._GraphMode(target_verify),
            is_idle_replay=is_idle_replay,
            # c4/c128 are absent for V4.1, so both verify sub-arms are skipped.
            has_compress=False,
            active_target_verify=False,
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(dsv4_backend, "is_npu_arch35", return_value=False)
            )
            stack.enter_context(
                patch.object(backend, "_build_dsv4_graph_replay_ctx", return_value=ctx)
            )
            for name in self._REFRESHES:
                stack.enter_context(patch.object(backend, name))
            backend._apply_dsv4_graph_metadata(None)

    def test_v41_verify_replay_raises_instead_of_corrupting(self):
        with self.assertRaisesRegex(RuntimeError, "target-verify"):
            self._apply(has_low_ratio=True)

    def test_v41_idle_verify_replay_is_not_blocked(self):
        # Idle ranks skip the model body, so nothing is scattered: no raise.
        self._apply(has_low_ratio=True, is_idle_replay=True)

    def test_decode_replay_is_untouched(self):
        self._apply(has_low_ratio=True, target_verify=False)

    def test_v4_verify_replay_is_untouched(self):
        # DeepSeek-V4 keeps its historical no-refresh behaviour here.
        self._apply(has_low_ratio=False)


class _FakeTorchOps:
    """``torch`` proxy that redirects ``torch.ops.custom`` to a recorder.

    Only ``ops`` is replaced; every other attribute falls through to the real
    torch, so the production writer's ``torch.bfloat16`` / ``torch.uint8`` keep
    working on a CPU-only box.
    """

    def __init__(self, custom):
        self._custom = custom

    def __getattr__(self, name):
        if name == "ops":
            return SimpleNamespace(custom=self._custom)
        return getattr(torch, name)


def _dsv41_llm_pool(row_dim, dtype):
    """A ``DSV41NPUSingleKVPool`` with just the ``kv_buffer`` the writers read."""
    pool = object.__new__(DSV41NPUSingleKVPool)
    pool.kv_buffer = [torch.zeros(2, 4, 1, row_dim, dtype=dtype)]
    return pool


class TestV41PoolWritePath(unittest.TestCase):
    """The V4.1 pool writers must hand ``kv_compress_epilog_v2`` exactly the ABI
    its doc specifies (cann-recipes-infer
    ``ops/ascendc/docs/custom-npu_kv_compress_epilog_v2.md`` L40/L60-L70): a 2-D
    ``[N, kvCacheCol]`` cache in the row dtype, a **bf16** ``[T, d]`` source, an
    int64 slot mapping (``-1`` skips a row), and one of the three legal
    ``(quant_mode, quant_group_size)`` pairs with ``x_scale`` pinned to 1.0.
    Stubbing the operator makes all of it CPU-checkable.
    """

    _SELF = SimpleNamespace(qk_nope_head_dim=448, qk_rope_head_dim=64)

    def _write(self, *, buf, loc, cache, compressed):
        recorder = MagicMock(name="kv_compress_epilog_v2")
        custom = SimpleNamespace(kv_compress_epilog_v2=recorder)
        with patch.object(dsv4_memory_pool, "torch", _FakeTorchOps(custom)):
            DSV4NPUTokenToKVPool._write_dsv41_packed_kv(
                self._SELF, buf=buf, loc=loc, cache=cache, compressed=compressed
            )
        return recorder

    def test_window_row_uses_the_mxfp8_group32_abi(self):
        buf = torch.zeros(2, 4, 1, 544, dtype=torch.float8_e4m3fn)
        recorder = self._write(
            buf=buf,
            loc=torch.tensor([[0, 1], [2, 3]], dtype=torch.int64),
            # float32 in: the writer must cast to the bf16 the op requires.
            cache=torch.zeros(4, 1, 512, dtype=torch.float32),
            compressed=False,
        )
        self.assertEqual(recorder.call_count, 1)
        args, kwargs = recorder.call_args
        self.assertEqual(tuple(args[0].shape), (8, 544))
        self.assertEqual(args[0].dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(args[1].shape), (4, 512))
        self.assertEqual(args[1].dtype, torch.bfloat16)
        self.assertEqual(args[2].dtype, torch.int64)
        self.assertEqual(args[2].tolist(), [0, 1, 2, 3])
        self.assertEqual(
            kwargs,
            {
                "quant_group_size": 32,
                "quant_mode": "mxfp8_bf16",
                "round_scale": True,
                "x_scale": 1.0,
            },
        )

    def test_compressed_row_uses_the_mxfp4_group16_abi(self):
        buf = torch.zeros(2, 4, 1, 320, dtype=torch.uint8)
        recorder = self._write(
            buf=buf,
            loc=torch.tensor([5, -1], dtype=torch.int64),
            cache=torch.zeros(2, 1, 512, dtype=torch.bfloat16),
            compressed=True,
        )
        args, kwargs = recorder.call_args
        self.assertEqual(tuple(args[0].shape), (8, 320))
        self.assertEqual(args[0].dtype, torch.uint8)
        # The -1 sentinel reaches the operator (which skips it); the writer must
        # not compress or drop it itself.
        self.assertEqual(args[2].tolist(), [5, -1])
        self.assertEqual(kwargs["quant_mode"], "mxfp4_bf16")
        self.assertEqual(kwargs["quant_group_size"], 16)
        self.assertEqual(kwargs["x_scale"], 1.0)

    def test_write_rejects_a_cache_that_is_not_head_dim_wide(self):
        with self.assertRaisesRegex(RuntimeError, "last dim 512"):
            self._write(
                buf=torch.zeros(2, 4, 1, 544, dtype=torch.float8_e4m3fn),
                loc=torch.zeros(1, dtype=torch.int64),
                cache=torch.zeros(1, 1, 448),
                compressed=False,
            )

    def test_write_rejects_a_slot_count_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "one slot per token"):
            self._write(
                buf=torch.zeros(2, 4, 1, 320, dtype=torch.uint8),
                loc=torch.zeros(2, dtype=torch.int64),
                cache=torch.zeros(3, 1, 512),
                compressed=True,
            )

    def test_an_empty_write_calls_no_operator(self):
        recorder = self._write(
            buf=torch.zeros(2, 4, 1, 320, dtype=torch.uint8),
            loc=torch.zeros(0, dtype=torch.int64),
            cache=torch.zeros(0, 1, 512),
            compressed=True,
        )
        self.assertEqual(recorder.call_count, 0)


class TestV41PoolWriteDispatch(unittest.TestCase):
    """``set_compress_buffer`` / ``set_swa_buffer`` must pick the V4.1 row format
    by the *pool class* (R8), never by the layer's ratio, and must not fall
    through to the V4 writer or the generic PA_ND store.
    """

    def _ns(self, *, ratio, pool):
        return SimpleNamespace(
            layer_mapping={
                7: SimpleNamespace(
                    compress_ratio=ratio, compress_layer_id=0, compress_kv_pool=pool
                )
            },
        )

    def test_compress_buffer_routes_a_v41_pool_to_the_fp4_writer(self):
        pool = _dsv41_llm_pool(320, torch.uint8)
        ns = self._ns(ratio=2, pool=pool)
        writer = MagicMock()
        ns._write_dsv41_packed_kv = writer
        kv = SimpleNamespace(device=SimpleNamespace(type="npu"))
        loc = torch.tensor([1, 2], dtype=torch.int64)
        DSV4NPUTokenToKVPool.set_compress_buffer(ns, 7, loc, kv, None, False)
        # A ratio-1 aggregate must take the same fp4 route as ratio 2.
        writer.assert_called_once_with(
            buf=pool.kv_buffer[0], loc=loc, cache=kv, compressed=True
        )

    def test_compress_buffer_rejects_a_scale_for_the_v41_row(self):
        pool = _dsv41_llm_pool(320, torch.uint8)
        ns = self._ns(ratio=1, pool=pool)
        ns._write_dsv41_packed_kv = MagicMock()
        kv = SimpleNamespace(device=SimpleNamespace(type="npu"))
        with self.assertRaisesRegex(AssertionError, "not from a separate scale"):
            DSV4NPUTokenToKVPool.set_compress_buffer(
                ns,
                7,
                torch.tensor([0], dtype=torch.int64),
                kv,
                torch.zeros(1, 1),
                False,
            )

    def test_swa_buffer_routes_a_v41_pool_to_the_fp8_writer(self):
        pool = _dsv41_llm_pool(544, torch.float8_e4m3fn)
        ns = SimpleNamespace(
            swa_kv_pool=pool, _swa_local_layer_id=lambda layer_id: layer_id
        )
        writer = MagicMock()
        ns._write_dsv41_packed_kv = writer
        loc = torch.tensor([3], dtype=torch.int64)
        cache = torch.zeros(1, 1, 512, dtype=torch.bfloat16)
        DSV4NPUTokenToKVPool.set_swa_buffer(ns, 0, loc, cache)
        writer.assert_called_once_with(
            buf=pool.kv_buffer[0], loc=loc, cache=cache, compressed=False
        )


def _metadata_recorder(shape):
    """Operator stub that records its call and answers a probe-shaped tensor."""
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.zeros(*shape, dtype=torch.int32)

    return record, calls


class TestKernelMetadataBuilders(unittest.TestCase):
    """The two eager metadata builders are the only place where the backend
    config + live ``seq_lens`` become the ds41 operators' dispatch arguments, and
    the aclgraph replay refresh calls the very same functions -- so a wrong
    argument is wrong on every step, and handing back a *new* tensor instead of
    refreshing the captured buffer would silently detach the graph.

    R51 (``pool.head_dim``) lived in the MQSMLA builder, so the width source gets
    its own test below.
    """

    def _backend(self, *, bs=2, tokens_per_req=1):
        backend = _low_ratio_backend()
        metadata = SimpleNamespace(block_tables=torch.zeros((bs, 2), dtype=torch.int32))
        metadata.kernel_metadata = {}
        backend._init_dsv41_graph_metadata(
            metadata, bs=bs, tokens_per_req=tokens_per_req, device="cpu"
        )
        # The graph path refreshes this from the launch metadata; both builders
        # read it straight off forward_metadata.
        metadata.actual_seq_lengths_q_pa = (
            torch.arange(bs + 1, dtype=torch.int32) * tokens_per_req
        )
        backend.forward_metadata = metadata
        return backend, metadata

    def _batch(self):
        return _batch(
            positions=[1, 2], out_cache_loc=[201, 202], seq_lens=[5, 6], batch_size=2
        )

    def test_mqsmla_metadata_is_one_table_per_ratio_from_the_config(self):
        backend, fm = self._backend()
        recorder, calls = _metadata_recorder((48,))
        with patch.object(DS41, "mixed_quant_sparse_flash_mla_metadata", recorder):
            out = backend._dsv41_attention_kernel_metadata(self._batch())

        self.assertEqual(set(out), {"win_metadata", "c1a_metadata", "c2a_metadata"})
        # aclgraph binds the address at capture: refresh in place, never rebind.
        for key in out:
            self.assertIs(out[key], fm.kernel_metadata[key])
        self.assertEqual(len(calls), 3)

        # Call order is (ratio 0, *low_ratios): the window table first. It must
        # carry has_cmp_kv=False and an all-zero compressed length -- the window
        # side has no compressed pool at all.
        win_args, win_kwargs = calls[0]
        self.assertIs(win_args[0], fm.dsv41_win_topk_length)
        self.assertIs(win_kwargs["has_cmp_kv"], False)
        self.assertTrue(torch.all(win_args[1] == 0))
        self.assertEqual(win_args[1].shape, win_args[0].shape)

        for (args, kwargs), ratio in zip(calls[1:], (1, 2)):
            self.assertEqual(tuple(args[0].shape), (2, 1))
            self.assertIs(kwargs["has_cmp_kv"], True)
            # The per-query compressed length is the graph-resident buffer, so
            # the mixed builder refreshes rather than allocates.
            self.assertIs(args[1], getattr(fm, f"dsv41_cmp_topk_length_c{ratio}"))

        for args, kwargs in calls:
            self.assertEqual(kwargs["num_heads_q"], 64)
            self.assertEqual(kwargs["num_heads_kv"], 1)
            self.assertEqual(kwargs["head_dim"], 512)
            self.assertEqual(kwargs["quant_mode"], 1)
            self.assertEqual(kwargs["layout_q"], "TND")
            self.assertEqual(kwargs["layout_kv"], "PA_BBND")
            self.assertIs(kwargs["has_ori_kv"], True)
            self.assertIs(kwargs["cu_seqlens_q"], fm.actual_seq_lengths_q_pa)

    def test_mqsmla_metadata_takes_head_dim_from_the_backend_config(self):
        """The DSV4 pool has no ``head_dim`` attribute (only
        ``qk_nope_head_dim``/``qk_rope_head_dim``), so the width must come from
        the backend config -- reading it off the pool is an AttributeError on the
        very first V4.1 forward (R51).
        """
        backend, _ = self._backend()
        backend._dsv4_head_dim = 384  # a value only the backend config carries
        backend.token_to_kv_pool = SimpleNamespace()  # exactly the real pool's API
        recorder, calls = _metadata_recorder((48,))
        with patch.object(DS41, "mixed_quant_sparse_flash_mla_metadata", recorder):
            backend._dsv41_attention_kernel_metadata(self._batch())
        for _, kwargs in calls:
            self.assertEqual(kwargs["head_dim"], 384)

    def test_qli_metadata_derives_the_lengths_and_the_residual_per_ratio(self):
        backend, fm = self._backend()
        recorder, calls = _metadata_recorder((1024,))
        with patch.object(DS41, "quant_lightning_indexer_metadata", recorder):
            out = backend._dsv41_index_kernel_metadata(self._batch())

        self.assertEqual(set(out), {"c1a_qli_metadata", "c2a_qli_metadata"})
        for key in out:
            self.assertIs(out[key], fm.kernel_metadata[key])
        self.assertEqual(len(calls), 2)

        r1 = calls[0][1]
        r2 = calls[1][1]
        # seqused_k is the compressed length, seq_lens // ratio (QLI README).
        self.assertEqual(r1["seqused_k"].tolist(), [5, 6])
        self.assertEqual(r2["seqused_k"].tolist(), [2, 3])
        # mask_mode=3 with cmp_ratio != 1 also needs the remainder; ratio 1 must
        # not send one at all (the operator picks its path off cmp_ratio).
        self.assertIsNone(r1["cmp_residual_k"])
        self.assertEqual(r2["cmp_residual_k"].tolist(), [1, 0])

        for kwargs, ratio in ((r1, 1), (r2, 2)):
            self.assertEqual(kwargs["cmp_ratio"], ratio)
            self.assertEqual(kwargs["mask_mode"], 3)
            self.assertEqual(kwargs["batch_size"], 2)
            # The index-side widths, not the attention-side ones (64 / 512).
            self.assertEqual(kwargs["num_heads_q"], 32)
            self.assertEqual(kwargs["num_heads_k"], 1)
            self.assertEqual(kwargs["head_dim"], 128)
            self.assertEqual(kwargs["topk"], 512)
            self.assertIsNone(kwargs["seqused_q"])
            self.assertIs(kwargs["cu_seqlens_q"], fm.actual_seq_lengths_q_pa)
            self.assertEqual(kwargs["layout_q"], "TND")
            self.assertEqual(kwargs["layout_k"], "PA_BBND")

    def test_qli_metadata_is_skipped_until_the_query_metadata_exists(self):
        # The index builder must not depend on the compress pass having run.
        backend, fm = self._backend()
        fm.actual_seq_lengths_q_pa = None
        recorder, calls = _metadata_recorder((1024,))
        with patch.object(DS41, "quant_lightning_indexer_metadata", recorder):
            self.assertEqual(backend._dsv41_index_kernel_metadata(self._batch()), {})
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
