import contextlib
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.utils import setup_state_kv_args
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DSV4_FP8_NOPE_ROW_BYTES,
    DeepSeekV4TokenToKVPool,
    DeepSeekV4UnifiedKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

NOPE_DIM = 448
ROPE_DIM = 64
ROPE_ROW_BYTES = ROPE_DIM * 2
PAGE_SIZE = 256
SWA_RING = 8
NUM_SLOTS = 2
NUM_BLOCKS = 4


class _StubMemorySaver:
    def region(self, _tag):
        return contextlib.nullcontext()


def _unified_pool(stage_ratios, *, fp8):
    return DeepSeekV4UnifiedKVPool(
        stage_ratios=stage_ratios,
        num_slots=NUM_SLOTS,
        num_blocks=NUM_BLOCKS,
        page_size=PAGE_SIZE,
        qk_nope_head_dim=NOPE_DIM,
        qk_rope_head_dim=ROPE_DIM,
        device="cpu",
        memory_saver_adapter=_StubMemorySaver(),
        custom_mem_pool=None,
        swa_ring_size=SWA_RING,
        fp8=fp8,
    )


def _token_pool(stage_ratios, *, fp8, indexer=None):
    uni = _unified_pool(stage_ratios, fp8=fp8)
    pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    pool._unified_kv = True
    pool._unified_kv_fp8 = fp8
    pool.page_size = PAGE_SIZE
    pool.compression_ratios = list(stage_ratios)
    pool._stage_start = 0
    pool._stage_end = len(stage_ratios)
    pool.unified_kv_pool = uni
    pool.unified_swa_window = 128
    pool.unified_swa_ring_size = SWA_RING
    pool.unified_swa_pages = uni.swa_pages
    # same insertion order as _init_compressed_pools
    pool.kv_pools = {4: None, 128: None}
    pool.index_pools = {} if indexer is None else {4: indexer}
    pool.compress_state_pools = [None] * len(stage_ratios)
    pool.indexer_compress_state_pools = [None] * len(stage_ratios)
    pool.get_state_buf_infos = lambda: ([], [], [])
    pool.get_request_state_buf_infos = lambda: ([], [], [])
    return pool


def _paint_rows(buf, start, end, salt):
    u8 = buf.view(torch.uint8).reshape(buf.shape[0], -1)
    for i in range(start, end):
        u8[i, 0] = (i + salt) % 256
        if u8.shape[1] > 1:
            u8[i, 1] = salt % 256


def _pp_mgr(start, end, ratios):
    mgr = object.__new__(CommonKVManager)
    mgr.kv_args = SimpleNamespace(
        prefill_start_layer=start,
        prefill_end_layer=end,
        mla_compression_ratios=ratios,
    )
    return mgr


class TestDSV4UnifiedFp8PdLayout(CustomTestCase):
    STAGE = [4, 128]

    def test_bf16_contiguous_and_ring_stay_one_pool(self):
        pool = _token_pool(self.STAGE, fp8=False)
        ptrs, _, items = pool.get_contiguous_buf_infos()
        self.assertEqual(len(ptrs), 2)
        self.assertEqual(items, [PAGE_SIZE // 4 * 1024, PAGE_SIZE // 128 * 1024])
        ring_ptrs, _, ring_items = pool.get_unified_swa_ring_buf_infos()
        self.assertEqual(len(ring_ptrs), 2)
        self.assertEqual(ring_items, [1024, 1024])

    def test_fp8_adds_rope_groups_with_128_vs_512_item_len(self):
        indexer_buf = torch.empty((2, 8), dtype=torch.uint8)
        indexer = SimpleNamespace(contiguous_page_row_buffers=lambda: [indexer_buf])
        pool = _token_pool(self.STAGE, fp8=True, indexer=indexer)
        ptrs, _, items = pool.get_contiguous_buf_infos()
        # C4_nope, C4_rope, C4_indexer, C128_nope, C128_rope
        self.assertEqual(len(ptrs), 5)
        c4_rows = PAGE_SIZE // 4
        c128_rows = PAGE_SIZE // 128
        self.assertEqual(
            items,
            [
                c4_rows * DSV4_FP8_NOPE_ROW_BYTES,
                c4_rows * ROPE_ROW_BYTES,
                indexer_buf[0].nbytes,
                c128_rows * DSV4_FP8_NOPE_ROW_BYTES,
                c128_rows * ROPE_ROW_BYTES,
            ],
        )
        self.assertEqual(items[1] / items[0], ROPE_ROW_BYTES / DSV4_FP8_NOPE_ROW_BYTES)
        ring_ptrs, _, ring_items = pool.get_unified_swa_ring_buf_infos()
        self.assertEqual(len(ring_ptrs), 4)
        self.assertEqual(
            ring_items,
            [
                DSV4_FP8_NOPE_ROW_BYTES,
                DSV4_FP8_NOPE_ROW_BYTES,
                ROPE_ROW_BYTES,
                ROPE_ROW_BYTES,
            ],
        )

    def test_fake_copy_moves_nope_and_rope(self):
        src = _token_pool(self.STAGE, fp8=True)
        dst = _token_pool(self.STAGE, fp8=True)
        swa = src.unified_kv_pool.swa_pages
        page_ids = [0, 1]
        ring_rows = [0, SWA_RING, SWA_RING + 3]

        for layer, salt in enumerate((11, 22)):
            nope = src.unified_kv_pool.kv_buffer[layer]
            rope = src.unified_kv_pool.kv_buffer_rope[layer]
            _paint_rows(nope, swa, nope.shape[0], salt)
            _paint_rows(rope, swa, rope.shape[0], salt + 50)
            _paint_rows(nope, 0, swa, salt + 7)
            _paint_rows(rope, 0, swa, salt + 57)

        src_kv = src.get_contiguous_buf_infos()
        self.assertEqual(len(src_kv[0]), 4)
        kv_groups = (
            (4, "nope"),
            (4, "rope"),
            (128, "nope"),
            (128, "rope"),
        )
        for i, (ratio, kind) in enumerate(kv_groups):
            local = self.STAGE.index(ratio)
            buf_s = (
                src.unified_kv_pool.kv_buffer[local]
                if kind == "nope"
                else src.unified_kv_pool.kv_buffer_rope[local]
            )
            buf_d = (
                dst.unified_kv_pool.kv_buffer[local]
                if kind == "nope"
                else dst.unified_kv_pool.kv_buffer_rope[local]
            )
            row_bytes = buf_s[0].nbytes
            rows_per_page = PAGE_SIZE // ratio
            self.assertEqual(src_kv[0][i], buf_s.data_ptr() + swa * row_bytes)
            self.assertEqual(src_kv[2][i], rows_per_page * row_bytes)
            s_u8 = buf_s.view(torch.uint8).reshape(buf_s.shape[0], -1)
            d_u8 = buf_d.view(torch.uint8).reshape(buf_d.shape[0], -1)
            for p in page_ids:
                a = swa + p * rows_per_page
                d_u8[a : a + rows_per_page].copy_(s_u8[a : a + rows_per_page])
                self.assertTrue(
                    torch.equal(
                        s_u8[a : a + rows_per_page], d_u8[a : a + rows_per_page]
                    )
                )

        src_ring = src.get_unified_swa_ring_buf_infos()
        self.assertEqual(len(src_ring[0]), 4)
        ring_src = list(src.unified_kv_pool.kv_buffer) + list(
            src.unified_kv_pool.kv_buffer_rope
        )
        ring_dst = list(dst.unified_kv_pool.kv_buffer) + list(
            dst.unified_kv_pool.kv_buffer_rope
        )
        for i, (buf_s, buf_d) in enumerate(zip(ring_src, ring_dst)):
            self.assertEqual(src_ring[0][i], buf_s.data_ptr())
            self.assertEqual(src_ring[2][i], buf_s[0].nbytes)
            s_u8 = buf_s.view(torch.uint8).reshape(buf_s.shape[0], -1)
            d_u8 = buf_d.view(torch.uint8).reshape(buf_d.shape[0], -1)
            for r in ring_rows:
                d_u8[r].copy_(s_u8[r])
                self.assertTrue(torch.equal(s_u8[r], d_u8[r]))

    def test_dspark_draft_ring_stays_single_pool(self):
        draft = _token_pool([0], fp8=False)
        draft.compression_ratios = [0]
        ptrs, _, items = draft.get_unified_swa_ring_buf_infos()
        self.assertEqual(len(ptrs), 1)
        self.assertEqual(items, [1024])
        self.assertIsNone(draft.unified_kv_pool.kv_buffer_rope[0])

    def test_fp8_draft_ring_includes_rope(self):
        draft = _token_pool([0], fp8=True)
        draft.compression_ratios = [0]
        ptrs, _, items = draft.get_unified_swa_ring_buf_infos()
        self.assertEqual(len(ptrs), 2)
        self.assertEqual(items, [DSV4_FP8_NOPE_ROW_BYTES, ROPE_ROW_BYTES])
        kv_args = KVArgs()
        target = _token_pool(self.STAGE, fp8=True)
        setup_state_kv_args(kv_args, target, draft)
        self.assertEqual(kv_args.state_types[-1], StateType.SWA_RING)
        self.assertEqual(kv_args.state_item_lens[-1], items)

    def test_get_buf_infos_and_hicache_still_refuse(self):
        uni = _unified_pool(self.STAGE, fp8=True)
        with self.assertRaises(NotImplementedError):
            uni.get_buf_infos()
        pool = _token_pool(self.STAGE, fp8=True)
        with self.assertRaises(NotImplementedError):
            pool.unified_region_buffers(4)


class TestDSV4UnifiedFp8PpSlice(CustomTestCase):
    RATIOS = [4, 4, 128, 4]

    def _slice(self, dst, start, end, state_type=None):
        mgr = _pp_mgr(start, end, self.RATIOS)
        src = list(range(100, 100 + 8))
        return mgr._mla_slice_ptrs_for_pp(src, dst, self.RATIOS, state_type)

    def test_bf16_kv_layout_unchanged(self):
        # [C4 x3 | indexer x3 | C128 x1]
        dst = list(range(7))
        _, sliced = self._slice(dst, 0, 2)
        self.assertEqual(sliced, [0, 1, 3, 4])
        _, sliced = self._slice(dst, 2, 3)
        self.assertEqual(sliced, [6])

    def test_fp8_kv_layout_five_groups(self):
        # [C4_nope x3 | C4_rope x3 | idx x3 | C128_nope x1 | C128_rope x1]
        dst = list(range(11))
        _, sliced = self._slice(dst, 0, 2)
        self.assertEqual(sliced, [0, 1, 3, 4, 6, 7])
        _, sliced = self._slice(dst, 2, 3)
        self.assertEqual(sliced, [9, 10])
        _, sliced = self._slice(dst, 3, 4)
        self.assertEqual(sliced, [2, 5, 8])

    def test_swa_ring_fp8_splits_nope_then_rope(self):
        dst = list(range(8))
        _, sliced = self._slice(dst, 1, 3, StateType.SWA_RING)
        self.assertEqual(sliced, [1, 2, 5, 6])

    def test_swa_ring_bf16_and_draft_len_not_double_sliced(self):
        dst = list(range(4))
        _, sliced = self._slice(dst, 1, 3, StateType.SWA_RING)
        self.assertEqual(sliced, [1, 2])
        # 1-layer fp8 draft: 2 ptrs, not 2 * len(mla_ratios)
        dst = [10, 11]
        _, sliced = self._slice(dst, 1, 3, StateType.SWA_RING)
        self.assertEqual(sliced, [11])


if __name__ == "__main__":
    unittest.main()
