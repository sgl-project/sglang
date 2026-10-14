import contextlib
import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
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

    def test_get_buf_infos_still_refuses(self):
        uni = _unified_pool(self.STAGE, fp8=True)
        with self.assertRaises(NotImplementedError):
            uni.get_buf_infos()
        # #37778 already ships the rope host pool; PD must not re-refuse it.
        pool = _token_pool(self.STAGE, fp8=True)
        pool.unified_region_buffers(4)
        self.assertIsNotNone(pool.unified_rope_region_buffers(4))


class TestDSV4UnifiedFp8PpSlice(CustomTestCase):
    RATIOS = [4, 4, 128, 4]
    # kv_data groups: 2*c4 + c128 bf16, 3*c4 + 2*c128 under the fp8 two-pool
    BF16_KV_LEN = 7
    FP8_KV_LEN = 11

    def _slice(self, dst, start, end, state_type=None, src=None):
        mgr = _pp_mgr(start, end, self.RATIOS)
        if src is None:
            src = list(range(100, 100 + 8))
        return mgr._mla_slice_ptrs_for_pp(src, dst, self.RATIOS, state_type)

    def test_bf16_kv_layout_unchanged(self):
        # [C4 x3 | indexer x3 | C128 x1]
        dst = list(range(self.BF16_KV_LEN))
        _, sliced = self._slice(dst, 0, 2)
        self.assertEqual(sliced, [0, 1, 3, 4])
        _, sliced = self._slice(dst, 2, 3)
        self.assertEqual(sliced, [6])

    def test_same_layout_peers_never_reach_the_slicer(self):
        # PD with pp_size=1 always lands here, whichever pool layout is in use;
        # the fp8 group counts are only ever produced on both sides at once.
        mgr = _pp_mgr(0, len(self.RATIOS), self.RATIOS)
        for n in (self.BF16_KV_LEN, self.FP8_KV_LEN):
            ptrs = list(range(n))
            src, dst, count = mgr.get_mla_kv_ptrs_with_pp(ptrs, list(ptrs))
            self.assertEqual((src, dst, count), (ptrs, ptrs, n))

    def test_mixed_fp8_and_bf16_peers_are_rejected(self):
        cases = (
            (None, self.BF16_KV_LEN, self.FP8_KV_LEN),
            (None, self.FP8_KV_LEN, self.BF16_KV_LEN),
            (StateType.SWA_RING, len(self.RATIOS), 2 * len(self.RATIOS)),
            (StateType.SWA_RING, 2 * len(self.RATIOS), len(self.RATIOS)),
        )
        for state_type, src_len, dst_len in cases:
            with self.subTest(state_type=state_type, src_len=src_len):
                with self.assertRaisesRegex(ValueError, "SGLANG_DSV4_UNIFIED_KV_FP8"):
                    self._slice(
                        list(range(dst_len)),
                        1,
                        3,
                        state_type,
                        src=list(range(100, 100 + src_len)),
                    )

    def test_swa_ring_bf16_slice_unchanged(self):
        # stage [1, 3) registers 2 of the 4 bf16 rings
        dst = list(range(4))
        _, sliced = self._slice(dst, 1, 3, StateType.SWA_RING, src=[100, 101])
        self.assertEqual(sliced, [1, 2])


if __name__ == "__main__":
    unittest.main()
