import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.utils import get_dsv41_spec_layout, setup_state_kv_args
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


def _unified_pool(stage_ratios, *, fp8, group_of=None):
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
        group_of=group_of,
    )


def _token_pool(stage_ratios, *, fp8, indexer=None, group_of=None):
    uni = _unified_pool(stage_ratios, fp8=fp8, group_of=group_of)
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


class TestDSV41UnifiedDSparkPdHandshake(CustomTestCase):
    def test_target_and_draft_rings_describe_a_dspark_layout(self):
        """V4.1 DSpark PD on unified KV ships both SWA rings as SWA_RING; the spec
        layout handshake must accept them instead of demanding two paged SWA states."""
        target = _token_pool([2, 4, 128], fp8=True, group_of=[0, None, None])
        draft = _token_pool([0], fp8=False)
        draft.get_swa_valid_from_buf_infos = lambda: ([], [], [])
        kv_args = SimpleNamespace(kv_layer_ids=[0, 1, 2], kv_item_lens=[1, 1, 1])
        setup_state_kv_args(kv_args, target, draft_token_to_kv_pool=draft)
        spec = SimpleNamespace(
            speculative_algorithm="DSPARK", speculative_num_draft_tokens=6
        )
        with patch("sglang.srt.disaggregation.utils.get_spec", return_value=spec):
            layout = get_dsv41_spec_layout(kv_args)
        self.assertEqual(
            layout["state_types"].count(StateType.SWA_RING.value), 2, layout
        )


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


if __name__ == "__main__":
    unittest.main()
