import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.static_kv_storage import _static_state_entries
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


def _pp_mgr(start, end, ratios):
    mgr = object.__new__(CommonKVManager)
    mgr.kv_args = SimpleNamespace(
        prefill_start_layer=start,
        prefill_end_layer=end,
        mla_compression_ratios=ratios,
    )
    return mgr


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


class TestUnifiedC4StateTransfer(CustomTestCase):
    def _pool(self, *, unified=True, ring_size=8, with_c4=True):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = unified
        pool.request_window = None
        pool.swa_kv_pool = None
        pool.layer_router = SimpleNamespace(sub_pool=lambda _: None)
        pool.compression_ratios = [4, 128]
        pool.get_ring_size = lambda _: ring_size
        pool.get_unified_swa_ring_buf_infos = lambda: ([100], [1024], [16])
        pool.compress_state_pools = []
        pool.indexer_compress_state_pools = []
        if with_c4:
            for pools, width in (
                (pool.compress_state_pools, 16),
                (pool.indexer_compress_state_pools, 4),
            ):
                pools.append(
                    SimpleNamespace(
                        ratio=4,
                        ring_size=ring_size,
                        request_scoped=False,
                        kv_score_buffer=SimpleNamespace(
                            kv_score=torch.arange(64 * width).reshape(64, width)
                        ),
                    )
                )
        return pool

    def test_unified_c4_is_separate_and_row_sized(self):
        pool = self._pool()
        self.assertEqual(pool.get_state_buf_infos(), ([], [], []))
        ptrs, lengths, items = pool.get_c4_state_buf_infos()
        self.assertEqual(len(ptrs), 2)
        self.assertEqual(lengths, [64 * 16 * 8, 64 * 4 * 8])
        self.assertEqual(items, [16 * 8, 4 * 8])

    def test_non_unified_layout_is_preserved(self):
        pool = self._pool(unified=False)
        self.assertEqual(pool.get_c4_state_buf_infos(), ([], [], []))
        self.assertEqual(pool.get_state_buf_infos()[2], [16 * 8 * 8, 4 * 8 * 8])

    def test_local_request_slots_and_ring_sizes_pair_by_position(self):
        src = self._pool(ring_size=8).get_c4_state_transfer_indices(2, 13)
        dst = self._pool(ring_size=16).get_c4_state_transfer_indices(5, 13)
        self.assertEqual(src.tolist(), [16, 17, 18, 19, 20])
        self.assertEqual(dst.tolist(), [88, 89, 90, 91, 92])

    @patch("sglang.srt.mem_cache.static_kv_storage.is_npu", return_value=False)
    def test_pp_stage_without_c4_keeps_the_component(self, _):
        schemas = []
        for with_c4 in (True, False):
            entries = _static_state_entries(
                self._pool(with_c4=with_c4),
                draft_token_to_kv_pool=None,
                total_kv_layers=None,
                state_pool=None,
            )
            schemas.append([e.state_type for e in entries.wire_entries])
            c4 = next(
                entries.regions[e.sub_pool]
                for e in entries.wire_entries
                if e.state_type == StateType.DSV4_C4_STATE
            )
            self.assertEqual(bool(c4.ptrs), with_c4)
        self.assertEqual(schemas[0], schemas[1])

    def test_pp_slices_both_attention_and_indexer_groups(self):
        ratios = [4, 128, 4, 128]
        mgr = _pp_mgr(2, 4, ratios)
        src, dst, count = mgr.get_mla_kv_ptrs_with_pp(
            [100, 101], [10, 11, 20, 21], StateType.DSV4_C4_STATE
        )
        self.assertEqual(src, [100, 101])
        self.assertEqual(dst, [11, 21])
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
