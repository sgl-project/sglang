"""Unit tests for srt/disaggregation/common/staging_buffer -- compute_head_slice_params.

Wrong head indices do not fail loudly: the transfer delivers the wrong channels
and the only symptom is a garbled end-to-end accuracy score. The replication
cases are the ones no end-to-end test can reach -- every heterogeneous-TP suite
runs tp <= total_kv_heads on both sides, where a modulo map and the correct
divide-by-replication map agree.

Expected values are derived by hand from the head-distribution rules, never by
calling the implementation, so a bug in it cannot make both sides agree.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


import unittest

from sglang.srt.disaggregation.common.staging_buffer import compute_head_slice_params
from sglang.test.test_utils import CustomTestCase


class TestHeadSliceParamsGather(CustomTestCase):
    SRC_TP, DST_TP, KV_HEADS = 4, 2, 8

    def _call(self, src_rank, dst_rank=0):
        return compute_head_slice_params(
            self.SRC_TP, self.DST_TP, src_rank, dst_rank, self.KV_HEADS
        )

    def test_prefill_ranks_tile_the_decode_head_range(self):
        self.assertEqual([self._call(r)[2] for r in range(4)], [0, 2, 0, 2])

    def test_rank_is_taken_modulo_the_tp_group(self):
        for src_rank in range(self.SRC_TP):
            self.assertEqual(self._call(src_rank), self._call(src_rank + self.SRC_TP))


class TestHeadSliceParamsScatter(CustomTestCase):
    SRC_TP, DST_TP, KV_HEADS = 2, 4, 8

    def _call(self, dst_rank, src_rank=0):
        return compute_head_slice_params(
            self.SRC_TP, self.DST_TP, src_rank, dst_rank, self.KV_HEADS
        )

    def test_decode_ranks_walk_the_prefill_head_range(self):
        self.assertEqual([self._call(d)[0] for d in range(4)], [0, 2, 0, 2])


class TestHeadSliceParamsScatterReplication(CustomTestCase):
    """More decode ranks than KV heads: consecutive decode ranks share a head."""

    SRC_TP, DST_TP, KV_HEADS = 1, 4, 2

    def _call(self, dst_rank):
        return compute_head_slice_params(
            self.SRC_TP, self.DST_TP, 0, dst_rank, self.KV_HEADS
        )

    def test_replicating_decode_ranks_read_the_same_head(self):
        starts = [self._call(d)[0] for d in range(4)]
        self.assertEqual(starts, [0, 0, 1, 1])
        self.assertNotEqual(starts, [0, 1, 0, 1], "a modulo map would give this")
        for dst_rank in range(self.DST_TP):
            self.assertEqual(
                self._call(dst_rank)[1], 1, "max(1, 2 // 4) clamps to one head"
            )


class TestHeadSliceParamsGatherReplication(CustomTestCase):
    """More prefill ranks than KV heads -- the src_replication branch."""

    SRC_TP, DST_TP, KV_HEADS = 8, 2, 4

    def _call(self, src_rank):
        return compute_head_slice_params(
            self.SRC_TP, self.DST_TP, src_rank, 0, self.KV_HEADS
        )

    def test_replicating_prefill_ranks_write_the_same_head(self):
        starts = [self._call(r)[2] for r in range(8)]
        self.assertEqual(starts, [0, 0, 1, 1, 0, 0, 1, 1])
        self.assertNotEqual(
            starts, [0, 1, 0, 1, 0, 1, 0, 1], "a modulo map would give this"
        )
        for src_rank in range(self.SRC_TP):
            self.assertEqual(
                self._call(src_rank)[1], 1, "max(1, 4 // 8) clamps to one head"
            )


class TestHeadSliceParamsEqualTp(CustomTestCase):
    def test_equal_tp_copies_the_whole_rank_slice(self):
        for rank in range(4):
            src_start, num_heads, dst_start, _ = compute_head_slice_params(
                4, 4, rank, rank, 8
            )
            self.assertEqual((src_start, num_heads, dst_start), (0, 2, 0))


if __name__ == "__main__":
    unittest.main()
