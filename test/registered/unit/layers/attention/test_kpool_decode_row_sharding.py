import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import dsa_dcp
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _shards_by_rank(rows, dcp_size, attn_tp_size):
    shards = []
    for rank in range(dcp_size):
        group = SimpleNamespace(world_size=dcp_size, rank_in_group=rank)
        with get_parallel().override(
            attn_dcp_size=dcp_size, attn_tp_size=attn_tp_size, dcp_group=group
        ):
            shards.append(dsa_dcp.kpool_decode_row_shard(rows))
    return shards


class TestKPoolDecodeRowSharding(CustomTestCase):
    def test_rank_major_gather_preserves_request_order(self):
        with (
            patch.object(dsa_dcp, "is_cuda", return_value=True),
            envs.SGLANG_OPT_DSA_KPOOL_DECODE_ROW_SHARD.override(True),
        ):
            for rows in (8, 64, 65):
                with self.subTest(rows=rows):
                    requests = list(range(rows))
                    gathered = []
                    shards = _shards_by_rank(rows, dcp_size=8, attn_tp_size=8)
                    stride = (rows + 7) // 8
                    for width, owned in shards:
                        self.assertEqual(width, stride)
                        local = requests[owned]
                        self.assertTrue(local)
                        gathered.extend(local + [-1] * (width - len(local)))
                    self.assertEqual(gathered, requests + [-1] * (stride * 8 - rows))

    def test_all_ranks_fall_back_together(self):
        cases = (
            (True, True, 64, 1, 8),  # DCP1
            (True, True, 64, 4, 8),  # DCP covers part of attention TP
            (True, True, 7, 8, 8),  # fewer rows than ranks
            (True, True, 9, 8, 8),  # uniform strides would leave ranks empty
            (False, True, 64, 8, 8),  # feature disabled
            (True, False, 64, 8, 8),  # non-CUDA device
        )
        for enabled, cuda, rows, dcp_size, attn_tp_size in cases:
            with (
                self.subTest(case=(enabled, cuda, rows, dcp_size, attn_tp_size)),
                patch.object(dsa_dcp, "is_cuda", return_value=cuda),
                envs.SGLANG_OPT_DSA_KPOOL_DECODE_ROW_SHARD.override(enabled),
            ):
                shards = _shards_by_rank(rows, dcp_size, attn_tp_size)
                self.assertEqual(shards, [None] * dcp_size)


if __name__ == "__main__":
    unittest.main()
