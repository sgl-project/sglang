"""CPU checks for rank-common Engram collective admission, not NCCL execution."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.layers import engram
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    set_dp_buffer_len,
    set_dp_buffer_len_from_batch,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEngramCollectiveAdmission(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(engram, "is_cuda", return_value=True))
        self.enterContext(
            patch.object(engram, "is_dp_gatherv_active", return_value=False)
        )
        self.enterContext(
            patch.object(
                engram, "get_device", return_value=SimpleNamespace(device="cuda")
            )
        )
        self.enterContext(
            get_parallel().override(
                tp_size=2,
                attn_dp_size=2,
                attn_dp_rank=0,
                attn_tp_size=1,
                attn_cp_size=1,
                num_dp_ranks=2,
            )
        )
        self.enterContext(get_flags().dp.override(use_world_group_for_gather=False))
        self.enterContext(envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(False))
        self.addCleanup(set_dp_buffer_len, 0, 0, False)

    def batch(self, counts):
        batch = SimpleNamespace(
            global_dp_buffer_len=sum(counts),
            global_num_tokens_padded_cpu=list(counts),
            global_num_tokens_gpu=None,
            dp_padding_mode=DpPaddingMode.SUM_LEN,
            is_extend_in_batch=True,
            can_run_dp_prefill_cuda_graph=False,
        )
        set_dp_buffer_len_from_batch(batch)
        return batch

    def test_uniform_threshold_and_uneven_or_idle_shards(self):
        for counts, expected in (
            ([32767, 32767], None),
            ([32768, 32768], 32768),
            ([33792, 33792], 33792),
            ([33792, 32768], None),
            ([33792, 0], None),
            ([0, 0], None),
        ):
            with self.subTest(counts=counts):
                self.assertEqual(
                    engram._uniform_sum_len_scatter_rows(
                        self.batch(counts), 2, sum(counts)
                    ),
                    expected,
                )

    def test_published_counts_and_global_rows_must_match(self):
        batch = self.batch([33792, 33792])
        self.assertIsNone(engram._uniform_sum_len_scatter_rows(batch, 2, 67583))
        batch.global_num_tokens_padded_cpu = [32768, 32768]
        self.assertIsNone(engram._uniform_sum_len_scatter_rows(batch, 2, 65536))

    def test_local_mode_does_not_select_the_collective(self):
        batch = self.batch([33792, 33792])
        for rank, mode in ((0, ForwardMode.EXTEND), (1, ForwardMode.IDLE)):
            batch.forward_mode = mode
            with get_parallel().override(attn_dp_rank=rank):
                self.assertEqual(
                    engram._uniform_sum_len_scatter_rows(batch, 2, 67584), 33792
                )
        batch.can_run_dp_prefill_cuda_graph = True
        self.assertIsNone(engram._uniform_sum_len_scatter_rows(batch, 2, 67584))

    def test_world_expansion_and_scoped_attention_fall_back(self):
        batch = self.batch([33792, 33792])
        with get_flags().dp.override(use_world_group_for_gather=True):
            self.assertEqual(
                engram._uniform_sum_len_scatter_rows(batch, 2, 67584), 33792
            )
            with get_parallel().override(num_dp_ranks=4):
                self.assertIsNone(engram._uniform_sum_len_scatter_rows(batch, 2, 67584))
        with get_parallel().override(attn_dp_size=1, attn_tp_size=2):
            self.assertIsNone(engram._uniform_sum_len_scatter_rows(batch, 2, 67584))

    def test_cpu_device_and_deterministic_mode_fall_back(self):
        batch = self.batch([33792, 33792])
        with patch.object(
            engram, "get_device", return_value=SimpleNamespace(device="cpu")
        ):
            self.assertIsNone(engram._uniform_sum_len_scatter_rows(batch, 2, 67584))
        with envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(True):
            self.assertIsNone(engram._uniform_sum_len_scatter_rows(batch, 2, 67584))


if __name__ == "__main__":
    unittest.main()
