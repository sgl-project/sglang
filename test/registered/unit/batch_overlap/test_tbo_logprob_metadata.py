"""Keep input-logprob metadata on the parent used after TBO hidden-state merge."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestTboLogprobMetadata(CustomTestCase):
    def test_input_logprobs_remain_on_parent_for_empty_and_populated_requests(self):
        for starts in ([3, 2], [1, 0]):
            with self.subTest(logprob_starts=starts):
                ids = torch.arange(5 - sum(starts), dtype=torch.int64)
                parent = ForwardBatch(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=2,
                    input_ids=torch.arange(5),
                    positions=torch.tensor([0, 1, 2, 0, 1]),
                    out_cache_loc=torch.arange(5),
                    req_pool_indices=torch.arange(2),
                    seq_lens=torch.tensor([3, 2]),
                    seq_lens_cpu=torch.tensor([3, 2]),
                    seq_lens_sum=5,
                    extend_num_tokens=5,
                    extend_seq_lens=torch.tensor([3, 2]),
                    extend_seq_lens_cpu=[3, 2],
                    extend_prefix_lens=torch.tensor([0, 0]),
                    extend_prefix_lens_cpu=[0, 0],
                    extend_start_loc=torch.tensor([0, 3]),
                    extend_logprob_start_lens_cpu=starts,
                    extend_input_logprob_token_ids_gpu=ids,
                    return_logprob=True,
                )
                with patch(
                    "sglang.srt.batch_overlap.two_batch_overlap.get_parallel",
                    return_value=SimpleNamespace(attn_tp_size=1, moe_dense_tp_size=1),
                ):
                    for lo, hi, seq_lo, seq_hi in (
                        (0, 0, 0, 0),
                        (0, 3, 0, 1),
                        (3, 5, 1, 2),
                    ):
                        child = TboForwardBatchPreparer.filter_batch(
                            parent,
                            start_token_index=lo,
                            end_token_index=hi,
                            start_seq_index=seq_lo,
                            end_seq_index=seq_hi,
                            out_num_token_non_padded=torch.tensor(hi - lo),
                        )
                        self.assertTrue(child.return_logprob)
                        self.assertIsNone(child.extend_input_logprob_token_ids_gpu)
                        torch.testing.assert_close(
                            child.input_ids, parent.input_ids[lo:hi]
                        )
                self.assertIs(parent.extend_input_logprob_token_ids_gpu, ids)
                torch.testing.assert_close(ids, torch.arange(5 - sum(starts)))


if __name__ == "__main__":
    unittest.main()
