"""Logits selection must ignore dummy requests on idle DP ranks."""

import unittest

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestIdlePrefillLogits(unittest.TestCase):
    def test_idle_rank_does_not_index_dummy_last_token(self):
        # MLP-sync turns an idle rank into a dummy zero-token EXTEND batch.
        empty = torch.empty(0, dtype=torch.int64)
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=empty,
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([0]),
            out_cache_loc=empty,
            seq_lens_sum=0,
            positions=empty,
            extend_seq_lens=torch.tensor([0]),
            extend_seq_lens_cpu=[0],
            _original_forward_mode=ForwardMode.IDLE,
            _original_batch_size=0,
        )
        hidden = torch.empty(0, 4)
        pruned, *_ = LogitsProcessor._get_pruned_states(
            None, hidden, None, None, LogitsMetadata.from_forward_batch(batch)
        )
        self.assertEqual(pruned.shape, (0, 4))
        # Attention and MLP execution still use the padded mode.
        self.assertEqual(batch.forward_mode, ForwardMode.EXTEND)


if __name__ == "__main__":
    unittest.main()
