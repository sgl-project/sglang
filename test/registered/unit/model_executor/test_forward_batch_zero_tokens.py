import unittest
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    compute_local_num_token_non_padded,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestForwardBatchZeroTokens(unittest.TestCase):
    def test_idle_dp_rank_avoids_device_arithmetic(self):
        count = torch.tensor(0, dtype=torch.int32)
        batch = object.__new__(ForwardBatch)
        batch.global_num_tokens_cpu = [7, 0]
        batch.num_token_non_padded = count
        batch.num_token_non_padded_cpu = 0

        with (
            get_parallel().override(attn_dp_rank=1),
            patch("sglang.srt.utils.common.require_mlp_tp_gather", return_value=True),
            patch(
                "sglang.srt.model_executor.forward_batch_info.compute_local_num_token_non_padded"
            ) as compute,
        ):
            batch.adjust_num_token_non_padded_for_attn_tp(object())

        compute.assert_not_called()
        self.assertIs(batch.num_token_non_padded, count)

    def test_idle_dp_rank_requires_zero_cpu_mirror(self):
        batch = object.__new__(ForwardBatch)
        batch.global_num_tokens_cpu = [7, 0]
        batch.num_token_non_padded = torch.tensor(1, dtype=torch.int32)
        batch.num_token_non_padded_cpu = 1

        with (
            get_parallel().override(attn_dp_rank=1),
            patch("sglang.srt.utils.common.require_mlp_tp_gather", return_value=True),
            self.assertRaises(AssertionError),
        ):
            batch.adjust_num_token_non_padded_for_attn_tp(object())

    def test_nonempty_rank_keeps_local_clamp(self):
        count = torch.tensor(7, dtype=torch.int32)

        with get_parallel().override(attn_tp_size=2, attn_tp_rank=1):
            local = compute_local_num_token_non_padded(
                global_num_token_non_padded=count,
                num_tokens_per_dp=8,
            )

        self.assertEqual(local.item(), 3)


if __name__ == "__main__":
    unittest.main()
