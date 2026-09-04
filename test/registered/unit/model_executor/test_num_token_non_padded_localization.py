"""Per-rank-local non-padded token count across attn x MoE parallelism layouts.

``compute_local_num_token_non_padded`` (GPU tensor) and
``compute_local_num_token_non_padded_cpu`` (host int) convert a dp-group-global
real-token count into this attention-TP rank's local count. Each rank owns a
contiguous ``padded_bucket // attn_tp_size`` slice of the padded sequence, so the
localizer clamps ``real - chunk * attn_tp_rank`` into ``[0, chunk]``: a replicated
(non-sharded) rank keeps the full count and SP ranks split it. The value is
identical whether the MoE runs TP or EP -- it is an attention-side quantity both
backends consume. This table locks the exact per-rank counts and that the GPU
tensor and host-int twin agree, so a change to the sharding math fails loudly.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import (
    compute_local_num_token_non_padded,
    compute_local_num_token_non_padded_cpu,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import CustomTestCase


class TestNumTokenNonPaddedLayoutTable(CustomTestCase):
    # (label, attn_tp_size, sharded, padded_bucket, real-per-dp-group,
    #  expected [per attn-tp rank] per dp group)
    _LAYOUTS = [
        # Each dp rank gets a 10-token request, cuda graph pads it to 16.
        ("TP4", 4, False, 16, [10], [[10, 10, 10, 10]]),
        ("TP4.SP4", 4, True, 16, [10], [[4, 4, 2, 0]]),
        ("DP4", 1, False, 16, [10, 10, 10, 10], [[10], [10], [10], [10]]),
        ("TP2.DP2.SP2", 2, True, 16, [10, 10], [[8, 2], [8, 2]]),
        # dp0/dp2 get 10 tokens, dp1/dp3 get 20; all padded to 32.
        ("DP4.EP4", 1, False, 32, [10, 20, 10, 20], [[10], [20], [10], [20]]),
        ("TP2.DP2.SP2.EP2", 2, True, 32, [10, 20], [[10, 0], [16, 4]]),
    ]

    def test_layouts_match_expected_per_rank(self):
        for label, attn_tp, sharded, bucket, dp_reals, expected in self._LAYOUTS:
            for dp_idx, real in enumerate(dp_reals):
                for rank in range(attn_tp):
                    want = expected[dp_idx][rank]
                    with (
                        self.subTest(layout=label, dp=dp_idx, rank=rank),
                        get_parallel().override(
                            attn_tp_size=attn_tp, attn_tp_rank=rank
                        ),
                    ):
                        got_cpu = compute_local_num_token_non_padded_cpu(
                            global_num_token_non_padded=real,
                            num_tokens_per_dp=bucket,
                            sharded=sharded,
                        )
                        got_gpu = compute_local_num_token_non_padded(
                            global_num_token_non_padded=torch.tensor(real),
                            num_tokens_per_dp=bucket,
                            sharded=sharded,
                        )
                        self.assertEqual(got_cpu, want)
                        self.assertEqual(int(got_gpu), want)


if __name__ == "__main__":
    unittest.main()
