import unittest

import torch

from sglang.srt.layers.moe.topk import (
    biased_grouped_topk_gpu,
)
from sglang.srt.layers.moe.topk import (
    biased_grouped_topk_impl as native_biased_grouped_topk,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=5, suite="stage-a-test-1-gpu-xpu")


# Nemotron-3 uses biased_grouped_topk
class TestBiasedGroupedTopK(CustomTestCase):
    def _run_single_test(
        self,
        M,
        E,
        G,
        topk,
        topk_group,
        renormalize,
        gating_dtype,
        bias_dtype,
        routed_scaling_factor,
    ):
        torch.manual_seed(1024)
        device = torch.device("xpu")

        # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
        hidden_states = torch.randn(M, 100, dtype=torch.bfloat16, device=device)
        gating_output = torch.randn(M, E, dtype=gating_dtype, device=device)
        correction_bias = torch.randn(E, dtype=bias_dtype, device=device)

        ref_topk_weights, ref_topk_ids = native_biased_grouped_topk(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            G,
            topk_group,
            routed_scaling_factor=routed_scaling_factor,
        )

        # fused version
        topk_weights, topk_ids = biased_grouped_topk_gpu(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            G,
            topk_group,
            0,
            routed_scaling_factor,
            None,
        )

        res = torch.zeros(M, E, dtype=torch.float, device=device)
        ref = torch.zeros(M, E, dtype=torch.float, device=device)
        res.scatter_(1, topk_ids.long(), topk_weights)
        ref.scatter_(1, ref_topk_ids.long(), ref_topk_weights)
        torch.testing.assert_close(res, ref)

    # Nemotron-3-Nano-30B-A3B uses fast biased_grouped_topk with num_expert_group = 1 and topk_group = 1
    def test_fast_biased_grouped_topk(self):
        # The test config is also from this nemotron model.
        E_num = 128
        num_expert_group = 1
        topk_value = 6
        topk_group = 1
        gating_dtype = torch.bfloat16
        bias_dtype = torch.float32
        renormalize = True
        routed_scaling_factor = 2.5

        bs = [1, 2, 4, 8]
        seq_len = 1024
        num_tokens = [b * seq_len for b in bs]

        for M in num_tokens:
            self._run_single_test(
                M,
                E_num,
                num_expert_group,
                topk_value,
                topk_group,
                renormalize,
                gating_dtype,
                bias_dtype,
                routed_scaling_factor,
            )


if __name__ == "__main__":
    unittest.main()
