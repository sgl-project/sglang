import unittest

import torch

from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestGdnPackedDecode(CustomTestCase):
    def test_wide_batch_matches_split_launch(self):
        """Batch times value heads can exceed the CUDA grid y limit."""
        torch.manual_seed(0)
        width, hv, k, v = 5120, 16, 16, 48
        state = torch.randn(width, hv, v, k, device="cuda")
        qkv = torch.randn(width, 8 * k + hv * v, device="cuda", dtype=torch.bfloat16)
        a = qkv.new_zeros(width, hv).normal_()
        b = torch.randn_like(a)
        a_log, dt_bias = torch.randn(2, hv, device="cuda")
        slots = torch.randperm(width, device="cuda")

        def decode(chunk_size):
            cache = state.clone()
            out = qkv.new_empty(width, 1, hv, v)
            for start in range(0, width, chunk_size):
                s = slice(start, start + chunk_size)
                fused_recurrent_gated_delta_rule_packed_decode(
                    qkv[s],
                    a[s],
                    b[s],
                    a_log,
                    dt_bias,
                    k**-0.5,
                    cache,
                    out[s],
                    slots[s],
                    use_qk_l2norm_in_kernel=True,
                )
            return out, cache

        torch.testing.assert_close(decode(width), decode(width // 2), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
