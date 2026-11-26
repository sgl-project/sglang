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
        torch.manual_seed(0)
        device = "cuda"
        rows, heads, value_heads, key_dim, value_dim = 5120, 4, 16, 16, 48
        qkv = torch.randn(
            rows,
            2 * heads * key_dim + value_heads * value_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        a = torch.randn(rows, value_heads, device=device, dtype=qkv.dtype)
        b = torch.randn_like(a)
        a_log = torch.randn(value_heads, device=device)
        dt_bias = torch.randn_like(a_log)
        slots = torch.randperm(rows, device=device)
        initial_state = torch.randn(
            rows, value_heads, value_dim, key_dim, device=device
        )

        def decode(chunks):
            state = initial_state.clone()
            out = qkv.new_empty((rows, 1, value_heads, value_dim))
            for chunk in chunks:
                fused_recurrent_gated_delta_rule_packed_decode(
                    mixed_qkv=qkv[chunk],
                    a=a[chunk],
                    b=b[chunk],
                    A_log=a_log,
                    dt_bias=dt_bias,
                    scale=key_dim**-0.5,
                    initial_state=state,
                    out=out[chunk],
                    ssm_state_indices=slots[chunk],
                    use_qk_l2norm_in_kernel=True,
                )
            return out, state

        expected = decode((slice(2560), slice(2560, None)))
        actual = decode((slice(None),))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
