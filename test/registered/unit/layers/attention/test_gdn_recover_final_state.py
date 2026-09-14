"""Compare the recovery-only Triton recurrence with a small PyTorch reference."""

import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_recover_final_state,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestGDNRecoverFinalState(CustomTestCase):
    def test_accepted_prefix_and_boundary_output_slots(self):
        torch.manual_seed(0)
        n, steps, heads, value_heads, width = 3, 4, 1, 2, 32
        k = torch.randn(1, n * steps, heads, width, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(
            1, n * steps, value_heads, width, device="cuda", dtype=torch.bfloat16
        )
        a = torch.randn(n * steps, value_heads, device="cuda")
        b = torch.randn(n * steps, value_heads, device="cuda")
        a_log = torch.randn(value_heads, device="cuda")
        bias = torch.randn(value_heads, device="cuda")
        accepted = torch.tensor([-1, 0, 3], device="cuda", dtype=torch.int32)
        src = torch.arange(n, device="cuda", dtype=torch.int32)
        for dtype in (torch.float32, torch.bfloat16):
            for separate in (False, True):
                with self.subTest(dtype=dtype, separate=separate):
                    state = (
                        torch.randn(
                            5, value_heads, width, width, device="cuda", dtype=dtype
                        )
                        * 0.1
                    )
                    expected = state.float().clone()
                    dst = (
                        torch.tensor([3, -1, 4], device="cuda", dtype=torch.int32)
                        if separate
                        else src
                    )
                    for row in range(n):
                        out_slot = int(dst[row])
                        if out_slot < 0:
                            continue
                        h = state[row].float().clone()
                        for step in range(int(accepted[row]) + 1):
                            pos = row * steps + step
                            key = (
                                k[0, pos]
                                .float()
                                .repeat_interleave(value_heads // heads, dim=0)
                            )
                            key = key / torch.sqrt(
                                key.square().sum(-1, keepdim=True) + 1e-6
                            )
                            decay = torch.exp(-a_log.exp() * F.softplus(a[pos] + bias))
                            h = h * decay[:, None, None]
                            delta = v[0, pos].float() - (h * key[:, None, :]).sum(-1)
                            delta = delta * b[pos].sigmoid()[:, None]
                            h = h + delta[:, :, None] * key[:, None, :]
                        expected[out_slot] = h
                    fused_sigmoid_gating_delta_rule_recover_final_state(
                        a_log,
                        a,
                        bias,
                        1.0,
                        20.0,
                        k,
                        v,
                        b,
                        state,
                        src,
                        accepted,
                        steps,
                        use_qk_l2norm_in_kernel=True,
                        output_state_indices=dst if separate else None,
                    )
                    torch.testing.assert_close(
                        state.float(),
                        expected.to(dtype).float(),
                        atol=0.005 if dtype == torch.bfloat16 else 1e-4,
                        rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
                    )


if __name__ == "__main__":
    unittest.main()
