"""MLU attention state merge correctness and output buffer coverage."""

import unittest

import torch

from sglang.kernels.ops.attention.merge_state import merge_state_mlu
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=15, suite="pr-test-1-mlu")


class TestMergeStateMLU(CustomTestCase):
    def test_merge_output_buffers(self):
        # Create independent prefix and suffix attention states.
        torch.manual_seed(17)
        prefix = torch.randn(7, 4, 128, device="mlu", dtype=torch.bfloat16)
        suffix = torch.randn_like(prefix)
        p_lse = torch.randn(7, 4, device="mlu", dtype=torch.float32)
        s_lse = torch.randn_like(p_lse)

        # Build the CPU reference with mathematical empty-state LSE (-inf).
        prefix[0] = 0
        p_lse[0] = -torch.inf
        expected_lse = torch.logaddexp(p_lse.cpu(), s_lse.cpu())
        expected = prefix.cpu().float() * (p_lse.cpu() - expected_lse).exp().unsqueeze(
            -1
        ) + suffix.cpu().float() * (s_lse.cpu() - expected_lse).exp().unsqueeze(-1)

        # torch_mlu_ops.flash_attention returns zero output and +inf LSE for empty KV.
        # update_out_and_lse must consume that sentinel without normalization.
        p_lse[0] = torch.inf
        for layout in ("contiguous", "transposed", "inplace"):
            with self.subTest(layout=layout):
                p = prefix.clone()
                pl = p_lse.clone()
                out = p if layout == "inplace" else torch.empty_like(p)
                out_lse = (
                    pl
                    if layout == "inplace"
                    else torch.empty(4, 7, device="mlu").T
                    if layout == "transposed"
                    else torch.empty_like(pl)
                )
                actual, actual_lse = merge_state_mlu(p, pl, suffix, s_lse, out, out_lse)

                # Check the caller-provided buffers and merged values.
                self.assertIs(actual, out)
                self.assertIs(actual_lse, out_lse)
                torch.testing.assert_close(
                    actual.cpu().float(), expected, atol=1e-2, rtol=1e-2
                )
                # LSE merging operates on FP32 inputs.
                torch.testing.assert_close(
                    actual_lse.cpu(), expected_lse, atol=1e-5, rtol=1e-5
                )


if __name__ == "__main__":
    unittest.main()
