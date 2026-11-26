import unittest

import torch

from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.kernels.ops.mamba.triton_ops.mamba_ssm import selective_state_update
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestReadOnlyMambaDecode(CustomTestCase):
    def test_aliased_rows_do_not_write_but_other_rows_advance(self):
        torch.manual_seed(0)
        device = "cuda"
        rows, dim, dstate, width = 3, 64, 16, 4
        x = torch.randn(rows, dim, device=device, dtype=torch.bfloat16)
        weight = torch.randn(dim, width, device=device, dtype=x.dtype)
        conv = torch.randn(rows, width - 1, dim, device=device, dtype=x.dtype)
        conv = conv.transpose(1, 2)
        ssm = torch.randn(rows, dim, dstate, device=device, dtype=x.dtype)
        conv[1] = conv[0]
        ssm[1] = ssm[0]
        dt = torch.rand_like(x)
        a = -torch.rand(dim, dstate, device=device)
        b = torch.randn(rows, dstate, device=device, dtype=x.dtype)
        c = torch.randn_like(b)

        def decode(indices, read_only=None):
            conv_state, ssm_state = conv.clone(), ssm.clone()
            out = torch.empty_like(x)
            convolved = causal_conv1d_update(
                x,
                conv_state,
                weight,
                activation="silu",
                conv_state_indices=indices,
                read_only_state_rows=read_only,
            )
            selective_state_update(
                ssm_state,
                convolved,
                dt,
                a,
                b,
                c,
                D=torch.ones(dim, device=device),
                dt_bias=torch.zeros(dim, device=device),
                state_batch_indices=indices,
                out=out,
                read_only_state_rows=read_only,
            )
            return out, conv_state, ssm_state

        expected, conv_written, ssm_written = decode(torch.arange(rows, device=device))
        actual, conv_aliased, ssm_aliased = decode(
            torch.tensor([0, 0, 2], device=device),
            torch.tensor([True, True, False], device=device),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for original, aliased, written in (
            (conv, conv_aliased, conv_written),
            (ssm, ssm_aliased, ssm_written),
        ):
            torch.testing.assert_close(aliased[:2], original[:2], rtol=0, atol=0)
            torch.testing.assert_close(aliased[2], written[2], rtol=0, atol=0)
            self.assertFalse(torch.equal(aliased[2], original[2]))


if __name__ == "__main__":
    unittest.main()
