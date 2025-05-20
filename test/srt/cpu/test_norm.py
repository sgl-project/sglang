import unittest
from typing import Optional, Tuple, Union

import torch
from sgl_kernel.common_ops import fused_add_rmsnorm_cpu as fused_add_rmsnorm
from sgl_kernel.common_ops import rmsnorm_cpu as rmsnorm

from sglang.test.test_utils import CustomTestCase


class TestNorm(CustomTestCase):
    def _forward_native(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        variance_epsilon: float = 1e-6,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x.to(orig_dtype) * weight
        if residual is None:
            return x
        else:
            return x, residual

    def _run_single_test(self, shape, dtype, device="cuda"):

        x = torch.randn(shape, dtype=dtype).to(device=device)
        hidden_size = x.size(-1)
        weight = torch.randn(hidden_size, dtype=dtype).to(device=device)
        variance_epsilon = 1e-6

        # TEST: rmsnorm
        out = rmsnorm(x, weight, variance_epsilon)
        ref_out = self._forward_native(x, weight, variance_epsilon)

        torch.testing.assert_close(out, ref_out)

        # TEST: fused_add_rmsnorm
        # flashinfer writes x and residual inplaced
        ref_x = x.clone()

        residual = torch.randn(shape, dtype=dtype).to(device=device)
        ref_residual = residual.clone()

        fused_add_rmsnorm(x, residual, weight, variance_epsilon)

        ref_x, ref_residual = self._forward_native(
            ref_x, weight, variance_epsilon, ref_residual
        )

        torch.testing.assert_close(x, ref_x)
        torch.testing.assert_close(residual, ref_residual)

    def test_norm(self):
        self._run_single_test([4096, 4096], torch.bfloat16, "cpu")
        self._run_single_test([1024, 4096], torch.bfloat16, "cpu")
        self._run_single_test([1024, 4096 + 13], torch.float16, "cpu")


if __name__ == "__main__":
    unittest.main()
