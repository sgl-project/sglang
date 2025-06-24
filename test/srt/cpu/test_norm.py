import itertools
import unittest
from typing import Optional, Tuple, Union

import sgl_kernel
import torch
from utils import make_non_contiguous, precision

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(0)


class TestNorm(CustomTestCase):
    M = [4096, 1024]
    N = [4096, 4096 + 13]
    dtype = [torch.float16, torch.bfloat16]

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

    def _norm_test(self, m, n, dtype):

        x = torch.randn([m, n], dtype=dtype)
        x = make_non_contiguous(x)
        hidden_size = x.size(-1)
        weight = torch.randn(hidden_size, dtype=dtype)
        variance_epsilon = 1e-6

        out = torch.ops.sgl_kernel.rmsnorm_cpu(x, weight, variance_epsilon)
        ref_out = self._forward_native(x, weight, variance_epsilon)

        atol = rtol = precision[ref_out.dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

        ref_x = x.clone()
        residual = torch.randn([m, hidden_size], dtype=dtype)
        ref_residual = residual.clone()

        torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(
            x, residual, weight, variance_epsilon
        )

        ref_x, ref_residual = self._forward_native(
            ref_x, weight, variance_epsilon, ref_residual
        )

        torch.testing.assert_close(x, ref_x, atol=atol, rtol=rtol)
        torch.testing.assert_close(residual, ref_residual, atol=atol, rtol=rtol)

    def _l2norm_test(self, m, n, dtype):

        x = torch.randn([m, n], dtype=dtype)
        hidden_size = x.size(-1)
        fake_ones_weight = torch.ones(hidden_size, dtype=dtype)
        variance_epsilon = 1e-6

        out = torch.ops.sgl_kernel.l2norm_cpu(x, variance_epsilon)
        ref_out = self._forward_native(x, fake_ones_weight, variance_epsilon)

        atol = rtol = precision[ref_out.dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    def test_norm(self):
        for params in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=params[0], n=params[1], dtype=params[2]):
                self._norm_test(*params)
                self._l2norm_test(*params)


if __name__ == "__main__":
    unittest.main()
