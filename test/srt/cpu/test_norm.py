import itertools
import unittest
from typing import Optional, Tuple, Union

import torch
from utils import make_non_contiguous, parametrize, precision

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


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


class TestFusedRMSNormGated(CustomTestCase):
    M = [4096, 1024]
    N = [4096, 4096 + 13]
    dtype = [torch.float16, torch.bfloat16]

    def _forward_native(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        variance_epsilon: float = 1e-6,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        hidden_states = weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * torch.nn.functional.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)

    def _norm_test(self, m, n, dtype):

        x = torch.randn([m, n], dtype=dtype)
        x = make_non_contiguous(x)
        batch_size = x.size(0)
        hidden_size = x.size(-1)
        weight = torch.randn(hidden_size, dtype=dtype)
        variance_epsilon = 1e-6
        gate = torch.randn([batch_size, hidden_size], dtype=dtype)

        out = torch.ops.sgl_kernel.fused_rmsnorm_gated_cpu(
            x, weight, gate, variance_epsilon
        )
        ref_out = self._forward_native(x, weight, variance_epsilon, gate)

        atol = rtol = precision[ref_out.dtype] * 2
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    def test_norm(self):
        for params in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=params[0], n=params[1], dtype=params[2]):
                self._norm_test(*params)


class TestLayerNorm(CustomTestCase):

    def _forward_native(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        variance_epsilon: float,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        (variance, mean) = torch.var_mean(x, dim=-1, keepdim=True, correction=0)
        x = (x - mean) * torch.rsqrt(variance + variance_epsilon)
        x = x.to(orig_dtype) * weight
        if residual is None:
            return x
        else:
            return x, residual

    @parametrize(
        m=[4096, 1024],
        n=[4096, 4109],
        dtype=[torch.float16, torch.bfloat16],
    )
    def test_norm(self, m: int, n: int, dtype: torch.dtype) -> None:
        x_ln = torch.randn([m, n], dtype=dtype)
        x_ln = make_non_contiguous(x_ln)
        ref_x_ln = x_ln.clone()
        hidden_size = x_ln.size(-1)
        weight = torch.randn(hidden_size, dtype=dtype)
        variance_epsilon = 1e-6

        torch.ops.sgl_kernel.layernorm_cpu(x_ln, weight, variance_epsilon)
        ref_ln_out = self._forward_native(ref_x_ln, weight, variance_epsilon)

        atol = rtol = precision[ref_ln_out.dtype]
        torch.testing.assert_close(x_ln, ref_ln_out, atol=atol, rtol=rtol)

        x_add_ln = torch.randn([m, n], dtype=dtype)
        x_add_ln = make_non_contiguous(x_add_ln)
        ref_x_add_ln = x_add_ln.clone()
        residual = torch.randn([m, hidden_size], dtype=dtype)
        ref_residual = residual.clone()

        torch.ops.sgl_kernel.fused_add_layernorm_cpu(
            x_add_ln, residual, weight, variance_epsilon
        )
        ref_add_ln_out, ref_residual = self._forward_native(
            ref_x_add_ln, weight, variance_epsilon, ref_residual
        )

        torch.testing.assert_close(x_add_ln, ref_add_ln_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(residual, ref_residual, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
