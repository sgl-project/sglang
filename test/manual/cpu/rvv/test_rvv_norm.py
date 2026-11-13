"""Unit tests for RVV norm kernels."""

import itertools
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

from .rvv_utils import has_sgl_kernel_op, helper_non_contiguous, precision

torch.manual_seed(1234)


def rmsnorm_native(x, weight, eps, residual=None):
    """Reference RMSNorm: out = x * rsqrt(mean(x^2) + eps) * weight."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype) * weight
    return x if residual is None else (x, residual)


def gemma_rmsnorm_native(x, weight, eps, residual=None):
    """Reference Gemma RMSNorm: scale = (1 + weight)."""
    orig_dtype = x.dtype
    if residual is not None:
        x = x + residual
        residual = x
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + weight.float())
    x = x.to(orig_dtype)
    return x if residual is None else (x, residual)


def gemma3_rmsnorm_native(x, weight, eps):
    """Reference Gemma3 RMSNorm: supports 2D and 4D inputs."""
    output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    output = output * (1.0 + weight.float())
    return output.type_as(x)


def fused_rmsnorm_gated_native(x, weight, gate, eps):
    """Reference fused gated RMSNorm: rms_norm(x) * weight * silu(gate)."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = weight * x.to(input_dtype)
    x = x * torch.nn.functional.silu(gate.to(torch.float32))
    return x.to(input_dtype)


def layernorm_native(x, weight, eps, residual=None):
    """Reference LayerNorm: mean+var two pass."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)
    variance, mean = torch.var_mean(x, dim=-1, keepdim=True, correction=0)
    x = (x - mean) * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype) * weight
    return x if residual is None else (x, residual)


@unittest.skipUnless(
    has_sgl_kernel_op("rmsnorm_cpu"),
    "sgl_kernel norm not available (non-RISC-V build)",
)
class TestRVVNormCore(CustomTestCase):
    """Test suite for RVV norm kernels."""

    M = [128, 129, 257, 1024, 4096]
    N = [4096, 4109]
    dtype = [torch.float16, torch.bfloat16]

    def _run_rmsnorm(self, m, n, x=None, dtype=torch.float16):
        if x is None:
            x = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        out = torch.ops.sgl_kernel.rmsnorm_cpu(x, weight, eps)
        ref = rmsnorm_native(x, weight, eps)
        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def _run_fused_add_rmsnorm(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        residual = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        ref_x = x.clone()
        ref_residual = residual.clone()

        torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(x, residual, weight, eps)
        ref_out, ref_res = rmsnorm_native(ref_x, weight, eps, ref_residual)

        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(x, ref_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(residual, ref_res, atol=atol, rtol=rtol)

    def _run_l2norm(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        eps = 1e-6

        out = torch.ops.sgl_kernel.l2norm_cpu(x, eps)
        # L2 norm matches RMSNorm with a unit weight vector.
        fake_weight = torch.ones(n, dtype=dtype)
        ref = rmsnorm_native(x, fake_weight, eps)

        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def _run_gemma_rmsnorm(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        out = torch.ops.sgl_kernel.gemma_rmsnorm_cpu(x, weight, eps)
        ref = gemma_rmsnorm_native(x, weight, eps)
        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def _run_gemma_fused_add(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        residual = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        ref_x = x.clone()
        ref_residual = residual.clone()

        torch.ops.sgl_kernel.gemma_fused_add_rmsnorm_cpu(x, residual, weight, eps)
        ref_out, ref_res = gemma_rmsnorm_native(ref_x, weight, eps, ref_residual)

        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(x, ref_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(residual, ref_res, atol=atol, rtol=rtol)

    def _run_gemma3_rmsnorm(self, m, n, dtype):
        x_2d = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        out_2d = torch.ops.sgl_kernel.gemma3_rmsnorm_cpu(x_2d, weight, eps)
        ref_2d = gemma3_rmsnorm_native(x_2d, weight, eps)
        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(ref_2d, out_2d, atol=atol, rtol=rtol)

        # Also cover the 4D layout used by attention blocks.
        x_4d = torch.randn([1, m, 2, n], dtype=dtype)
        out_4d = torch.ops.sgl_kernel.gemma3_rmsnorm_cpu(x_4d, weight, eps)
        ref_4d = gemma3_rmsnorm_native(x_4d, weight, eps)
        torch.testing.assert_close(ref_4d, out_4d, atol=atol, rtol=rtol)

    def test_rmsnorm(self):
        """N+13 odd shape catches tail handling; M range covers VLEN=256 boundary."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._run_rmsnorm(m, n, dtype=dt)

    def test_fused_add_rmsnorm(self):
        """Both output tensors (normed x and updated residual) must match reference."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._run_fused_add_rmsnorm(m, n, dt)

    def test_l2norm(self):
        """L2Norm equals RMSNorm with unit weight; shared shape matrix catches tails."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._run_l2norm(m, n, dt)

    def test_gemma_rmsnorm(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._run_gemma_rmsnorm(m, n, dt)

    def test_gemma_fused_add_rmsnorm(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._run_gemma_fused_add(m, n, dt)

    def test_gemma3_rmsnorm(self):
        """Gemma3 adds (1 + w) scaling; both 2D and 4D layouts must match reference."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._run_gemma3_rmsnorm(m, n, dt)

    def test_gemma3_rmsnorm_4d_multi_batch(self):
        """Case: Gemma3 RMSNorm 4D with batch_size > 1."""
        for dt in self.dtype:
            n = 4096
            x = torch.randn([2, 4, 3, n], dtype=dt)
            weight = torch.randn(n, dtype=dt)
            eps = 1e-6
            out = torch.ops.sgl_kernel.gemma3_rmsnorm_cpu(x, weight, eps)
            ref = gemma3_rmsnorm_native(x, weight, eps)
            atol = rtol = precision["pointwise_default"][dt]
            with self.subTest(dtype=dt):
                torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_rmsnorm_non_contiguous(self):
        """Case: RMSNorm with non-contiguous input tensors."""
        for dt in self.dtype:
            x = helper_non_contiguous(torch.randn(256, 4096, dtype=dt))
            with self.subTest(dtype=dt, shape=x.shape):
                self._run_rmsnorm(x.shape[0], x.shape[1], x=x, dtype=dt)

    def test_fused_add_rmsnorm_non_contiguous(self):
        """Case: fused-add-RMSNorm with non-contiguous input tensors."""
        for dt in self.dtype:
            x = helper_non_contiguous(torch.randn(256, 4096, dtype=dt))
            residual = helper_non_contiguous(torch.randn(256, 4096, dtype=dt))
            weight = torch.randn(4096, dtype=dt)
            eps = 1e-6
            ref_x, ref_res = rmsnorm_native(x.clone(), weight, eps, residual.clone())
            x_copy, res_copy = x.clone(), residual.clone()
            torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(x_copy, res_copy, weight, eps)
            atol = rtol = precision["pointwise_default"][dt]
            with self.subTest(dtype=dt):
                torch.testing.assert_close(x_copy, ref_x, atol=atol, rtol=rtol)
                torch.testing.assert_close(res_copy, ref_res, atol=atol, rtol=rtol)

    def test_rmsnorm_small_n(self):
        """Case: RMSNorm with N smaller than one full m4 vector (< 8 for VLEN=256)."""
        for n, dt in itertools.product([1, 4, 7], self.dtype):
            with self.subTest(n=n, dtype=dt):
                self._run_rmsnorm(4, n, dtype=dt)

    def test_gemma_rmsnorm_non_contiguous(self):
        """Case: Gemma RMSNorm with non-contiguous input tensors."""
        for dt in self.dtype:
            x = helper_non_contiguous(torch.randn(256, 4096, dtype=dt))
            weight = torch.randn(4096, dtype=dt)
            eps = 1e-6
            out = torch.ops.sgl_kernel.gemma_rmsnorm_cpu(x, weight, eps)
            ref = gemma_rmsnorm_native(x, weight, eps)
            atol = rtol = precision["pointwise_default"][dt]
            with self.subTest(dtype=dt):
                torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)


@unittest.skipUnless(
    has_sgl_kernel_op("rmsnorm_cpu"),
    "sgl_kernel norm not available (non-RISC-V build)",
)
class TestRVVNormLayer(CustomTestCase):
    """Test suite for RVV LayerNorm kernels."""

    M = [128, 129, 257, 1024, 4096]
    N = [4096, 4109]
    dtype = [torch.float16, torch.bfloat16]

    def _run_layernorm(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        out = torch.ops.sgl_kernel.layernorm_cpu(x, weight, None, eps)
        ref = layernorm_native(x, weight, eps)

        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    def _run_fused_add_layernorm(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        residual = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        ref_x = x.clone()
        ref_residual = residual.clone()

        out = torch.ops.sgl_kernel.fused_add_layernorm_cpu(
            x, residual, weight, None, eps
        )
        ref_out, ref_res = layernorm_native(ref_x, weight, eps, ref_residual)

        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(residual, ref_res, atol=atol, rtol=rtol)

    def test_layernorm(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._run_layernorm(m, n, dt)

    def test_layernorm_with_bias(self):
        """Case: LayerNorm with non-None bias tensor."""
        for m, n, dt in itertools.product([128, 257], [4096, 4109], self.dtype):
            x = torch.randn([m, n], dtype=dt)
            weight = torch.randn(n, dtype=dt)
            bias = torch.randn(n, dtype=dt)
            eps = 1e-6
            out = torch.ops.sgl_kernel.layernorm_cpu(x, weight, bias, eps)
            ref = layernorm_native(x, weight, eps)
            ref = ref + bias
            tol_map = precision.get("norm_layer_bias", precision["pointwise_default"])
            atol = rtol = tol_map[dt]
            if dt == torch.float16:
                atol = rtol = max(atol, 1e-2)
            with self.subTest(m=m, n=n, dtype=dt):
                torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    def test_layernorm_non_contiguous(self):
        """Case: LayerNorm with non-contiguous input tensors."""
        for dt in self.dtype:
            x = helper_non_contiguous(torch.randn(256, 4096, dtype=dt))
            weight = torch.randn(4096, dtype=dt)
            eps = 1e-6
            out = torch.ops.sgl_kernel.layernorm_cpu(x, weight, None, eps)
            ref = layernorm_native(x, weight, eps)
            atol = rtol = precision["pointwise_default"][dt]
            with self.subTest(dtype=dt):
                torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    def test_fused_add_layernorm(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._run_fused_add_layernorm(m, n, dt)


@unittest.skipUnless(
    has_sgl_kernel_op("rmsnorm_cpu"),
    "sgl_kernel norm not available (non-RISC-V build)",
)
class TestRVVNormFusedGated(CustomTestCase):
    """Test suite for RVV fused gated RMSNorm kernel."""

    M = [128, 129, 257, 1024, 4096]
    N = [4096, 4109]
    dtype = [torch.float16, torch.bfloat16]

    def _run_rmsnorm_gated(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        gate = torch.randn([m, n], dtype=dtype)
        eps = 1e-6

        out = torch.ops.sgl_kernel.fused_rmsnorm_gated_cpu(x, weight, gate, eps)
        ref = fused_rmsnorm_gated_native(x, weight, gate, eps)

        atol = rtol = precision["pointwise_default"][dtype] * 2
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_fused_rmsnorm_gated(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._run_rmsnorm_gated(m, n, dt)

    def test_fused_rmsnorm_gated_non_contiguous(self):
        """Case: fused gated RMSNorm with non-contiguous input tensors."""
        for dt in self.dtype:
            x = helper_non_contiguous(torch.randn(256, 4096, dtype=dt))
            gate = torch.randn(256, 4096, dtype=dt)  # gate must be contiguous
            weight = torch.randn(4096, dtype=dt)
            eps = 1e-6
            out = torch.ops.sgl_kernel.fused_rmsnorm_gated_cpu(x, weight, gate, eps)
            ref = fused_rmsnorm_gated_native(x, weight, gate, eps)
            atol = rtol = precision["pointwise_default"][dt] * 2
            with self.subTest(dtype=dt):
                torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
