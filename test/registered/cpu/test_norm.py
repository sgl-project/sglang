import sys
from typing import Optional, Tuple, Union

import pytest
import torch
from utils import make_non_contiguous, precision

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-b-test-cpu")
register_cpu_ci(est_time=10, suite="base-b-test-cpu-arm64")

torch.manual_seed(1234)

DTYPES = [torch.float16, torch.bfloat16]
DTYPE_IDS = ["float16", "bfloat16"]
eps = 1e-6


class TestNorm:

    def _forward_native(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        variance_epsilon: float = eps,
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

    def _norm(self, x, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def _gemma3_rmsnorm_native(
        self, x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float = eps
    ):
        output = self._norm(x.float(), variance_epsilon)
        output = output * (1.0 + weight.float())
        return output.type_as(x)

    def _gemma_rmsnorm_native(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        variance_epsilon: float = eps,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x * (1.0 + weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize("hidden_size", [2048, 512])
    @pytest.mark.parametrize("batch_size", [32, 121])
    def test_l2norm(self, batch_size, hidden_size, dtype):

        x = torch.randn([batch_size, hidden_size], dtype=dtype)
        fake_ones_weight = torch.ones(hidden_size, dtype=dtype)

        out = torch.ops.sgl_kernel.l2norm_cpu(x, eps)
        ref_out = self._forward_native(x, fake_ones_weight, eps)

        atol = rtol = precision[ref_out.dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize("hidden_size", [2048, 512])
    @pytest.mark.parametrize("batch_size", [32, 121])
    @pytest.mark.parametrize("seq_len", [None, 2], ids=["2d", "3d"])
    def test_rmsnorm(self, seq_len, batch_size, hidden_size, dtype):

        if seq_len is None:
            x = torch.randn([batch_size, hidden_size], dtype=dtype)
        else:
            x = torch.randn([batch_size, seq_len, hidden_size], dtype=dtype)
        x = make_non_contiguous(x)
        residual = torch.randn(x.shape, dtype=dtype)
        weight = torch.randn(hidden_size, dtype=dtype)

        out = torch.ops.sgl_kernel.rmsnorm_cpu(x, weight, eps)
        ref_out = self._forward_native(x, weight, eps)

        atol = rtol = precision[ref_out.dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

        ref_x = x.clone()
        ref_residual = residual.clone()

        torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(x, residual, weight, eps)

        ref_x, ref_residual = self._forward_native(ref_x, weight, eps, ref_residual)

        torch.testing.assert_close(x, ref_x, atol=atol, rtol=rtol)
        torch.testing.assert_close(residual, ref_residual, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bfloat16"])
    @pytest.mark.parametrize("hidden_size", [2048, 256, 33])
    @pytest.mark.parametrize("batch_size", [32, 121])
    def test_gemma_rmsnorm(self, batch_size, hidden_size, dtype):

        x = torch.randn([batch_size, hidden_size], dtype=dtype)
        x = make_non_contiguous(x)
        weight = torch.randn(hidden_size, dtype=dtype)

        out = torch.ops.sgl_kernel.gemma_rmsnorm_cpu(x, weight, eps)
        ref_out = self._gemma_rmsnorm_native(x, weight, eps)

        atol = rtol = precision[ref_out.dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

        ref_x = x.clone()
        residual = torch.randn([batch_size, hidden_size], dtype=dtype)
        ref_residual = residual.clone()

        torch.ops.sgl_kernel.gemma_fused_add_rmsnorm_cpu(x, residual, weight, eps)

        ref_x, ref_residual = self._gemma_rmsnorm_native(
            ref_x, weight, eps, ref_residual
        )

        torch.testing.assert_close(x, ref_x, atol=atol, rtol=rtol)
        torch.testing.assert_close(residual, ref_residual, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bfloat16"])
    @pytest.mark.parametrize("hidden_size", [128, 256])
    @pytest.mark.parametrize("batch_size", [32, 121])
    def test_gemma3_rmsnorm(self, batch_size, hidden_size, dtype):
        x_list = [
            torch.randn([batch_size, hidden_size], dtype=dtype),
            torch.randn([batch_size, 16, 2, hidden_size], dtype=dtype),
        ]
        for x in x_list:
            x = make_non_contiguous(x)
            weight = torch.randn(hidden_size, dtype=dtype)
            out = torch.ops.sgl_kernel.gemma3_rmsnorm_cpu(x, weight, eps)
            ref_out = self._gemma3_rmsnorm_native(x, weight, eps)

            atol = rtol = precision[ref_out.dtype]
            torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    def _gemma4_rmsnorm_native(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        variance_epsilon: float = eps,
        scale_shift: float = 0.0,
        with_scale: bool = True,
    ):
        output = self._norm(x.float(), variance_epsilon)
        if with_scale:
            output = output * (weight.float() + scale_shift)
        return output.type_as(x)

    @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bfloat16"])
    @pytest.mark.parametrize("hidden_size", [128, 2048])
    @pytest.mark.parametrize("batch_size", [32, 121])
    @pytest.mark.parametrize("scale_shift", [0.0, 1.0], ids=["shift0.0", "shift1.0"])
    @pytest.mark.parametrize("with_scale", [True, False], ids=["scale", "no-scale"])
    def test_gemma4_rmsnorm(
        self, batch_size, hidden_size, dtype, scale_shift, with_scale
    ):
        x_list = [
            torch.randn([batch_size, hidden_size], dtype=dtype),
            torch.randn([batch_size, 4, hidden_size], dtype=dtype),
        ]

        for x in x_list:
            x = make_non_contiguous(x)
            weight = torch.randn(hidden_size, dtype=dtype)

            out = torch.ops.sgl_kernel.gemma4_rmsnorm_cpu(
                x, weight, eps, scale_shift, with_scale
            )
            ref_out = self._gemma4_rmsnorm_native(
                x, weight, eps, scale_shift, with_scale
            )

            atol = rtol = precision[ref_out.dtype]
            torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)


class TestFusedRMSNormGated:

    def _forward_native(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        variance_epsilon: float = eps,
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

    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize("hidden_size", [64, 1024 + 13])
    @pytest.mark.parametrize("batch_size", [32, 121])
    def test_fused_rmsnorm_gated(self, batch_size, hidden_size, dtype):
        x = torch.randn([batch_size, hidden_size], dtype=dtype)
        x = make_non_contiguous(x)
        weight = torch.randn(hidden_size, dtype=dtype)
        gate = torch.randn([batch_size, hidden_size], dtype=dtype)

        out = torch.ops.sgl_kernel.fused_rmsnorm_gated_cpu(x, weight, gate, eps)
        ref_out = self._forward_native(x, weight, eps, gate)

        atol = rtol = precision[ref_out.dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)


class TestLayerNorm:

    def _forward_native(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        variance_epsilon: float = eps,
        residual: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance, mean = torch.var_mean(x, dim=-1, keepdim=True, correction=0)
        x = (x - mean) * torch.rsqrt(variance + variance_epsilon)
        x = x * weight.to(torch.float32)
        if bias is not None:
            x = x + bias.to(torch.float32)
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bfloat16"])
    @pytest.mark.parametrize("batch_size", [32, 121])
    @pytest.mark.parametrize("hidden_size", [128, 4096, 533])
    @pytest.mark.parametrize("has_bias", [False, True], ids=["no-bias", "bias"])
    def test_layernorm(
        self,
        batch_size: int,
        hidden_size: int,
        has_bias: bool,
        dtype: torch.dtype,
    ) -> None:
        x_list = [
            torch.randn([batch_size, hidden_size], dtype=dtype),
            torch.randn([batch_size, 3, hidden_size], dtype=dtype),
        ]

        for x in x_list:
            x = make_non_contiguous(x)
            weight = torch.randn(hidden_size, dtype=dtype)
            bias = torch.randn(hidden_size, dtype=dtype) if has_bias else None

            ln_out = torch.ops.sgl_kernel.layernorm_cpu(x, weight, bias, eps)
            ref_ln_out = self._forward_native(x, weight, eps, residual=None, bias=bias)

            atol = rtol = precision[ref_ln_out.dtype]
            torch.testing.assert_close(ln_out, ref_ln_out, atol=atol, rtol=rtol)

            residual = torch.randn(x.shape, dtype=dtype)
            ref_residual = residual.clone()

            add_ln_out = torch.ops.sgl_kernel.fused_add_layernorm_cpu(
                x, residual, weight, bias, eps
            )
            ref_add_ln_out, ref_residual = self._forward_native(
                x, weight, eps, residual=ref_residual, bias=bias
            )

            torch.testing.assert_close(add_ln_out, ref_add_ln_out, atol=atol, rtol=rtol)
            torch.testing.assert_close(residual, ref_residual, atol=atol, rtol=rtol)


class TestFusedQKGemmaRMSNorm:

    def _gemma_rmsnorm_per_head_native(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        head_dim: int,
        variance_epsilon: float = eps,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.to(torch.float32).reshape(-1, head_dim)
        variance = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_f = x_f * torch.rsqrt(variance + variance_epsilon)
        x_f = x_f * (1.0 + weight.to(torch.float32))
        return x_f.to(orig_dtype).reshape_as(x)

    @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bfloat16"])
    @pytest.mark.parametrize(
        "batch_size,num_head,num_head_kv,head_dim",
        [
            (8, 4, 2, 128),
            (17, 8, 2, 64),
            (5, 3, 1, 96),
        ],
    )
    def test_fused_qk_gemma_rmsnorm_cpu(
        self, batch_size: int, num_head: int, num_head_kv: int, head_dim: int, dtype
    ):
        q = torch.randn([batch_size, num_head * head_dim], dtype=dtype)
        k = torch.randn([batch_size, num_head_kv * head_dim], dtype=dtype)

        # Keep last dim contiguous but make base storage non-contiguous to stress stride handling.
        q = make_non_contiguous(q)
        k = make_non_contiguous(k)

        q_weight = torch.randn(head_dim, dtype=dtype)
        k_weight = torch.randn(head_dim, dtype=dtype)

        q_out, k_out = torch.ops.sgl_kernel.fused_qk_gemma_rmsnorm_cpu(
            q, k, q_weight, k_weight, eps, head_dim
        )

        ref_q_out = self._gemma_rmsnorm_per_head_native(q, q_weight, head_dim, eps)
        ref_k_out = self._gemma_rmsnorm_per_head_native(k, k_weight, head_dim, eps)

        atol = rtol = precision[ref_q_out.dtype]
        torch.testing.assert_close(q_out, ref_q_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(k_out, ref_k_out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
