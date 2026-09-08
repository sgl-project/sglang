import sys
from typing import Optional, Tuple, Union

import pytest
import sgl_kernel  # noqa: F401
import torch

from sglang.kernels.ops.diffusion.modulate.scale_shift_triton import (
    expand_scale_shift_cpu_param,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import make_non_contiguous, precision

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")
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


class TestFusedQKRMSNorm:
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize(
        "batch_size,q_size,k_size,v_size",
        [(1, 256, 64, 64), (17, 512, 128, 128)],
    )
    def test_fused_qk_rmsnorm(
        self, batch_size: int, q_size: int, k_size: int, v_size: int, dtype
    ):
        """Q and K split views must be normalized over their distinct full widths."""
        qkv = torch.randn([batch_size, q_size + k_size + v_size], dtype=dtype)
        q, k, _ = qkv.split([q_size, k_size, v_size], dim=-1)
        q_weight = torch.randn(q_size, dtype=dtype)
        k_weight = torch.randn(k_size, dtype=dtype)

        q_out, k_out = torch.ops.sgl_kernel.fused_qk_rmsnorm_cpu(
            q, k, q_weight, k_weight, eps
        )
        ref_q_out = TestNorm()._forward_native(q, q_weight, eps)
        ref_k_out = TestNorm()._forward_native(k, k_weight, eps)

        atol = rtol = precision[dtype]
        torch.testing.assert_close(q_out, ref_q_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(k_out, ref_k_out, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize(
        "batch_size,q_size,k_size,tp_world_size",
        [(1, 256, 64, 2), (17, 512, 128, 4)],
    )
    def test_fused_qk_rmsnorm_tp(
        self,
        batch_size: int,
        q_size: int,
        k_size: int,
        tp_world_size: int,
        dtype,
    ):
        q = torch.randn([batch_size, q_size], dtype=dtype)
        k = torch.randn([batch_size, k_size], dtype=dtype)
        q_weight = torch.randn(q_size, dtype=dtype)
        k_weight = torch.randn(k_size, dtype=dtype)

        q_shards = q.chunk(tp_world_size, dim=-1)
        k_shards = k.chunk(tp_world_size, dim=-1)
        q_weight_shards = q_weight.chunk(tp_world_size)
        k_weight_shards = k_weight.chunk(tp_world_size)
        local_sum_sq = [
            torch.ops.sgl_kernel.fused_qk_rmsnorm_sumsq_cpu(q_shard, k_shard)
            for q_shard, k_shard in zip(q_shards, k_shards)
        ]
        global_sum_sq = torch.stack(local_sum_sq).sum(dim=0)

        shard_outputs = [
            torch.ops.sgl_kernel.fused_qk_rmsnorm_apply_from_stats_cpu(
                q_shard,
                k_shard,
                q_weight_shard,
                k_weight_shard,
                global_sum_sq,
                tp_world_size,
                eps,
            )
            for q_shard, k_shard, q_weight_shard, k_weight_shard in zip(
                q_shards, k_shards, q_weight_shards, k_weight_shards
            )
        ]
        q_out = torch.cat([output[0] for output in shard_outputs], dim=-1)
        k_out = torch.cat([output[1] for output in shard_outputs], dim=-1)

        ref_q_out = TestNorm()._forward_native(q, q_weight, eps)
        ref_k_out = TestNorm()._forward_native(k, k_weight, eps)
        atol = rtol = precision[dtype]
        torch.testing.assert_close(q_out, ref_q_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(k_out, ref_k_out, atol=atol, rtol=rtol)

        assert global_sum_sq.shape == (batch_size, 2)
        assert global_sum_sq.dtype == torch.float32


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
    def test_fused_qk_gemma_rmsnorm(
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

    @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bfloat16"])
    @pytest.mark.parametrize(
        "batch_size,num_head,num_head_kv,head_dim",
        [
            (8, 4, 2, 128),
            (17, 8, 2, 64),
            (5, 3, 1, 96),
        ],
    )
    def test_fused_qk_gemma_rmsnorm_with_gate(
        self, batch_size: int, num_head: int, num_head_kv: int, head_dim: int, dtype
    ):
        q = torch.randn([batch_size, num_head, head_dim], dtype=dtype)
        gate = torch.randn([batch_size, num_head, head_dim], dtype=dtype)
        q_gate = torch.cat((q, gate), dim=-1).reshape(
            batch_size, num_head * head_dim * 2
        )
        k = torch.randn([batch_size, num_head_kv * head_dim], dtype=dtype)

        q_gate = make_non_contiguous(q_gate)
        k = make_non_contiguous(k)

        q_weight = torch.randn(head_dim, dtype=dtype)
        k_weight = torch.randn(head_dim, dtype=dtype)

        q_out, k_out, gate_out = (
            torch.ops.sgl_kernel.fused_qk_gemma_rmsnorm_with_gate_cpu(
                q_gate, k, q_weight, k_weight, eps, head_dim, num_head
            )
        )

        ref_q_out = self._gemma_rmsnorm_per_head_native(q, q_weight, head_dim, eps)
        ref_k_out = self._gemma_rmsnorm_per_head_native(k, k_weight, head_dim, eps)

        atol = rtol = precision[ref_q_out.dtype]
        torch.testing.assert_close(
            q_out, ref_q_out.reshape(-1, head_dim), atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            k_out, ref_k_out.reshape(-1, head_dim), atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            gate_out, gate.reshape(-1, head_dim), atol=atol, rtol=rtol
        )


class TestFusedScaleShiftKernels:
    def rmsnorm_ref(
        self,
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        x_fp32 = x.float()
        variance = x_fp32.square().mean(dim=-1, keepdim=True)
        out = x_fp32 * torch.rsqrt(variance + eps)

        if weight is not None:
            out = out * weight.float()

        return out

    @pytest.mark.parametrize(
        "input_dtype,param_dtype",
        [
            (torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, torch.float32),
            (torch.float16, torch.float16),
            (torch.float16, torch.float32),
        ],
    )
    def test_fused_scale_shift(
        self,
        input_dtype,
        param_dtype,
    ):
        B, S, D = 2, 4, 67
        x = torch.randn(B, S, D, dtype=input_dtype)
        scale = torch.randn(B, 1, D, dtype=param_dtype)
        shift = torch.randn(B, S, D, dtype=param_dtype)

        scale_expanded = scale.expand_as(x)
        shift_expanded = shift.expand_as(x)

        out = torch.ops.sgl_kernel.fused_scale_shift_cpu(
            x, scale_expanded, shift_expanded, 1.0
        )

        ref = (x.float() * (1.0 + scale.float()) + shift.float()).to(input_dtype)

        torch.testing.assert_close(
            out, ref, atol=precision[input_dtype], rtol=precision[input_dtype]
        )

    @pytest.mark.parametrize(
        "input_dtype,param_dtype",
        [
            (torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, torch.float32),
            (torch.float16, torch.float16),
            (torch.float16, torch.float32),
        ],
    )
    def test_fused_scale_shift_4d(self, input_dtype, param_dtype):
        B, F, frame_seqlen, D = 2, 3, 4, 67
        S = F * frame_seqlen

        x = torch.randn(B, S, D, dtype=input_dtype)
        scale = torch.randn(B, F, 1, D, dtype=param_dtype)
        shift = torch.randn(B, F, 1, D, dtype=param_dtype)

        out = torch.ops.sgl_kernel.fused_scale_shift_cpu(
            x,
            expand_scale_shift_cpu_param(scale, x),
            expand_scale_shift_cpu_param(shift, x),
            1.0,
        )

        scale_ref = scale.expand(B, F, frame_seqlen, D).reshape(B, S, D)
        shift_ref = shift.expand(B, F, frame_seqlen, D).reshape(B, S, D)

        ref = (x.float() * (1.0 + scale_ref.float()) + shift_ref.float()).to(
            input_dtype
        )

        torch.testing.assert_close(
            out, ref, atol=precision[input_dtype], rtol=precision[input_dtype]
        )

    @pytest.mark.parametrize(
        "input_dtype,param_dtype,norm_dtype",
        [
            (torch.bfloat16, torch.bfloat16, torch.float32),
            (torch.bfloat16, torch.float32, torch.float32),
            (torch.float16, torch.float16, torch.float32),
            (torch.float16, torch.float32, torch.float32),
        ],
    )
    def test_fused_norm_scale_shift(
        self,
        input_dtype,
        param_dtype,
        norm_dtype,
    ):
        B, S, D = 2, 4, 67

        x = torch.randn(B, S, D, dtype=input_dtype)
        weight = torch.randn(D, dtype=norm_dtype)
        scale = torch.randn(B, 1, D, dtype=param_dtype)
        shift = torch.randn(B, S, D, dtype=param_dtype)

        scale_expanded = scale.expand_as(x)
        shift_expanded = shift.expand_as(x)

        out = torch.ops.sgl_kernel.fused_norm_scale_shift_cpu(
            x,
            weight,
            None,
            scale_expanded,
            shift_expanded,
            "rms",
            eps=eps,
        )

        normalized = self.rmsnorm_ref(x, weight, eps).to(input_dtype).float()

        ref = (normalized * (1.0 + scale.float()) + shift.float()).to(input_dtype)

        torch.testing.assert_close(
            out, ref, atol=precision[input_dtype], rtol=precision[input_dtype]
        )

    @pytest.mark.parametrize(
        "input_dtype,gate_dtype,norm_dtype,param_dtype,norm_type,use_gate,use_norm",
        [
            # Same-dtype gate.
            (torch.bfloat16, torch.bfloat16, None, torch.bfloat16, "rms", True, False),
            (torch.float16, torch.float16, None, torch.float16, "rms", True, False),
            # FP32 gate + FP32 affine norm.
            (
                torch.bfloat16,
                torch.float32,
                torch.float32,
                torch.bfloat16,
                "layer",
                True,
                True,
            ),
            (
                torch.float16,
                torch.float32,
                torch.float32,
                torch.float16,
                "layer",
                True,
                True,
            ),
            # FP32 scale/shift + no gate + no affine norm.
            (torch.bfloat16, None, None, torch.float32, "layer", False, False),
            (torch.float16, None, None, torch.float32, "layer", False, False),
        ],
    )
    def test_fused_scale_residual_norm_scale_shift(
        self,
        input_dtype,
        gate_dtype,
        norm_dtype,
        param_dtype,
        norm_type,
        use_gate,
        use_norm,
    ):
        B, S, D = 2, 4, 67

        x = torch.randn(B, S, D, dtype=input_dtype)
        residual = torch.randn(B, S, D, dtype=input_dtype)

        gate = torch.randn(D, dtype=gate_dtype) if use_gate else None

        weight = torch.randn(D, dtype=norm_dtype) if use_norm else None

        bias = (
            torch.randn(D, dtype=norm_dtype)
            if use_norm and norm_type == "layer"
            else None
        )

        scale = torch.randn(B, 1, D, dtype=param_dtype)
        shift = torch.randn(B, S, D, dtype=param_dtype)

        scale_expanded = scale.expand_as(x)
        shift_expanded = shift.expand_as(x)

        gate_expanded = gate.view(1, 1, D).expand_as(x) if gate is not None else None

        out, residual_out = (
            torch.ops.sgl_kernel.fused_scale_residual_norm_scale_shift_cpu(
                residual,
                x,
                gate_expanded,
                weight,
                bias,
                scale_expanded,
                shift_expanded,
                norm_type,
                eps=eps,
            )
        )

        if gate is None:
            residual_fp32 = residual.float() + x.float()
        else:
            residual_fp32 = residual.float() + x.float() * gate.float()

        ref_residual = residual_fp32.to(input_dtype)
        norm_input = ref_residual.float()

        if norm_type == "rms":
            normalized = self.rmsnorm_ref(norm_input, weight, eps)
        else:
            normalized = torch.nn.functional.layer_norm(
                norm_input,
                (D,),
                weight.float() if weight is not None else None,
                bias.float() if bias is not None else None,
                eps,
            )

        # Match CUDA activation boundary after norm.
        normalized = normalized.to(input_dtype).float()

        ref_out = (normalized * (1.0 + scale.float()) + shift.float()).to(input_dtype)

        torch.testing.assert_close(
            residual_out,
            ref_residual,
            atol=precision[input_dtype],
            rtol=precision[input_dtype],
        )

        torch.testing.assert_close(
            out, ref_out, atol=precision[input_dtype], rtol=precision[input_dtype]
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
