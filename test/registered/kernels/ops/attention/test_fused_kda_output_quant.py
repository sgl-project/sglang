"""Correctness checks for standalone KDA output finalization."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.fla.fused_kda_output_quant import (
    fused_kda_output_quant,
)
from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
from sglang.kernels.ops.quantization.fp8_kernel import (
    fp8_dtype,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_sm90_supported()
    or not hasattr(torch, "float8_e4m3fn"),
    reason="requires SM90 and E4M3 support",
)


def _make_inputs(num_tokens: int, extra_gate_columns: int):
    torch.manual_seed(20260909)
    num_heads, head_dim = 8, 128
    x = torch.randn(
        1,
        num_tokens,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    x[:, 0, 0].zero_()
    gate_storage = torch.randn(
        num_tokens,
        num_heads * head_dim + extra_gate_columns,
        dtype=torch.bfloat16,
        device="cuda",
    )
    gate = gate_storage[:, : num_heads * head_dim].unflatten(-1, (num_heads, head_dim))
    weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")
    return x, gate, weight


@pytest.mark.parametrize(
    "num_tokens,extra_gate_columns", [(1, 37), (6, 37), (8, 37), (6, 0)]
)
def test_fused_kda_output_quant_exactly_matches_legacy_kernels(
    num_tokens: int, extra_gate_columns: int
):
    num_heads, head_dim, eps = 8, 128, 1e-5
    x, gate, weight = _make_inputs(num_tokens, extra_gate_columns)

    # rms_norm_gated writes into its input when no output dtype is supplied.
    norm_out = rms_norm_gated(
        x.clone(),
        gate,
        weight,
        bias=None,
        activation="sigmoid",
        eps=eps,
    )
    q_ref, scale_ref = sglang_per_token_group_quant_fp8(
        norm_out.reshape(num_tokens, num_heads * head_dim).contiguous(), head_dim
    )

    for _ in range(3):
        q_fused, scale_fused = fused_kda_output_quant(x, gate, weight, eps)
        torch.cuda.synchronize()

        assert q_fused.shape == (num_tokens, num_heads * head_dim)
        assert q_fused.dtype == fp8_dtype and q_fused.is_contiguous()
        assert scale_fused.shape == (num_tokens, num_heads)
        assert scale_fused.dtype == torch.float32 and scale_fused.is_contiguous()
        assert scale_fused.stride() == (num_heads, 1)
        assert torch.equal(q_fused, q_ref)
        assert torch.equal(scale_fused, scale_ref)

        dequant_fused = q_fused.float().view(num_tokens, num_heads, head_dim)
        dequant_fused *= scale_fused[..., None]
        dequant_ref = q_ref.float().view(num_tokens, num_heads, head_dim)
        dequant_ref *= scale_ref[..., None]
        assert torch.equal(dequant_fused, dequant_ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
