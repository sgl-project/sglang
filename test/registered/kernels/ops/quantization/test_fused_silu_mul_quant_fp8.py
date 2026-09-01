"""Correctness tests for fused_silu_mul_quant_fp8 kernel.

The reference is computed in pure PyTorch: silu(gate) * up followed by
per_token_group_quant_fp8. The fused kernel performs both steps in a single
Triton launch, so we verify that the output fp8 codes and scales match the
two-step baseline within FP8 quantization tolerance.

Test strategy:
  - scale_diff: max abs difference of per-group scales (should be ~0)
  - fp8_match: percentage of FP8 outputs that are bit-exact
  - cosine_sim: cosine similarity of dequantized outputs (should be > 0.9999)
  - rel_err: mean relative error of dequantized outputs (should be < 0.01)
"""

import pytest
import torch

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    fused_silu_mul_quant_fp8,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

G = 128
EPS = 1e-10


# --------------------------------------------------------------------------- #
# Pure-torch reference (two-step baseline).
# --------------------------------------------------------------------------- #
def ref_silu_mul_quant(x: torch.Tensor, gs: int):
    """Two-step baseline: silu_and_mul + per_token_group_quant_fp8."""
    d = x.shape[-1] // 2
    result = torch.nn.functional.silu(x[..., :d]) * x[..., d:]
    fp8_out, scale = per_token_group_quant_fp8(result, gs)
    return fp8_out, scale


def dequantize(fp8_codes: torch.Tensor, scales: torch.Tensor, gs: int):
    """Dequantize fp8 codes back to float32 using per-group scales."""
    return fp8_codes.float() * scales.repeat_interleave(gs, dim=1).float()


# --------------------------------------------------------------------------- #
# Parametrized test configurations.
# --------------------------------------------------------------------------- @@
SIZES = [
    # (num_tokens, hidden_dim, group_size)
    (1, 2048, 128),
    (16, 2048, 128),
    (256, 2048, 128),
    (4096, 2048, 128),
    (8192, 2048, 128),
    (256, 4096, 128),
    (256, 8192, 128),
    (256, 2048, 64),
    (256, 2048, 256),
]

DTYPES = [torch.bfloat16, torch.float16]


@pytest.mark.parametrize("num_tokens, hidden_dim, group_size", SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_silu_mul_quant_fp8_correctness(
    num_tokens, hidden_dim, group_size, dtype
):
    """Verify fused kernel matches the two-step baseline."""
    torch.manual_seed(42)
    x = torch.randn(num_tokens, 2 * hidden_dim, device="cuda", dtype=dtype) * 3.0

    # Baseline
    ref_fp8, ref_scale = ref_silu_mul_quant(x, group_size)

    # Fused
    fused_fp8, fused_scale = fused_silu_mul_quant_fp8(x, group_size)

    # Scale difference
    scale_diff = (ref_scale.float() - fused_scale.float()).abs().max().item()
    assert scale_diff < 0.01, (
        f"scale_diff={scale_diff} exceeds tolerance for "
        f"shape=({num_tokens}, {hidden_dim}), gs={group_size}, dtype={dtype}"
    )

    # FP8 bit-exact match rate
    fp8_match = (ref_fp8 == fused_fp8).float().mean().item()
    assert fp8_match > 0.95, (
        f"fp8_match={fp8_match:.1%} below 95% for "
        f"shape=({num_tokens}, {hidden_dim}), gs={group_size}, dtype={dtype}"
    )

    # Dequantized cosine similarity
    ref_dq = dequantize(ref_fp8, ref_scale, group_size)
    fused_dq = dequantize(fused_fp8, fused_scale, group_size)
    cosine_sim = torch.nn.functional.cosine_similarity(
        ref_dq.flatten().unsqueeze(0), fused_dq.flatten().unsqueeze(0)
    ).item()
    assert cosine_sim > 0.9999, (
        f"cosine_sim={cosine_sim} below 0.9999 for "
        f"shape=({num_tokens}, {hidden_dim}), gs={group_size}, dtype={dtype}"
    )

    # Relative error
    denom = ref_dq.flatten().abs().clamp(min=1e-6)
    rel_err = ((ref_dq.flatten() - fused_dq.flatten()).abs() / denom).mean().item()
    assert rel_err < 0.01, (
        f"rel_err={rel_err} exceeds 0.01 for "
        f"shape=({num_tokens}, {hidden_dim}), gs={group_size}, dtype={dtype}"
    )


def test_fused_silu_mul_quant_fp8_zero_input():
    """All-zero input should produce all-zero output and near-zero scales."""
    x = torch.zeros(16, 2 * 2048, device="cuda", dtype=torch.bfloat16)
    fp8_out, scale = fused_silu_mul_quant_fp8(x, G)
    assert (
        scale.max().item() < 1e-5
    ), f"Expected near-zero scale, got {scale.max().item()}"
    assert fp8_out.float().abs().max().item() == 0.0, "Expected all-zero fp8 output"


def test_fused_silu_mul_quant_fp8_negative_input():
    """Negative input values should be handled correctly."""
    x = -torch.randn(16, 2 * 2048, device="cuda", dtype=torch.bfloat16) * 3.0
    ref_fp8, ref_scale = ref_silu_mul_quant(x, G)
    fused_fp8, fused_scale = fused_silu_mul_quant_fp8(x, G)
    scale_diff = (ref_scale.float() - fused_scale.float()).abs().max().item()
    assert scale_diff < 0.01, f"Negative input scale_diff={scale_diff}"


def test_fused_silu_mul_quant_fp8_contiguity():
    """Non-contiguous input should raise an assertion error."""
    x = torch.randn(16, 2 * 2048, device="cuda", dtype=torch.bfloat16).t().t()
    x_non_contig = x[:, ::2]  # non-contiguous slice
    with pytest.raises(AssertionError):
        fused_silu_mul_quant_fp8(x_non_contig, G)


@pytest.mark.parametrize("group_size", [1, 32, 64, 128, 256, 512])
def test_fused_silu_mul_quant_fp8_group_sizes(group_size):
    """Test various group sizes."""
    hidden_dim = group_size * 4  # ensure divisibility
    x = torch.randn(8, 2 * hidden_dim, device="cuda", dtype=torch.bfloat16)
    ref_fp8, ref_scale = ref_silu_mul_quant(x, group_size)
    fused_fp8, fused_scale = fused_silu_mul_quant_fp8(x, group_size)
    assert ref_fp8.shape == fused_fp8.shape
    assert ref_scale.shape == fused_scale.shape
    scale_diff = (ref_scale.float() - fused_scale.float()).abs().max().item()
    assert scale_diff < 0.01, f"gs={group_size} scale_diff={scale_diff}"
