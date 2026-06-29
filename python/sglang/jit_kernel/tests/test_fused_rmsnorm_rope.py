# SPDX-License-Identifier: Apache-2.0
"""Tests for fused RMSNorm + interleaved RoPE kernel."""

import pytest
import torch

from sglang.jit_kernel.diffusion.triton.fused_rmsnorm_rope import fused_rmsnorm_rope


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def _ref_fused_rmsnorm_rope(x, weight, cos, sin, head_dim, eps):
    """PyTorch reference implementation: RMSNorm then interleaved RoPE."""
    orig_dtype = x.dtype
    x_fp32 = x.float()

    # RMSNorm
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps) * weight.float()

    # Interleaved RoPE
    shape = x_normed.shape
    x_normed = x_normed.view(*shape[:-1], -1, head_dim)
    x1 = x_normed[..., ::2]  # even elements
    x2 = x_normed[..., 1::2]  # odd elements

    cos_b = cos.unsqueeze(0).unsqueeze(2)  # [1, S, 1, head_dim//2]
    sin_b = sin.unsqueeze(0).unsqueeze(2)

    o1 = x1 * cos_b - x2 * sin_b
    o2 = x1 * sin_b + x2 * cos_b

    out = torch.stack((o1, o2), dim=-1).flatten(-2).flatten(2)
    return out.to(orig_dtype)


def _make_test_inputs(B, S, D, head_dim, dtype, device):
    torch.manual_seed(42)
    x = torch.randn(B, S, D, dtype=dtype, device=device)
    weight = torch.randn(D, dtype=dtype, device=device) * 0.5 + 1.0

    head_dim_half = head_dim // 2
    angles = torch.randn(S, head_dim_half, device=device, dtype=torch.float32) * 0.5
    cos = angles.cos()
    sin = angles.sin()

    return x, weight, cos, sin


# Test shapes: (B, S, D, head_dim)
# Includes edge cases: S=1, D not multiple of largest BLOCK_D (1024), large B
TEST_SHAPES = [
    (1, 6, 1536, 128),  # audio: small batch, typical shape
    (1, 1024, 1536, 128),  # audio: longer sequence
    (1, 256, 5120, 128),  # video: typical shape
    (2, 512, 5120, 128),  # video: batch=2
    (4, 256, 5120, 128),  # video: larger batch
    (1, 1, 1536, 128),  # edge: S=1
    (1, 16, 3072, 128),  # D not multiple of 1024
]


@pytest.mark.parametrize("B,S,D,head_dim", TEST_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_fused_rmsnorm_rope_correctness(B, S, D, head_dim, dtype):
    eps = 1e-6
    x, weight, cos, sin = _make_test_inputs(B, S, D, head_dim, dtype, "cuda")

    out = fused_rmsnorm_rope(x, weight, cos, sin, head_dim, eps)
    ref = _ref_fused_rmsnorm_rope(x, weight, cos, sin, head_dim, eps)

    atol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(
        out,
        ref,
        atol=atol,
        rtol=atol,
        msg=f"Mismatch at B={B} S={S} D={D} head_dim={head_dim} dtype={dtype}",
    )


@pytest.mark.parametrize("B,S,D,head_dim", TEST_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_fused_rmsnorm_rope_output_dtype_shape(B, S, D, head_dim, dtype):
    eps = 1e-6
    x, weight, cos, sin = _make_test_inputs(B, S, D, head_dim, dtype, "cuda")

    out = fused_rmsnorm_rope(x, weight, cos, sin, head_dim, eps)

    assert out.dtype == x.dtype, f"Expected {x.dtype}, got {out.dtype}"
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_fused_rmsnorm_rope_non_cuda_raises():
    """CPU tensor should raise AssertionError."""
    x = torch.randn(1, 4, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, dtype=torch.bfloat16)
    cos = torch.randn(4, 64, dtype=torch.float32)
    sin = torch.randn(4, 64, dtype=torch.float32)

    with pytest.raises(AssertionError, match="CUDA"):
        fused_rmsnorm_rope(x, weight, cos, sin, head_dim=128, eps=1e-6)


def test_fused_rmsnorm_rope_odd_hidden_dim_raises():
    """Odd hidden dimension should raise AssertionError."""
    x = torch.randn(1, 4, 127, dtype=torch.bfloat16, device="cuda")  # odd D
    weight = torch.randn(127, dtype=torch.bfloat16, device="cuda")
    cos = torch.randn(4, 64, dtype=torch.float32, device="cuda")
    sin = torch.randn(4, 64, dtype=torch.float32, device="cuda")

    with pytest.raises(AssertionError, match="even"):
        fused_rmsnorm_rope(x, weight, cos, sin, head_dim=128, eps=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
