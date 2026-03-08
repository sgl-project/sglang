# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

sys.path.insert(
    0,
    __import__("os").path.join(
        __import__("os").path.dirname(__file__), "..", "..", "kernels"
    ),
)
from import_fused_norm import import_fused_norm_module

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="nightly-1-gpu", nightly=True)

if not torch.cuda.is_available():
    pytest.skip(
        reason="CUDA is required for fused RMSNorm+RoPE kernel tests.",
        allow_module_level=True,
    )


triton_ops = import_fused_norm_module()


def reference_rmsnorm_rope(x, weight, cos, sin, head_dim, eps):
    """Sequential reference: RMSNorm then interleaved RoPE using PyTorch ops."""
    orig_dtype = x.dtype
    x_fp32 = x.float()

    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps) * weight.float()

    B, S, D = x_normed.shape
    x_normed = x_normed.view(B, S, -1, head_dim)

    x1 = x_normed[..., ::2]
    x2 = x_normed[..., 1::2]

    cos_b = cos.unsqueeze(0).unsqueeze(2)
    sin_b = sin.unsqueeze(0).unsqueeze(2)

    o1 = x1 * cos_b - x2 * sin_b
    o2 = x1 * sin_b + x2 * cos_b

    out = torch.stack((o1, o2), dim=-1).flatten(-2).flatten(2)
    return out.to(orig_dtype)


def make_test_inputs(B, S, D, head_dim, dtype, device):
    torch.manual_seed(42)
    x = torch.randn(B, S, D, dtype=dtype, device=device)
    weight = torch.randn(D, dtype=dtype, device=device) * 0.5 + 1.0

    head_dim_half = head_dim // 2
    angles = torch.randn(S, head_dim_half, device=device, dtype=torch.float32) * 0.5
    cos = angles.cos()
    sin = angles.sin()

    return x, weight, cos, sin


DIMS_AND_HEADS = [
    (1536, 12),  # audio: dim=1536, num_heads=12, head_dim=128
    (5120, 40),  # video: dim=5120, num_heads=40, head_dim=128
]


@pytest.mark.parametrize("dim,num_heads", DIMS_AND_HEADS)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [16, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_fused_rmsnorm_rope_tp1(dim, num_heads, batch_size, seq_len, dtype):
    device = "cuda"
    head_dim = dim // num_heads
    eps = 1e-6

    x, weight, cos, sin = make_test_inputs(
        batch_size, seq_len, dim, head_dim, dtype, device
    )

    ref_out = reference_rmsnorm_rope(x, weight, cos, sin, head_dim, eps)
    fused_out = triton_ops.fused_rmsnorm_rope(x, weight, cos, sin, head_dim, eps)

    torch.testing.assert_close(fused_out, ref_out, rtol=5e-2, atol=1e-2)


@pytest.mark.parametrize("dim,num_heads", DIMS_AND_HEADS)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [16, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_fused_rmsnorm_rope_output_dtype(dim, num_heads, batch_size, seq_len, dtype):
    device = "cuda"
    head_dim = dim // num_heads
    eps = 1e-6

    x, weight, cos, sin = make_test_inputs(
        batch_size, seq_len, dim, head_dim, dtype, device
    )
    fused_out = triton_ops.fused_rmsnorm_rope(x, weight, cos, sin, head_dim, eps)

    assert fused_out.dtype == x.dtype, f"Expected {x.dtype}, got {fused_out.dtype}"
    assert fused_out.shape == x.shape, f"Expected {x.shape}, got {fused_out.shape}"


@pytest.mark.parametrize("dim,num_heads", DIMS_AND_HEADS)
@pytest.mark.parametrize("seq_len", [16, 128])
@torch.inference_mode()
def test_fused_rmsnorm_rope_tp_simulation(dim, num_heads, seq_len):
    device = "cuda"
    dtype = torch.bfloat16
    head_dim = dim // num_heads
    eps = 1e-6
    B = 1
    tp_size = 2

    x, weight, cos, sin = make_test_inputs(B, seq_len, dim, head_dim, dtype, device)

    ref_out = reference_rmsnorm_rope(x, weight, cos, sin, head_dim, eps)

    x_shards = x.tensor_split(tp_size, dim=-1)
    weight_shards = weight.tensor_split(tp_size)

    outputs = []
    for rank in range(tp_size):
        x_local = x_shards[rank].contiguous()
        w_local = weight_shards[rank].float()
        x_2d = x_local.reshape(-1, x_local.shape[-1]).contiguous()
        M, D_local = x_2d.shape

        mean_sq = torch.empty(M, dtype=torch.float32, device=device)
        triton_ops._compute_local_mean_sq_kernel[(M,)](
            mean_sq,
            x_2d,
            M,
            D_local,
            x_2d.stride(0),
        )

        x_other = x_shards[1 - rank].contiguous()
        x_other_2d = x_other.reshape(-1, x_other.shape[-1]).contiguous()
        mean_sq_other = torch.empty(M, dtype=torch.float32, device=device)
        triton_ops._compute_local_mean_sq_kernel[(M,)](
            mean_sq_other,
            x_other_2d,
            M,
            D_local,
            x_other_2d.stride(0),
        )
        global_mean_sq = (mean_sq + mean_sq_other) / 2.0

        rstd = torch.rsqrt(global_mean_sq + eps)
        out = torch.empty_like(x_2d)
        head_dim_half = head_dim // 2
        triton_ops._fused_apply_norm_rope_kernel[(M,)](
            out,
            x_2d,
            w_local,
            cos,
            sin,
            rstd,
            M,
            D_local,
            seq_len,
            head_dim_half,
            x_2d.stride(0),
            out.stride(0),
            cos.stride(0),
        )
        outputs.append(out.view(x_local.shape))

    fused_out = torch.cat(outputs, dim=-1)

    torch.testing.assert_close(fused_out, ref_out, rtol=5e-2, atol=1e-2)


if __name__ == "__main__":
    test_fused_rmsnorm_rope_tp1(5120, 40, 1, 256, torch.bfloat16)
    test_fused_rmsnorm_rope_tp_simulation(5120, 40, 128)
