"""Benchmark: fused RMSNorm+RoPE vs separate RMSNorm + RoPE."""

import torch
import triton.testing

from sglang.jit_kernel.diffusion.triton.fused_rmsnorm_rope import fused_rmsnorm_rope


def _ref_rmsnorm_rope(x, weight, cos, sin, head_dim, eps):
    """PyTorch reference: separate RMSNorm + interleaved RoPE."""
    orig_dtype = x.dtype
    x_fp32 = x.float()

    # RMSNorm
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps) * weight.float()

    # Interleaved RoPE
    shape = x_normed.shape
    x_normed = x_normed.view(*shape[:-1], -1, head_dim)
    x1 = x_normed[..., ::2]
    x2 = x_normed[..., 1::2]

    cos_b = cos.unsqueeze(0).unsqueeze(2)
    sin_b = sin.unsqueeze(0).unsqueeze(2)

    o1 = x1 * cos_b - x2 * sin_b
    o2 = x1 * sin_b + x2 * cos_b

    out = torch.stack((o1, o2), dim=-1).flatten(-2).flatten(2)
    return out.to(orig_dtype)


SHAPES = [
    # (B, S, D)  — representative diffusion shapes
    (1, 6, 1536),  # audio: small batch
    (1, 256, 5120),  # video: typical shape
    (1, 1024, 5120),  # video: longer sequence
    (2, 256, 5120),  # video: batch=2
]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "S", "D"],
        x_vals=SHAPES,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton Fused", "PyTorch"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="µs (median)",
        plot_name="fused-rmsnorm-rope",
        args={"head_dim": 128, "eps": 1e-6},
    )
)
def benchmark(B, S, D, provider, head_dim, eps):
    dtype = torch.bfloat16
    torch.manual_seed(42)

    x = torch.randn(B, S, D, dtype=dtype, device="cuda")
    weight = torch.randn(D, dtype=dtype, device="cuda") * 0.5 + 1.0

    head_dim_half = head_dim // 2
    angles = torch.randn(S, head_dim_half, device="cuda", dtype=torch.float32) * 0.5
    cos = angles.cos()
    sin = angles.sin()

    if provider == "triton":
        fn = lambda: fused_rmsnorm_rope(x, weight, cos, sin, head_dim, eps)
    else:
        fn = lambda: _ref_rmsnorm_rope(x, weight, cos, sin, head_dim, eps)

    ms, *_ = triton.testing.do_bench_cudagraph(fn, quantiles=[0.5, 0.2, 0.8])
    return ms * 1000  # µs


if __name__ == "__main__":
    benchmark.run(print_data=True)
