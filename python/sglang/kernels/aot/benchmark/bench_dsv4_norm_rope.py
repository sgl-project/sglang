"""Benchmark for DeepSeek-V4 fused norm + RoPE kernels."""

import itertools

import sgl_kernel
import torch
import triton
import triton.testing

try:
    from sglang.utils import is_in_ci

    IS_CI = is_in_ci()
except ImportError:
    IS_CI = False

batch_sizes = [1] if IS_CI else [1, 4, 16, 64, 256]
num_heads_list = [8] if IS_CI else [8, 16, 64]
head_dims = [192] if IS_CI else [128, 192]

configs = list(itertools.product(batch_sizes, num_heads_list, head_dims))


def torch_rmsnorm_rope(
    q: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor, eps: float
) -> torch.Tensor:
    """Naive PyTorch reference: RMSNorm + RoPE."""
    rms = torch.sqrt(q.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    q_normed = (q.float() / rms).to(q.dtype)
    return q_normed


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "num_heads", "head_dim"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sglang", "torch"],
        line_names=["SGL Kernel", "PyTorch"],
        styles=[("green", "-"), ("red", "--")],
        ylabel="µs (median)",
        plot_name="dsv4-q-norm-rope-performance",
        args={},
    )
)
def benchmark_q_norm_rope(batch_size, num_heads, head_dim, provider):
    torch.manual_seed(42)
    eps = 1e-6
    max_pos = 8192
    rope_dim = 64

    q_input = torch.randn(
        batch_size, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    q_output = torch.empty_like(q_input)
    freqs_cis = torch.randn(max_pos, rope_dim, dtype=torch.float32, device="cuda")
    positions = torch.randint(
        0, max_pos, (batch_size,), dtype=torch.int32, device="cuda"
    )

    if provider == "sglang":
        fn = lambda: sgl_kernel.dsv4_fused_q_norm_rope(
            q_input, freqs_cis, positions, eps, q_output
        )
    else:
        fn = lambda: torch_rmsnorm_rope(q_input, freqs_cis, positions, eps)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark_q_norm_rope.run(print_data=True)
