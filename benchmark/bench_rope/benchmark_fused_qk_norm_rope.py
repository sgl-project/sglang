"""Benchmark the Triton fused QK normalization, partial RoPE and gate split."""

import torch
import triton.testing

from sglang.kernels.jit.benchmark.utils import get_benchmark_range, run_benchmark

NUM_TOKENS_RANGE = get_benchmark_range(
    full_range=[1, 64, 256, 1024, 4096], ci_range=[64, 512]
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=NUM_TOKENS_RANGE,
        line_arg="provider",
        line_vals=["gated"],
        line_names=["Gated partial RoPE"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="gated-qknorm-rope-performance",
        args={},
    )
)
def bench_gated_qknorm_rope(num_tokens, provider):
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )

    q = torch.randn(num_tokens, 32 * 512, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(num_tokens, 2 * 256, device="cuda", dtype=q.dtype)
    weight = torch.ones(256, device="cuda", dtype=q.dtype)
    cache = torch.randn(max(1, num_tokens), 64, device="cuda", dtype=q.dtype)
    positions = torch.arange(num_tokens, device="cuda")
    return run_benchmark(
        lambda: fused_qk_gemma_rmsnorm_rope_gate(
            q, k, weight, weight, cache, positions, 1e-6, 32, 2, 256, 64
        )
    )


if __name__ == "__main__":
    bench_gated_qknorm_rope.run(print_data=True)
