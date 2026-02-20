"""
Benchmark: moe_sum_reduce JIT vs AOT (sgl_kernel)

Measures throughput (µs) across num_tokens × hidden_dim configurations,
covering the BF16 vectorized fast path and the general warp-per-token path.

Run:
    python python/sglang/jit_kernel/benchmark/bench_moe_sum_reduce.py

Output columns: num_tokens | hidden_dim | topk  (µs)
"""

import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.moe_sum_reduce import moe_sum_reduce as moe_sum_reduce_jit

try:
    from sgl_kernel import moe_sum_reduce as moe_sum_reduce_aot

    AOT_AVAILABLE = True
except ImportError:
    moe_sum_reduce_aot = None
    AOT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

NUM_TOKENS_RANGE = get_benchmark_range(
    full_range=[1, 64, 128, 256, 512, 1024, 2048, 4096],
    ci_range=[64, 512, 2048],
)

# (hidden_dim, topk)
CONFIGS_EXPERT = get_benchmark_range(
    full_range=[
        (4096, 4),   # common: 4k hidden, topk=4
        (7168, 8),   # DeepSeek-style: 7k hidden, topk=8
        (2048, 2),   # smaller hidden
    ],
    ci_range=[(4096, 4)],
)

_configs_product = list(itertools.product(NUM_TOKENS_RANGE, CONFIGS_EXPERT))
CONFIGS = [(nt, hd, tk) for nt, (hd, tk) in _configs_product]

LINE_VALS = ["jit", "aot"] if AOT_AVAILABLE else ["jit"]
LINE_NAMES = ["JIT (new)", "AOT sgl_kernel"] if AOT_AVAILABLE else ["JIT (new)"]
STYLES = [("blue", "--"), ("orange", "-")] if AOT_AVAILABLE else [("blue", "--")]


# ---------------------------------------------------------------------------
# Benchmark function
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "hidden_dim", "topk"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="moe-sum-reduce-performance",
        args={},
    )
)
def benchmark(num_tokens: int, hidden_dim: int, topk: int, provider: str):
    dtype = torch.bfloat16
    device = "cuda"

    x = torch.randn((num_tokens, topk, hidden_dim), dtype=dtype, device=device)
    out = torch.empty((num_tokens, hidden_dim), dtype=dtype, device=device)

    if provider == "jit":
        fn = lambda: moe_sum_reduce_jit(x, out, 1.0)
    elif provider == "aot":
        fn = lambda: moe_sum_reduce_aot(x, out, 1.0)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run_benchmark(fn)


# ---------------------------------------------------------------------------
# Quick correctness diff
# ---------------------------------------------------------------------------


def calculate_diff():
    if not AOT_AVAILABLE:
        print("sgl_kernel not available — skipping AOT diff check")
        return

    print("Correctness diff (JIT vs AOT):")
    for num_tokens, hidden_dim, topk in [(64, 4096, 4), (512, 7168, 8), (1024, 2048, 2)]:
        x = torch.randn((num_tokens, topk, hidden_dim), dtype=torch.bfloat16, device="cuda")
        out_jit = torch.empty((num_tokens, hidden_dim), dtype=torch.bfloat16, device="cuda")
        out_aot = torch.empty((num_tokens, hidden_dim), dtype=torch.bfloat16, device="cuda")

        moe_sum_reduce_jit(x, out_jit, 1.0)
        moe_sum_reduce_aot(x, out_aot, 1.0)

        match = torch.allclose(out_jit, out_aot, atol=1e-2, rtol=1e-3)
        status = "OK" if match else "MISMATCH"
        print(
            f"  tokens={num_tokens:5d}  hidden={hidden_dim:5d}  topk={topk}  "
            f"output={'✓' if match else '✗'}  [{status}]"
        )


if __name__ == "__main__":
    calculate_diff()
    print()
    benchmark.run(print_data=True)
