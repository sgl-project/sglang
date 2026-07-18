"""
Benchmark: moe_sum JIT vs AOT (sgl_kernel)

Measures throughput (µs) across num_tokens × hidden_dim configurations.

Run:
    python python/sglang/jit_kernel/benchmark/bench_moe_sum.py

Output columns: num_tokens | hidden_dim | topk  (µs)
"""

import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.moe_sum import moe_sum as moe_sum_jit

try:
    from sgl_kernel import moe_sum as moe_sum_aot

    AOT_AVAILABLE = True
except ImportError:
    moe_sum_aot = None
    AOT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

NUM_TOKENS_RANGE = get_benchmark_range(
    full_range=[1, 64, 128, 512, 1024, 2048, 4096],
    ci_range=[64, 512, 2048],
)

# (hidden_dim, topk)
EXPERT_CONFIGS = get_benchmark_range(
    full_range=[
        (4096, 2),  # common: topk=2
        (4096, 4),  # common: topk=4
        (7168, 8),  # DeepSeek-style
    ],
    ci_range=[(4096, 4)],
)

_configs_product = list(itertools.product(NUM_TOKENS_RANGE, EXPERT_CONFIGS))
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
        plot_name="moe-sum-performance",
        args={},
    )
)
def benchmark(num_tokens: int, hidden_dim: int, topk: int, provider: str):
    dtype = torch.bfloat16
    device = "cuda"

    x = torch.randn((num_tokens, topk, hidden_dim), dtype=dtype, device=device)
    out = torch.empty((num_tokens, hidden_dim), dtype=dtype, device=device)

    if provider == "jit":
        fn = lambda: moe_sum_jit(x, out)
    elif provider == "aot":
        fn = lambda: moe_sum_aot(x, out)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run_benchmark(fn)


# ---------------------------------------------------------------------------
# Quick correctness diff
# ---------------------------------------------------------------------------


def calculate_diff():
    # JIT uses float32 accumulation; compare against float32 reference.
    # AOT accumulates in bf16 for topk<=4 and uses at::sum_out (fp32) for topk>4,
    # so JIT vs AOT would show false mismatches for small topk in bf16.
    print("Correctness diff (JIT vs float32 reference):")
    for num_tokens, hidden_dim, topk in [
        (64, 4096, 2),
        (512, 4096, 4),
        (1024, 7168, 8),
    ]:
        x = torch.randn(
            (num_tokens, topk, hidden_dim), dtype=torch.bfloat16, device="cuda"
        )
        out_jit = torch.empty(
            (num_tokens, hidden_dim), dtype=torch.bfloat16, device="cuda"
        )
        ref = x.float().sum(dim=1).bfloat16()

        moe_sum_jit(x, out_jit)

        match = torch.allclose(out_jit, ref, atol=1e-3, rtol=1e-3)
        status = "OK" if match else "MISMATCH"
        print(
            f"  tokens={num_tokens:5d}  hidden={hidden_dim:5d}  topk={topk}  "
            f"output={'✓' if match else '✗'}  [{status}]"
        )


if __name__ == "__main__":
    calculate_diff()
    print()
    benchmark.run(print_data=True)
