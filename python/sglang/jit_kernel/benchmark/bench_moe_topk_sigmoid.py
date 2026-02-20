"""
Benchmark: moe_topk_sigmoid JIT vs AOT (sgl_kernel)

Measures throughput (µs) across num_tokens and num_experts configurations,
covering static-dispatch (power-of-2) and dynamic-fallback paths.

Run:
    python python/sglang/jit_kernel/benchmark/bench_moe_topk_sigmoid.py

Output columns: num_tokens | num_experts | topk  (µs)
"""

import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.moe_topk_sigmoid import topk_sigmoid as topk_sigmoid_jit

try:
    from sgl_kernel import topk_sigmoid as topk_sigmoid_aot

    AOT_AVAILABLE = True
except ImportError:
    topk_sigmoid_aot = None
    AOT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

NUM_TOKENS_RANGE = get_benchmark_range(
    full_range=[1, 16, 64, 128, 256, 512, 1024, 2048, 4096],
    ci_range=[64, 512, 2048],
)

# (num_experts, topk) — static dispatch + dynamic fallback
EXPERT_CONFIGS = get_benchmark_range(
    full_range=[
        (64, 2),  # static: small expert count
        (128, 4),  # static: common config
        (256, 4),  # static: large power-of-2
    ],
    ci_range=[(128, 4)],
)

_configs_product = list(itertools.product(NUM_TOKENS_RANGE, EXPERT_CONFIGS))
CONFIGS = [(nt, ne, tk) for nt, (ne, tk) in _configs_product]

LINE_VALS = ["jit", "aot"] if AOT_AVAILABLE else ["jit"]
LINE_NAMES = ["JIT (new)", "AOT sgl_kernel"] if AOT_AVAILABLE else ["JIT (new)"]
STYLES = [("blue", "--"), ("orange", "-")] if AOT_AVAILABLE else [("blue", "--")]


# ---------------------------------------------------------------------------
# Benchmark function
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts", "topk"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="moe-topk-sigmoid-performance",
        args={},
    )
)
def benchmark(num_tokens: int, num_experts: int, topk: int, provider: str):
    dtype = torch.float32
    device = "cuda"

    gating = torch.randn((num_tokens, num_experts), dtype=dtype, device=device)
    topk_w = torch.empty((num_tokens, topk), dtype=torch.float32, device=device)
    topk_i = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)

    if provider == "jit":
        fn = lambda: topk_sigmoid_jit(topk_w, topk_i, gating)
    elif provider == "aot":
        fn = lambda: topk_sigmoid_aot(topk_w, topk_i, gating)
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
    for num_tokens, num_experts, topk in [(64, 128, 4), (512, 256, 4), (1024, 64, 2)]:
        gating = torch.randn(
            (num_tokens, num_experts), dtype=torch.float32, device="cuda"
        )
        topk_w_jit = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
        topk_i_jit = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
        topk_w_aot = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
        topk_i_aot = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

        topk_sigmoid_jit(topk_w_jit, topk_i_jit, gating)
        topk_sigmoid_aot(topk_w_aot, topk_i_aot, gating)

        weights_match = torch.allclose(topk_w_jit, topk_w_aot, atol=1e-3, rtol=1e-3)
        indices_match = torch.equal(topk_i_jit, topk_i_aot)
        status = "OK" if (weights_match and indices_match) else "MISMATCH"
        print(
            f"  tokens={num_tokens:5d}  experts={num_experts:3d}  topk={topk}  "
            f"weights={'✓' if weights_match else '✗'}  "
            f"indices={'✓' if indices_match else '✗'}  [{status}]"
        )


if __name__ == "__main__":
    calculate_diff()
    print()
    benchmark.run(print_data=True)
