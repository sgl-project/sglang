"""
Benchmark: prepare_moe_input JIT vs AOT (sgl_kernel)

Measures throughput (µs) for prepare_moe_input, shuffle_rows,
and apply_shuffle_mul_sum across typical MoE configurations.

Run:
    python python/sglang/jit_kernel/benchmark/bench_prepare_moe_input.py
"""

import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.prepare_moe_input import (
    apply_shuffle_mul_sum as apply_shuffle_mul_sum_jit,
)
from sglang.jit_kernel.prepare_moe_input import (
    prepare_moe_input as prepare_moe_input_jit,
)
from sglang.jit_kernel.prepare_moe_input import shuffle_rows as shuffle_rows_jit

try:
    from sgl_kernel import apply_shuffle_mul_sum as apply_shuffle_mul_sum_aot
    from sgl_kernel import prepare_moe_input as prepare_moe_input_aot
    from sgl_kernel import shuffle_rows as shuffle_rows_aot

    AOT_AVAILABLE = True
except ImportError:
    apply_shuffle_mul_sum_aot = None
    prepare_moe_input_aot = None
    shuffle_rows_aot = None
    AOT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

NUM_TOKENS_RANGE = get_benchmark_range(
    full_range=[1, 64, 256, 1024, 4096],
    ci_range=[64, 512],
)

# (topk, num_experts) typical MoE configs
MOE_CONFIGS = get_benchmark_range(
    full_range=[(2, 8), (4, 16), (8, 64)],
    ci_range=[(4, 16)],
)

LINE_VALS = ["jit", "aot"] if AOT_AVAILABLE else ["jit"]
LINE_NAMES = ["JIT (new)", "AOT sgl_kernel"] if AOT_AVAILABLE else ["JIT (new)"]
STYLES = [(("blue", "--")), (("orange", "-"))] if AOT_AVAILABLE else [("blue", "--")]


# ---------------------------------------------------------------------------
# Benchmark: prepare_moe_input
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "topk", "num_experts"],
        x_vals=[(nt, tk, ne) for nt, (tk, ne) in itertools.product(NUM_TOKENS_RANGE, MOE_CONFIGS)],
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="prepare-moe-input-performance",
        args={},
    )
)
def bench_prepare_moe_input(num_tokens: int, topk: int, num_experts: int, provider: str):
    device = "cuda"
    n, k = 4096, 4096
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device)

    def alloc():
        return (
            torch.empty(num_experts + 1, dtype=torch.int32, device=device),
            torch.empty((num_experts, 3), dtype=torch.int32, device=device),
            torch.empty((num_experts, 3), dtype=torch.int32, device=device),
            torch.empty(num_tokens * topk, dtype=torch.int32, device=device),
            torch.empty(num_tokens * topk, dtype=torch.int32, device=device),
        )

    eo, ps1, ps2, ip, op = alloc()

    if provider == "jit":
        fn = lambda: prepare_moe_input_jit(topk_ids, eo, ps1, ps2, ip, op, num_experts, n, k)
    elif provider == "aot":
        fn = lambda: prepare_moe_input_aot(topk_ids, eo, ps1, ps2, ip, op, num_experts, n, k)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run_benchmark(fn)


# ---------------------------------------------------------------------------
# Benchmark: shuffle_rows
# ---------------------------------------------------------------------------

SHUFFLE_CONFIGS = get_benchmark_range(
    full_range=[(4096, 256), (4096, 4096), (32768, 512)],
    ci_range=[(4096, 4096)],
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_rows", "num_cols"],
        x_vals=SHUFFLE_CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="shuffle-rows-performance",
        args={},
    )
)
def bench_shuffle_rows(num_rows: int, num_cols: int, provider: str):
    device = "cuda"
    dtype = torch.bfloat16
    input_t = torch.randn((num_rows, num_cols), dtype=dtype, device=device)
    dst2src = torch.randperm(num_rows, device=device).to(torch.int32)

    if provider == "jit":
        fn = lambda: shuffle_rows_jit(input_t, dst2src, (num_rows, num_cols))
    elif provider == "aot":
        fn = lambda: shuffle_rows_aot(input_t, dst2src, (num_rows, num_cols))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run_benchmark(fn)


# ---------------------------------------------------------------------------
# Benchmark: apply_shuffle_mul_sum
# ---------------------------------------------------------------------------

APPLY_CONFIGS = get_benchmark_range(
    full_range=[(512, 2, 4096), (1024, 4, 4096), (2048, 8, 4096)],
    ci_range=[(512, 4, 4096)],
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m", "topk", "k"],
        x_vals=APPLY_CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="apply-shuffle-mul-sum-performance",
        args={},
    )
)
def bench_apply_shuffle_mul_sum(m: int, topk: int, k: int, provider: str):
    device = "cuda"
    dtype = torch.bfloat16
    perm = torch.randperm(m * topk, device=device).to(torch.int32)
    input_t = torch.randn((m * topk, k), dtype=dtype, device=device)
    factors = torch.rand((m * topk,), dtype=dtype, device=device)
    output = torch.empty((m, k), dtype=dtype, device=device)

    if provider == "jit":
        fn = lambda: apply_shuffle_mul_sum_jit(input_t, output, perm, factors)
    elif provider == "aot":
        fn = lambda: apply_shuffle_mul_sum_aot(input_t, output, perm, factors)
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

    device = "cuda"
    print("Correctness diff (JIT vs AOT):")

    # prepare_moe_input
    for num_tokens, topk, num_experts in [(64, 4, 8), (256, 2, 16)]:
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device
        )
        n, k = 256, 512

        def alloc():
            return (
                torch.empty(num_experts + 1, dtype=torch.int32, device=device),
                torch.empty((num_experts, 3), dtype=torch.int32, device=device),
                torch.empty((num_experts, 3), dtype=torch.int32, device=device),
                torch.empty(num_tokens * topk, dtype=torch.int32, device=device),
                torch.empty(num_tokens * topk, dtype=torch.int32, device=device),
            )

        eo_jit, ps1_jit, ps2_jit, ip_jit, op_jit = alloc()
        eo_aot, ps1_aot, ps2_aot, ip_aot, op_aot = alloc()
        prepare_moe_input_jit(topk_ids, eo_jit, ps1_jit, ps2_jit, ip_jit, op_jit, num_experts, n, k)
        prepare_moe_input_aot(topk_ids, eo_aot, ps1_aot, ps2_aot, ip_aot, op_aot, num_experts, n, k)

        match = torch.equal(eo_jit, eo_aot) and torch.equal(ps1_jit, ps1_aot)
        status = "OK" if match else "MISMATCH"
        print(
            f"  prepare_moe_input tokens={num_tokens:4d} topk={topk} experts={num_experts:3d}"
            f"  [{status}]"
        )

    # apply_shuffle_mul_sum
    for dtype, m, topk, k in [(torch.bfloat16, 64, 4, 1024), (torch.float16, 128, 2, 512)]:
        perm = torch.randperm(m * topk, device=device).to(torch.int32)
        input_t = torch.randn((m * topk, k), dtype=dtype, device=device)
        factors = torch.rand((m * topk,), dtype=dtype, device=device)
        out_jit = torch.empty((m, k), dtype=dtype, device=device)
        out_aot = torch.empty((m, k), dtype=dtype, device=device)
        apply_shuffle_mul_sum_jit(input_t, out_jit, perm, factors)
        apply_shuffle_mul_sum_aot(input_t, out_aot, perm, factors)
        match = torch.allclose(out_jit, out_aot, atol=1e-2, rtol=1e-3)
        status = "OK" if match else "MISMATCH"
        print(
            f"  apply_shuffle_mul_sum dtype={dtype} m={m} topk={topk} k={k}  [{status}]"
        )


if __name__ == "__main__":
    calculate_diff()
    print()
    bench_prepare_moe_input.run(print_data=True)
    bench_shuffle_rows.run(print_data=True)
    bench_apply_shuffle_mul_sum.run(print_data=True)
