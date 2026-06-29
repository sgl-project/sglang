"""
Benchmark: kimi_k2_moe_fused_gate JIT vs AOT (sgl_kernel)

Measures throughput (µs) of the Kimi K2 fused MoE gate kernel across
num_rows spanning both the small-token path (≤512) and large-token path (>512).

Run:
    python python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py

Output columns: num_rows  (µs)
"""

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.kimi_k2_moe_fused_gate import (
    kimi_k2_moe_fused_gate as kimi_k2_jit,
)

try:
    from sgl_kernel import kimi_k2_moe_fused_gate as kimi_k2_aot

    AOT_AVAILABLE = True
except ImportError:
    kimi_k2_aot = None
    AOT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

# Covers both the small-token (≤512) and large-token (>512) kernel paths
NUM_ROWS_RANGE = get_benchmark_range(
    full_range=[1, 4, 16, 64, 128, 256, 512, 513, 1024, 2048, 4096, 8192],
    ci_range=[64, 512, 1024],
)

NUM_EXPERTS = 384
TOPK = 6  # Kimi K2 uses topk=6

LINE_VALS = ["jit", "aot"] if AOT_AVAILABLE else ["jit"]
LINE_NAMES = ["JIT (new)", "AOT sgl_kernel"] if AOT_AVAILABLE else ["JIT (new)"]
STYLES = [("blue", "--"), ("orange", "-")] if AOT_AVAILABLE else [("blue", "--")]


# ---------------------------------------------------------------------------
# Benchmark function
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_rows"],
        x_vals=NUM_ROWS_RANGE,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="kimi-k2-moe-fused-gate-performance",
        args={},
    )
)
def benchmark(num_rows: int, provider: str):
    device = "cuda"
    routed_scaling_factor = 2.872

    inp = torch.rand((num_rows, NUM_EXPERTS), dtype=torch.float32, device=device)
    bias = torch.rand(NUM_EXPERTS, dtype=torch.float32, device=device)

    kwargs = dict(
        topk=TOPK,
        renormalize=True,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=False,
    )

    if provider == "jit":
        fn = lambda: kimi_k2_jit(inp, bias, **kwargs)
    elif provider == "aot":
        fn = lambda: kimi_k2_aot(inp, bias, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run_benchmark(fn)


# ---------------------------------------------------------------------------
# Quick correctness diff (printed before benchmark table)
# ---------------------------------------------------------------------------


def calculate_diff():
    if not AOT_AVAILABLE:
        print("sgl_kernel not available — skipping AOT diff check")
        return

    print("Correctness diff (JIT vs AOT):")
    for num_rows in [1, 64, 512, 513, 1024, 4096]:
        inp = torch.rand((num_rows, NUM_EXPERTS), dtype=torch.float32, device="cuda")
        bias = torch.rand(NUM_EXPERTS, dtype=torch.float32, device="cuda")
        kwargs = dict(
            topk=TOPK,
            renormalize=True,
            routed_scaling_factor=2.872,
            apply_routed_scaling_factor_on_output=False,
        )
        w_jit, i_jit = kimi_k2_jit(inp, bias, **kwargs)
        w_aot, i_aot = kimi_k2_aot(inp, bias, **kwargs)
        weights_match = torch.allclose(
            w_jit.sort(dim=-1)[0], w_aot.sort(dim=-1)[0].float(), rtol=1e-3, atol=1e-4
        )
        kernel_used = "small" if num_rows <= 512 else "large"
        status = "OK" if weights_match else "MISMATCH"
        print(
            f"  rows={num_rows:5d}  kernel={kernel_used:5s}  "
            f"weights={'✓' if weights_match else '✗'}  [{status}]"
        )


if __name__ == "__main__":
    calculate_diff()
    print()
    benchmark.run(print_data=True)
