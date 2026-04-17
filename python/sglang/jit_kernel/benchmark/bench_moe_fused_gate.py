"""
Benchmark: moe_fused_gate JIT vs AOT (sgl_kernel)

Measures throughput (µs) of the fused MoE gate kernel across:
  - num_rows (batch × seq_len): the primary performance dimension
  - Expert configs: static-dispatch cases and the dynamic fallback

Run:
    python python/sglang/jit_kernel/benchmark/bench_moe_fused_gate.py

Output columns: num_rows | num_experts | num_expert_group | topk_group | topk  (µs)
"""

import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.moe_fused_gate import moe_fused_gate as moe_jit

try:
    from sgl_kernel import moe_fused_gate as moe_aot

    AOT_AVAILABLE = True
except ImportError:
    moe_aot = None
    AOT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

NUM_ROWS_RANGE = get_benchmark_range(
    full_range=[1, 4, 16, 64, 256, 512, 1024, 2048, 4096, 8192],
    ci_range=[64, 1024],
)

# (num_experts, num_expert_group, topk_group, topk_routed)
# Covers: two static-dispatch configs + one dynamic fallback
EXPERT_CONFIGS = get_benchmark_range(
    full_range=[
        (128, 4, 2, 4),  # static: VPT=32
        (256, 8, 4, 8),  # static: VPT=32 — DeepSeek V3 (most important)
        (512, 16, 8, 16),  # dynamic fallback: VPT=32, outside static switch
    ],
    ci_range=[
        (256, 8, 4, 8),
    ],
)

# Flatten (num_rows, (ne, neg, tg, tk)) → (num_rows, ne, neg, tg, tk)
_configs_product = list(itertools.product(NUM_ROWS_RANGE, EXPERT_CONFIGS))
CONFIGS = [(nr, ne, neg, tg, tk) for nr, (ne, neg, tg, tk) in _configs_product]

LINE_VALS = ["jit", "aot"] if AOT_AVAILABLE else ["jit"]
LINE_NAMES = ["JIT (new)", "AOT sgl_kernel"] if AOT_AVAILABLE else ["JIT (new)"]
STYLES = [("blue", "--"), ("orange", "-")] if AOT_AVAILABLE else [("blue", "--")]


# ---------------------------------------------------------------------------
# Benchmark function
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "num_rows",
            "num_experts",
            "num_expert_group",
            "topk_group",
            "topk_routed",
        ],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="moe-fused-gate-performance",
        args={},
    )
)
def benchmark(
    num_rows: int,
    num_experts: int,
    num_expert_group: int,
    topk_group: int,
    topk_routed: int,
    provider: str,
):
    dtype = torch.float32
    device = "cuda"
    topk = topk_routed  # no fused shared experts in the perf baseline

    inp = torch.rand((num_rows, num_experts), dtype=dtype, device=device)
    bias = torch.rand(num_experts, dtype=dtype, device=device)

    kwargs = dict(
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
    )

    if provider == "jit":
        fn = lambda: moe_jit(inp, bias, **kwargs)
    elif provider == "aot":
        fn = lambda: moe_aot(inp, bias, **kwargs)
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
    configs_to_check = [
        (128, (128, 4, 2, 4)),
        (1024, (256, 8, 4, 8)),
        (512, (512, 16, 8, 16)),
    ]
    for num_rows, (ne, neg, tg, tk) in configs_to_check:
        inp = torch.rand((num_rows, ne), dtype=torch.float32, device="cuda")
        bias = torch.rand(ne, dtype=torch.float32, device="cuda")
        kwargs = dict(num_expert_group=neg, topk_group=tg, topk=tk)
        w_jit, i_jit = moe_jit(inp, bias, **kwargs)
        w_aot, i_aot = moe_aot(inp, bias, **kwargs)
        ids_match = torch.equal(i_jit, i_aot)
        weights_match = torch.allclose(w_jit, w_aot, rtol=1e-4, atol=1e-4)
        status = "OK" if (ids_match and weights_match) else "MISMATCH"
        print(
            f"  rows={num_rows:5d}  experts={ne}  group={neg}  "
            f"ids={'✓' if ids_match else '✗'}  weights={'✓' if weights_match else '✗'}  [{status}]"
        )


if __name__ == "__main__":
    calculate_diff()
    print()
    benchmark.run(print_data=True)
