"""
Benchmark: fused_qknorm_rope JIT vs AOT (sgl_kernel)

Measures throughput (µs) for fused_qk_norm_rope across typical
LLM configurations (head_dim × num_heads × num_tokens).

Run:
    python python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py
"""

import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.fused_qknorm_rope import (
    fused_qk_norm_rope as fused_qk_norm_rope_jit,
)

try:
    from sgl_kernel import fused_qk_norm_rope as fused_qk_norm_rope_aot

    AOT_AVAILABLE = True
except ImportError:
    fused_qk_norm_rope_aot = None
    AOT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

NUM_TOKENS_RANGE = get_benchmark_range(
    full_range=[1, 64, 256, 1024, 4096],
    ci_range=[64, 512],
)

# (head_dim, num_heads_q, num_heads_k, num_heads_v) — typical MoE/dense configs
MODEL_CONFIGS = get_benchmark_range(
    full_range=[
        (64, 32, 8, 8),  # small
        (128, 32, 8, 8),  # typical (e.g. LLaMA-style GQA)
        (256, 16, 4, 4),  # large head_dim
    ],
    ci_range=[(128, 32, 8, 8)],
)

LINE_VALS = ["jit", "aot"] if AOT_AVAILABLE else ["jit"]
LINE_NAMES = ["JIT (new)", "AOT sgl_kernel"] if AOT_AVAILABLE else ["JIT (new)"]
STYLES = [("blue", "--"), ("orange", "-")] if AOT_AVAILABLE else [("blue", "--")]


# ---------------------------------------------------------------------------
# Benchmark: fused_qk_norm_rope (interleave style, no YaRN)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "head_dim", "num_heads_q", "num_heads_k", "num_heads_v"],
        x_vals=[
            (nt, hd, nq, nk, nv)
            for nt, (hd, nq, nk, nv) in itertools.product(
                NUM_TOKENS_RANGE, MODEL_CONFIGS
            )
        ],
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="fused-qknorm-rope-performance",
        args={},
    )
)
def bench_fused_qknorm_rope(
    num_tokens: int,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    provider: str,
):
    device = "cuda"
    total_heads = num_heads_q + num_heads_k + num_heads_v

    qkv = torch.randn(
        (num_tokens, total_heads * head_dim), dtype=torch.bfloat16, device=device
    )
    q_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    k_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    common_kwargs = dict(
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        eps=1e-5,
        q_weight=q_weight,
        k_weight=k_weight,
        base=10000.0,
        is_neox=False,
        position_ids=position_ids,
        factor=1.0,
        low=1.0,
        high=32.0,
        attention_factor=1.0,
        rotary_dim=head_dim,
    )

    if provider == "jit":
        fn = lambda: fused_qk_norm_rope_jit(qkv.clone(), **common_kwargs)
    elif provider == "aot":
        fn = lambda: fused_qk_norm_rope_aot(qkv.clone(), **common_kwargs)
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

    for head_dim, is_neox in [(64, False), (128, False), (128, True), (256, False)]:
        num_tokens = 32
        num_heads_q, num_heads_k, num_heads_v = 4, 2, 2
        total_heads = num_heads_q + num_heads_k + num_heads_v

        qkv = torch.randn(
            (num_tokens, total_heads * head_dim), dtype=torch.bfloat16, device=device
        )
        q_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
        k_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
        position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

        common = dict(
            num_heads_q=num_heads_q,
            num_heads_k=num_heads_k,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            eps=1e-5,
            q_weight=q_weight,
            k_weight=k_weight,
            base=10000.0,
            is_neox=is_neox,
            position_ids=position_ids,
            factor=1.0,
            low=1.0,
            high=32.0,
            attention_factor=1.0,
            rotary_dim=head_dim,
        )

        qkv_jit = qkv.clone()
        fused_qk_norm_rope_jit(qkv_jit, **common)
        qkv_aot = qkv.clone()
        fused_qk_norm_rope_aot(qkv_aot, **common)

        match = torch.allclose(qkv_jit.float(), qkv_aot.float(), atol=1e-2, rtol=1e-2)
        status = "OK" if match else "MISMATCH"
        max_err = (qkv_jit.float() - qkv_aot.float()).abs().max().item()
        print(
            f"  head_dim={head_dim:3d} is_neox={str(is_neox):5s}  "
            f"max_err={max_err:.2e}  [{status}]"
        )


if __name__ == "__main__":
    calculate_diff()
    print()
    bench_fused_qknorm_rope.run(print_data=True)
