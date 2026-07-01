"""
Benchmark: fused_qknorm_rope JIT vs AOT (sgl_kernel)

Measures latency (us) and bandwidth (GB/s) for fused_qk_norm_rope across
typical LLM configurations (head_dim x num_heads x num_tokens) plus a set
of real production shapes.

Run:
    python test/registered/jit/benchmark/bench_fused_qknorm_rope.py
"""

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.fused_qknorm_rope import (
    fused_qk_norm_rope as fused_qk_norm_rope_jit,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

try:
    from sgl_kernel import fused_qk_norm_rope as fused_qk_norm_rope_aot

    AOT_AVAILABLE = True
except ImportError:
    fused_qk_norm_rope_aot = None
    AOT_AVAILABLE = False

LINE_VALS = ["jit", "aot"] if AOT_AVAILABLE else ["jit"]

FN_MAP = {
    "jit": fused_qk_norm_rope_jit,
    "aot": fused_qk_norm_rope_aot,
}

# (head_dim, num_heads_q, num_heads_k, num_heads_v) -- typical MoE/dense configs
MODEL_CONFIGS = [
    (64, 32, 8, 8),  # small
    (128, 32, 8, 8),  # typical (e.g. Qwen3-8B)
    (256, 16, 4, 4),  # large head_dim
]

# Real production shapes (self-attention; num_heads_k == num_heads_v == num_heads_q).
# Format: (num_tokens, num_heads_q, num_heads_k, num_heads_v, head_dim, rotary_dim)
PRODUCTION_SHAPES = [
    (4096, 24, 24, 24, 128, 128),  # flux_1024
    (4096, 32, 32, 32, 128, 128),  # qwen_image_1024
    (4096, 32, 32, 32, 128, 64),  # qwen_image_partial
    (4096, 30, 30, 30, 128, 128),  # zimage_1024
    (4096, 24, 24, 24, 128, 128),  # batch2_medium (B=2, T=2048)
]


def _make_inputs(num_tokens, num_heads_q, num_heads_k, num_heads_v, head_dim):
    total_heads = num_heads_q + num_heads_k + num_heads_v
    qkv = torch.randn(
        (num_tokens, total_heads * head_dim), dtype=torch.bfloat16, device="cuda"
    )
    q_weight = torch.ones(head_dim, dtype=torch.bfloat16, device="cuda")
    k_weight = torch.ones(head_dim, dtype=torch.bfloat16, device="cuda")
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    return qkv, q_weight, k_weight, position_ids


def _bench(qkv, q_weight, k_weight, position_ids, common_kwargs, impl):
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(qkv,),
        input_kwargs=dict(
            q_weight=q_weight,
            k_weight=k_weight,
            position_ids=position_ids,
            **common_kwargs,
        ),
        graph_clone_args=(0,),  # qkv is read + written in-place
        graph_clone_kwargs=None,  # weights / position_ids are read-only
        memory_output=(qkv,),  # in-place write to qkv
    )


# ---------------------------------------------------------------------------
# Benchmark: fused_qk_norm_rope (interleave style, no YaRN) -- swept configs
# ---------------------------------------------------------------------------


@marker.parametrize(
    "head_dim,num_heads_q,num_heads_k,num_heads_v",
    MODEL_CONFIGS,
    [(128, 32, 8, 8)],
)
@marker.parametrize("num_tokens", [1, 64, 256, 1024, 4096], [64, 512])
@marker.benchmark("impl", LINE_VALS)
def benchmark(
    num_tokens: int,
    head_dim: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    impl: str,
):
    qkv, q_weight, k_weight, position_ids = _make_inputs(
        num_tokens, num_heads_q, num_heads_k, num_heads_v, head_dim
    )
    common_kwargs = dict(
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        eps=1e-5,
        base=10000.0,
        is_neox=False,
        factor=1.0,
        low=1.0,
        high=32.0,
        attention_factor=1.0,
        rotary_dim=head_dim,
    )
    return _bench(qkv, q_weight, k_weight, position_ids, common_kwargs, impl)


# ---------------------------------------------------------------------------
# Benchmark: fused_qk_norm_rope -- real production shapes
# ---------------------------------------------------------------------------


@marker.parametrize(
    "num_tokens,num_heads_q,num_heads_k,num_heads_v,head_dim,rotary_dim",
    PRODUCTION_SHAPES,
    PRODUCTION_SHAPES[:1],
)
@marker.benchmark("impl", LINE_VALS)
def benchmark_production(
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    rotary_dim: int,
    impl: str,
):
    qkv, q_weight, k_weight, position_ids = _make_inputs(
        num_tokens, num_heads_q, num_heads_k, num_heads_v, head_dim
    )
    common_kwargs = dict(
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        eps=1e-5,
        base=10000.0,
        is_neox=False,
        factor=1.0,
        low=1.0,
        high=32.0,
        attention_factor=1.0,
        rotary_dim=rotary_dim,
    )
    return _bench(qkv, q_weight, k_weight, position_ids, common_kwargs, impl)


if __name__ == "__main__":
    print("fused-qknorm-rope swept configs:")
    benchmark.run()
    print("\nfused-qknorm-rope production shapes:")
    benchmark_production.run()
