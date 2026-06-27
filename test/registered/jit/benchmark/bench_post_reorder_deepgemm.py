"""Benchmark: fused deep_gemm MoE combine (post_reorder_deepgemm) vs the legacy
post_reorder_triton_kernel + separate ``output *= routed_scaling_factor`` pass.

The legacy path launches one program per token (poor occupancy at decode batch
sizes) plus a second full-tensor scaling kernel; the fused path uses a 2-D
persistent grid, fp32 accumulation, and folds the scaling into the store.
"""

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.srt.layers.moe.ep_moe.kernels import (
    post_reorder_deepgemm,
    post_reorder_triton_kernel,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, suite="base-b-kernel-benchmark-1-gpu-large")

HIDDEN = 6144
NUM_EXPERTS = 129  # 128 routed + 1 fused shared
TOP_K = 5
RSF = 2.0


def _build(num_tokens):
    m_max = (num_tokens // 256 + 1) * 256
    down_output = torch.randn(
        NUM_EXPERTS * m_max, HIDDEN, dtype=torch.bfloat16, device="cuda"
    )
    # timing only: a representative (bounded, valid) access pattern is enough.
    topk_ids = torch.randint(
        0, NUM_EXPERTS, (num_tokens, TOP_K), dtype=torch.int32, device="cuda"
    )
    src2dst = (
        topk_ids.long() * m_max
        + torch.randint(0, m_max, (num_tokens, TOP_K), device="cuda")
    ).to(torch.int32)
    topk_weights = torch.rand(num_tokens, TOP_K, dtype=torch.float32, device="cuda")
    return down_output, src2dst, topk_ids, topk_weights


def _new(down_output, src2dst, topk_ids, topk_weights):
    num_tokens = topk_ids.shape[0]
    out = torch.empty(num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    post_reorder_deepgemm(
        down_output,
        out,
        src2dst,
        topk_ids,
        topk_weights,
        TOP_K,
        num_tokens,
        HIDDEN,
        RSF,
    )
    return out


def _old(down_output, src2dst, topk_ids, topk_weights):
    num_tokens = topk_ids.shape[0]
    out = torch.empty(num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    post_reorder_triton_kernel[(num_tokens,)](
        down_output, out, src2dst, topk_ids, topk_weights, TOP_K, HIDDEN, BLOCK_SIZE=512
    )
    out *= RSF
    return out


FN_MAP = {"fused": _new, "legacy": _old}


@marker.parametrize("num_tokens", [1, 8, 64, 256, 1024, 4096, 16384], [64, 4096])
@marker.benchmark("impl", ["fused", "legacy"])
def benchmark(num_tokens: int, impl: str):
    down_output, src2dst, topk_ids, topk_weights = _build(num_tokens)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(down_output, src2dst, topk_ids, topk_weights),
        graph_clone_args=(0,),  # down_output is the read-heavy input
        memory_args=None,
    )


if __name__ == "__main__":
    benchmark.run()
