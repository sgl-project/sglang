"""K3 MoE finalize (top-16 weighted unpermute) vs the trtllm-fork baseline.

The baseline (moe_finalize_fuse_shared, forked from trtllm-gen's finalize
kernels) serializes the 16 gathers behind one CTA per token at decode sizes;
the K3 kernel spreads one 16B output vector per thread across fixed 128-thread
blocks, so small-T decode fills SMs instead. gemm2_out is deliberately NOT
cloned per iteration: in production it was just written by the grouped GEMM,
so an L2-hot read is the realistic regime.
"""

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.kimi_k3 import moe_finalize
from sglang.jit_kernel.moe_finalize_fuse_shared import moe_finalize_fuse_shared
from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")

TOPK = 16


def build_permuted_layout(num_tokens: int, hidden: int, num_experts: int = 896, tile: int = 8):
    gen = torch.Generator(device="cpu").manual_seed(num_tokens)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, generator=gen)[:TOPK] for _ in range(num_tokens)]
    )
    counts = torch.bincount(topk_ids.flatten(), minlength=num_experts)
    padded = (counts + tile - 1) // tile * tile
    bases = torch.cumsum(padded, 0) - padded
    fill = torch.zeros(num_experts, dtype=torch.long)
    idx = torch.empty(num_tokens * TOPK, dtype=torch.int32)
    for i, e in enumerate(topk_ids.flatten().tolist()):
        idx[i] = bases[e] + fill[e]
        fill[e] += 1
    gemm2_out = torch.randn(int(padded.sum()), hidden, dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(num_tokens, TOPK, generator=gen).to(
        device="cuda", dtype=torch.bfloat16
    )
    return gemm2_out, idx.to("cuda"), weights


def run_k3(gemm2_out, idx, weights):
    return moe_finalize(gemm2_out, idx, weights)


def run_trtllm_fork(gemm2_out, idx, weights):
    return moe_finalize_fuse_shared(
        gemm2_out, idx, weights, None, TOPK, enable_pdl=is_arch_support_pdl()
    )


FN_MAP = {
    "k3": run_k3,
    "trtllm_fork": run_trtllm_fork,
}


@marker.parametrize("num_tokens", [2**x for x in range(14)], [1, 64])
@marker.parametrize("hidden", [3584], [3584])
@marker.benchmark("impl", ["k3", "trtllm_fork"])
def benchmark(num_tokens: int, hidden: int, impl: str):
    gemm2_out, idx, weights = build_permuted_layout(num_tokens, hidden)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(gemm2_out, idx, weights),
        memory_args=(gemm2_out[:num_tokens * 16], idx, weights),
    )


if __name__ == "__main__":
    benchmark.run()
