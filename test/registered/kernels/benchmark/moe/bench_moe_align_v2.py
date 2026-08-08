"""moe_align + count&sort, over batch size for a few fixed (topk, num_experts).

    aot     sglang.kernels.ops.moe.moe_align_block_size -- the sgl_kernel wheel
            path, the same baseline the in-tree JIT copy is measured against
    v2      kernels/jit/csrc/moe/moe_align_v2.cuh -- drop-in for that same
            signature. Small batches take one fused launch; above its capacity a
            two-launch histogram/scan + scatter path takes over.
    triton  kernels/ops/moe/moe_align_small_numel.py (PR #32395), the single-CTA
            tiny-numel variant; skipped above its numel <= 64 gate

Routing is uniform random over all experts, with no -1. That deliberately leaves
out two things the real runtime does, both of which make bucket 0 or one routed
bucket huge and hammer a single atomic counter: EP filtering
(``token_dispatcher/standard.py:186-238``) and DeepSeek's fused shared expert
(``models/deepseek_v2.py:571-597``). Add them back before trusting any conclusion
about atomic contention.

    python test/registered/kernels/benchmark/moe/bench_moe_align_v2.py
"""

import torch
import triton

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.moe import moe_align_block_size as aot_moe_align_block_size
from sglang.kernels.ops.moe.moe_align_small_numel import (
    SMALL_NUMEL_LIMIT,
    moe_align_small_numel,
)
from sglang.kernels.ops.moe.moe_align_v2 import CTA_SIZE as V2_CTA_SIZE
from sglang.kernels.ops.moe.moe_align_v2 import moe_align_block_size as v2_moe_align_block_size
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

# (topk, num_experts) as the triton runner sees them. DeepSeek-V4 fuses its shared
# expert into the routed set, so both numbers are one higher than its config.
SHAPES = [
    (7, 257),  # DeepSeek-V4-Flash
    (7, 385),  # DeepSeek-V4-Pro
    (16, 896),  # Kimi-K3
]

BATCH_SIZES = [2**n for n in range(15)]  # 1 .. 16384


def _make_topk_ids(batch_size, topk, num_experts, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(
        0,
        num_experts,
        (batch_size, topk),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )


def _alloc(numel, block_size, num_buckets):
    """Buffers sized exactly as the moe_runner call site sizes them."""
    if numel < num_buckets:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + num_buckets * (block_size - 1)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    return (
        torch.empty((max_num_tokens_padded,), dtype=torch.int32, device="cuda"),
        torch.empty((max_num_m_blocks,), dtype=torch.int32, device="cuda"),
        torch.empty((1,), dtype=torch.int32, device="cuda"),
        torch.empty((num_buckets + 1,), dtype=torch.int32, device="cuda"),
    )


# aot and v2 share one signature; triton drops the two trailing arguments.
def _run_aot(ids, num_buckets, block_size, sorted_ids, expert_ids, post_pad, cumsum):
    aot_moe_align_block_size(
        ids, num_buckets, block_size, sorted_ids, expert_ids, post_pad, cumsum, True
    )


def _run_v2(ids, num_buckets, block_size, sorted_ids, expert_ids, post_pad, cumsum):
    v2_moe_align_block_size(
        ids, num_buckets, block_size, sorted_ids, expert_ids, post_pad, cumsum, True
    )


def _run_triton(ids, num_buckets, block_size, sorted_ids, expert_ids, post_pad, cumsum):
    moe_align_small_numel(ids, num_buckets, block_size, sorted_ids, expert_ids, post_pad)


FN_MAP = {
    "aot": _run_aot,
    "v2": _run_v2,
    "triton": _run_triton,
}


@marker.parametrize("block_size", [64, 128], [128])
@marker.parametrize("topk,num_experts", SHAPES, [SHAPES[-1]])
@marker.parametrize("batch_size", BATCH_SIZES, [1, 4096])
@marker.benchmark("impl", ["aot", "v2", "triton"])
def benchmark(
    block_size: int, topk: int, num_experts: int, batch_size: int, impl: str
):
    num_buckets = num_experts + 1
    numel = batch_size * topk

    if impl == "triton" and numel > SMALL_NUMEL_LIMIT:
        marker.skip(f"numel {numel} > SMALL_NUMEL_LIMIT {SMALL_NUMEL_LIMIT}")
    if impl == "v2" and num_buckets > V2_CTA_SIZE:
        marker.skip(f"num_buckets {num_buckets} > CTA size {V2_CTA_SIZE}")

    ids = _make_topk_ids(batch_size, topk, num_experts, seed=numel)
    sorted_ids, expert_ids, post_pad, cumsum = _alloc(numel, block_size, num_buckets)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(
            ids,
            num_buckets,
            block_size,
            sorted_ids,
            expert_ids,
            post_pad,
            cumsum,
        ),
        # topk_ids is the only read tensor; the rest are written in place but
        # still need rotating so a shared buffer does not stay L2-hot.
        graph_clone_args=(0, 3, 4, 5, 6),
        memory_args=(ids,),
        memory_output=(sorted_ids, expert_ids, post_pad),
        warmup_iters=20,
        replay_iters=200,
    )


if __name__ == "__main__":
    benchmark.run()
