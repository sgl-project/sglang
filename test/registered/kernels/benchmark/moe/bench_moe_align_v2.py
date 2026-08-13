import torch
import triton
from sgl_kernel import moe_align_block_size as aot_moe_align_block_size

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.moe.moe_align import moe_align_block_size_out
from sglang.kernels.ops.moe.moe_align_small_numel import (
    SMALL_NUMEL_LIMIT,
    moe_align_small_numel,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

SHAPES = [
    (8, 129),  # DeepSeek-V3 + fused expert
    (7, 257),  # DeepSeek-V4-Flash + fused expert
    (7, 385),  # DeepSeek-V4-Pro + fused expert
    (16, 896),  # Kimi-K3
]


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


# aot, v1 and v2 share one signature; triton drops the two trailing arguments.
def _run_aot(ids, num_buckets, block_size, sorted_ids, expert_ids, post_pad, cumsum):
    # 8 args, never 9: older wheels bind no ignore_invalid_expert parameter.
    aot_moe_align_block_size(
        ids, num_buckets, block_size, sorted_ids, expert_ids, post_pad, cumsum, True
    )


def _run_v1(ids, num_buckets, block_size, sorted_ids, expert_ids, post_pad, cumsum):
    moe_align_block_size_out(
        ids,
        num_buckets,
        block_size,
        sorted_ids,
        expert_ids,
        post_pad,
        cumsum,
        True,
        version=1,
    )


def _run_v2(ids, num_buckets, block_size, sorted_ids, expert_ids, post_pad, cumsum):
    moe_align_block_size_out(
        ids,
        num_buckets,
        block_size,
        sorted_ids,
        expert_ids,
        post_pad,
        cumsum,
        True,
        version=2,
    )


def _run_triton(ids, num_buckets, block_size, sorted_ids, expert_ids, post_pad, cumsum):
    moe_align_small_numel(
        ids, num_buckets, block_size, sorted_ids, expert_ids, post_pad
    )


FN_MAP = {
    "aot": _run_aot,
    "v1": _run_v1,
    "v2": _run_v2,
    "triton": _run_triton,
}


@marker.parametrize("block_size", [16, 128], [16])
@marker.parametrize("topk,num_experts", SHAPES, [SHAPES[0]])
@marker.parametrize("batch_size", [2**n for n in range(14)], [1, 4096])
@marker.benchmark("impl", ["aot", "v1", "v2", "triton"])
def benchmark(block_size: int, topk: int, num_experts: int, batch_size: int, impl: str):
    num_buckets = num_experts + 1
    numel = batch_size * topk

    if impl == "triton" and numel > SMALL_NUMEL_LIMIT:
        marker.skip(f"numel {numel} > SMALL_NUMEL_LIMIT {SMALL_NUMEL_LIMIT}")

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
        graph_clone_args=(0, 3, 4, 5, 6),
        memory_args=(ids,),
        memory_output=(sorted_ids, expert_ids, post_pad),
    )


if __name__ == "__main__":
    benchmark.run()
