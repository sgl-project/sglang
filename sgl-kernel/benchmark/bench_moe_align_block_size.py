import argparse
import itertools
import os

import torch
import triton
import triton.language as tl
from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

try:
    from vllm import _custom_ops as ops

    VLLM_AVAILABLE = True
except ImportError:
    ops = None
    VLLM_AVAILABLE = False

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

USE_RANDOM_PERM = False


def ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)
    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)


@triton.jit
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    grid = (num_experts,)
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    tokens_per_thread = ceil_div(numel, num_experts)

    moe_align_block_size_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
    )
    moe_align_block_size_stage2[grid](
        tokens_cnts,
        num_experts,
    )
    moe_align_block_size_stage3[(1,)](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )
    moe_align_block_size_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )


def calculate_diff(num_tokens, num_experts=256, block_size=128, topk=8):
    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, dtype=torch.int32, device="cuda")[:topk]
            for _ in range(num_tokens)
        ]
    )

    # SGL kernel uses dynamic padding optimization
    max_num_tokens_padded_sgl = topk_ids.numel() + num_experts * (block_size - 1)
    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded_sgl = topk_ids.numel() * block_size
    sorted_ids_cuda = torch.empty(
        (max_num_tokens_padded_sgl,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids_cuda.fill_(topk_ids.numel())
    max_num_m_blocks_sgl = max_num_tokens_padded_sgl // block_size
    expert_ids_cuda = torch.zeros(
        (max_num_m_blocks_sgl,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad_cuda = torch.empty(
        (1), dtype=torch.int32, device=topk_ids.device
    )
    cumsum_buffer = torch.zeros(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )

    # Triton and vLLM use original padding calculation
    max_num_tokens_padded_triton = topk_ids.numel() + num_experts * (block_size - 1)
    max_num_m_blocks_triton = max_num_tokens_padded_triton // block_size
    sorted_ids_triton = torch.empty(
        (max_num_tokens_padded_triton,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids_triton.fill_(topk_ids.numel())
    expert_ids_triton = torch.zeros(
        (max_num_m_blocks_triton,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad_triton = torch.empty_like(num_tokens_post_pad_cuda)

    sorted_ids_vllm = torch.empty_like(sorted_ids_triton)
    sorted_ids_vllm.fill_(topk_ids.numel())
    expert_ids_vllm = torch.zeros_like(expert_ids_triton)
    num_tokens_post_pad_vllm = torch.empty_like(num_tokens_post_pad_cuda)

    # compare the performance of cuda, triton and vllm implementation
    sgl_moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_cuda,
        expert_ids_cuda,
        num_tokens_post_pad_cuda,
        cumsum_buffer,
    )
    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_triton,
        expert_ids_triton,
        num_tokens_post_pad_triton,
    )

    if VLLM_AVAILABLE:
        try:
            ops.moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids_vllm,
                expert_ids_vllm,
                num_tokens_post_pad_vllm,
            )
            print(f"✅ VLLM implementation works with {num_experts} experts!")
            vllm_works = True
        except Exception as e:
            print(f"❌ VLLM implementation failed with {num_experts} experts: {e}")
            vllm_works = False
    else:
        print("⚠️ vLLM not available, skipping vLLM test")
        vllm_works = False

    if torch.allclose(expert_ids_cuda, expert_ids_triton) and torch.allclose(
        num_tokens_post_pad_cuda, num_tokens_post_pad_triton
    ):
        print("✅ SGL and Triton implementations match")
    else:
        print("❌ SGL and Triton implementations do not match")
        print("SGL expert_ids:", expert_ids_cuda)
        print("Triton expert_ids:", expert_ids_triton)
        print("SGL num_tokens_post_pad:", num_tokens_post_pad_cuda)
        print("Triton num_tokens_post_pad:", num_tokens_post_pad_triton)

    if (
        vllm_works
        and torch.allclose(expert_ids_cuda, expert_ids_vllm)
        and torch.allclose(num_tokens_post_pad_cuda, num_tokens_post_pad_vllm)
    ):
        print("✅ SGL and VLLM implementations match")
    else:
        if not vllm_works:
            print("⚠️ VLLM comparison skipped due to failure")
        else:
            print("❌ SGL and VLLM implementations do not match")
            print("SGL expert_ids:", expert_ids_cuda)
            print("VLLM expert_ids:", expert_ids_vllm)
            print("SGL num_tokens_post_pad:", num_tokens_post_pad_cuda)
            print("VLLM num_tokens_post_pad:", num_tokens_post_pad_vllm)


# Test range
num_tokens_range = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
num_experts_range = [8, 32, 64, 128, 256]
topk_range = [1, 2, 4, 8]

configs = list(itertools.product(num_tokens_range, num_experts_range, topk_range))


def get_topk_ids(num_tokens: int, num_experts: int, topk: int) -> torch.Tensor:
    topk_ids = torch.zeros((num_tokens, topk), dtype=torch.int32, device="cuda")
    for i in range(num_tokens):
        topk_ids[i, :] = torch.randperm(num_experts, dtype=torch.int32, device="cuda")[
            :topk
        ]
    return topk_ids


def sgl_moe_align_block_size_with_empty(
    topk_ids,
    num_experts,
    block_size,
    sorted_ids,
    expert_ids,
    num_tokens_post_pad,
    pad_sorted_token_ids=False,
):
    if not pad_sorted_token_ids:
        sorted_ids.fill_(topk_ids.numel())

    cumsum_buffer = torch.empty(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )

    sgl_moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids.clone(),
        expert_ids.clone(),
        num_tokens_post_pad.clone(),
        cumsum_buffer,
        pad_sorted_token_ids,
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts", "topk"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl", "sgl_fusion", "triton"],
        line_names=["SGL", "SGL Fusion", "Triton"],
        styles=[("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel="us",
        plot_name="moe-align-block-size-performance",
        args={},
    )
)
def benchmark(num_tokens, num_experts, topk, provider):
    block_size = 128

    if USE_RANDOM_PERM:
        topk_ids = get_topk_ids(num_tokens, num_experts, topk)
    else:
        topk_ids = torch.randint(
            0,
            num_experts,
            (num_tokens, topk),
            dtype=torch.int32,
            device="cuda",
        )

    # Calculate max_num_tokens_padded based on provider
    if provider == "sgl" or provider == "sgl_fusion":
        # Apply dynamic padding optimization for SGL kernel
        max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
        if topk_ids.numel() < num_experts:
            max_num_tokens_padded = topk_ids.numel() * block_size
    else:  # triton
        # Use original padding calculation for Triton
        max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)

    # Create tensors
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "sgl":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: sgl_moe_align_block_size_with_empty(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
            ),
            quantiles=quantiles,
        )
    elif provider == "sgl_fusion":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: sgl_moe_align_block_size_with_empty(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                pad_sorted_token_ids=True,
            ),
            quantiles=quantiles,
        )
    elif provider == "triton":
        sorted_ids.fill_(topk_ids.numel())
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: moe_align_block_size_triton(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids.clone(),
                expert_ids.clone(),
                num_tokens_post_pad.clone(),
            ),
            quantiles=quantiles,
        )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/moe_align_blocks/",
        help="Path to save moe align benchmark results",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=256,
        choices=[8, 16, 32, 64, 128, 256],
        help="Number of experts for benchmark",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        choices=[2, 4, 8],
        help="Top-k value for benchmark",
    )
    parser.add_argument(
        "--skip_full_benchmark",
        action="store_true",
        help="Only run the calculate_diff function, skip full benchmarking",
    )
    args = parser.parse_args()

    # Simplify for CI environment
    if IS_CI:
        num_tokens = 256  # Smaller for CI
        num_experts = 8  # Smaller for CI
        topk = 2  # Smaller for CI
    else:
        num_tokens = 1024
        num_experts = args.num_experts
        topk = args.topk

    calculate_diff(num_tokens=num_tokens, num_experts=num_experts, topk=topk)

    if not args.skip_full_benchmark and not IS_CI:  # Skip full benchmark in CI
        print(f"\n📊 Running performance benchmark for {args.num_experts} experts...")
        benchmark.run(print_data=True)
