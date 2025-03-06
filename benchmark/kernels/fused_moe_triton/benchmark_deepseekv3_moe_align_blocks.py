import argparse
import itertools

import torch
import triton
import triton.language as tl
from sgl_kernel import moe_align_block_size

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


def calculate_diff(batch_size, seq_len):
    num_experts = 256
    block_size = 128
    topk = 8

    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, dtype=torch.int32, device="cuda")[:topk]
            for _ in range(batch_size * seq_len)
        ]
    )

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids_cuda = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids_cuda.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids_cuda = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad_cuda = torch.empty(
        (1), dtype=torch.int32, device=topk_ids.device
    )
    token_cnts_buffer = torch.zeros(
        (num_experts + 1) * num_experts, dtype=torch.int32, device=topk_ids.device
    )
    cumsum_buffer = torch.zeros(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )

    sorted_ids_triton = torch.empty_like(sorted_ids_cuda)
    sorted_ids_triton.fill_(topk_ids.numel())
    expert_ids_triton = torch.zeros_like(expert_ids_cuda)
    num_tokens_post_pad_triton = torch.empty_like(num_tokens_post_pad_cuda)

    # compare the performance of cuda and triton implementation
    moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_cuda,
        expert_ids_cuda,
        num_tokens_post_pad_cuda,
        token_cnts_buffer,
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

    if torch.allclose(expert_ids_cuda, expert_ids_triton) and torch.allclose(
        num_tokens_post_pad_cuda, num_tokens_post_pad_triton
    ):
        print("✅ CUDA and Triton implementations match")
    else:
        print("❌ CUDA and Triton implementations do not match")
        print("CUDA expert_ids:", expert_ids_cuda)
        print("Triton expert_ids:", expert_ids_triton)
        print("CUDA num_tokens_post_pad:", num_tokens_post_pad_cuda)
        print("Triton num_tokens_post_pad:", num_tokens_post_pad_triton)


batch_size_range = [2**i for i in range(0, 8)]
seq_length_range = [2**i for i in range(0, 16)]
configs = list(itertools.product(batch_size_range, seq_length_range))


def get_topk_ids(num_tokens: int, num_experts: int, topk: int) -> torch.Tensor:
    topk_ids = torch.zeros((num_tokens, topk), dtype=torch.int32, device="cuda")
    for i in range(num_tokens):
        topk_ids[i, :] = torch.randperm(num_experts, dtype=torch.int32, device="cuda")[
            :topk
        ]
    return topk_ids


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["cuda", "triton"],
        line_names=["CUDA", "Triton"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="us",
        plot_name="moe-align-block-size-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, provider):
    num_experts = 256
    block_size = 128
    topk = 8

    if USE_RANDOM_PERM:
        topk_ids = get_topk_ids(batch_size * seq_len, num_experts, topk)
    else:
        topk_ids = torch.randint(
            0,
            num_experts,
            (batch_size * seq_len, topk),
            dtype=torch.int32,
            device="cuda",
        )

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    token_cnts_buffer = torch.zeros(
        (num_experts + 1) * num_experts, dtype=torch.int32, device=topk_ids.device
    )
    cumsum_buffer = torch.zeros(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )

    quantiles = [0.5, 0.2, 0.8]
    if provider == "cuda":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids.clone(),
                expert_ids.clone(),
                num_tokens_post_pad.clone(),
                token_cnts_buffer,
                cumsum_buffer,
            ),
            quantiles=quantiles,
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
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
    args = parser.parse_args()

    calculate_diff(batch_size=4, seq_len=1024)

    benchmark.run(print_data=True)
