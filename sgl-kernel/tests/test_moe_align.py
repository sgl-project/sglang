import itertools

import pytest
import torch
import triton
import triton.language as tl
from sgl_kernel import moe_align_block_size


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


@pytest.mark.parametrize(
    "block_size,num_tokens,topk,num_experts",
    list(
        itertools.product(
            [32, 64, 128, 256],  # block_size
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],  # num_tokens
            [1, 2, 4, 8, 16, 32, 64],  # topk
            [64, 160, 256, 257, 260, 264],  #  num_experts
        )
    ),
)
def test_moe_align_block_size_compare_implementations(
    block_size, num_tokens, topk, num_experts
):
    # For DeepSeek V3, we have 256 experts

    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, dtype=torch.int32, device="cuda")[:topk]
            for _ in range(num_tokens)
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
    token_cnts_buffer = torch.empty(
        (num_experts + 1) * num_experts,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    cumsum_buffer = torch.empty(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )

    sorted_ids_triton = torch.empty_like(sorted_ids_cuda)
    sorted_ids_triton.fill_(topk_ids.numel())
    expert_ids_triton = torch.zeros_like(expert_ids_cuda)
    num_tokens_post_pad_triton = torch.empty_like(num_tokens_post_pad_cuda)

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

    assert torch.allclose(expert_ids_cuda, expert_ids_triton), (
        f"Expert IDs mismatch for block_size={block_size}, "
        f"num_tokens={num_tokens}, topk={topk}\n"
        f"CUDA expert_ids: {expert_ids_cuda}\n"
        f"Triton expert_ids: {expert_ids_triton}"
    )

    assert torch.allclose(num_tokens_post_pad_cuda, num_tokens_post_pad_triton), (
        f"Num tokens post pad mismatch for block_size={block_size}, "
        f"num_tokens={num_tokens}, topk={topk}\n"
        f"CUDA num_tokens_post_pad: {num_tokens_post_pad_cuda}\n"
        f"Triton num_tokens_post_pad: {num_tokens_post_pad_triton}"
    )


if __name__ == "__main__":
    pytest.main([__file__])
