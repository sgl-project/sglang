import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.moe.moe_align import moe_align_block_size
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.moe_align import ceil_div, moe_align_block_size_triton

register_cuda_ci(est_time=28, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=50, stage="nightly", runner_config="1-gpu-large")


MOE_ALIGN_CASES = get_ci_test_range(
    list(
        itertools.product(
            [32, 64, 128, 256],  # block_size
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],  # num_tokens
            [1, 2, 4, 8, 16, 32, 64],  # topk
            [64, 128, 160, 256, 257, 260, 264],  # num_experts
            [True, False],  # pad_sorted_token_ids
        )
    ),
    [
        (32, 1, 1, 64, True),
        (128, 48, 1, 128, True),
        (256, 103, 4, 256, False),
        (128, 128, 1, 256, True),
        (256, 4096, 8, 256, False),
    ],
)


@pytest.mark.parametrize(
    "block_size,num_tokens,topk,num_experts,pad_sorted_token_ids",
    MOE_ALIGN_CASES,
)
def test_moe_align_block_size_compare_implementations(
    block_size, num_tokens, topk, num_experts, pad_sorted_token_ids
):
    topk_ids = torch.argsort(torch.rand(num_tokens, num_experts, device="cuda"), dim=1)[
        :, :topk
    ]

    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    # The kernel pads sorted_ids with int4 stores.
    max_num_tokens_padded = (max_num_tokens_padded + 3) // 4 * 4

    sorted_ids_cuda = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    if not pad_sorted_token_ids:
        sorted_ids_cuda.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids_cuda = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad_cuda = torch.empty(
        (1), dtype=torch.int32, device=topk_ids.device
    )
    cumsum_buffer = torch.empty(
        num_experts + 2, dtype=torch.int32, device=topk_ids.device
    )

    sorted_ids_triton = torch.empty_like(sorted_ids_cuda)
    sorted_ids_triton.fill_(topk_ids.numel())
    expert_ids_triton = torch.zeros_like(expert_ids_cuda)
    num_tokens_post_pad_triton = torch.empty_like(num_tokens_post_pad_cuda)

    moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids_cuda,
        expert_ids_cuda,
        num_tokens_post_pad_cuda,
        cumsum_buffer,
        pad_sorted_token_ids,
    )

    moe_align_block_size_triton(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids_triton,
        expert_ids_triton,
        num_tokens_post_pad_triton,
    )

    assert torch.allclose(expert_ids_cuda, expert_ids_triton, atol=0, rtol=0), (
        f"Expert IDs mismatch for block_size={block_size}, "
        f"num_tokens={num_tokens}, topk={topk}\n"
        f"CUDA expert_ids: {expert_ids_cuda}\n"
        f"Triton expert_ids: {expert_ids_triton}"
    )

    assert torch.allclose(
        num_tokens_post_pad_cuda, num_tokens_post_pad_triton, atol=0, rtol=0
    ), (
        f"Num tokens post pad mismatch for block_size={block_size}, "
        f"num_tokens={num_tokens}, topk={topk}\n"
        f"CUDA num_tokens_post_pad: {num_tokens_post_pad_cuda}\n"
        f"Triton num_tokens_post_pad: {num_tokens_post_pad_triton}"
    )

    # Select an expert to check
    expert_idx = expert_ids_cuda.max().item()

    # Get the first and last block id where expert_ids_cuda == expert_idx
    matching_indices = torch.where(expert_ids_cuda == expert_idx)[0]
    block_sorted_start = matching_indices[0].item() * block_size
    block_sorted_end = min(
        (matching_indices[-1].item() + 1) * block_size,
        num_tokens_post_pad_cuda.item(),
    )

    selected_sorted_ids_cuda = sorted_ids_cuda[
        block_sorted_start:block_sorted_end
    ].sort()[0]
    selected_sorted_ids_triton = sorted_ids_triton[
        block_sorted_start:block_sorted_end
    ].sort()[0]

    assert torch.allclose(
        selected_sorted_ids_cuda,
        selected_sorted_ids_triton,
        atol=0,
        rtol=0,
    ), (
        f"Sorted IDs mismatch for block_size={block_size}, "
        f"num_tokens={num_tokens}, topk={topk}\n"
        f"CUDA sorted_ids: {selected_sorted_ids_cuda}\n"
        f"Triton sorted_ids: {selected_sorted_ids_triton}"
    )


@pytest.mark.parametrize(
    "block_size,num_tokens,topk,num_experts",
    list(
        itertools.product(
            [64, 128],  # block_size
            [1, 8, 32, 256],  # num_tokens
            [8],  # topk
            [
                1025,
                2048,
                4095,
            ],  # num_experts (>1024 to exercise v2 kernel, max 4095 real experts)
        )
    ),
)
def test_moe_align_block_size_v2_large_num_experts(
    block_size, num_tokens, topk, num_experts
):
    """Test moe_align_block_size v2 kernel for >1024 experts against Triton reference."""
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )

    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    # The kernel pads sorted_ids with int4 stores.
    max_num_tokens_padded = (max_num_tokens_padded + 3) // 4 * 4

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
    cumsum_buffer = torch.empty(
        num_experts + 2, dtype=torch.int32, device=topk_ids.device
    )

    sorted_ids_triton = torch.empty_like(sorted_ids_cuda)
    sorted_ids_triton.fill_(topk_ids.numel())
    expert_ids_triton = torch.zeros_like(expert_ids_cuda)
    num_tokens_post_pad_triton = torch.empty_like(num_tokens_post_pad_cuda)

    moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids_cuda,
        expert_ids_cuda,
        num_tokens_post_pad_cuda,
        cumsum_buffer,
        True,
    )

    moe_align_block_size_triton(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids_triton,
        expert_ids_triton,
        num_tokens_post_pad_triton,
    )

    assert torch.equal(num_tokens_post_pad_cuda, num_tokens_post_pad_triton), (
        f"Num tokens post pad mismatch: CUDA={num_tokens_post_pad_cuda.item()}, "
        f"Triton={num_tokens_post_pad_triton.item()}"
    )

    ntp = num_tokens_post_pad_cuda.item()
    num_blocks = ntp // block_size

    assert torch.equal(expert_ids_cuda[:num_blocks], expert_ids_triton[:num_blocks]), (
        f"Expert IDs mismatch for block_size={block_size}, "
        f"num_tokens={num_tokens}, topk={topk}, num_experts={num_experts}"
    )

    # Compare sorted_token_ids per expert block (order within block may differ)
    for b in range(num_blocks):
        s, e = b * block_size, (b + 1) * block_size
        block_cuda = sorted_ids_cuda[s:e].sort().values
        block_triton = sorted_ids_triton[s:e].sort().values
        assert torch.equal(block_cuda, block_triton), (
            f"Block {b} sorted_ids mismatch for num_experts={num_experts}, "
            f"num_tokens={num_tokens}"
        )


@pytest.mark.parametrize("remainder", [1, 2, 3])
@pytest.mark.parametrize(
    "num_experts,num_tokens",
    [
        (32, 8),  # small-batch kernel: <= 64 buckets and numel < 1024
        (128, 640),  # generic kernel: <= 1024 buckets
        (1025, 640),  # v2 kernel: > 1024 buckets
    ],
)
def test_moe_align_block_size_pad_fill_stays_in_bounds(
    num_experts, num_tokens, remainder
):
    """The pad fill writes sorted_token_ids with int4 stores; a buffer whose
    length is not a multiple of 4 must be filled to its end and no further."""
    block_size, topk = 64, 8
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    needed = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    length = needed + (remainder - needed) % 4
    assert length % 4 == remainder

    # sorted_ids is a prefix view of a longer buffer; a fill that runs past the
    # view lands in the guard words behind it.
    backing = torch.full((length + 4,), -1, dtype=torch.int32, device="cuda")
    sorted_ids = backing[:length]
    expert_ids = torch.zeros(
        (ceil_div(length, block_size),), dtype=torch.int32, device="cuda"
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device="cuda")
    cumsum_buffer = torch.empty(num_experts + 2, dtype=torch.int32, device="cuda")

    moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    torch.cuda.synchronize()

    guard = backing[length:]
    assert torch.equal(guard, torch.full_like(guard, -1)), (
        f"pad fill wrote past sorted_token_ids for length={length}: {guard.tolist()}"
    )
    tail = sorted_ids[num_tokens_post_pad.item() :]
    assert torch.equal(tail, torch.full_like(tail, topk_ids.numel())), (
        f"pad fill left part of the tail unfilled for length={length}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
