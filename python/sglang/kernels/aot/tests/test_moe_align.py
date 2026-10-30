import itertools
import sys

import pytest
import torch
from sgl_kernel import moe_align_block_size, moe_sum

from sglang.test.kernels.moe_align import moe_align_block_size_triton


def is_hip() -> bool:
    return torch.version.hip is not None


_is_hip = is_hip()


@pytest.mark.parametrize(
    "block_size,num_tokens,topk,num_experts,pad_sorted_token_ids",
    list(
        itertools.product(
            [32, 256],  # block_size
            [1, 16, 4096],  # num_tokens
            [1, 8, 64],  # topk
            [64, 160, 257, 264],  # num_experts
            [True, False],  # pad_sorted_token_ids
        )
    ),
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
        (matching_indices[-1].item() + 1) * block_size, num_tokens_post_pad_cuda.item()
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


@pytest.mark.parametrize("m", [1, 33, 64, 222])
@pytest.mark.parametrize("topk", [2, 6])
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.skipif(_is_hip, reason="Skip for AMD GPU")
def test_moe_sum(m: int, topk: int, k: int, dtype: torch.dtype):
    input = torch.randn((m, topk, k), device="cuda", dtype=dtype)
    actual = torch.empty((m, k), device="cuda", dtype=dtype)

    expected = input.sum(dim=1)
    moe_sum(input, actual)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
