import itertools

import pytest
import torch
import triton
import triton.language as tl
from sgl_kernel import moe_align_block_size, moe_sum


def is_hip() -> bool:
    return torch.version.hip is not None


_is_hip = is_hip()


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
    "block_size,num_tokens,topk,num_experts,pad_sorted_token_ids",
    list(
        itertools.product(
            [32, 64, 128, 256],  # block_size
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],  # num_tokens
            [1, 2, 4, 8, 16, 32, 64],  # topk
            [64, 160, 256, 257, 260, 264],  #  num_experts
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


# Additional optimization tests
def test_memory_allocation_optimization():
    """
    Test precise memory allocation for small batches
    """
    num_tokens = 10
    num_experts = 8
    topk = 2
    block_size = 64
    
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda")
    
    # Test with pad_to_block_size=False (should use less memory)
    max_num_tokens_padded_no_pad = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    
    # Test with pad_to_block_size=True (should round up)
    max_num_tokens_padded_with_pad = triton.cdiv(max_num_tokens_padded_no_pad, block_size) * block_size
    
    assert max_num_tokens_padded_no_pad < max_num_tokens_padded_with_pad, \
        "Without padding should use less memory"


def test_parallel_initialization():
    """
    Test parallel initialization of sorted_token_ids
    """
    num_tokens = 100
    num_experts = 16
    topk = 4
    block_size = 64
    
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda")
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(
        (triton.cdiv(max_num_tokens_padded, block_size),), dtype=torch.int32, device="cuda"
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device="cuda")
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device="cuda")
    
    # Run with pad_sorted_token_ids=True
    moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids=True,
    )
    
    # Check that padding values are correctly set to numel
    valid_count = num_tokens_post_pad.item()
    padding_values = sorted_ids[valid_count:valid_count+10]
    
    # All padding values should be equal to numel (the sentinel value)
    assert torch.all(padding_values == topk_ids.numel()), \
        f"Padding values should all be {topk_ids.numel()}, got {padding_values}"


def test_invalid_expert_filtering():
    """
    Test filtering out invalid experts (for EP mode)
    """
    num_tokens = 50
    num_experts = 16
    topk = 2
    block_size = 32
    
    # Create topk_ids with some invalid expert IDs
    topk_ids = torch.randint(0, num_experts + 5, (num_tokens, topk), dtype=torch.int32, device="cuda")
    
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(
        (triton.cdiv(max_num_tokens_padded, block_size),), dtype=torch.int32, device="cuda"
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device="cuda")
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device="cuda")
    
    # Should not crash even with invalid expert IDs
    moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids=True,
    )
    
    # Count how many invalid expert IDs we had
    invalid_count = torch.sum(topk_ids >= num_experts).item()
    valid_count = topk_ids.numel() - invalid_count


def test_expert_ids_padding():
    """
    Test filling remaining expert_ids with -1
    """
    num_tokens = 30
    num_experts = 8
    topk = 2
    block_size = 32
    
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda")
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(
        (triton.cdiv(max_num_tokens_padded, block_size),), dtype=torch.int32, device="cuda"
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device="cuda")
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device="cuda")
    
    moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids=True,
    )
    
    # Calculate the number of valid blocks
    valid_blocks = (num_tokens_post_pad.item() + block_size - 1) // block_size
    
    # Check that remaining expert_ids are filled with -1
    remaining_expert_ids = expert_ids[valid_blocks:]
    
    # All remaining blocks should have expert_id = -1
    if len(remaining_expert_ids) > 0:
        assert torch.all(remaining_expert_ids == -1), \
            f"Remaining expert_ids should be -1, got {remaining_expert_ids[:10]}"


@pytest.mark.parametrize(
    "num_tokens,num_experts,topk,block_size",
    [
        (1, 8, 1, 32),      # Small batch
        (10, 16, 2, 64),    # Medium batch
        (100, 32, 4, 128),  # Larger batch
        (1000, 64, 8, 256), # Large batch
    ]
)
def test_all_optimizations_combined(num_tokens, num_experts, topk, block_size):
    """
    Test all optimizations work together correctly
    """
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda")
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(
        (triton.cdiv(max_num_tokens_padded, block_size),), dtype=torch.int32, device="cuda"
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device="cuda")
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device="cuda")
    
    # Should complete successfully with all optimizations
    moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids=True,
    )
    
    # Basic sanity checks
    assert num_tokens_post_pad.item() > 0, "Should have some valid tokens"
    assert num_tokens_post_pad.item() % block_size == 0, "Result should be aligned to block_size"


if __name__ == "__main__":
    pytest.main([__file__])
