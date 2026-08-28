"""Numerical regression tests for direct-paged classic MoE LoRA kernels.

The tests deliberately use non-contiguous physical-page assignments.  They
compare the paged kernel against the existing flat expert-buffer path for both
a partial tail page and a mixed r8/r64 launch, so page lookup, tail masking and
per-adapter invalid-rank-block skipping are covered without a model-specific
golden output.
"""

import unittest

import torch

from sglang.jit_kernel.moe_lora_align import moe_lora_align_block_size
from sglang.kernels.ops.moe.fused_moe_lora_kernel import fused_moe_lora
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _align_tokens(
    topk_ids: torch.Tensor,
    seg_indptr: torch.Tensor,
    req_to_lora: torch.Tensor,
    adapter_enabled: torch.Tensor,
    num_experts: int,
    block_size: int,
):
    num_slots = adapter_enabled.numel()
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    max_num_tokens_padded = _ceil_div(max_num_tokens_padded, block_size) * block_size
    max_num_m_blocks = _ceil_div(max_num_tokens_padded, block_size)
    device = topk_ids.device

    sorted_token_ids = torch.empty(
        num_slots * max_num_tokens_padded, dtype=torch.int32, device=device
    )
    expert_ids = torch.empty(
        num_slots * max_num_m_blocks, dtype=torch.int32, device=device
    )
    num_tokens_post_padded = torch.empty(num_slots, dtype=torch.int32, device=device)
    lora_ids = torch.arange(num_slots, dtype=torch.int32, device=device)

    moe_lora_align_block_size(
        topk_ids,
        seg_indptr,
        req_to_lora,
        num_experts,
        block_size,
        num_slots,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        adapter_enabled,
        lora_ids,
        None,
    )
    return (
        sorted_token_ids.view(num_slots, -1),
        expert_ids.view(num_slots, -1),
        num_tokens_post_padded,
        lora_ids,
    )


def _run_fused(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    lora_ids: torch.Tensor,
    adapter_enabled: torch.Tensor,
    max_rank: int,
    page_table: torch.Tensor | None = None,
    lora_ranks: torch.Tensor | None = None,
    page_rank_size: int = 0,
    shrink_block_size_n: int = 16,
    expand_block_size_k: int = 16,
    output_offset: int = 0,
):
    fused_moe_lora(
        output=output,
        qcurr_hidden_states=hidden_states,
        lora_a_stacked=[lora_a],
        lora_b_stacked=[lora_b],
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        max_lora_rank=max_rank,
        top_k_num=topk_weights.shape[1],
        lora_ids=lora_ids,
        adapter_enabled=adapter_enabled,
        shrink_block_size_m=16,
        shrink_block_size_n=shrink_block_size_n,
        shrink_block_size_k=32,
        shrink_group_size_m=1,
        shrink_num_warps=4,
        shrink_num_stages=2,
        shrink_split_k=1,
        expand_block_size_m=16,
        expand_block_size_n=16,
        expand_block_size_k=expand_block_size_k,
        expand_group_size_m=1,
        expand_num_warps=4,
        expand_num_stages=2,
        expand_split_k=1,
        mul_routed_weight=True,
        offset=output_offset,
        page_table=page_table,
        lora_ranks=lora_ranks,
        page_rank_size=page_rank_size,
    )


def _assert_direct_paged_matches_flat(
    dtype: torch.dtype,
    actual_rank_values=(0, 4, 6),
    page_rank_size: int = 4,
    page_table_values=None,
    shrink_block_size_n: int = 16,
    expand_block_size_k: int = 16,
    output_offset: int = 0,
):
    torch.manual_seed(7)
    device = torch.device("cuda")
    num_slots, num_experts = 3, 3
    num_tokens, top_k = 12, 2
    input_dim, output_dim = 32, 24
    if page_table_values is None:
        page_table_values = [[-1, -1], [2, -1], [3, 0]]
    max_pages = len(page_table_values[0])
    padded_rank = page_rank_size * max_pages
    actual_ranks = torch.tensor(actual_rank_values, dtype=torch.int32, device=device)
    adapter_enabled = (actual_ranks > 0).to(torch.int32)

    page_table = torch.tensor(page_table_values, dtype=torch.int32, device=device)
    num_physical_pages = int(page_table.max().item()) + 1

    flat_a = torch.zeros(
        num_slots,
        num_experts,
        padded_rank,
        input_dim,
        dtype=dtype,
        device=device,
    )
    flat_b = torch.zeros(
        num_slots,
        num_experts,
        output_dim,
        padded_rank,
        dtype=dtype,
        device=device,
    )
    for slot, rank in enumerate(actual_ranks.tolist()):
        if rank:
            flat_a[slot, :, :rank].normal_(mean=0.0, std=0.1)
            flat_b[slot, :, :, :rank].normal_(mean=0.0, std=0.1)

    paged_a = torch.zeros(
        num_physical_pages,
        num_experts,
        page_rank_size,
        input_dim,
        dtype=dtype,
        device=device,
    )
    paged_b = torch.zeros(
        num_physical_pages,
        num_experts,
        output_dim,
        page_rank_size,
        dtype=dtype,
        device=device,
    )
    for slot, rank in enumerate(actual_ranks.tolist()):
        for logical_page in range(_ceil_div(rank, page_rank_size)):
            physical_page = int(page_table[slot, logical_page])
            rank_start = logical_page * page_rank_size
            rank_stop = min(rank_start + page_rank_size, rank)
            width = rank_stop - rank_start
            paged_a[physical_page, :, :width].copy_(
                flat_a[slot, :, rank_start:rank_stop]
            )
            paged_b[physical_page, :, :, :width].copy_(
                flat_b[slot, :, :, rank_start:rank_stop]
            )

    topk_ids = torch.tensor(
        [[0, 1], [2, 1], [1, 0], [2, 0]] * 3,
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.rand(num_tokens, top_k, device=device)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    seg_indptr = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device=device)
    req_to_lora = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    alignment = _align_tokens(
        topk_ids,
        seg_indptr,
        req_to_lora,
        adapter_enabled,
        num_experts,
        block_size=16,
    )
    hidden_states = torch.randn(num_tokens, input_dim, dtype=dtype, device=device)

    full_output_dim = output_offset + output_dim + 3
    base_output = torch.randn(
        num_tokens, top_k, full_output_dim, dtype=dtype, device=device
    )
    flat_delta = torch.zeros_like(base_output)
    flat_output = base_output.clone()
    paged_output = base_output.clone()
    _run_fused(
        flat_delta,
        hidden_states,
        flat_a,
        flat_b,
        topk_weights,
        *alignment,
        adapter_enabled,
        padded_rank,
        output_offset=output_offset,
    )
    _run_fused(
        flat_output,
        hidden_states,
        flat_a,
        flat_b,
        topk_weights,
        *alignment,
        adapter_enabled,
        padded_rank,
        output_offset=output_offset,
    )
    _run_fused(
        paged_output,
        hidden_states,
        paged_a,
        paged_b,
        topk_weights,
        *alignment,
        adapter_enabled,
        padded_rank,
        page_table,
        actual_ranks,
        page_rank_size,
        shrink_block_size_n,
        expand_block_size_k,
        output_offset,
    )

    torch.testing.assert_close(flat_output, base_output + flat_delta)
    torch.testing.assert_close(paged_output, flat_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(paged_output[:4], base_output[:4])
    if output_offset:
        torch.testing.assert_close(
            paged_output[..., :output_offset], base_output[..., :output_offset]
        )
    torch.testing.assert_close(
        paged_output[..., output_offset + output_dim :],
        base_output[..., output_offset + output_dim :],
    )


class TestPagedMoEKernelCorrectness(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_direct_paged_matches_flat_moe_lora(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                _assert_direct_paged_matches_flat(dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_direct_paged_mixed_r8_r64_matches_flat(self):
        # Slot 1 is r8 and owns one page. Slot 2 is r64 and owns eight
        # deliberately non-contiguous pages. Both slots execute in one launch.
        page_table = [
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [10, -1, -1, -1, -1, -1, -1, -1],
            [7, 0, 9, 2, 11, 4, 6, 1],
        ]
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                _assert_direct_paged_matches_flat(
                    dtype,
                    actual_rank_values=(0, 8, 64),
                    page_rank_size=8,
                    page_table_values=page_table,
                    shrink_block_size_n=64,
                    expand_block_size_k=64,
                    output_offset=3,
                )


if __name__ == "__main__":
    unittest.main()
