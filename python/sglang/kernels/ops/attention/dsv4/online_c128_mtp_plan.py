from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _plan_online_c128_mtp_prefill_kernel(
    prefix_lens_ptr,
    req_pool_indices_ptr,
    plan_c_ptr,
    plan_w_ptr,
    batch_size,
    active_batch_size,
    state_slot_offset,
    num_draft_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = batch_id < batch_size
    active = in_bounds & (batch_id < active_batch_size)

    prefix_len = tl.load(prefix_lens_ptr + batch_id, mask=in_bounds, other=0).to(
        tl.int32
    )
    main_slot = tl.load(req_pool_indices_ptr + batch_id, mask=in_bounds, other=0).to(
        tl.int32
    )
    end_pos = prefix_len + num_draft_tokens
    chunk_end = (prefix_len // 128 + 1) * 128
    ragged_base = batch_id * num_draft_tokens
    write_slot = main_slot + state_slot_offset

    has_compress = active & (chunk_end <= end_pos)
    compress_len = chunk_end - prefix_len
    compress_word = ragged_base + compress_len - 1 + (compress_len << 16)

    # Exact-boundary requests need no trailing write plan. Requests that do
    # not cross a boundary write all draft tokens; crossing requests write only
    # the tail after the close-chunk segment.
    has_write = active & (chunk_end != end_pos)
    write_len = tl.where(chunk_end < end_pos, end_pos - chunk_end, num_draft_tokens)
    write_word = ragged_base + num_draft_tokens - 1 + (write_len << 16)

    plan_c_row = batch_id * 4
    tl.store(
        plan_c_ptr + plan_c_row,
        tl.where(has_compress, chunk_end, -1),
        mask=in_bounds,
    )
    tl.store(
        plan_c_ptr + plan_c_row + 1,
        tl.where(has_compress, compress_word, 0),
        mask=in_bounds,
    )
    tl.store(
        plan_c_ptr + plan_c_row + 2,
        tl.where(has_compress, write_slot, -1),
        mask=in_bounds,
    )
    tl.store(
        plan_c_ptr + plan_c_row + 3,
        tl.where(has_compress, main_slot, -1),
        mask=in_bounds,
    )

    plan_w_row = batch_id * 4
    tl.store(
        plan_w_ptr + plan_w_row,
        tl.where(has_write, end_pos, -1),
        mask=in_bounds,
    )
    tl.store(
        plan_w_ptr + plan_w_row + 1,
        tl.where(has_write, write_word, 0),
        mask=in_bounds,
    )
    tl.store(
        plan_w_ptr + plan_w_row + 2,
        tl.where(has_write, write_slot, -1),
        mask=in_bounds,
    )
    tl.store(
        plan_w_ptr + plan_w_row + 3,
        tl.where(has_write, main_slot, -1),
        mask=in_bounds,
    )


def plan_online_c128_mtp_prefill(
    prefix_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    active_batch_size: int,
    num_draft_tokens: int,
    state_slot_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build fixed-shape online-C128 target-verify plans on the current stream."""
    assert prefix_lens.ndim == req_pool_indices.ndim == 1
    assert prefix_lens.shape == req_pool_indices.shape
    batch_size = int(prefix_lens.shape[0])
    device = req_pool_indices.device
    assert prefix_lens.device == device
    assert prefix_lens.is_cuda and req_pool_indices.is_cuda
    assert prefix_lens.is_contiguous() and req_pool_indices.is_contiguous()
    assert prefix_lens.dtype in (torch.int32, torch.int64)
    assert req_pool_indices.dtype in (torch.int32, torch.int64)

    plan_c_i32 = torch.empty((batch_size, 4), dtype=torch.int32, device=device)
    plan_w_i32 = torch.empty_like(plan_c_i32)
    if batch_size:
        block_size = 256
        _plan_online_c128_mtp_prefill_kernel[(triton.cdiv(batch_size, block_size),)](
            prefix_lens,
            req_pool_indices,
            plan_c_i32,
            plan_w_i32,
            batch_size,
            active_batch_size,
            state_slot_offset,
            num_draft_tokens=num_draft_tokens,
            BLOCK_SIZE=block_size,
        )
    return plan_c_i32.view(torch.uint8), plan_w_i32.view(torch.uint8)
