"""Device-side KV layout builder for compact speculative trees."""

import torch
import triton
import triton.language as tl


@triton.jit
def _build_tree_query_kv_layout_kernel(
    tree_mask_ptr,
    req_to_token_ptr,
    req_pool_indices_ptr,
    seq_lens_ptr,
    kv_slots_ptr,
    kv_lens_ptr,
    tree_batch_size,
    req_to_token_stride: tl.constexpr,
    kv_slots_stride: tl.constexpr,
    NUM_DRAFT_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    query_idx = tl.program_id(1)
    key_idx = tl.arange(0, BLOCK_SIZE)
    valid_key = key_idx < NUM_DRAFT_TOKENS
    is_real_req = req_idx < tree_batch_size

    mask_offset = (req_idx * NUM_DRAFT_TOKENS + query_idx) * NUM_DRAFT_TOKENS + key_idx
    visible = tl.load(
        tree_mask_ptr + mask_offset,
        mask=is_real_req & valid_key,
        other=0,
    ).to(tl.int1)
    visible_i32 = visible.to(tl.int32)
    compact_col = tl.cumsum(visible_i32, axis=0) - 1
    kv_len = tl.sum(visible_i32, axis=0)

    req_pool_idx = tl.load(req_pool_indices_ptr + req_idx)
    seq_len = tl.load(seq_lens_ptr + req_idx)
    tree_slots = tl.load(
        req_to_token_ptr
        + req_pool_idx.to(tl.int64) * req_to_token_stride
        + seq_len.to(tl.int64)
        + key_idx,
        mask=is_real_req & valid_key & visible,
        other=0,
    )

    row = req_idx * NUM_DRAFT_TOKENS + query_idx
    tl.store(
        kv_slots_ptr + row * kv_slots_stride + compact_col,
        tree_slots.to(tl.int32),
        mask=is_real_req & valid_key & visible,
    )

    # CUDA-graph padding rows use one harmless slot. Their outputs are discarded,
    # but attention kernels still require every query to have valid metadata.
    is_padding_root = (~is_real_req) & (key_idx == 0)
    tl.store(
        kv_slots_ptr + row * kv_slots_stride + key_idx,
        0,
        mask=is_padding_root,
    )
    tl.store(kv_lens_ptr + row, tl.where(is_real_req, kv_len, 1))


def build_tree_query_kv_layout(
    tree_mask: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_slots: torch.Tensor,
    kv_lens: torch.Tensor,
    num_draft_tokens: int,
) -> None:
    """Build each tree query's ordered visible-KV slot list.

    ``tree_mask`` is flattened from ``[num_reqs, D, D]``. The output contains
    raw physical token slots and their per-query lengths; it does not encode a
    particular attention backend's page size or metadata type.
    """
    tree_size_per_req = num_draft_tokens * num_draft_tokens
    if tree_mask.numel() % tree_size_per_req != 0:
        raise ValueError("The compact tree mask has an invalid size.")
    tree_batch_size = tree_mask.numel() // tree_size_per_req
    batch_size = req_pool_indices.numel()
    if tree_batch_size > batch_size:
        raise ValueError("The compact tree batch exceeds the attention batch.")
    if kv_slots.shape != (batch_size * num_draft_tokens, num_draft_tokens):
        raise ValueError("The tree-query KV-slot buffer has an invalid shape.")
    if kv_lens.numel() != batch_size * num_draft_tokens:
        raise ValueError("The tree-query KV-length buffer has an invalid shape.")

    block_size = triton.next_power_of_2(num_draft_tokens)
    _build_tree_query_kv_layout_kernel[(batch_size, num_draft_tokens)](
        tree_mask,
        req_to_token,
        req_pool_indices,
        seq_lens,
        kv_slots,
        kv_lens,
        tree_batch_size,
        req_to_token.stride(0),
        kv_slots.stride(0),
        NUM_DRAFT_TOKENS=num_draft_tokens,
        BLOCK_SIZE=block_size,
    )
