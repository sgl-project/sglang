from typing import Sequence

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_mla_rows_into_pack_kernel(
    src_metadata,
    row_indices,
    pack,
    num_rows,
    BLOCK_SIZE: tl.constexpr,
):
    layer_id = tl.program_id(0)
    block_id = tl.program_id(1)
    metadata_offset = layer_id * 3
    src = tl.load(src_metadata + metadata_offset).to(pack.dtype)
    row_nbytes = tl.load(src_metadata + metadata_offset + 1)
    pack_offset = tl.load(src_metadata + metadata_offset + 2)

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    layer_nbytes = num_rows * row_nbytes
    mask = offsets < layer_nbytes
    row = offsets // row_nbytes
    byte = offsets % row_nbytes
    src_row = tl.load(row_indices + row, mask=mask, other=0)
    values = tl.load(src + src_row * row_nbytes + byte, mask=mask)
    tl.store(pack + pack_offset + offsets, values, mask=mask)


def copy_mla_rows_into_pack(
    kv_data_ptrs: Sequence[int],
    row_indices: torch.Tensor,
    pack: torch.Tensor,
    token_item_lens: Sequence[int],
) -> None:
    if len(kv_data_ptrs) != len(token_item_lens):
        raise ValueError(
            "kv_data_ptrs and token_item_lens length mismatch: "
            f"{len(kv_data_ptrs)} vs {len(token_item_lens)}"
        )
    if not kv_data_ptrs:
        return

    n = int(row_indices.numel())
    metadata = []
    offset = 0
    for ptr, item_len in zip(kv_data_ptrs, token_item_lens):
        item_len = int(item_len)
        if item_len <= 0:
            raise ValueError(f"MLA token item length must be positive, got {item_len}")
        metadata.extend((int(ptr), item_len, offset))
        offset += n * item_len

    src_metadata = torch.tensor(metadata, dtype=torch.int64, device=pack.device)
    max_item_len = max(int(item_len) for item_len in token_item_lens)
    grid = (len(kv_data_ptrs), triton.cdiv(n * max_item_len, 1024))
    _copy_mla_rows_into_pack_kernel[grid](
        src_metadata,
        row_indices,
        pack,
        n,
        BLOCK_SIZE=1024,
    )


@triton.jit(do_not_specialize=["num_tokens", "dcp_size", "dcp_rank"])
def _copy_dsa_pages_into_pack_kernel(
    metadata,
    pack,
    num_tokens,
    dcp_size,
    dcp_rank,
    BLOCK_SIZE: tl.constexpr,
):
    # Runtime DCP values and sizes share one specialization across peers/tails.
    source = tl.load(metadata).to(pack.dtype)
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    page = offset // 8448
    byte = offset % 8448
    is_key = byte < 8192
    slot = tl.where(is_key, byte // 128, (byte - 8192) // 4)
    field_byte = tl.where(is_key, byte % 128, (byte - 8192) % 4)
    local_token = page * 64 + slot
    valid = local_token < num_tokens
    source_token = local_token * dcp_size + dcp_rank
    source_page = tl.load(metadata + 1 + source_token // 64, mask=valid, other=0)
    source_byte = (
        source_page * 8448
        + tl.where(is_key, (source_token % 64) * 128, 8192 + (source_token % 64) * 4)
        + field_byte
    )
    value = tl.load(source + source_byte, mask=valid, other=0)
    packed_bytes = tl.cdiv(num_tokens, 64) * 8448
    tl.store(pack + offset, value, mask=offset < packed_bytes)


def copy_dsa_pages_into_pack(
    metadata: torch.Tensor,
    pack: torch.Tensor,
    num_tokens: int,
    dcp_size: int,
    dcp_rank: int,
) -> None:
    """Gather [source pointer, source page IDs] into native DSA target pages.

    metadata is a small int64 tensor. The uint8 output aliases a registered
    pack buffer; no context-sized intermediate or output tensor is allocated.
    """
    if (
        metadata.dtype != torch.int64
        or metadata.ndim != 1
        or not metadata.is_contiguous()
    ):
        raise ValueError("DSA pack metadata must be contiguous int64")
    if pack.dtype != torch.uint8 or pack.ndim != 1 or not pack.is_contiguous():
        raise ValueError("DSA pack output must be contiguous uint8")
    if dcp_size <= 1 or not 0 <= dcp_rank < dcp_size or num_tokens < 0:
        raise ValueError("Invalid DSA DCP pack geometry")
    if num_tokens == 0:
        return
    required = triton.cdiv(num_tokens, 64) * 8448
    if required > pack.numel():
        raise ValueError("DSA pack output is too small")
    max_source_token = (num_tokens - 1) * dcp_size + dcp_rank
    if metadata.numel() < 2 + max_source_token // 64:
        raise ValueError("DSA pack source page list is too short")
    _copy_dsa_pages_into_pack_kernel[(triton.cdiv(required, 1024),)](
        metadata,
        pack,
        num_tokens,
        dcp_size,
        dcp_rank,
        BLOCK_SIZE=1024,
    )
