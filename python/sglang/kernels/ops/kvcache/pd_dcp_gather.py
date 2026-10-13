from typing import Optional, Sequence

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
    metadata_offset = layer_id * 4
    src = tl.load(src_metadata + metadata_offset).to(pack.dtype)
    row_nbytes = tl.load(src_metadata + metadata_offset + 1)
    pack_offset = tl.load(src_metadata + metadata_offset + 2)
    src_row_stride = tl.load(src_metadata + metadata_offset + 3)

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    layer_nbytes = num_rows * row_nbytes
    mask = offsets < layer_nbytes
    row = offsets // row_nbytes
    byte = offsets % row_nbytes
    src_row = tl.load(row_indices + row, mask=mask, other=0)
    values = tl.load(src + src_row * src_row_stride + byte, mask=mask)
    tl.store(pack + pack_offset + offsets, values, mask=mask)


@triton.jit
def _copy_pack_into_mla_rows_kernel(
    dst_metadata,
    row_indices,
    pack,
    num_rows,
    BLOCK_SIZE: tl.constexpr,
):
    layer_id = tl.program_id(0)
    block_id = tl.program_id(1)
    metadata_offset = layer_id * 4
    dst = tl.load(dst_metadata + metadata_offset).to(pack.dtype)
    row_nbytes = tl.load(dst_metadata + metadata_offset + 1)
    pack_offset = tl.load(dst_metadata + metadata_offset + 2)
    dst_row_stride = tl.load(dst_metadata + metadata_offset + 3)

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    layer_nbytes = num_rows * row_nbytes
    mask = offsets < layer_nbytes
    row = offsets // row_nbytes
    byte = offsets % row_nbytes
    dst_row = tl.load(row_indices + row, mask=mask, other=0)
    values = tl.load(pack + pack_offset + offsets, mask=mask)
    tl.store(dst + dst_row * dst_row_stride + byte, values, mask=mask)


def mla_pack_required_bytes(
    row_count: int,
    token_item_lens: Sequence[int],
    pack_offset_bytes: int = 0,
) -> int:
    row_count = int(row_count)
    offset = int(pack_offset_bytes)
    if row_count < 0 or offset < 0:
        raise ValueError(f"Invalid pack layout: rows={row_count}, offset={offset}")
    widths = [int(item_len) for item_len in token_item_lens]
    if any(item_len <= 0 for item_len in widths):
        raise ValueError(f"MLA token item lengths must be positive, got {widths}")
    return offset + row_count * sum(widths)


def copy_mla_rows_into_pack(
    kv_data_ptrs: Sequence[int],
    row_indices: torch.Tensor,
    pack: torch.Tensor,
    token_item_lens: Sequence[int],
    src_token_item_lens: Optional[Sequence[int]] = None,
    pack_offset_bytes: int = 0,
) -> None:
    if src_token_item_lens is None:
        src_token_item_lens = token_item_lens
    if not (len(kv_data_ptrs) == len(token_item_lens) == len(src_token_item_lens)):
        raise ValueError(
            "KV pointers, copy widths, and source strides length mismatch: "
            f"{len(kv_data_ptrs)}, {len(token_item_lens)}, {len(src_token_item_lens)}"
        )
    if not kv_data_ptrs:
        return

    n = int(row_indices.numel())
    required = mla_pack_required_bytes(n, token_item_lens, pack_offset_bytes)
    capacity = int(pack.numel()) * int(pack.element_size())
    if required > capacity:
        raise ValueError(
            f"MLA pack overflow before gather: need={required}, capacity={capacity}"
        )
    metadata = []
    offset = int(pack_offset_bytes)
    for ptr, item_len, src_item_len in zip(
        kv_data_ptrs, token_item_lens, src_token_item_lens
    ):
        ptr = int(ptr)
        item_len = int(item_len)
        src_item_len = int(src_item_len)
        if ptr <= 0 or src_item_len < item_len:
            raise ValueError(
                f"Invalid MLA source range: ptr={ptr}, width={item_len}, "
                f"stride={src_item_len}"
            )
        metadata.extend((ptr, item_len, offset, src_item_len))
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


def copy_pack_into_mla_rows(
    pack: torch.Tensor,
    kv_data_ptrs: Sequence[int],
    row_indices: torch.Tensor,
    token_item_lens: Sequence[int],
    dst_token_item_lens: Optional[Sequence[int]] = None,
    pack_offset_bytes: int = 0,
) -> None:
    if dst_token_item_lens is None:
        dst_token_item_lens = token_item_lens
    if not (len(kv_data_ptrs) == len(token_item_lens) == len(dst_token_item_lens)):
        raise ValueError(
            "KV pointers, copy widths, and destination strides length mismatch: "
            f"{len(kv_data_ptrs)}, {len(token_item_lens)}, "
            f"{len(dst_token_item_lens)}"
        )
    if not kv_data_ptrs:
        return

    n = int(row_indices.numel())
    required = mla_pack_required_bytes(n, token_item_lens, pack_offset_bytes)
    capacity = int(pack.numel()) * int(pack.element_size())
    if required > capacity:
        raise ValueError(
            f"MLA pack overflow before scatter: need={required}, capacity={capacity}"
        )
    metadata = []
    offset = int(pack_offset_bytes)
    for ptr, item_len, dst_item_len in zip(
        kv_data_ptrs, token_item_lens, dst_token_item_lens
    ):
        ptr = int(ptr)
        item_len = int(item_len)
        dst_item_len = int(dst_item_len)
        if ptr <= 0 or dst_item_len < item_len:
            raise ValueError(
                f"Invalid MLA destination range: ptr={ptr}, width={item_len}, "
                f"stride={dst_item_len}"
            )
        metadata.extend((ptr, item_len, offset, dst_item_len))
        offset += n * item_len

    dst_metadata = torch.tensor(metadata, dtype=torch.int64, device=pack.device)
    max_item_len = max(int(item_len) for item_len in token_item_lens)
    grid = (len(kv_data_ptrs), triton.cdiv(n * max_item_len, 1024))
    _copy_pack_into_mla_rows_kernel[grid](
        dst_metadata,
        row_indices,
        pack,
        n,
        BLOCK_SIZE=1024,
    )
