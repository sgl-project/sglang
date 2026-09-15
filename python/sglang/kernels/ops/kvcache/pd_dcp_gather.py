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
