from typing import Optional, Sequence

import torch
import triton
import triton.language as tl


# G1-PATCH (!33, 2026-09-09)：gather kernel grid 维度对调 + 8B 向量化。
# 问题 1：原版 grid=(层数, ceil(n×1152/1024))，CUDA grid.y 上限 65,535 → 单次 pack > 58,253 行即崩溃。
#   对调后块数→grid.x（上限 2^31-1），层数→grid.y（≤61）。
# 问题 2：原版每 lane 搬 1 字节，吞吐 ~500GB/s；item_len 全为 8 的倍数时走 int64 向量化路径，吞吐 ~5 倍。
# 上游 f1065423 已新增 src_row_stride（metadata 第 4 字段）与 src_token_item_lens 参数，本版保留并合并。


@triton.jit
def _copy_mla_rows_into_pack_kernel(
    src_metadata,
    row_indices,
    pack,
    num_rows,
    USE_INT64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 字节路径：grid 对调（块数→x，层数→y），任意 item_len 均正确
    block_id = tl.program_id(0)
    layer_id = tl.program_id(1)
    metadata_offset = layer_id * 4
    src = tl.load(src_metadata + metadata_offset).to(pack.dtype)
    row_nbytes = tl.load(src_metadata + metadata_offset + 1)
    pack_offset = tl.load(src_metadata + metadata_offset + 2)
    src_row_stride = tl.load(src_metadata + metadata_offset + 3)

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if USE_INT64:
        offsets = offsets.to(tl.int64)
    layer_nbytes = num_rows * row_nbytes
    mask = offsets < layer_nbytes
    row = offsets // row_nbytes
    byte = offsets % row_nbytes
    src_row = tl.load(row_indices + row, mask=mask, other=0)
    values = tl.load(src + src_row * src_row_stride + byte, mask=mask)
    tl.store(pack + pack_offset + offsets, values, mask=mask)


@triton.jit
def _copy_mla_rows_into_pack_kernel_vec8(
    src_metadata,
    row_indices,
    pack,
    num_rows,
    USE_INT64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 8B 向量化路径（要求 item_len 与 src_item_len 均为 8 的倍数，host 侧保证）：
    # pack 为 int64 视图，row_qwords = row_nbytes // 8，源行步长 = src_item_len // 8
    block_id = tl.program_id(0)
    layer_id = tl.program_id(1)
    metadata_offset = layer_id * 4
    src = tl.load(src_metadata + metadata_offset).to(tl.pointer_type(tl.uint64))
    row_qwords = tl.load(src_metadata + metadata_offset + 1) // 8
    pack_offset = tl.load(src_metadata + metadata_offset + 2) // 8
    src_row_stride = tl.load(src_metadata + metadata_offset + 3) // 8

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if USE_INT64:
        offsets = offsets.to(tl.int64)
    layer_qwords = num_rows * row_qwords
    mask = offsets < layer_qwords
    row = offsets // row_qwords
    qword = offsets % row_qwords
    src_row = tl.load(row_indices + row, mask=mask, other=0)
    values = tl.load(src + src_row * src_row_stride + qword, mask=mask)
    tl.store(pack + pack_offset + offsets, values, mask=mask)


def copy_mla_rows_into_pack(
    kv_data_ptrs: Sequence[int],
    row_indices: torch.Tensor,
    pack: torch.Tensor,
    token_item_lens: Sequence[int],
    src_token_item_lens: Optional[Sequence[int]] = None,
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
    metadata = []
    offset = 0
    for ptr, item_len, src_item_len in zip(
        kv_data_ptrs, token_item_lens, src_token_item_lens
    ):
        item_len = int(item_len)
        if item_len <= 0:
            raise ValueError(f"MLA token item length must be positive, got {item_len}")
        metadata.extend((int(ptr), item_len, offset, int(src_item_len)))
        offset += n * item_len

    src_metadata = torch.tensor(metadata, dtype=torch.int64, device=pack.device)
    max_item_len = max(int(item_len) for item_len in token_item_lens)

    if all(int(item_len) % 8 == 0 for item_len in token_item_lens) and all(
        int(src_item_len) % 8 == 0 for src_item_len in src_token_item_lens
    ):
        # 8B 向量化：每 lane 搬 8 字节
        qwords = n * (max_item_len // 8)
        blocks_per_layer = triton.cdiv(qwords, 512)
        grid = (blocks_per_layer, len(kv_data_ptrs))
        _copy_mla_rows_into_pack_kernel_vec8[grid](
            src_metadata,
            row_indices,
            pack.view(torch.int64),
            n,
            USE_INT64=(n * max_item_len) >= 8 * 2**31,
            BLOCK_SIZE=512,
        )
        return

    # 字节回退路径：grid 对调（块数→x，层数→y），消除 58,253 行上限
    blocks_per_layer = triton.cdiv(n * max_item_len, 1024)
    grid = (blocks_per_layer, len(kv_data_ptrs))
    _copy_mla_rows_into_pack_kernel[grid](
        src_metadata,
        row_indices,
        pack,
        n,
        USE_INT64=(n * max_item_len) >= 2**31,
        BLOCK_SIZE=1024,
    )
