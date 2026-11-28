from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

SUPPORTED_GROUP_TOPK = (128, 160, 192, 224, 256, 512)


@cache_once
def _jit_kpool_topk_transform_module(group_topk: int) -> Module:
    assert group_topk in SUPPORTED_GROUP_TOPK, (
        "fast_kpool_topk_transform supports pool-level topk "
        f"{SUPPORTED_GROUP_TOPK}, got {group_topk}"
    )
    return load_jit(
        f"kpool_topk_transform_{group_topk}",
        cuda_files=["dsa/kpool_topk_transform.cuh"],
        cuda_wrappers=[("kpool_topk_transform", "KpoolTopKTransformKernel::transform")],
        extra_cuda_cflags=[f"-DSGL_GROUP_TOPK={group_topk}"],
    )


def fast_kpool_topk_transform_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    kpool: int,
    topk: int,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert topk % kpool == 0
    group_topk = topk // kpool
    assert group_topk in SUPPORTED_GROUP_TOPK, (
        "fast_kpool_topk_transform supports pool-level topk "
        f"{SUPPORTED_GROUP_TOPK}, got {group_topk}"
    )
    assert score.dim() == 2
    assert page_table is None or topk_indices_offset is None
    assert page_table_row_index is None or page_table is not None
    if seq_lens is not None:
        assert seq_lens.dim() == 1
        assert seq_lens.shape[0] == score.shape[0]
    if page_table_row_index is not None:
        assert page_table_row_index.dim() == 1
        assert page_table_row_index.shape[0] == score.shape[0]

    out_cols = topk + (kpool - 1 if seq_lens is not None else 0)
    dst_token_indices = score.new_empty((score.shape[0], out_cols), dtype=torch.int32)

    module = _jit_kpool_topk_transform_module(group_topk)
    module.kpool_topk_transform(
        score,
        lengths,
        dst_token_indices,
        kpool,
        page_table,
        topk_indices_offset,
        row_starts,
        seq_lens,
        page_table_row_index,
    )
    return dst_token_indices


@triton.jit(do_not_specialize=["page_stride", "page_width"])
def _map_groups(
    groups,
    rejected,
    pool_lens,
    seq_lens,
    page_table,
    page_rows,
    offsets,
    out,
    page_stride,
    page_width,
    PAGED: tl.constexpr,
    GROUP_TOPK: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    rank = tl.arange(0, BLOCK_GROUPS)
    lane = tl.arange(0, 4)
    TOPK: tl.constexpr = GROUP_TOPK * 4
    WIDTH: tl.constexpr = TOPK + 3
    live = tl.load(rejected + row) == 0
    pool_len = tl.load(pool_lens + row)
    seq_len = tl.load(seq_lens + row)
    history_len = tl.minimum(pool_len, GROUP_TOPK)
    history = live & (rank < history_len)
    group = tl.load(groups + row * GROUP_TOPK + rank, mask=history, other=-1)
    valid_history = history & (group >= 0) & (group < pool_len)
    tail_token = pool_len * 4 + lane
    valid_tail = live & (lane < seq_len % 4)
    # Short rows place the tail in the first unused group of the main output.
    # Full rows use the extra columns at the end.
    short_tail = (
        (rank[:, None] == history_len)
        & (rank[:, None] < GROUP_TOPK)
        & valid_tail[None, :]
    )
    end_tail = (history_len == GROUP_TOPK) & valid_tail
    raw = tl.where(short_tail, tail_token[None, :], group[:, None] * 4 + lane[None, :])
    valid = valid_history[:, None] | short_tail
    if PAGED:
        page_row = tl.load(page_rows + row).to(tl.int64)
        page = page_table + page_row * page_stride
        slot = tl.load(
            page + raw,
            mask=valid & (raw < page_width),
            other=-1,
        )
        tail_slot = tl.load(
            page + tail_token, mask=end_tail & (tail_token < page_width), other=-1
        )
    else:
        offset = tl.load(offsets + row)
        slot = tl.where(valid, raw + offset, -1)
        tail_slot = tl.where(end_tail, tail_token + offset, -1)
    row_out = out + row * WIDTH
    tl.store(
        row_out + rank[:, None] * 4 + lane[None, :],
        slot,
        mask=rank[:, None] < GROUP_TOPK,
    )
    tl.store(row_out + TOPK + lane, tail_slot, mask=lane < 3)


def map_kpool4_groups(
    groups: torch.Tensor,
    rejected: torch.Tensor,
    pool_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor | None,
    page_rows: torch.Tensor | None,
    offsets: torch.Tensor | None,
) -> torch.Tensor:
    """Expand preselected KPool4 groups and map complete history and tail slots."""
    group_topk = groups.shape[1]
    width = group_topk * 4 + 3
    out = groups.new_empty((groups.shape[0], width))
    _map_groups[(groups.shape[0],)](
        groups,
        rejected,
        pool_lens.contiguous(),
        seq_lens.contiguous(),
        page_table,
        page_rows,
        None if offsets is None else offsets.reshape(-1).contiguous(),
        out,
        0 if page_table is None else page_table.stride(0),
        0 if page_table is None else page_table.shape[1],
        page_table is not None,
        group_topk,
        triton.next_power_of_2(max(group_topk, 1)),
        num_warps=8,
    )
    return out
