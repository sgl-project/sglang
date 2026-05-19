from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

TOPK = 2048


@cache_once
def _jit_fast_topk_module() -> Module:
    return load_jit(
        "fast_topk",
        cuda_files=["elementwise/fast_topk.cuh"],
        cuda_wrappers=[
            ("fast_topk", "fast_topk"),
            ("fast_topk_transform", "fast_topk_transform"),
            ("fast_topk_transform_ragged", "fast_topk_transform_ragged"),
        ],
    )


def _dummy_row_starts(device: torch.device) -> torch.Tensor:
    return torch.zeros(1, dtype=torch.int32, device=device)


@register_custom_op(
    op_name="jit_fast_topk",
    mutates_args=["indices"],
)
def _fast_topk_op(
    score: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    row_starts: torch.Tensor,
    has_row_starts: bool,
) -> None:
    module = _jit_fast_topk_module()
    module.fast_topk(score, indices, lengths, row_starts, has_row_starts)


def fast_topk_v2(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert topk == TOPK
    assert score.dim() == 2
    topk_indices = score.new_empty((score.size(0), topk), dtype=torch.int32)
    has_row_starts = row_starts is not None
    _fast_topk_op(
        score,
        topk_indices,
        lengths,
        row_starts if has_row_starts else _dummy_row_starts(score.device),
        has_row_starts,
    )
    return topk_indices


@register_custom_op(
    op_name="jit_fast_topk_transform",
    mutates_args=["dst_page_table"],
)
def _fast_topk_transform_op(
    score: torch.Tensor,
    lengths: torch.Tensor,
    dst_page_table: torch.Tensor,
    src_page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    row_starts: torch.Tensor,
    has_row_starts: bool,
    is_decode: bool,
) -> None:
    module = _jit_fast_topk_module()
    module.fast_topk_transform(
        score, lengths, dst_page_table, src_page_table,
        cu_seqlens_q, row_starts, has_row_starts, is_decode,
    )


def fast_topk_transform_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table_size_1: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert topk == TOPK
    assert score.dim() == 2
    src_page_table = page_table_size_1
    dst_page_table = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    prefill_bs = cu_seqlens_q.size(0) - 1
    B = score.size(0)
    has_row_starts = row_starts is not None
    is_decode = not has_row_starts and prefill_bs == B
    _fast_topk_transform_op(
        score, lengths, dst_page_table, src_page_table,
        cu_seqlens_q,
        row_starts if has_row_starts else _dummy_row_starts(score.device),
        has_row_starts,
        is_decode,
    )
    return dst_page_table


@register_custom_op(
    op_name="jit_fast_topk_transform_ragged",
    mutates_args=["topk_indices_ragged"],
)
def _fast_topk_transform_ragged_op(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk_indices_ragged: torch.Tensor,
    topk_indices_offset: torch.Tensor,
    row_starts: torch.Tensor,
    has_row_starts: bool,
) -> None:
    module = _jit_fast_topk_module()
    module.fast_topk_transform_ragged(
        score, lengths, topk_indices_ragged, topk_indices_offset,
        row_starts, has_row_starts,
    )


def fast_topk_transform_ragged_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk_indices_offset: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert topk == TOPK
    assert score.dim() == 2
    topk_indices_ragged = score.new_empty(
        (score.shape[0], topk), dtype=torch.int32
    )
    has_row_starts = row_starts is not None
    _fast_topk_transform_ragged_op(
        score, lengths, topk_indices_ragged, topk_indices_offset,
        row_starts if has_row_starts else _dummy_row_starts(score.device),
        has_row_starts,
    )
    return topk_indices_ragged
