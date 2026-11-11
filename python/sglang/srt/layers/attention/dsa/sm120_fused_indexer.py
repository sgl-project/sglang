"""FlashInfer's fused SM120 DSA indexer: FP8 MQA scoring and exact top-k in one
pass, in place of DeepGEMM logits followed by a top-k over the logits.

The kernel returns request-local key positions ranked like ``fp8_mqa_logits``
followed by an exact top-k (same scores bit for bit; ties keep the larger
position). ``to_kv_slots`` maps them to the physical page_size=1 KV slots that
the fused top-k transform hands the attention backend.
"""

import logging
from functools import lru_cache
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_HEADS = (32, 64)
_HEAD_DIM = 128
_MAX_TOPK = 4096


def model_shape_supported(
    *,
    n_heads: int,
    head_dim: int,
    index_topk: int,
    num_init_tokens: int,
    num_local_tokens: int,
) -> bool:
    """Static per-model half of the gate. Forced init/local tokens are masked on
    the logits, which the fused kernel never writes."""
    return (
        n_heads in _HEADS
        and head_dim == _HEAD_DIM
        and 0 < index_topk <= _MAX_TOPK
        and num_init_tokens == 0
        and num_local_tokens == 0
    )


@lru_cache(maxsize=1)
def runtime_ok() -> bool:
    """Whether this process serves the fused indexer: SM12x and a FlashInfer
    that ships it. Declines are logged, as a warning when asked for by name."""
    requested = get_exec().kernel.enable_dsa_fused_indexer
    if requested is False or not is_cuda():
        return False

    def _refuse(reason: str) -> bool:
        log = logger.warning if requested is True else logger.info
        log("SM120 fused DSA indexer disabled: %s", reason)
        return False

    if torch.cuda.get_device_capability()[0] != 12:
        return False
    try:
        from flashinfer.dsa_indexer import dsa_indexer_det_topk  # noqa: F401
    except ImportError:
        return _refuse("FlashInfer has no dsa_indexer")
    logger.info("SM120 fused DSA indexer enabled")
    return True


def topk_paged(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk: int,
    num_rows: int,
) -> torch.Tensor:
    """One query row per page-table row (decode). ``q_fp8`` [T, H, 128] e4m3,
    ``weights`` [T, H] fp32, ``kv_cache`` [pages, page_size, 1, 132] uint8,
    ``context_lens`` int32 [num_rows], ``block_table`` int32 [num_rows, P].
    Rows past ``num_rows`` (hidden-state padding) are all ``-1``."""
    from flashinfer.dsa_indexer import dsa_indexer_det_topk

    out = torch.full((q_fp8.shape[0], topk), -1, dtype=torch.int32, device=q_fp8.device)
    if num_rows == 0:
        return out
    dsa_indexer_det_topk(
        q_fp8[:num_rows].unsqueeze(1),
        kv_cache,
        weights[:num_rows].contiguous(),
        None,
        None,
        top_k=topk,
        indices=out[:num_rows],
        return_scores=False,
        backend="sm120",
        block_table=block_table[:num_rows],
        context_lens=context_lens[:num_rows].reshape(num_rows, 1).to(torch.int32),
    )
    return out


def topk_ragged(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    topk: int,
    out: torch.Tensor,
) -> None:
    """Extend rows over the gathered keys. Each request's rows are the tail of its
    keys (``cu_seqlens_k`` counts prefix plus extend), so row ``u`` of a
    request with ``Lq`` rows and ``Lk`` keys sees its first ``Lk - Lq + u + 1``."""
    from flashinfer.dsa_indexer import dsa_indexer_det_topk

    rows = q_fp8.shape[0]
    dsa_indexer_det_topk(
        q_fp8,
        k_fp8,
        weights.contiguous(),
        cu_seqlens_q.to(torch.int32),
        cu_seqlens_k.to(torch.int32),
        top_k=topk,
        max_seqlen_k=max_seqlen_k,
        indices=out[:rows],
        return_scores=False,
        backend="sm120",
        k_scale=k_scale,
    )


@triton.jit
def _to_kv_slots_kernel(
    positions,
    page_table,
    table_rows,
    out,
    stride_pos,
    stride_table,
    stride_out,
    topk,
    PAGE_SIZE: tl.constexpr,
    HAS_TABLE_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    in_row = cols < topk
    pos = tl.load(positions + row * stride_pos + cols, mask=in_row, other=-1)
    table_row = tl.load(table_rows + row).to(tl.int64) if HAS_TABLE_ROWS else row
    valid = in_row & (pos >= 0)
    page = tl.load(
        page_table + table_row * stride_table + pos // PAGE_SIZE,
        mask=valid,
        other=0,
    )
    slot = tl.where(valid, page * PAGE_SIZE + pos % PAGE_SIZE, -1)
    tl.store(out + row * stride_out + cols, slot, mask=in_row)


def to_kv_slots(
    positions: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    table_rows: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Request-local positions [T, k] (``-1`` padded) to physical page_size=1 KV
    slots through ``page_table`` [R, P] of ``page_size``-token pages. Row ``t``
    reads table row ``table_rows[t]``, or ``t`` when ``table_rows`` is None.
    ``out`` may be ``positions``: each element is read before it is written."""
    if out is None:
        out = torch.empty_like(positions)
    rows, topk = positions.shape
    if rows == 0:
        return out
    block = 512
    _to_kv_slots_kernel[(rows, triton.cdiv(topk, block))](
        positions,
        page_table,
        table_rows if table_rows is not None else page_table,
        out,
        positions.stride(0),
        page_table.stride(0),
        out.stride(0),
        topk,
        PAGE_SIZE=page_size,
        HAS_TABLE_ROWS=table_rows is not None,
        BLOCK=block,
    )
    return out
