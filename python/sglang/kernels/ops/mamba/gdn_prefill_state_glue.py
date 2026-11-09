"""Eager GDN prefill state copies, one launch per family: the prefix-cache track copies
(SGLANG_ENABLE_GDN_PREFILL_FUSED_TRACK) and the chunked kernel's gather / scatter of the
pool rows (SGLANG_ENABLE_GDN_PREFILL_FUSED_STATE_COPY).

The eager extend writes each track slot with torch advanced indexing: a gather,
an index_put and an int64 cast of the index tensors per copy. These kernels read
the same index tensors and move the same bytes in one launch:

* ``gdn_track_conv_snapshot``: conv_pool[dst[r], c, j] = x[tok[r, j], c], the
  window ``conv_states[dst] = x[tok].transpose(1, 2)`` takes before the conv.
* ``gdn_track_state_rows``: pool[h_dst[i]] = h[h_src[i]] (chunk checkpoints of
  unaligned rows), then pool[f_dst[i]] = pool[f_src[i]] (final states of aligned
  rows), the two copies ``_track_mamba_state_extend`` makes after the chunked
  kernel. Destinations are track slots, never a live source slot.
"""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK_CONV = 1024
# 16 KB of fp16 state per program; a 48 x 128 x 128 row is 96 programs.
_BLOCK_ROW = 8192
_logged: set = set()


def _log_once(key: str, msg: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.info(msg, *args)


@triton.jit
def _gdn_track_conv_snapshot_kernel(
    conv_pool,  # [slots, dim, STATE_LEN]
    x,  # [tokens, dim] raw pre-conv input, unit column stride
    tok,  # [rows, STATE_LEN] token rows
    dst,  # [rows] track slots
    stride_pool_slot,
    stride_pool_dim,
    stride_pool_pos,
    stride_x_token,
    stride_tok_row,
    stride_tok_pos,
    dim,
    STATE_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < dim
    slot = tl.load(dst + r).to(tl.int64)
    out = conv_pool + slot * stride_pool_slot + offs.to(tl.int64) * stride_pool_dim
    for j in tl.static_range(STATE_LEN):
        t = tl.load(tok + r * stride_tok_row + j * stride_tok_pos).to(tl.int64)
        val = tl.load(x + t * stride_x_token + offs, mask=mask)
        tl.store(out + j * stride_pool_pos, val, mask=mask)


@triton.jit
def _gdn_track_state_rows_kernel(
    pool,  # [slots, row_elems] (row view)
    h,  # [checkpoints, row_elems] (row view); pool when there are no h rows
    h_src,
    h_dst,
    f_src,
    f_dst,
    n_h,
    row_elems,
    stride_pool_slot,
    stride_h_row,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < row_elems
    if r < n_h:
        src = h + tl.load(h_src + r).to(tl.int64) * stride_h_row
        slot = tl.load(h_dst + r).to(tl.int64)
    else:
        src = pool + tl.load(f_src + (r - n_h)).to(tl.int64) * stride_pool_slot
        slot = tl.load(f_dst + (r - n_h)).to(tl.int64)
    val = tl.load(src + offs, mask=mask)
    tl.store(pool + slot * stride_pool_slot + offs, val, mask=mask)


@triton.jit
def _gdn_copy_rows_kernel(
    src,
    dst,
    src_idx,  # [rows] source row ids (SRC_INDEXED), else row r
    dst_idx,  # [rows] destination row ids (DST_INDEXED), else row r
    row_elems,
    stride_src,
    stride_dst,
    SRC_INDEXED: tl.constexpr,
    DST_INDEXED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < row_elems
    s_row = r.to(tl.int64)
    d_row = r.to(tl.int64)
    if SRC_INDEXED:
        s_row = tl.load(src_idx + r).to(tl.int64)
    if DST_INDEXED:
        d_row = tl.load(dst_idx + r).to(tl.int64)
    val = tl.load(src + s_row * stride_src + offs, mask=mask)
    tl.store(dst + d_row * stride_dst + offs, val, mask=mask)


def _copy_rows(src, dst, rows, src_idx=None, dst_idx=None) -> None:
    row_elems = src[0].numel()
    idx = src_idx if src_idx is not None else dst_idx
    _gdn_copy_rows_kernel[(rows, triton.cdiv(row_elems, _BLOCK_ROW))](
        src,
        dst,
        src_idx if src_idx is not None else idx,
        dst_idx if dst_idx is not None else idx,
        row_elems,
        src.stride(0),
        dst.stride(0),
        SRC_INDEXED=src_idx is not None,
        DST_INDEXED=dst_idx is not None,
        BLOCK=_BLOCK_ROW,
    )


def gdn_gather_state_rows(pool: torch.Tensor, slots: torch.Tensor) -> Optional[torch.Tensor]:
    """``pool[slots].contiguous()`` in one copy launch (None when pool rows are not dense)."""
    if not _row_major_tail(pool):
        _log_once("gather-miss", "SGLANG_ENABLE_GDN_PREFILL_FUSED_STATE_COPY: pool rows not dense")
        return None
    out = torch.empty((slots.shape[0], *pool.shape[1:]), dtype=pool.dtype, device=pool.device)
    if slots.shape[0]:
        _copy_rows(pool, out, slots.shape[0], src_idx=slots)
    _log_once(
        "gather",
        "SGLANG_ENABLE_GDN_PREFILL_FUSED_STATE_COPY: chunked-GDN state rows gathered and "
        "scattered by one copy launch each (first: %d rows of %d elements)",
        slots.shape[0],
        pool[0].numel(),
    )
    return out


def gdn_scatter_state_rows(pool: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor) -> None:
    """``pool.index_copy_(0, slots, rows)`` in one copy launch (rows: dense, pool's dtype)."""
    if slots.shape[0]:
        _copy_rows(rows, pool, slots.shape[0], dst_idx=slots)


def _row_major_tail(t: torch.Tensor) -> bool:
    """Every dim after the first is dense (one row = one contiguous span)."""
    expected = 1
    for size, stride in zip(reversed(t.shape[1:]), reversed(t.stride()[1:])):
        if size != 1 and stride != expected:
            return False
        expected *= size
    return True


def gdn_track_conv_snapshot(
    *,
    conv_pool: torch.Tensor,
    x: torch.Tensor,
    token_rows: torch.Tensor,
    dst_slots: torch.Tensor,
) -> bool:
    """One launch for ``conv_pool[dst_slots] = x[token_rows].transpose(1, 2)``.
    Returns False (nothing written) when the layout is not the one handled."""
    rows = dst_slots.shape[0]
    reason = None
    if conv_pool.dim() != 3 or x.dim() != 2 or token_rows.dim() != 2:
        reason = "ranks"
    elif conv_pool.dtype != x.dtype:
        reason = f"dtypes {conv_pool.dtype} / {x.dtype}"
    elif x.stride(1) != 1 or token_rows.shape != (rows, conv_pool.shape[2]):
        reason = "layout"
    elif x.shape[1] != conv_pool.shape[1]:
        reason = "dims"
    if reason is not None:
        _log_once(
            "conv-miss",
            "SGLANG_ENABLE_GDN_PREFILL_FUSED_TRACK: conv window kept on torch (%s)",
            reason,
        )
        return False
    if rows == 0:
        return True
    dim = x.shape[1]
    _gdn_track_conv_snapshot_kernel[(rows, triton.cdiv(dim, _BLOCK_CONV))](
        conv_pool,
        x,
        token_rows,
        dst_slots,
        conv_pool.stride(0),
        conv_pool.stride(1),
        conv_pool.stride(2),
        x.stride(0),
        token_rows.stride(0),
        token_rows.stride(1),
        dim,
        STATE_LEN=conv_pool.shape[2],
        BLOCK=_BLOCK_CONV,
    )
    _log_once(
        "conv",
        "SGLANG_ENABLE_GDN_PREFILL_FUSED_TRACK: conv windows of the track slots in one "
        "launch (first: %d rows x %d channels)",
        rows,
        dim,
    )
    return True


def gdn_track_state_rows(
    *,
    pool: torch.Tensor,
    h: Optional[torch.Tensor],
    h_src: torch.Tensor,
    h_dst: torch.Tensor,
    f_src: torch.Tensor,
    f_dst: torch.Tensor,
) -> bool:
    """One launch for ``pool[h_dst] = h[h_src]`` and ``pool[f_dst] = pool[f_src]``
    (h: [checkpoints, *pool.shape[1:]]). Returns False (nothing written) when the
    layout is not the one handled."""
    n_h, n_f = h_src.shape[0], f_src.shape[0]
    reason = None
    if not _row_major_tail(pool):
        reason = "pool rows not dense"
    elif n_h and h is None:
        reason = "no checkpoints"
    elif n_h and (
        h.dtype != pool.dtype
        or tuple(h.shape[1:]) != tuple(pool.shape[1:])
        or not _row_major_tail(h)
    ):
        reason = "checkpoint layout"
    if reason is not None:
        _log_once(
            "state-miss",
            "SGLANG_ENABLE_GDN_PREFILL_FUSED_TRACK: state rows kept on torch (%s)",
            reason,
        )
        return False
    if n_h + n_f == 0:
        return True
    row_elems = pool[0].numel()
    src_h = h if n_h else pool
    _gdn_track_state_rows_kernel[(n_h + n_f, triton.cdiv(row_elems, _BLOCK_ROW))](
        pool,
        src_h,
        h_src if n_h else f_src,
        h_dst if n_h else f_dst,
        f_src if n_f else h_src,
        f_dst if n_f else h_dst,
        n_h,
        row_elems,
        pool.stride(0),
        src_h.stride(0),
        BLOCK=_BLOCK_ROW,
    )
    _log_once(
        "state",
        "SGLANG_ENABLE_GDN_PREFILL_FUSED_TRACK: track states in one launch "
        "(first: %d checkpoint rows + %d final rows of %d elements)",
        n_h,
        n_f,
        row_elems,
    )
    return True
