"""Fused Triton kernel for Quest's per-layer scatter-pack step.

Replaces this PyTorch op chain in the decode forward path
(``flashinfer_quest_backend.py`` + ``flashinfer_hisparse_backend.py``):

    page_table = page_table.clamp(min=0)
    actual_lens_bs = self._actual_lens_buf[:bs]
    j = self._range_top_k.unsqueeze(0).expand(bs, self.top_k)
    valid = j < actual_lens_bs.unsqueeze(1)
    dest_real = self._kv_indptr_buf[:bs].unsqueeze(1) + j
    dest = torch.where(
        valid, dest_real,
        torch.full_like(dest_real, self._scatter_scratch_idx),
    )
    self._kv_indices_buf.scatter_(
        0, dest.view(-1).long(), page_table.view(-1)
    )

The kernel writes ``page_table[i, j]`` (clamped to ≥0) into
``kv_indices_buf[kv_indptr[i] + j]`` for ``j < actual_lens[i]``, and into
``kv_indices_buf[scratch_idx]`` (a harmless throwaway slot) otherwise.
One launch per layer per step replaces ~7 small PyTorch ops, mostly
helping at small bs where launch overhead dominates the work.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quest_scatter_pack_kernel(
    page_table_ptr,        # [bs, TOP_K] int32 (any signed int)
    actual_lens_ptr,       # [bs] int32
    kv_indptr_ptr,         # [bs + 1] int32 (cumsum of actual_lens)
    kv_indices_buf_ptr,    # [scratch_idx + 1] int32
    scratch_idx,           # int (target for invalid positions)
    NUM_REQS,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    if pid_b >= NUM_REQS:
        return

    actual_len = tl.load(actual_lens_ptr + pid_b)
    indptr = tl.load(kv_indptr_ptr + pid_b)

    j_off = pid_k * BLOCK + tl.arange(0, BLOCK)
    j_mask = j_off < TOP_K

    src = tl.load(
        page_table_ptr + pid_b * TOP_K + j_off,
        mask=j_mask,
        other=0,
    )
    # Clamp to ≥0 (the caller used to do this with a separate op).
    src = tl.maximum(src, 0)

    valid = j_off < actual_len
    dst = tl.where(valid, indptr + j_off, scratch_idx)
    tl.store(kv_indices_buf_ptr + dst, src, mask=j_mask)


@triton.jit
def _quest_only_gather_scatter_kernel(
    topk_positions_ptr,    # [bs, TOP_K] int32
    req_pool_indices_ptr,  # [bs] int64
    req_to_token_ptr,      # [max_reqs, max_context_len] int32
    actual_lens_ptr,       # [bs] int32
    kv_indptr_ptr,         # [bs + 1] int32
    kv_indices_buf_ptr,    # int32
    scratch_idx,
    NUM_REQS,
    REQ_TO_TOKEN_STRIDE: tl.constexpr,  # max_context_len
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Quest_only-specific fused: gather pool addresses via req_to_token, then
    scatter-pack into kv_indices_buf.  Replaces:

        kv_indices_padded = self.req_to_token[
            forward_batch.req_pool_indices.unsqueeze(1),
            topk_token_positions.long(),
        ].to(torch.int32)
        # ... scatter pack as in _quest_scatter_pack_kernel ...

    Fuses the advanced-index gather + dtype cast + scatter-pack into one
    kernel launch (replaces ~4 ops per layer).
    """
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    if pid_b >= NUM_REQS:
        return

    rpi = tl.load(req_pool_indices_ptr + pid_b)
    actual_len = tl.load(actual_lens_ptr + pid_b)
    indptr = tl.load(kv_indptr_ptr + pid_b)

    j_off = pid_k * BLOCK + tl.arange(0, BLOCK)
    j_mask = j_off < TOP_K

    # Load the top-k logical positions for this req.
    topk_pos = tl.load(
        topk_positions_ptr + pid_b * TOP_K + j_off,
        mask=j_mask,
        other=0,
    ).to(tl.int64)

    # Gather: req_to_token[rpi, topk_pos]
    rt_offsets = rpi * REQ_TO_TOKEN_STRIDE + topk_pos
    pool_idx = tl.load(
        req_to_token_ptr + rt_offsets,
        mask=j_mask,
        other=0,
    )

    # Scatter-pack: write to kv_indices_buf[kv_indptr[i] + j] (or scratch_idx
    # if j >= actual_lens[i]).
    valid = j_off < actual_len
    dst = tl.where(valid, indptr + j_off, scratch_idx)
    tl.store(kv_indices_buf_ptr + dst, pool_idx, mask=j_mask)


def quest_only_gather_scatter(
    topk_positions: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    actual_lens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices_buf: torch.Tensor,
    scratch_idx: int,
    top_k: int,
) -> None:
    """Fused gather (req_to_token) + scatter-pack for the quest_only backend.

    Args:
      topk_positions: ``[bs, top_k]`` int32 — Quest top-k LOGICAL positions
        (in [0, seq_len)) for each req.
      req_pool_indices: ``[bs]`` int64 — request pool slots.
      req_to_token: ``[max_reqs, max_context_len]`` int32 — pool address per
        (req, position).
      actual_lens, kv_indptr, kv_indices_buf, scratch_idx, top_k:
        same as :func:`quest_scatter_pack`.
    """
    bs = int(topk_positions.shape[0])
    if bs == 0:
        return
    assert topk_positions.shape[1] == top_k
    assert req_to_token.dim() == 2

    BLOCK = min(top_k, 256)
    grid = (bs, triton.cdiv(top_k, BLOCK))

    _quest_only_gather_scatter_kernel[grid](
        topk_positions,
        req_pool_indices,
        req_to_token,
        actual_lens,
        kv_indptr,
        kv_indices_buf,
        scratch_idx,
        bs,
        REQ_TO_TOKEN_STRIDE=req_to_token.shape[1],
        TOP_K=top_k,
        BLOCK=BLOCK,
        num_warps=4,
    )


def quest_scatter_pack(
    page_table: torch.Tensor,
    actual_lens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices_buf: torch.Tensor,
    scratch_idx: int,
    top_k: int,
) -> None:
    """Pack the top-k page indices per request into kv_indices_buf.

    Args:
      page_table: ``[bs, top_k]`` int32 — top-k pool addresses (or device
        buffer indices in the hisparse path).  Negative entries are
        clamped to 0 (kept for parity with the previous PyTorch path).
      actual_lens: ``[bs]`` int32 — ``min(seq_lens, top_k)`` per request.
      kv_indptr: ``[bs + 1]`` int32 — cumsum of ``actual_lens`` (the
        prefix-sum target slot for each request in kv_indices_buf).
      kv_indices_buf: ``[scratch_idx + 1]`` int32 — destination buffer.
        Will be written at positions ``[kv_indptr[i] : kv_indptr[i] +
        actual_lens[i])`` for each ``i``, plus ``scratch_idx`` for any
        residual positions ``j ∈ [actual_lens[i], top_k)``.
      scratch_idx: int — index of the throwaway slot in kv_indices_buf
        (callers reserve ``max_bs * top_k`` for this).
      top_k: int — must match ``page_table.shape[1]``.
    """
    bs = int(page_table.shape[0])
    if bs == 0:
        return
    assert page_table.shape[1] == top_k, (
        f"page_table.shape[1]={page_table.shape[1]} != top_k={top_k}"
    )

    BLOCK = min(top_k, 256)
    grid = (bs, triton.cdiv(top_k, BLOCK))

    _quest_scatter_pack_kernel[grid](
        page_table,
        actual_lens,
        kv_indptr,
        kv_indices_buf,
        scratch_idx,
        bs,
        TOP_K=top_k,
        BLOCK=BLOCK,
        num_warps=4,
    )
