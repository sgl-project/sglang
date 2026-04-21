"""Triton kernel for the hisa coord-transform step.

Fuses the post-``fast_topk_v2`` conversion pipeline:
    relevant_safe = relevant.clamp(min=0)
    slot          = relevant_safe // k_block_size
    abs_block     = topk_block_indices.gather(-1, slot)
    raw           = abs_block * k_block_size + (relevant_safe % k_block_size)
    # RAGGED only:  raw = raw - ks;  valid = raw in [0, ke-ks)
    # PAGED only:   raw = raw;       valid = raw < seq_len
    final         = where(valid & (relevant != -1), raw, -1)

into a single pass that:
- keeps all intermediates in registers (no int64 abs_block/raw tensors, no
  bool mask tensor, etc.) — peak memory savings of ~5×M×index_topk bytes
- launches once instead of ~6 torch ops
- stays in int32 throughout (safe: max value is K_total < 2^31 for reasonable
  contexts)

Used by :class:`HisaIndexer` in both ``_get_topk_ragged`` (prefill) and
``_get_topk_paged`` (decode).
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def hisa_coord_transform_kernel(
    relevant_ptr,           # [M, INDEX_TOPK] int32 — fast_topk_v2 output
    topk_block_ptr,         # [M, BLOCK_TOPK] int32 — abs block IDs
    ks_ptr,                 # [M] int32  — per-query K start (RAGGED); unused on PAGED
    lens_ptr,               # [M] int32  — per-query ke (RAGGED) or seq_len (PAGED)
    out_ptr,                # [M, INDEX_TOPK] int32
    K_BLOCK_SIZE: tl.constexpr,
    INDEX_TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    RAGGED: tl.constexpr,
):
    m = tl.program_id(0)
    offs = tl.arange(0, INDEX_TOPK)

    # Load this row's fast_topk_v2 output.
    r = tl.load(relevant_ptr + m * INDEX_TOPK + offs)
    r_is_valid = r != -1
    r_safe = tl.maximum(r, 0)

    # Gather: abs_block[i] = topk_block_indices[m, r_safe[i] // k_block_size].
    slot = r_safe // K_BLOCK_SIZE
    # mask guards against bogus loads when r == -1 (slot undefined). r_safe
    # actually clamps to 0 so slot is 0 → always in range — but the mask lets
    # the compiler skip the load and avoids any potential OOB on edge cases.
    abs_block = tl.load(
        topk_block_ptr + m * BLOCK_TOPK + slot, mask=r_is_valid, other=0
    )

    # raw = abs_block * k_block_size + (r_safe % k_block_size).
    # Use r_safe - slot * K_BLOCK_SIZE to avoid a second modulo.
    raw = abs_block * K_BLOCK_SIZE + (r_safe - slot * K_BLOCK_SIZE)

    if RAGGED:
        ks = tl.load(ks_ptr + m)
        ke = tl.load(lens_ptr + m)
        raw_rel = raw - ks
        valid = (raw_rel >= 0) & (raw_rel < (ke - ks))
    else:
        seq_len = tl.load(lens_ptr + m)
        raw_rel = raw
        valid = raw < seq_len

    out = tl.where(valid & r_is_valid, raw_rel, -1)
    tl.store(out_ptr + m * INDEX_TOPK + offs, out)


def hisa_coord_transform(
    relevant: torch.Tensor,
    topk_block_indices: torch.Tensor,
    lens: torch.Tensor,
    k_block_size: int,
    ks: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused coord transform. ``ks is None`` selects PAGED semantics.

    Args:
        relevant: [M, index_topk] int32 — fast_topk_v2 output (positions in
                  the sparse score array, -1 for invalid).
        topk_block_indices: [M, block_topk] int — absolute block IDs.
        lens: [M] int32 — ``ke`` for RAGGED, ``seq_lens`` for PAGED.
        k_block_size: int — hisa block size (e.g. 128).
        ks: [M] int32 or None — RAGGED per-query K start. None → PAGED decode
            (no ks-subtract; mask ``raw < seq_len``).
        out: optional pre-allocated output buffer.

    Returns:
        [M, index_topk] int32. RAGGED: ks-relative positions; PAGED: absolute
        per-request K positions. ``-1`` for invalid / out-of-range entries.
    """
    assert relevant.ndim == 2 and relevant.dtype == torch.int32, (
        f"relevant must be [M, index_topk] int32, got {relevant.shape} {relevant.dtype}"
    )
    M, index_topk = relevant.shape
    block_topk = topk_block_indices.shape[-1]

    # Normalize dtypes — the triton kernel assumes int32 throughout.
    if topk_block_indices.dtype != torch.int32:
        topk_block_indices = topk_block_indices.to(torch.int32)
    if lens.dtype != torch.int32:
        lens = lens.to(torch.int32)

    ragged = ks is not None
    if ragged and ks.dtype != torch.int32:
        ks = ks.to(torch.int32)
    # In PAGED mode the kernel ignores ks_ptr; pass any valid pointer.
    ks_arg = ks if ragged else lens

    if out is None:
        out = torch.empty((M, index_topk), dtype=torch.int32, device=relevant.device)
    else:
        assert out.shape == (M, index_topk) and out.dtype == torch.int32

    hisa_coord_transform_kernel[(M,)](
        relevant,
        topk_block_indices,
        ks_arg,
        lens,
        out,
        K_BLOCK_SIZE=k_block_size,
        INDEX_TOPK=index_topk,
        BLOCK_TOPK=block_topk,
        RAGGED=ragged,
    )
    return out
