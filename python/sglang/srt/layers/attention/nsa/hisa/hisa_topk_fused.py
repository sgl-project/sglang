"""HISA fused topk kernels (single .so, multiple variants).

Two output contracts, selected by the indexer's dispatch based on
``SGLANG_NSA_FUSE_TOPK``:

  - ``hisa_topk_coord_transform_fused`` — ``FUSE_TOPK=0`` path. Fuses
    ``fast_topk_v2 + hisa_coord_transform`` into one CUDA kernel.
    ``ks=None`` -> PAGED (output = absolute K positions, masked by
    ``lens=seq_lens``); ``ks!=None`` -> RAGGED (output = ``raw - ks``,
    masked by ``ks <= raw < lens=ke``). Downstream applies the
    ``transform_index_page_table_*`` step.

  - ``hisa_topk_transform_paged`` / ``..._ragged`` — ``FUSE_TOPK=1`` path.
    Additionally fuses the final ``page_table_1`` gather (PAGED) or
    ``+ topk_indices_offset`` (RAGGED) into the same kernel, matching
    upstream's ``fast_topk_transform_{fused,ragged_fused}`` contract so
    the indexer's output can be used as a transformed page table directly.

The .cu is JIT-compiled on first import via ``torch.utils.cpp_extension.load``
and cached under ``$TORCH_EXTENSIONS_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

from sglang.srt.environ import envs

_HERE = Path(__file__).resolve().parent
_CSRC = _HERE / "csrc"


def _load_module():
    load(
        name="hisa_topk_fused",
        sources=[str(_CSRC / "hisa_topk_fused.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        is_python_module=False,
        verbose=bool(int(os.environ.get("HISA_TOPK_FUSED_VERBOSE", "0"))),
    )
    return None


# Side-effect: load + register torch ops on import.
_module = _load_module()


# Topk size matches upstream's ``fast_topk_v2`` contract — DSv3.2 index_topk.
TOPK = 2048


# ---------------------------------------------------------------------------
# topk + coord-transform (output = HISA's token-position contract)
# ---------------------------------------------------------------------------


def hisa_topk_coord_transform_fused(
    score: torch.Tensor,  # [B, sparse_len] f32
    topk_block_idx: torch.Tensor,  # [B, block_topk] i32
    k_block_size: int,
    ke: torch.Tensor,  # [B] i32 — ke for RAGGED, seq_lens for PAGED
    ks: Optional[torch.Tensor] = None,  # [B] i32 — None → PAGED, else RAGGED
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused topk-2048 + coord transform. ``ks is None`` selects PAGED.

    Equivalent to::

        relevant = fast_topk_v2(score, full_lens, 2048)
        topk_result = hisa_coord_transform(
            relevant, topk_block_idx, lens=lens,
            k_block_size=k_block_size, ks=ks,
        )

    Output ``[B, 2048]`` i32:
      - PAGED (``ks=None``): absolute per-request K positions
        (``raw = topk_block_idx[batch, slot] * K_BLK + (i % K_BLK)``,
        masked by ``raw < lens[batch]``).
      - RAGGED (``ks!=None``): ks-relative positions (``raw - ks[batch]``,
        masked by ``ks <= raw < lens[batch]``).
      ``-1`` for invalid / out-of-range entries.
    """
    if topk_block_idx.dtype != torch.int32:
        topk_block_idx = topk_block_idx.to(torch.int32)
    if ke.dtype != torch.int32:
        ke = ke.to(torch.int32)
    B, sparse_len = score.shape
    full_lens = torch.full(
        (B,),
        sparse_len,
        dtype=torch.int32,
        device=score.device,
    )
    if out is None:
        out = torch.empty((B, TOPK), dtype=torch.int32, device=score.device)
    if ks is None:
        # PAGED: lens = seq_lens.
        torch.ops.hisa_topk_fused.topk_coord_transform_fused_paged(
            score,
            full_lens,
            topk_block_idx,
            ke,
            out,
            k_block_size,
        )
    else:
        # RAGGED: lens = ke.
        if ks.dtype != torch.int32:
            ks = ks.to(torch.int32)
        torch.ops.hisa_topk_fused.topk_coord_transform_fused_ragged(
            score,
            full_lens,
            topk_block_idx,
            ks,
            ke,
            out,
            k_block_size,
        )
    return out


# ---------------------------------------------------------------------------
# topk + page_table-style transform (SGLANG_NSA_FUSE_TOPK=1 path).
# Output contract matches upstream's ``fast_topk_transform_{fused,
# ragged_fused}`` so the consumer at nsa_backend.py:1357 (`page_table_1 =
# topk_indices`) works unchanged.
# ---------------------------------------------------------------------------


def hisa_topk_transform_paged(
    score: torch.Tensor,  # [M, sparse_len] f32 (M = total query tokens)
    topk_block_idx: torch.Tensor,  # [M, block_topk] i32 (per-token)
    seq_lens: torch.Tensor,  # [M] i32 — per-token K end (seq_lens)
    page_table_1: torch.Tensor,  # [B, max_seqlen_k] i32 (per-batch!)
    cu_seqlens_q: torch.Tensor,  # [B+1] i32 (cumulative; arange(B+1) for decode)
    k_block_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused topk-2048 + coord transform + page_table_1 gather (PAGED).

    Mirrors upstream's ``fast_topk_transform_fused`` contract: output is
    [M, 2048] of physical page-table indices, matching the
    ``SGLANG_NSA_FUSE_TOPK=1`` consumer at ``nsa_backend.py:1357``.

    Internally dispatches:
      - decode kernel (1:1 mapping) when ``prefill_bs = cu_seqlens_q.size(0)-1 == M``
      - prefill kernel (cu_seqlens_q-based batch lookup) otherwise

    For pure decode, pass ``cu_seqlens_q = torch.arange(B+1, dtype=int32)``
    (which makes M==B and selects the decode kernel).
    """
    if topk_block_idx.dtype != torch.int32:
        topk_block_idx = topk_block_idx.to(torch.int32)
    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)
    if cu_seqlens_q.dtype != torch.int32:
        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
    M, sparse_len = score.shape
    full_lens = torch.full(
        (M,),
        sparse_len,
        dtype=torch.int32,
        device=score.device,
    )
    if out is None:
        out = torch.empty((M, TOPK), dtype=torch.int32, device=score.device)
    torch.ops.hisa_topk_fused.topk_transform_paged(
        score,
        full_lens,
        topk_block_idx,
        seq_lens,
        page_table_1,
        cu_seqlens_q,
        out,
        k_block_size,
    )
    return out


def hisa_topk_transform_ragged(
    score: torch.Tensor,  # [B, sparse_len] f32
    topk_block_idx: torch.Tensor,  # [B, block_topk] i32
    ks: torch.Tensor,  # [B] i32 — per-row kv start
    ke: torch.Tensor,  # [B] i32 — per-row kv end
    topk_indices_offset: torch.Tensor,  # [B] i32 — per-row global flat offset
    k_block_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused topk-2048 + coord transform + ragged offset (SGLANG_NSA_FUSE_TOPK=1).

    Equivalent to::

        relevant = fast_topk_v2(score, full_lens, 2048)
        raw_rel  = hisa_coord_transform(relevant, topk_block_idx,
                                        lens=ke, k_block_size=..., ks=ks)
        out      = raw_rel + topk_indices_offset[batch]   (-1 where invalid)

    Output ``[B, 2048]`` i32: global flat indices into ``page_table_1_flattened``,
    matching upstream's ``fast_topk_transform_ragged_fused`` contract.
    """
    if topk_block_idx.dtype != torch.int32:
        topk_block_idx = topk_block_idx.to(torch.int32)
    if ks.dtype != torch.int32:
        ks = ks.to(torch.int32)
    if ke.dtype != torch.int32:
        ke = ke.to(torch.int32)
    if topk_indices_offset.dtype != torch.int32:
        topk_indices_offset = topk_indices_offset.to(torch.int32)
    B, sparse_len = score.shape
    full_lens = torch.full(
        (B,),
        sparse_len,
        dtype=torch.int32,
        device=score.device,
    )
    if out is None:
        out = torch.empty((B, TOPK), dtype=torch.int32, device=score.device)
    torch.ops.hisa_topk_fused.topk_transform_ragged(
        score,
        full_lens,
        topk_block_idx,
        ks,
        ke,
        topk_indices_offset,
        out,
        k_block_size,
    )
    return out


# ---------------------------------------------------------------------------
# Unified dispatch — mirror upstream NSAIndexerMetadata.topk_transform style.
# ---------------------------------------------------------------------------


def hisa_topk_transform_dispatch(
    metadata,  # NSAIndexerMetadata-like
    block_sparse_logits: torch.Tensor,  # [M, sparse_len] f32
    topk_block_indices: torch.Tensor,  # [M, block_topk] i32
    k_block_size: int,
    *,
    ke: torch.Tensor,  # mirrors upstream's `ke_offset`
    ks: Optional[torch.Tensor] = None,  # mirrors upstream's `row_starts`
) -> torch.Tensor:
    """HISA's topk-transform dispatch, parallel to upstream's
    ``NSAIndexerMetadata.topk_transform`` (nsa_backend.py:212-276).

    Routing:
      - ``FUSE_TOPK=0`` or ``force_unfused_topk`` → ``hisa_topk_coord_transform_fused``
        (caller's ``ks`` selects PAGED vs RAGGED kernel inside).
      - ``FUSE_TOPK=1`` + PAGED method → ``hisa_topk_transform_paged``
        (output: page_table_1 indices). Uses ``cu_seqlens_q`` from metadata
        to handle both decode (1:1) and prefill (M > B) within one kernel.
      - ``FUSE_TOPK=1`` + RAGGED method → ``hisa_topk_transform_ragged``
        (output: ``raw_rel + topk_indices_offset``).

    Method (PAGED / RAGGED) is read from ``metadata.topk_transform_method``.
    Both methods cover decode AND prefill; "PAGED" / "RAGGED" describes the
    *output contract*, not the forward mode.
    """
    from sglang.srt.layers.attention.nsa_backend import TopkTransformMethod

    method = metadata.topk_transform_method
    fuse_topk = envs.SGLANG_NSA_FUSE_TOPK.get()
    force_unfused = getattr(metadata, "force_unfused_topk", False)

    if fuse_topk and not force_unfused:
        raise NotImplementedError(
            "SGLANG_NSA_FUSE_TOPK=1 path is not yet implemented for hisa, please export SGLANG_NSA_FUSE_TOPK=0."
        )

    if not fuse_topk or force_unfused:
        # FUSE_TOPK=0: route by caller intent (ks=None → PAGED, else RAGGED).
        return hisa_topk_coord_transform_fused(
            block_sparse_logits,
            topk_block_indices,
            k_block_size,
            ks=ks,
            ke=ke,
        )
    elif method == TopkTransformMethod.PAGED:
        # cu_seqlens_q from attn_metadata is already the cumulative form
        # ([B+1] i32; for decode it's arange(B+1)).
        return hisa_topk_transform_paged(
            score=block_sparse_logits,
            topk_block_idx=topk_block_indices,
            seq_lens=ke,
            page_table_1=metadata.get_page_table_1(),
            cu_seqlens_q=metadata.attn_metadata.cu_seqlens_q,
            k_block_size=k_block_size,
        )
    elif method == TopkTransformMethod.RAGGED:
        topk_indices_offset = metadata.attn_metadata.topk_indices_offset
        if topk_indices_offset is None:
            raise RuntimeError(
                "SGLANG_NSA_FUSE_TOPK=1 RAGGED requires "
                "metadata.attn_metadata.topk_indices_offset to be populated."
            )
        if ks is None:
            raise RuntimeError(
                "SGLANG_NSA_FUSE_TOPK=1 RAGGED requires `ks` to be provided "
                "by the caller (per-row kv start)."
            )
        return hisa_topk_transform_ragged(
            score=block_sparse_logits,
            topk_block_idx=topk_block_indices,
            ks=ks,
            ke=ke,
            topk_indices_offset=topk_indices_offset,
            k_block_size=k_block_size,
        )
    else:
        assert False, (
            f"Unreachable hisa_topk_transform_dispatch state: "
            f"fuse_topk={fuse_topk} method={method!r}"
        )
