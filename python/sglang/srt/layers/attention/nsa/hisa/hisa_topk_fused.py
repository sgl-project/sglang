"""HISA fused topk kernels (single .so, multiple variants).

Currently exposes:

  - ``hisa_topk_coord_transform_fused``: fuses ``fast_topk_v2`` +
    ``hisa_coord_transform`` into one CUDA kernel. ``ks=None`` selects
    PAGED (output = absolute K positions, masked by ``lens=seq_lens``);
    ``ks!=None`` selects RAGGED (output = ``raw - ks``, masked by
    ``ks <= raw < lens=ke``). Mirrors ``hisa_coord_transform``'s API.
    This is the ``SGLANG_NSA_FUSE_TOPK=0`` path that ``HisaIndexer`` calls.

Reserved names (kernel bodies are stubs; calling raises at the C++ interface):

  - ``hisa_topk_transform_paged`` / ``..._ragged``:
    will fuse the topk + a final ``page_table_1`` gather (paged) or
    ``+ topk_indices_offset`` (ragged), matching upstream's
    ``fast_topk_transform_fused`` / ``..._ragged_fused`` output contract.
    To be implemented when ``SGLANG_NSA_FUSE_TOPK=1`` support is added.

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
    score: torch.Tensor,                # [B, sparse_len] f32
    topk_block_idx: torch.Tensor,       # [B, block_topk] i32
    k_block_size: int,
    ke: torch.Tensor,                 # [B] i32 — ke for RAGGED, seq_lens for PAGED
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
        (B,), sparse_len, dtype=torch.int32, device=score.device,
    )
    if out is None:
        out = torch.empty((B, TOPK), dtype=torch.int32, device=score.device)
    if ks is None:
        # PAGED: lens = seq_lens.
        torch.ops.hisa_topk_fused.topk_coord_transform_fused_paged(
            score, full_lens, topk_block_idx, ke, out, k_block_size,
        )
    else:
        # RAGGED: lens = ke.
        if ks.dtype != torch.int32:
            ks = ks.to(torch.int32)
        torch.ops.hisa_topk_fused.topk_coord_transform_fused_ragged(
            score, full_lens, topk_block_idx, ks, ke, out, k_block_size,
        )
    return out


# ---------------------------------------------------------------------------
# topk + page_table-style transform (RESERVED — for SGLANG_NSA_FUSE_TOPK=1).
# Kernel body is a stub; the C++ interface raises at call time. These names
# match upstream's ``topk_transform_*`` family naming so swapping later
# only changes the implementation, not the API surface.
# ---------------------------------------------------------------------------

def hisa_topk_transform_paged(*args, **kwargs):
    raise NotImplementedError(
        "hisa_topk_transform_paged is reserved for the SGLANG_NSA_FUSE_TOPK=1 "
        "path (page_table_1 output). Use hisa_topk_coord_transform_fused_paged "
        "for the current (token-position) variant."
    )


def hisa_topk_transform_ragged(*args, **kwargs):
    raise NotImplementedError(
        "hisa_topk_transform_ragged is reserved for the SGLANG_NSA_FUSE_TOPK=1 "
        "path (ragged page-table-style output). Use "
        "hisa_topk_coord_transform_fused_ragged for the current "
        "(token-position) variant."
    )


# ---------------------------------------------------------------------------
# Unified dispatch — mirror upstream NSAIndexerMetadata.topk_transform style.
# ---------------------------------------------------------------------------

def hisa_topk_transform_dispatch(
    metadata,                                   # NSAIndexerMetadata-like
    block_sparse_logits: torch.Tensor,          # [M, sparse_len] f32
    topk_block_indices: torch.Tensor,           # [M, block_topk] i32
    k_block_size: int,
    *,
    ke: torch.Tensor,                         # ke for RAGGED, seq_lens for PAGED
    ks: Optional[torch.Tensor] = None,          # None for PAGED, [M] for RAGGED
) -> torch.Tensor:
    """Unify HISA's topk-transform implementations behind a single dispatch,
    mirroring upstream's ``NSAIndexerMetadata.topk_transform``.

    Method (PAGED / RAGGED) is read from ``metadata.topk_transform_method``.

    Branches:
      - ``SGLANG_NSA_FUSE_TOPK=1`` + PAGED   — paged transform [TODO]
      - ``SGLANG_NSA_FUSE_TOPK=1`` + RAGGED  — ragged transform [TODO]
      - default                              — ``hisa_topk_coord_transform_fused``
    """
    from sglang.srt.layers.attention.nsa_backend import TopkTransformMethod

    method = metadata.topk_transform_method
    fuse_topk = envs.SGLANG_NSA_FUSE_TOPK.get()

    if fuse_topk and method == TopkTransformMethod.PAGED:
        raise NotImplementedError(
            "SGLANG_NSA_FUSE_TOPK=1 + PAGED not yet implemented; "
            "reserved at topk_transform_paged_kernel stub in hisa_topk_fused.cu."
        )
    elif fuse_topk and method == TopkTransformMethod.RAGGED:
        raise NotImplementedError(
            "SGLANG_NSA_FUSE_TOPK=1 + RAGGED not yet implemented; "
            "reserved at topk_transform_ragged_kernel stub in hisa_topk_fused.cu."
        )
    elif not fuse_topk:
        return hisa_topk_coord_transform_fused(
            block_sparse_logits, topk_block_indices, k_block_size, ks=ks, ke=ke,
        )
    else:
        assert False, (
            f"Unreachable hisa_topk_transform_dispatch state: "
            f"fuse_topk={fuse_topk} method={method!r}"
        )
