# SPDX-License-Identifier: Apache-2.0
"""KVarN tile-level store (pure PyTorch).

Quantizes one tile of K (or V) per call.

Inputs to the K path are tile-shaped ``[D, group]`` (channels × tokens —
the KIVI K-axis orientation) **after** Hadamard rotation.  Inputs to the
V path are tile-shaped ``[group, D]`` (tokens × channels — the KIVI
V-axis orientation) also after Hadamard rotation.

The output is a packed record matching the cache layout from
``KVarNConfig``.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.quantization.kvarn.sinkhorn import (
    variance_normalize,
)


def _rtn_range(t: torch.Tensor, dim: int):
    """Per-row range (min/max)."""
    return t.amin(dim=dim, keepdim=True), t.amax(dim=dim, keepdim=True)


def _asymmetric_rtn_per_row(
    tile: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row asymmetric RTN over the full row.

    Args:
        tile: [R, C] fp32.
        bits: 2, 3 or 4.

    Returns:
        q     [R, C] int32 in [0, 2^bits - 1]
        scale [R, 1] fp32
        zp    [R, 1] fp32  (= row minimum)
    """
    qmax = (1 << bits) - 1
    lo = tile.amin(dim=1, keepdim=True)
    hi = tile.amax(dim=1, keepdim=True)
    scale = ((hi - lo) / qmax).clamp_min(1e-10)
    zp = lo
    q = torch.clamp(torch.round((tile - zp) / scale), 0, qmax).to(torch.int32)
    return q, scale, zp


def _pack_4bit(q: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit ints (last dim even) into uint8 pairs, two-per-byte."""
    assert q.shape[-1] % 2 == 0
    q = q.to(torch.uint8) & 0xF
    lo = q[..., 0::2]
    hi = q[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def _pack_lowbit(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack `bits`-bit ints into uint8, PACK=8//bits values per byte."""
    pack = 8 // bits
    C = q.shape[-1]
    assert C % pack == 0, f"last dim {C} must be divisible by {pack} for {bits}-bit"
    q = (q.to(torch.uint8) & ((1 << bits) - 1)).reshape(*q.shape[:-1], C // pack, pack)
    out = q[..., 0].clone()
    for j in range(1, pack):
        out = out | (q[..., j] << (j * bits))
    return out.to(torch.uint8)


# ─── K tile store: per-channel RTN, [D, group] orientation ──────────────────


def kvarn_store_tile_k(
    k_tile_rotated: torch.Tensor,
    bits: int,
    sinkhorn_iters: int = 16,
) -> dict[str, torch.Tensor]:
    """Quantize one rotated K tile ``[D, group]``.

    Returns dict with:
        q_packed_uint8 : ``[D, group/2]`` uint8
        s_col_K        : ``[D]``        fp16
        zp_K           : ``[D]``        fp16
        s_row_K        : ``[group]``    fp16
    """
    assert bits == 4, "Stage 3a only validates 4-bit; lower-bit follow-ups TBD"
    tile = k_tile_rotated.float()
    D, G = tile.shape

    balanced, s_col_sinkhorn, s_row_sinkhorn = variance_normalize(
        tile, iterations=sinkhorn_iters
    )
    s_chan = s_row_sinkhorn  # [D, 1]
    s_tok = s_col_sinkhorn  # [1, G]

    q, rtn_scale, rtn_zp = _asymmetric_rtn_per_row(balanced, bits=bits)

    s_col_K = (s_chan * rtn_scale).squeeze(-1)  # [D]
    zp_K = (s_chan * rtn_zp).squeeze(-1)  # [D]
    s_row_K = s_tok.squeeze(0)  # [G]

    q_packed = _pack_4bit(q)  # [D, G/2]

    return {
        "q_packed_uint8": q_packed,
        "s_col_K": s_col_K.to(torch.float16),
        "zp_K": zp_K.to(torch.float16),
        "s_row_K": s_row_K.to(torch.float16),
    }


def kvarn_store_tile_k_batch_from_sinkhorn(
    balanced: torch.Tensor,
    s_col: torch.Tensor,
    s_row: torch.Tensor,
    bits: int,
) -> dict[str, torch.Tensor]:
    """Batched K-path RTN + scale absorption + 4-bit packing.

    Args:
        balanced : ``[N, D, group]`` fp32 — sinkhorn-balanced K tiles.
        s_col    : ``[N, group]`` fp32 — per-token sinkhorn scale (axis-1).
        s_row    : ``[N, D]``     fp32 — per-channel sinkhorn scale (axis-0).
        bits     : key bit-width (4).
    """
    qmax = (1 << bits) - 1
    N, R, C = balanced.shape
    lo, hi = _rtn_range(balanced, dim=2)
    scale = ((hi - lo) / qmax).clamp_min(1e-10)
    zp = lo
    q = torch.clamp(torch.round((balanced - zp) / scale), 0, qmax).to(torch.int32)
    s_col_K = (s_row * scale.squeeze(-1)).to(torch.float16)  # [N, D]
    zp_K = (s_row * zp.squeeze(-1)).to(torch.float16)
    s_row_K = s_col.to(torch.float16)  # [N, group]
    q_packed = _pack_lowbit(q, bits)
    return {
        "q_packed_uint8": q_packed,
        "s_col_K": s_col_K,
        "zp_K": zp_K,
        "s_row_K": s_row_K,
    }


# ─── V tile store: per-token RTN, [group, D] orientation ────────────────────


def kvarn_store_tile_v(
    v_tile_rotated: torch.Tensor,
    bits: int,
    sinkhorn_iters: int = 16,
) -> dict[str, torch.Tensor]:
    """Quantize one rotated V tile ``[group, D]``.

    Returns dict with:
        q_packed_uint8 : ``[group, D/2]`` uint8
        s_col_V        : ``[D]``          fp16
        s_row_V        : ``[group]``      fp16
        zp_V           : ``[group]``      fp16
    """
    assert bits == 4, "Stage 3a only validates 4-bit; lower-bit follow-ups TBD"
    tile = v_tile_rotated.float()
    G, D = tile.shape

    balanced, s_col_sinkhorn, s_row_sinkhorn = variance_normalize(
        tile, iterations=sinkhorn_iters
    )
    s_chan = s_col_sinkhorn  # [1, D]
    s_tok = s_row_sinkhorn  # [G, 1]

    q, rtn_scale, rtn_zp = _asymmetric_rtn_per_row(balanced, bits=bits)

    s_row_V = (s_tok * rtn_scale).squeeze(-1)  # [G]
    zp_V = (s_tok * rtn_zp).squeeze(-1)  # [G]
    s_col_V = s_chan.squeeze(0)  # [D]

    q_packed = _pack_4bit(q)  # [G, D/2]

    return {
        "q_packed_uint8": q_packed,
        "s_col_V": s_col_V.to(torch.float16),
        "s_row_V": s_row_V.to(torch.float16),
        "zp_V": zp_V.to(torch.float16),
    }


def kvarn_store_tile_v_batch_from_sinkhorn(
    balanced: torch.Tensor,
    s_col: torch.Tensor,
    s_row: torch.Tensor,
    bits: int,
) -> dict[str, torch.Tensor]:
    """Batched V-path RTN + scale absorption + 4-bit packing.

    Args:
        balanced : ``[N, group, D]`` fp32 — sinkhorn-balanced V tiles.
        s_col    : ``[N, D]``     fp32 — per-channel sinkhorn scale (axis-1).
        s_row    : ``[N, group]`` fp32 — per-token-in-tile sinkhorn scale (axis-0).
        bits     : value bit-width (4).
    """
    qmax = (1 << bits) - 1
    N, R, C = balanced.shape
    lo, hi = _rtn_range(balanced, dim=2)
    scale = ((hi - lo) / qmax).clamp_min(1e-10)
    zp = lo
    q = torch.clamp(torch.round((balanced - zp) / scale), 0, qmax).to(torch.int32)
    s_row_V = (s_row * scale.squeeze(-1)).to(torch.float16)  # [N, group]
    zp_V = (s_row * zp.squeeze(-1)).to(torch.float16)
    s_col_V = s_col.to(torch.float16)  # [N, D]
    q_packed = _pack_lowbit(q, bits)
    return {
        "q_packed_uint8": q_packed,
        "s_col_V": s_col_V,
        "s_row_V": s_row_V,
        "zp_V": zp_V,
    }
