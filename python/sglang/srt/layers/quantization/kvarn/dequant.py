# SPDX-License-Identifier: Apache-2.0
"""KVarN tile-level dequant reference (pure PyTorch).

Inverse of the store functions.

Dequant identities:
    K_rot[d, g] = (q_K[d, g] * s_col_K[d] + zp_K[d]) * s_row_K[g]
    V_rot[g, d] = (q_V[g, d] * s_row_V[g] + zp_V[g]) * s_col_V[d]
"""

from __future__ import annotations

import torch


def _unpack_4bit(packed: torch.Tensor, original_last_dim: int) -> torch.Tensor:
    """Inverse of ``_pack_4bit``."""
    assert original_last_dim % 2 == 0
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    out = torch.empty(
        *packed.shape[:-1],
        original_last_dim,
        dtype=torch.uint8,
        device=packed.device,
    )
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def _unpack_lowbit(
    packed: torch.Tensor, original_last_dim: int, bits: int
) -> torch.Tensor:
    """Inverse of ``_pack_lowbit`` for any ``bits`` in {2, 4}."""
    if bits == 4:
        return _unpack_4bit(packed, original_last_dim)
    pack = 8 // bits
    mask = (1 << bits) - 1
    assert original_last_dim % pack == 0
    out = torch.empty(
        *packed.shape[:-1],
        original_last_dim,
        dtype=torch.uint8,
        device=packed.device,
    )
    for j in range(pack):
        out[..., j::pack] = (packed >> (j * bits)) & mask
    return out


def kvarn_dequant_tile_k(
    q_packed_uint8: torch.Tensor,
    s_col_K: torch.Tensor,
    zp_K: torch.Tensor,
    s_row_K: torch.Tensor,
    group: int,
    bits: int = 4,
) -> torch.Tensor:
    """Dequantize one K tile back to the rotated ``[D, group]`` frame.

    Args:
        q_packed_uint8 : ``[D, group // (8//bits)]`` uint8.
        s_col_K        : ``[D]`` fp16
        zp_K           : ``[D]`` fp16
        s_row_K        : ``[group]`` fp16
        group          : tile width in tokens.
        bits           : quant bit-width of K (default 4).

    Returns:
        ``[D, group]`` fp32 dequantized tile in the rotated frame.
    """
    q = _unpack_lowbit(q_packed_uint8, group, bits).float()
    s_col = s_col_K.float().unsqueeze(-1)
    zp = zp_K.float().unsqueeze(-1)
    s_row = s_row_K.float().unsqueeze(0)
    return (q * s_col + zp) * s_row


def kvarn_dequant_tile_v(
    q_packed_uint8: torch.Tensor,
    s_col_V: torch.Tensor,
    s_row_V: torch.Tensor,
    zp_V: torch.Tensor,
    head_dim: int,
    bits: int = 4,
) -> torch.Tensor:
    """Dequantize one V tile back to the rotated ``[group, D]`` frame.

    Args:
        q_packed_uint8 : ``[group, D // (8//bits)]`` uint8.
        s_col_V        : ``[D]``     fp16
        s_row_V        : ``[group]`` fp16
        zp_V           : ``[group]`` fp16
        head_dim       : tile width in channels.
        bits           : quant bit-width of V (default 4).

    Returns:
        ``[group, D]`` fp32 dequantized tile in the rotated frame.
    """
    q = _unpack_lowbit(q_packed_uint8, head_dim, bits).float()
    s_row = s_row_V.float().unsqueeze(-1)
    zp = zp_V.float().unsqueeze(-1)
    s_col = s_col_V.float().unsqueeze(0)
    return (q * s_row + zp) * s_col


def kvarn_dequant_tile_k_batch(
    q_packed_uint8: torch.Tensor,
    s_col_K: torch.Tensor,
    zp_K: torch.Tensor,
    s_row_K: torch.Tensor,
    group: int,
    bits: int = 4,
) -> torch.Tensor:
    """Batched dequant.  q_packed_uint8: ``[N, D, group/pack]``.

    Returns ``[N, D, group]`` fp32.
    """
    q = _unpack_lowbit(q_packed_uint8, group, bits).float()
    s_col = s_col_K.float().unsqueeze(-1)
    zp = zp_K.float().unsqueeze(-1)
    s_row = s_row_K.float().unsqueeze(1)
    return (q * s_col + zp) * s_row


def kvarn_dequant_tile_v_batch(
    q_packed_uint8: torch.Tensor,
    s_col_V: torch.Tensor,
    s_row_V: torch.Tensor,
    zp_V: torch.Tensor,
    head_dim: int,
    bits: int = 4,
) -> torch.Tensor:
    """Batched dequant.  q_packed_uint8: ``[N, group, D/pack]``.

    Returns ``[N, group, D]`` fp32.
    """
    q = _unpack_lowbit(q_packed_uint8, head_dim, bits).float()
    s_row = s_row_V.float().unsqueeze(-1)
    zp = zp_V.float().unsqueeze(-1)
    s_col = s_col_V.float().unsqueeze(1)
    return (q * s_row + zp) * s_col
