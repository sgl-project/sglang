# Adapted from KIVI (https://github.com/jy-yuan/KIVI), quant/new_pack.py
from __future__ import annotations

import math

import torch


def _safe_scale(mx: torch.Tensor, mn: torch.Tensor, max_int: int) -> torch.Tensor:
    scale = (mx - mn) / max_int
    return torch.where(scale == 0, torch.ones_like(scale), scale)


def pack_tensor(data: torch.Tensor, bits: int, pack_dim: int) -> torch.Tensor:
    feat_per_int = 32 // bits
    if bits not in (2, 4, 8):
        raise ValueError(f"Unsupported bits={bits}. Only 2/4/8 are supported.")
    if data.shape[pack_dim] % feat_per_int != 0:
        raise ValueError(
            f"Dimension {pack_dim} ({data.shape[pack_dim]}) must be divisible by {feat_per_int}."
        )

    # Vectorized pack: reshape to [..., num_packed, feat_per_int] along pack_dim
    # then bit-pack in a single reduction.
    moved = data.movedim(pack_dim, -1).to(torch.int32).contiguous()
    orig_shape = moved.shape[:-1]
    moved = moved.view(*orig_shape, moved.shape[-1] // feat_per_int, feat_per_int)
    shifts = (
        torch.arange(feat_per_int, device=data.device, dtype=torch.int32) * bits
    ).view(*([1] * (moved.ndim - 1)), feat_per_int)
    packed = torch.sum(moved << shifts, dim=-1, dtype=torch.int32)
    return packed.movedim(-1, pack_dim).contiguous()


def unpack_tensor(v_code: torch.Tensor, bits: int, pack_dim: int) -> torch.Tensor:
    if bits not in (2, 4, 8):
        raise ValueError(f"Unsupported bits={bits}. Only 2/4/8 are supported.")
    feat_per_int = 32 // bits
    shape = v_code.shape
    new_shape = (
        shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim + 1 :]
    )
    i = torch.arange(new_shape[pack_dim], device=v_code.device) // feat_per_int
    j = torch.arange(new_shape[pack_dim], device=v_code.device) % feat_per_int
    mask_bits = 0xFF >> (8 - bits)

    packed_indices = [slice(None)] * len(new_shape)
    packed_indices[pack_dim] = i
    if pack_dim == 2:
        return (v_code[tuple(packed_indices)] >> (j * bits)[None, None, :, None]).to(
            torch.int16
        ) & mask_bits
    if pack_dim == 3:
        return (
            (v_code[tuple(packed_indices)] >> (j * bits)).to(torch.int16)
        ) & mask_bits
    raise ValueError(f"Unsupported pack_dim={pack_dim}.")


def quant_and_pack_kcache(
    k: torch.Tensor, group_size: int, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(k.shape) != 4:
        raise ValueError("Expected k to be a 4D tensor [B, H, T, D].")
    bsz, nheads, seqlen, _ = k.shape
    if seqlen % group_size != 0:
        raise ValueError(
            f"Sequence length {seqlen} must be divisible by group_size {group_size}."
        )

    num_groups = seqlen // group_size
    max_int = 2**bits - 1
    data = k.view(bsz, nheads, num_groups, group_size, -1)
    mn = torch.min(data, dim=-2, keepdim=True)[0]
    mx = torch.max(data, dim=-2, keepdim=True)[0]
    scale = _safe_scale(mx, mn, max_int)
    data = data - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_().to(torch.int32).view_as(k)
    code = pack_tensor(data, bits=bits, pack_dim=2)
    return code, scale, mn


def quant_and_pack_vcache(
    v: torch.Tensor, group_size: int, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(v.shape) != 4:
        raise ValueError("Expected v to be a 4D tensor [B, H, T, D].")
    if v.shape[-1] % group_size != 0:
        raise ValueError(
            f"Head dim {v.shape[-1]} must be divisible by group_size {group_size}."
        )

    num_groups = v.shape[-1] // group_size
    max_int = 2**bits - 1
    data = v.view(v.shape[:-1] + (num_groups, group_size))
    mn = torch.min(data, dim=-1, keepdim=True)[0]
    mx = torch.max(data, dim=-1, keepdim=True)[0]
    scale = _safe_scale(mx, mn, max_int)
    data = data - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_().to(torch.int32).view_as(v)
    code = pack_tensor(data, bits=bits, pack_dim=3)
    return code, scale, mn


def unpack_and_dequant_kcache(
    k_code: torch.Tensor,
    scale: torch.Tensor,
    mn: torch.Tensor,
    group_size: int,
    bits: int,
) -> torch.Tensor:
    data = unpack_tensor(k_code, bits=bits, pack_dim=2)
    shape = data.shape
    num_groups = shape[2] // group_size
    data = data.view(shape[:2] + (num_groups, group_size) + shape[3:])
    data = data.to(scale.dtype)
    data = data * scale + mn
    return data.view(shape)


def unpack_and_dequant_vcache(
    v_code: torch.Tensor,
    scale: torch.Tensor,
    mn: torch.Tensor,
    group_size: int,
    bits: int,
) -> torch.Tensor:
    data = unpack_tensor(v_code, bits=bits, pack_dim=3)
    shape = data.shape
    num_groups = shape[-1] // group_size
    data = data.view(shape[:-1] + (num_groups, group_size))
    data = data.to(scale.dtype)
    data = data * scale + mn
    return data.view(shape)


def kivi_roundtrip_kv_chunk(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    *,
    k_bits: int,
    v_bits: int,
    k_group_size: int,
    v_group_size: int,
    residual_length: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize+dequantize a single-request KV chunk with KIVI-style residuals.

    Args:
        cache_k/cache_v: [T, H, D]
    """
    if cache_k.numel() == 0:
        return cache_k, cache_v

    feat_per_int_k = 32 // k_bits
    t, h, d = cache_k.shape
    if residual_length < 0:
        raise ValueError(
            f"residual_length must be non-negative, got {residual_length}."
        )

    # KIVI keeps recent/residual tokens in full precision. Do not pad incomplete
    # groups with zeros: padding changes min/max and can quantize unrelated tokens
    # together in serving batches.
    quant_len = max(0, t - residual_length)
    align = math.lcm(k_group_size, feat_per_int_k)
    quant_len = (quant_len // align) * align
    if quant_len == 0:
        return cache_k, cache_v

    k_in = cache_k[:quant_len]
    v_in = cache_v[:quant_len]

    k_4d = k_in.permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, H, T, D]
    v_4d = v_in.permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, H, T, D]

    k_code, k_scale, k_mn = quant_and_pack_kcache(k_4d, k_group_size, k_bits)
    v_code, v_scale, v_mn = quant_and_pack_vcache(v_4d, v_group_size, v_bits)
    k_dq = unpack_and_dequant_kcache(k_code, k_scale, k_mn, k_group_size, k_bits)
    v_dq = unpack_and_dequant_vcache(v_code, v_scale, v_mn, v_group_size, v_bits)

    k_out = cache_k.clone()
    v_out = cache_v.clone()
    k_out[:quant_len] = k_dq.squeeze(0).permute(1, 0, 2).contiguous()
    v_out[:quant_len] = v_dq.squeeze(0).permute(1, 0, 2).contiguous()
    return k_out.to(cache_k.dtype), v_out.to(cache_v.dtype)
