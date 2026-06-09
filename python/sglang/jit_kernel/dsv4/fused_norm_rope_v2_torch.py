"""
Pure-PyTorch implementation of FusedNormRopeKernel (fused_norm_rope_v2.cuh).

Two head-dim variants:
  Indexer  (kHeadDim=128): RMSNorm + RoPE + 128-pt WHT + FP8/FP4 → indexer cache
  FlashMLA (kHeadDim=512): RMSNorm + RoPE → FlashMLA paged cache

Two forward modes:
  CompressExtend (is_decode=False): reads CompressPlan
    position  = plan.seq_len - compress_ratio
    out_loc   = out_loc[plan.ragged_id]
    skip when plan.is_invalid()

  CompressDecode (is_decode=True): reads DecodePlan
    position  = plan.seq_len - compress_ratio
    out_loc   = out_loc[token_idx]
    skip when plan.seq_len % compress_ratio != 0

Cache layouts
─────────────
Indexer FP8 page: 132 * page_size bytes
  slot (132 B): [fp8 × 128] + [fp32 scale × 1]
  value_ptr = page_ptr + offset * 128
  scale_ptr = page_ptr + 128*page_size + offset*4

Indexer FP4 page: 68 * page_size bytes
  slot (68 B): [fp4 × 128 = 64 bytes] + [UE8M0 scale × 4]
  value_ptr = page_ptr + offset * 64
  scale_ptr = page_ptr + 64*page_size + offset*4

FlashMLA page: ceil(584*page_size/576)*576 bytes
  slot (576 B): [fp8 nope × 448] + [bf16 rope × 64] = 448+128=576 B
  scale region: page_ptr + 576*page_size + offset*8  (7 UE8M0 bytes)
  value_ptr = page_ptr + offset * 576
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FP8_E4M3_MAX = 448.0
_HEAD_DIM_IDX = 128
_HEAD_DIM_MLA = 512
_ROPE_DIM = 64
_NOPE_DIM_MLA = _HEAD_DIM_MLA - _ROPE_DIM
_NOPE_WARPS_MLA = 7
_EPW_MLA = _NOPE_DIM_MLA // _NOPE_WARPS_MLA  # 64


# ---------------------------------------------------------------------------
# Plan decoders (same binary layout as compress_v2.cuh)
# ---------------------------------------------------------------------------


def _decode_plan_c(plan_c: torch.Tensor):
    """(N,16) uint8 → (seq_len, ragged_id, buffer_len, rp0, rp1)."""
    raw = plan_c.contiguous()
    i32 = raw.view(torch.int32).reshape(-1, 4)
    seq_len = i32[:, 0]
    rp0 = i32[:, 2]
    rp1 = i32[:, 3]
    i16 = raw.view(torch.int16).reshape(-1, 8)
    ragged = i16[:, 2].to(torch.int32) & 0xFFFF
    buf_len = i16[:, 3].to(torch.int32) & 0xFFFF
    return seq_len, ragged, buf_len, rp0, rp1


def _decode_plan_d(plan_d: torch.Tensor):
    """(N,16) uint8 → (seq_len, write_loc, rp0, rp1)."""
    i32 = plan_d.contiguous().view(torch.int32).reshape(-1, 4)
    return i32[:, 0], i32[:, 1], i32[:, 2], i32[:, 3]


# ---------------------------------------------------------------------------
# UE8M0 helpers (vectorized, tensor-based)
# ---------------------------------------------------------------------------


def _cast_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Encode positive fp32 tensor as UE8M0 (extract biased exponent byte)."""
    bits = x.float().clamp(min=1e-38).view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def _inv_scale_ue8m0(ue8m0: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 → inv_scale = 2^(127 - e). Returns float32 tensor."""
    inv_exp = (254 - ue8m0.to(torch.int32)).clamp(min=0)
    inv_bits = (inv_exp << 23).to(torch.int32)
    inv_f = inv_bits.view(torch.float32)
    return torch.where(ue8m0 >= 254, torch.zeros_like(inv_f), inv_f)


# ---------------------------------------------------------------------------
# Generic FP8 E4M3 conversion helpers
# ---------------------------------------------------------------------------


def _fp8_clip(val: torch.Tensor, fp8_max: float) -> torch.Tensor:
    return torch.clamp(val, -fp8_max, fp8_max)


def cvt_float_to_fp8_e4m3(x: torch.Tensor, fnuz: bool = False) -> torch.Tensor:
    """
    Software float -> fp8 e4m3 encoded byte.
    Returns uint8 tensor with fp8 bytes.
    """
    x = x.to(torch.float32)
    fp8_max = 240.0 if fnuz else 448.0
    x = _fp8_clip(x, fp8_max)

    is_zero = x == 0.0
    u = x.view(torch.int32)
    sign = ((u >> 31) & 1).to(torch.uint8) << 7
    exp32 = ((u >> 23) & 0xFF).to(torch.int32) - 127
    mant23 = (u & 0x7FFFFF).to(torch.int32)

    if fnuz:
        bias = 8
        max_exp = 15
        min_sub = -10
        min_norm = -7
        saturate = 0x7F
        sign_if_underflow = torch.zeros_like(sign)
    else:
        bias = 7
        max_exp = 15
        min_sub = -9
        min_norm = -6
        saturate = 0x7E
        sign_if_underflow = sign

    exp8 = torch.zeros_like(exp32)
    mant3 = torch.zeros_like(exp32)

    under = exp32 < min_sub

    sub = (exp32 >= min_sub) & (exp32 < min_norm)
    if sub.any():
        shift = (-(bias - 1) - exp32).to(torch.int32)
        base = (0x800000 | mant23).to(torch.int32)
        subnorm_mant = (base >> (shift + 20)).to(torch.int32)
        round_bit = ((base >> (shift + 19)) & 1).to(torch.int32)
        subnorm_mant = subnorm_mant + round_bit

        mant3_sub = (subnorm_mant & 0x07).to(torch.int32)
        exp8_sub = torch.zeros_like(mant3_sub)

        overflow = subnorm_mant > 7
        exp8_sub = torch.where(overflow, torch.ones_like(exp8_sub), exp8_sub)
        mant3_sub = torch.where(overflow, torch.zeros_like(mant3_sub), mant3_sub)

        exp8 = torch.where(sub, exp8_sub, exp8)
        mant3 = torch.where(sub, mant3_sub, mant3)

    norm = exp32 >= min_norm
    if norm.any():
        exp8_norm = exp32 + bias
        mant3_norm = (mant23 >> 20).to(torch.int32)
        round_bit = ((mant23 >> 19) & 1).to(torch.int32)
        mant3_norm = mant3_norm + round_bit

        mant_over = mant3_norm > 7
        mant3_norm = torch.where(mant_over, torch.zeros_like(mant3_norm), mant3_norm)
        exp8_norm = exp8_norm + mant_over.to(torch.int32)

        sat = exp8_norm >= max_exp
        exp8_norm = torch.where(
            sat, torch.full_like(exp8_norm, (saturate >> 3)), exp8_norm
        )
        mant3_norm = torch.where(
            sat, torch.full_like(mant3_norm, (saturate & 0x7)), mant3_norm
        )

        exp8 = torch.where(norm, exp8_norm, exp8)
        mant3 = torch.where(norm, mant3_norm, mant3)

    out = sign | ((exp8.to(torch.uint8) & 0x1F) << 3) | (mant3.to(torch.uint8) & 0x07)
    out = torch.where(under, sign_if_underflow, out)
    out = torch.where(is_zero, torch.zeros_like(out), out)
    return out


def pack_fp8x2_e4m3(
    x: torch.Tensor, y: torch.Tensor, fnuz: bool = False
) -> torch.Tensor:
    x8 = cvt_float_to_fp8_e4m3(x, fnuz=fnuz).to(torch.uint16)
    y8 = cvt_float_to_fp8_e4m3(y, fnuz=fnuz).to(torch.uint16)
    return x8 | (y8 << 8)


# ---------------------------------------------------------------------------
# FP4 helpers
# ---------------------------------------------------------------------------


def quant_fp4_e2m1_scalar(x: torch.Tensor) -> torch.Tensor:
    ax = torch.clamp(x.abs(), max=6.0)
    idx = torch.zeros_like(ax, dtype=torch.int32)
    idx += (ax > 0.25).to(torch.int32)
    idx += (ax > 0.75).to(torch.int32)
    idx += (ax > 1.25).to(torch.int32)
    idx += (ax > 1.75).to(torch.int32)
    idx += (ax > 2.5).to(torch.int32)
    idx += (ax > 3.5).to(torch.int32)
    idx += (ax > 5.0).to(torch.int32)
    neg = (x < 0.0) & (idx != 0)
    idx = torch.where(neg, idx | 0x8, idx)
    return idx.to(torch.uint8)


# ---------------------------------------------------------------------------
# WHT-128 (vectorized, matches fused_norm_rope_indexer part 3)
# ---------------------------------------------------------------------------


def _fwht128(x: torch.Tensor) -> torch.Tensor:
    """
    Unnormalized 128-point Walsh-Hadamard Transform over last dimension.
    Implements the same iterative butterfly as the CUDA __shfl_xor_sync stages.
    """
    N = x.shape[-1]
    leading = x.shape[:-1]
    h = 1
    while h < N:
        x = x.reshape(*leading, N // (2 * h), 2, h)
        u, v = x[..., 0, :], x[..., 1, :]
        x = torch.stack([u + v, u - v], dim=-2).reshape(*leading, N)
        h *= 2
    return x


# ---------------------------------------------------------------------------
# Page-byte helpers
# ---------------------------------------------------------------------------


def _indexer_page_bytes(page_size: int, use_fp4: bool) -> int:
    return (68 if use_fp4 else 132) * page_size


def _flashmla_page_bytes(page_size: int) -> int:
    return math.ceil(584 * page_size / 576) * 576


# ---------------------------------------------------------------------------
# Indexer variant  (kHeadDim=128)
# ---------------------------------------------------------------------------


def _fused_norm_rope_indexer(
    input: torch.Tensor,  # (N, 128)
    weight: torch.Tensor,  # (128,)
    eps: float,
    freqs_cis: torch.Tensor,  # (max_pos, 64) fp32 interleaved
    active_idx: torch.Tensor,  # (A,) int64
    positions: torch.Tensor,  # (A,) int32
    out_locs: torch.Tensor,  # (A,) int32
    kvcache: torch.Tensor,  # (npages, page_bytes) uint8
    page_size: int,
    use_fp4: bool,
) -> None:
    """
    Fused RMSNorm + RoPE + 128-pt WHT + FP8/FP4 quantisation → indexer cache.
    """
    if active_idx.numel() == 0:
        return

    A = active_idx.numel()
    head_dim = 128
    rope_dim = 64
    nope_dim = head_dim - rope_dim
    page_bits = int(math.log2(page_size))

    # ---- Part 1: RMSNorm with weight ----------------------------------------
    x = input[active_idx].float()
    rms = x.pow(2).mean(dim=-1, keepdim=True)
    norm_factor = torch.rsqrt(rms + eps)
    x = x * norm_factor * weight.float().unsqueeze(0)

    # ---- Part 2: RoPE on last rope_dim=64 elements --------------------------
    freq = freqs_cis[positions.long()]
    freq_re = freq[:, 0::2]
    freq_im = freq[:, 1::2]

    x_nope = x[:, :nope_dim]
    x_rope = x[:, nope_dim:]
    xr = x_rope[:, 0::2]
    xi = x_rope[:, 1::2]
    rot_re = xr * freq_re - xi * freq_im
    rot_im = xr * freq_im + xi * freq_re
    rotated = torch.stack([rot_re, rot_im], dim=-1).flatten(-2)
    x = torch.cat([x_nope, rotated], dim=-1)

    # ---- Part 3: 128-pt Walsh-Hadamard Transform ----------------------------
    x = _fwht128(x) / math.sqrt(head_dim)

    flat_cache = kvcache.view(-1)
    out_locs_l = out_locs.long()
    pages = out_locs_l >> page_bits
    offsets = out_locs_l & (page_size - 1)
    page_bytes = _indexer_page_bytes(page_size, use_fp4)

    if not use_fp4:
        # ---- FP8 path -------------------------------------------------------
        abs_max = x.abs().amax(dim=-1, keepdim=True)
        scale = abs_max.clamp(min=1e-4) / _FP8_E4M3_MAX

        fp8_supported = hasattr(torch, "float8_e4m3fn")
        if fp8_supported:
            inv_scale = 1.0 / scale
            fp8_vals = (
                (x * inv_scale)
                .clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
                .to(torch.float8_e4m3fn)
            )
            fp8_bytes = fp8_vals.view(torch.uint8)
        else:
            ue8m0 = _cast_to_ue8m0(scale.squeeze(-1))
            inv_scale = _inv_scale_ue8m0(ue8m0).unsqueeze(-1)
            fp8_bytes = cvt_float_to_fp8_e4m3(x * inv_scale)

        scale_f32 = scale.squeeze(1).float()

        value_base = pages * page_bytes + offsets * 128
        scale_base = pages * page_bytes + 128 * page_size + offsets * 4

        fp8_cols = torch.arange(128, device=kvcache.device)
        fp8_idx = value_base.unsqueeze(1) + fp8_cols.unsqueeze(0)
        flat_cache.index_put_((fp8_idx.reshape(-1),), fp8_bytes.reshape(-1))

        scale_bytes = scale_f32.view(torch.uint8).reshape(A, 4)
        sc_cols = torch.arange(4, device=kvcache.device)
        sc_idx = scale_base.unsqueeze(1) + sc_cols.unsqueeze(0)
        flat_cache.index_put_((sc_idx.reshape(-1),), scale_bytes.reshape(-1))
    else:
        # ---- FP4 path -------------------------------------------------------
        xg = x.reshape(A, 4, 32)
        abs_max = xg.abs().amax(dim=-1)
        scale = abs_max.clamp(min=1e-4) / 6.0
        ue8m0 = _cast_to_ue8m0(scale)
        inv_scale = _inv_scale_ue8m0(ue8m0)
        q4 = quant_fp4_e2m1_scalar((xg * inv_scale.unsqueeze(-1)).reshape(A, 128))

        lo = q4[:, 0::2] & 0xF
        hi = (q4[:, 1::2] & 0xF) << 4
        packed = (lo | hi).contiguous()  # (A, 64)

        value_base = pages * page_bytes + offsets * 64
        scale_base = pages * page_bytes + 64 * page_size + offsets * 4

        val_cols = torch.arange(64, device=kvcache.device)
        val_idx = value_base.unsqueeze(1) + val_cols.unsqueeze(0)
        flat_cache.index_put_((val_idx.reshape(-1),), packed.reshape(-1))

        sc_cols = torch.arange(4, device=kvcache.device)
        sc_idx = scale_base.unsqueeze(1) + sc_cols.unsqueeze(0)
        flat_cache.index_put_((sc_idx.reshape(-1),), ue8m0.reshape(-1))


# ---------------------------------------------------------------------------
# FlashMLA variant  (kHeadDim=512)
# ---------------------------------------------------------------------------


def _fused_norm_rope_flashmla(
    input: torch.Tensor,  # (N, 512)
    weight: torch.Tensor,  # (512,)
    eps: float,
    freqs_cis: torch.Tensor,  # (max_pos, 64) fp32 interleaved
    active_idx: torch.Tensor,  # (A,) int64
    positions: torch.Tensor,  # (A,) int32
    out_locs: torch.Tensor,  # (A,) int32
    kvcache: torch.Tensor,  # (npages, page_bytes) uint8
    page_size: int,
) -> None:
    """
    Fused RMSNorm + RoPE → FlashMLA paged cache.
    Matches fused_norm_rope_flashmla kernel parts 1-2.
    Slot layout (576 B): [fp8 nope × 448 | bf16 rope × 64 (=128 B)]
    Scale region: page_ptr + 576*page_size + offset*8  (7 UE8M0 bytes)
    """
    if active_idx.numel() == 0:
        return

    A = active_idx.numel()
    head_dim = 512
    rope_dim = 64
    nope_dim = head_dim - rope_dim
    page_bits = int(math.log2(page_size))
    page_bytes = _flashmla_page_bytes(page_size)

    # ---- Part 1: RMSNorm with weight ----------------------------------------
    x = input[active_idx].float()
    rms = x.pow(2).mean(dim=-1, keepdim=True)
    norm_factor = torch.rsqrt(rms + eps)
    x = x * norm_factor * weight.float().unsqueeze(0)

    # ---- Part 2a: RoPE on last rope_dim=64 elements -------------------------
    freq = freqs_cis[positions.long()]
    freq_re = freq[:, 0::2]
    freq_im = freq[:, 1::2]

    x_nope = x[:, :nope_dim]
    x_rope = x[:, nope_dim:]
    xr = x_rope[:, 0::2]
    xi = x_rope[:, 1::2]
    rot_re = xr * freq_re - xi * freq_im
    rot_im = xr * freq_im + xi * freq_re
    rope_out = torch.stack([rot_re, rot_im], dim=-1).flatten(-2)

    # ---- Part 2b: per-warp FP8 quant for nope region ------------------------
    x_warps = x_nope.reshape(A, _NOPE_WARPS_MLA, _EPW_MLA)
    abs_max = x_warps.abs().amax(dim=-1)
    scale_raw = abs_max.clamp(min=1e-4) / _FP8_E4M3_MAX
    ue8m0 = _cast_to_ue8m0(scale_raw)
    inv_sc = _inv_scale_ue8m0(ue8m0)

    fp8_supported = hasattr(torch, "float8_e4m3fn")
    if fp8_supported:
        fp8_nope = (
            (x_warps * inv_sc.unsqueeze(-1))
            .clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
        fp8_bytes = fp8_nope.reshape(A, nope_dim).view(torch.uint8)
    else:
        fp8_bytes = cvt_float_to_fp8_e4m3(
            (x_warps * inv_sc.unsqueeze(-1)).reshape(A, nope_dim)
        )

    # ---- Part 2c: scatter into kvcache --------------------------------------
    flat_cache = kvcache.view(-1)
    out_locs_l = out_locs.long()
    pages = out_locs_l >> page_bits
    offsets = out_locs_l & (page_size - 1)

    value_base = pages * page_bytes + offsets * 576
    scale_base = pages * page_bytes + 576 * page_size + offsets * 8

    nope_cols = torch.arange(nope_dim, device=kvcache.device)
    nope_idx = value_base.unsqueeze(1) + nope_cols.unsqueeze(0)
    flat_cache.index_put_((nope_idx.reshape(-1),), fp8_bytes.reshape(-1))

    rope_bf16 = rope_out.to(torch.bfloat16)
    rope_bytes = rope_bf16.view(torch.uint8)
    rope_cols = torch.arange(128, device=kvcache.device)
    rope_idx = (value_base + nope_dim).unsqueeze(1) + rope_cols.unsqueeze(0)
    flat_cache.index_put_((rope_idx.reshape(-1),), rope_bytes.reshape(-1))

    scale_cols = torch.arange(_NOPE_WARPS_MLA, device=kvcache.device)
    scale_idx = scale_base.unsqueeze(1) + scale_cols.unsqueeze(0)
    flat_cache.index_put_((scale_idx.reshape(-1),), ue8m0.reshape(-1))


# ---------------------------------------------------------------------------
# Public entry point: FusedNormRopeKernel::forward
# ---------------------------------------------------------------------------


def compress_norm_rope_store_torch(
    input: torch.Tensor,  # (N, head_dim)
    plan: torch.Tensor,  # (N, 16) uint8
    weight: torch.Tensor,  # (head_dim,)
    eps: float,
    freqs_cis: torch.Tensor,  # (max_pos, rope_dim) fp32 interleaved
    out_loc: torch.Tensor,  # (M,) int32
    kvcache: torch.Tensor,  # (npages, page_bytes) uint8
    is_decode: bool,
    compress_ratio: int,
    page_size: int,
    use_fp4: bool,
) -> None:
    """
    Drop-in for FusedNormRopeKernel::forward.

    Dispatches to the Indexer (head_dim=128) or FlashMLA (head_dim=512) path.
    Matches both CompressExtend (is_decode=False) and CompressDecode (is_decode=True).
    """
    N = input.shape[0]
    head_dim = input.shape[1]
    assert head_dim in (
        _HEAD_DIM_IDX,
        _HEAD_DIM_MLA,
    ), f"head_dim must be 128 or 512, got {head_dim}"
    assert freqs_cis.shape[-1] == _ROPE_DIM, f"freqs_cis last dim must be {_ROPE_DIM}"
    assert (page_size & (page_size - 1)) == 0, "page_size must be a power of 2"

    if use_fp4:
        assert head_dim == _HEAD_DIM_IDX, "FP4 is only supported for head_dim=128"

    if N == 0:
        return

    # ---- Resolve active tokens and their positions/out_locs ----------------
    if not is_decode:
        seq_len_c, ragged_c, _, _, _ = _decode_plan_c(plan)

        invalid = (seq_len_c.to(torch.int64) & 0xFFFF_FFFF) == 0xFFFF_FFFF
        active = (~invalid).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return

        seq_a = seq_len_c[active].to(torch.int64) & 0xFFFF_FFFF
        pos_a = (seq_a - compress_ratio).to(torch.int32)
        ragged_a = ragged_c[active].long()
        out_locs = out_loc[ragged_a].int()
    else:
        seq_len_d, _, _, _ = _decode_plan_d(plan)

        seq_u32 = seq_len_d.to(torch.int64) & 0xFFFF_FFFF
        active = (seq_u32 % compress_ratio == 0).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return

        seq_a = seq_u32[active]
        pos_a = (seq_a - compress_ratio).to(torch.int32)
        out_locs = out_loc[active].int()

    # ---- Dispatch to head-dim variant --------------------------------------
    if head_dim == _HEAD_DIM_IDX:
        _fused_norm_rope_indexer(
            input,
            weight,
            eps,
            freqs_cis,
            active,
            pos_a,
            out_locs,
            kvcache,
            page_size,
            use_fp4,
        )
    else:
        _fused_norm_rope_flashmla(
            input,
            weight,
            eps,
            freqs_cis,
            active,
            pos_a,
            out_locs,
            kvcache,
            page_size,
        )
