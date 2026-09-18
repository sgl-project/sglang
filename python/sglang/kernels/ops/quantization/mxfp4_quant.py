# SPDX-License-Identifier: Apache-2.0
"""Fused MXFP4 quantize + scatter-store kernel for the KV cache.

One triton kernel replaces the eager ``MXFP4KVQuantizeUtil.batched_quantize``
+ indexed-scatter write path: per 32-element block it computes amax -> E8M0
scale -> saturating ties-to-even E2M1 codes -> nibble pack, then scatters the
packed bytes and scale bytes into the paged pool at ``loc`` (data + scale move
together). Slot 0 stays reserved (CUDA-graph padding contract, mirrored from
``reserved_skip_index=0``). No host sync: CUDA-graph safe.

Bit-exactness scope: for bf16/fp16 inputs the kernel is bit-exact with the
eager OCP codec (``MXFP4KVQuantizeUtil``) and the independent oracle
(``mxfp4_quantize_reference``). The scale exponent is extracted from the fp32
amax bit pattern instead of ``floor(log2(amax))``; the two agree for
bf16/fp16-derived amax because the smallest relative mantissa gap (2^-11 for
fp16) is orders of magnitude larger than log2f's ~2^-22 relative error, so
the floor can never flip. fp32 inputs (gap 2^-23) are NOT covered and must
keep using the eager codec -- ``mxfp4_fused_store_supported`` gates this.

NaN/Inf semantics follow OCP MX v1.0 §6.3 exactly like the eager codec:
NaN in block -> scale byte 0xFF and all-zero codes; Inf in block (no NaN) ->
scale byte 254 (2^127) with sign-saturated elements; all-zero block -> scale
byte 0 (2^-127).
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

MXFP4_BLOCK_SIZE = 32
# Kernel-side mirrors: @triton.jit functions may only read tl.constexpr globals.
_MXFP4_BLOCK = tl.constexpr(MXFP4_BLOCK_SIZE)
_E8M0_NAN_BYTE = tl.constexpr(255)
_E2M1_MAX = tl.constexpr(6.0)


@triton.jit
def e8m0_to_f32(sbytes):
    """Bit-exact E8M0 byte -> fp32 scale factor 2^(byte - 127).

    Bit construction instead of ``tl.exp2``: the fp32 pattern ``byte << 23``
    is exact for bytes 1..254, byte 0 (2^-127) needs the subnormal pattern
    0x00400000, and byte 255 (E8M0 NaN) maps to a quiet NaN. The approximate
    ex2 instruction flushes subnormal results (FTZ), which silently zeroed
    the 2^-127 scale; plain fp32 multiplies downstream do not flush.
    """
    b = sbytes.to(tl.int32)
    bits = b << 23
    bits = tl.where(b == 0, 0x00400000, bits)
    bits = tl.where(b == 255, 0x7FC00000, bits)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _mxfp4_quant_store_kernel(
    K,
    V,
    Loc,
    KC,
    VC,
    KSF,
    VSF,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_kc_slot,
    stride_kc_h,
    stride_vc_slot,
    stride_vc_h,
    stride_ksf_slot,
    stride_ksf_h,
    stride_vsf_slot,
    stride_vsf_h,
    NKV: tl.constexpr,
    D: tl.constexpr,
    SF_ACTUAL: tl.constexpr,
    SF: tl.constexpr,
    PADDED: tl.constexpr,
    PACKED_STORE: tl.constexpr,
):
    t = tl.program_id(0)
    is_v = tl.program_id(1)
    h = tl.program_id(2)

    loc = tl.load(Loc + t).to(tl.int64)

    blk = tl.arange(0, SF)
    offs = blk[:, None] * _MXFP4_BLOCK + tl.arange(0, _MXFP4_BLOCK)[None, :]
    load_mask = offs < D

    if is_v == 0:
        x = tl.load(
            K + t * stride_kt + h * stride_kh + offs, mask=load_mask, other=0.0
        ).to(tl.float32)
    else:
        x = tl.load(
            V + t * stride_vt + h * stride_vh + offs, mask=load_mask, other=0.0
        ).to(tl.float32)

    # --- block amax with OCP NaN/Inf classification (bit patterns)
    xb = x.to(tl.int32, bitcast=True)
    abs_bits = xb & 0x7FFFFFFF
    is_nan = abs_bits > 0x7F800000
    is_inf = abs_bits == 0x7F800000
    finite_abs = tl.where(is_nan | is_inf, 0.0, tl.abs(x))
    amax = tl.max(finite_abs, axis=1)
    any_nan = tl.max(is_nan.to(tl.int32), axis=1) != 0
    any_inf = tl.max(is_inf.to(tl.int32), axis=1) != 0

    # scale_exp = floor(log2(amax)) - 2, exact via the fp32 exponent field
    # (see module docstring); zero/subnormal amax -> E8M0 min; inf blocks ->
    # E8M0 max; clamp to [-127, 127].
    amax_bits = amax.to(tl.int32, bitcast=True)
    amax_exp = (amax_bits >> 23) & 0xFF
    scale_exp = tl.where(amax_exp == 0, -127, amax_exp - 129)
    scale_exp = tl.maximum(scale_exp, -127)
    scale_exp = tl.where(any_inf, 127, scale_exp)
    scale_bytes = tl.where(any_nan, _E8M0_NAN_BYTE, scale_exp + 127)

    # scaled = x * 2^-scale_exp. Multiplying by the exact power of two equals
    # the eager codec's division bit-for-bit (both correctly rounded, power-of-
    # two operand), then OCP nan_to_num + saturate to the E2M1 max.
    inv_scale = e8m0_to_f32(127 - scale_exp)
    prod = x * inv_scale[:, None]
    prod = tl.where(prod != prod, 0.0, prod)
    prod = tl.where(prod == float("inf"), _E2M1_MAX, prod)
    prod = tl.where(prod == float("-inf"), -_E2M1_MAX, prod)
    abs_p = tl.minimum(tl.abs(prod), _E2M1_MAX)

    # Saturating round-to-nearest ties-to-even onto (0,.5,1,1.5,2,3,4,6):
    # each branch has uniform value spacing, so rint() on the normalized
    # coordinate reproduces the eager distance-table tie rules exactly
    # (0.25->0, 0.75->2, 1.25->2, 1.75->4, 2.5->4, 3.5->6, 5->6).
    mag_lo = libdevice.rint(abs_p * 2.0)
    mag_mid = 4.0 + libdevice.rint(abs_p - 2.0)
    mag_hi = 6.0 + libdevice.rint((abs_p - 4.0) * 0.5)
    mag = tl.where(abs_p < 2.0, mag_lo, tl.where(abs_p < 4.0, mag_mid, mag_hi))

    sign = (prod.to(tl.int32, bitcast=True) >> 31) & 1
    code = mag.to(tl.int32) | (sign << 3)
    code = tl.where(any_nan[:, None], 0, code)
    code = tl.where(load_mask, code, 0)

    # Nibble pack: low = even-index element, high = odd-index element.
    pairs = tl.reshape(code, (PADDED // 2, 2))
    weights = 1 + 15 * tl.arange(0, 2)
    packed = tl.sum(pairs * weights[None, :], axis=1).to(tl.uint8)

    # Slot 0 is reserved for CUDA-graph padding writes: keep its bytes.
    if loc != 0:
        offs_p = tl.arange(0, PADDED // 2)
        store_mask_p = offs_p < PACKED_STORE
        store_mask_s = blk < SF_ACTUAL
        if is_v == 0:
            tl.store(
                KC + loc * stride_kc_slot + h * stride_kc_h + offs_p,
                packed,
                mask=store_mask_p,
            )
            tl.store(
                KSF + loc * stride_ksf_slot + h * stride_ksf_h + blk,
                scale_bytes.to(tl.uint8),
                mask=store_mask_s,
            )
        else:
            tl.store(
                VC + loc * stride_vc_slot + h * stride_vc_h + offs_p,
                packed,
                mask=store_mask_p,
            )
            tl.store(
                VSF + loc * stride_vsf_slot + h * stride_vsf_h + blk,
                scale_bytes.to(tl.uint8),
                mask=store_mask_s,
            )


_FUSED_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def mxfp4_fused_store_supported(k, v, loc) -> bool:
    """Whether (k, v, loc) can go through the fused kernel bit-exactly.

    Gate: CUDA, bf16/fp16 K/V of identical shape, contiguous last dim, and a
    matching per-token loc vector. Everything else (CPU tests, fp32 inputs,
    exotic strides) falls back to the eager OCP codec.
    """
    return (
        k.is_cuda
        and v.is_cuda
        and loc.is_cuda
        and k.ndim == 3
        and v.shape == k.shape
        and k.dtype in _FUSED_SUPPORTED_DTYPES
        and v.dtype == k.dtype
        and k.stride(2) == 1
        and v.stride(2) == 1
        and loc.ndim == 1
        and loc.shape[0] == k.shape[0]
    )


def quant_store_kv_mxfp4(
    k: torch.Tensor,
    v: torch.Tensor,
    loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_sf: torch.Tensor,
    v_sf: torch.Tensor,
) -> None:
    """Fused quantize + scatter for MXFP4 KV pools.

    Args:
        k / v: (num_tokens, kv_heads, head_dim) bf16/fp16 projections.
        loc: (num_tokens,) physical slot ids (int32/int64); slot 0 is skipped.
        k_cache / v_cache: (slots, kv_heads, ceil(head_dim/2)) uint8 pools.
        k_sf / v_sf: (slots, kv_heads, ceil(head_dim/32)) uint8 (or
            float8_e8m0fnu-viewed) scale pools.
    """
    if not mxfp4_fused_store_supported(k, v, loc):
        raise ValueError("quant_store_kv_mxfp4 called with unsupported inputs")
    num_tokens, nkv, d = k.shape
    if num_tokens == 0:
        return
    sf_actual = (d + MXFP4_BLOCK_SIZE - 1) // MXFP4_BLOCK_SIZE
    sf = triton.next_power_of_2(sf_actual)
    packed_store = (d + 1) // 2

    k_cache = k_cache.view(torch.uint8)
    v_cache = v_cache.view(torch.uint8)
    k_sf = k_sf.view(torch.uint8)
    v_sf = v_sf.view(torch.uint8)

    _mxfp4_quant_store_kernel[(num_tokens, 2, nkv)](
        k,
        v,
        loc,
        k_cache,
        v_cache,
        k_sf,
        v_sf,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        k_sf.stride(0),
        k_sf.stride(1),
        v_sf.stride(0),
        v_sf.stride(1),
        NKV=nkv,
        D=d,
        SF_ACTUAL=sf_actual,
        SF=sf,
        PADDED=sf * MXFP4_BLOCK_SIZE,
        PACKED_STORE=packed_store,
    )
