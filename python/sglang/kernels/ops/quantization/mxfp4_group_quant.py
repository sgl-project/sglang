# MXF4 (e2m1) per-token-group-32 activation quantization for the dwdp compact
# grouped-GEMM path (W4A4 experiment, SGLANG_USE_DEEPGEMM_W4A4=1).
#
# Produces the tensor format accepted by deep_gemm's
# m_grouped_fp8_fp4_gemm_nt_contiguous with recipe_a=(1, 32):
#   q : (M, K // 2)   int8   — packed e2m1, low nibble = even element
#   sf: (M, K // 128) int32  — packed ue8m0 exponents, 4 per word, mn-major
#        (storage (K // 128, ceil(M / 4) * 4) contiguous, transposed view)
#
# The e2m1 / ue8m0 coding primitives are shared with the DSA indexer fp4
# K-cache quantizer (kernels/ops/attention/dsv4/fp4_indexer.py).

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    _ceil_ue8m0_exp,
    _fp4_e2m1_code,
)


@triton.jit
def _quant_mxfp4_group32_kernel(
    x_ptr,  # (M, K) bf16 row-major
    q_ptr,  # (M, K // 2) int8 row-major
    sf_ptr,  # (K // 128, M_al) int32, column per row
    sf_row_stride,
    K,
    BLOCK_K: tl.constexpr,  # elements per program, multiple of 128
):
    m = tl.program_id(0)
    kc = tl.program_id(1)

    offs = kc * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs < K
    # int64 row bases: M * K can exceed int32 and wrap into an illegal access.
    x_base = m.to(tl.int64) * K
    q_base = m.to(tl.int64) * (K // 2)
    x = tl.load(x_ptr + x_base + offs, mask=mask, other=0.0).to(tl.float32)

    NG: tl.constexpr = BLOCK_K // 32
    # Per-group (1, 32) amax -> ue8m0 exponent -> scale.
    absx = tl.reshape(tl.abs(x), (NG, 32))
    amax = tl.max(absx, axis=1)
    sf = tl.maximum(amax / 6.0, 1.0e-4)
    exp = _ceil_ue8m0_exp(sf)  # (NG,) int32, 1..254
    scale = (exp << 23).to(tl.float32, bitcast=True)
    scale_b = tl.reshape(tl.broadcast_to(scale[:, None], (NG, 32)), (BLOCK_K,))

    code = _fp4_e2m1_code(x / scale_b)  # (BLOCK_K,) uint8 nibble codes

    # Pack two e2m1 codes per byte: low nibble = even element.
    c2 = tl.reshape(code, (BLOCK_K // 2, 2))
    lo, hi = tl.split(c2)
    packed = (lo & 0x0F) | ((hi & 0x0F) << 4)
    q_offs = kc * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    # Mask the tail block: K need only be a multiple of 128, so the last
    # program can carry fewer than BLOCK_K valid elements.
    tl.store(q_ptr + q_base + q_offs, packed.to(tl.int8), mask=q_offs < (K // 2))

    # Pack four ue8m0 exponents per int32 scale word (byte b = group 4i+b).
    e2d = tl.reshape(exp, (NG // 4, 4))
    sh = tl.arange(0, 4) * 8
    word = tl.sum(e2d << sh[None, :], axis=1)
    w_offs = kc * (NG // 4) + tl.arange(0, NG // 4)
    tl.store(sf_ptr + w_offs * sf_row_stride + m, word, mask=w_offs < (K // 128))


def quant_mxfp4_group32(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize (M, K) bf16 activations to packed e2m1 + packed ue8m0 (1, 32)
    scales in the layout deep_gemm's contiguous fp4 GEMM expects."""
    assert x.dim() == 2
    M, K = x.shape
    assert K % 128 == 0, f"K={K} must be a multiple of 128 for (1,32) fp4 packing"

    q = torch.empty((M, K // 2), device=x.device, dtype=torch.int8)
    kb4 = K // 128
    m_al = triton.cdiv(M, 4) * 4
    # zeros: the kernel only writes m < M, so the padding columns
    # [M, m_al) must not hold garbage — deep_gemm / ep_scatter read the
    # full m_al columns and garbage bytes decode to NaN ue8m0 scales.
    sf_storage = torch.zeros((kb4, m_al), device=x.device, dtype=torch.int32)
    sf = sf_storage.transpose(0, 1)[:M, :]  # (M, kb4), stride(-2) == 1

    BLOCK_K = 1024 if K >= 1024 else K
    grid = (M, triton.cdiv(K, BLOCK_K))
    _quant_mxfp4_group32_kernel[grid](
        x, q, sf_storage, sf_storage.stride(0), K, BLOCK_K=BLOCK_K
    )
    return q, sf


# ---------------------------------------------------------------------------
# v2: hardware-cvt based quantization + fused silu-mul-quant.
# The comparison-chain codegen is replaced by the SM100 `cvt.rn.satfinite
# .e2m1x2.f32` instruction (numeric difference vs the chain is only the sign
# of zero, which UMMA decodes identically).
# ---------------------------------------------------------------------------


@triton.jit
def _fp4_code_hw(x):
    # One fp32 -> one e2m1 nibble via the hardware converter (both cvt inputs
    # are the same value, so the low nibble holds the signed code).
    r = tl.inline_asm_elementwise(
        asm="{\n.reg .b8 byte0;\ncvt.rn.satfinite.e2m1x2.f32 byte0, $1, $1;\ncvt.u32.u8 $0, byte0;\n}",
        constraints="=r,f",
        args=[x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    return (r & 0x0F).to(tl.uint8)


@triton.jit
def _quant_mxfp4_group32_kernel_v2(
    x_ptr,
    q_ptr,
    sf_ptr,
    sf_row_stride,
    K,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)
    kc = tl.program_id(1)
    offs = kc * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs < K
    # int64 row bases: M * K can exceed int32 and wrap into an illegal access.
    x_base = m.to(tl.int64) * K
    q_base = m.to(tl.int64) * (K // 2)
    x = tl.load(x_ptr + x_base + offs, mask=mask, other=0.0).to(tl.float32)

    NG: tl.constexpr = BLOCK_K // 32
    absx = tl.reshape(tl.abs(x), (NG, 32))
    amax = tl.max(absx, axis=1)
    sf = tl.maximum(amax / 6.0, 1.0e-4)
    exp = _ceil_ue8m0_exp(sf)
    scale = (exp << 23).to(tl.float32, bitcast=True)
    scale_b = tl.reshape(tl.broadcast_to(scale[:, None], (NG, 32)), (BLOCK_K,))

    code = _fp4_code_hw(x / scale_b)

    c2 = tl.reshape(code, (BLOCK_K // 2, 2))
    lo, hi = tl.split(c2)
    packed = (lo & 0x0F) | ((hi & 0x0F) << 4)
    q_offs = kc * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    tl.store(q_ptr + q_base + q_offs, packed.to(tl.int8), mask=q_offs < (K // 2))

    e2d = tl.reshape(exp, (NG // 4, 4))
    sh = tl.arange(0, 4) * 8
    word = tl.sum(e2d << sh[None, :], axis=1)
    w_offs = kc * (NG // 4) + tl.arange(0, NG // 4)
    # Mask the tail block: the last program may hold fewer than BLOCK_K
    # valid elements, i.e. fewer than NG // 4 valid scale words.
    tl.store(sf_ptr + w_offs * sf_row_stride + m, word, mask=w_offs < (K // 128))


def quant_mxfp4_group32_v2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    M, K = x.shape
    assert K % 128 == 0
    q = torch.empty((M, K // 2), device=x.device, dtype=torch.int8)
    kb4 = K // 128
    m_al = triton.cdiv(M, 4) * 4
    # zeros, not empty: see the padding-columns note in quant_mxfp4_group32.
    sf_storage = torch.zeros((kb4, m_al), device=x.device, dtype=torch.int32)
    sf = sf_storage.transpose(0, 1)[:M, :]
    BLOCK_K = 2048 if K >= 2048 else K
    grid = (M, triton.cdiv(K, BLOCK_K))
    _quant_mxfp4_group32_kernel_v2[grid](
        x, q, sf_storage, sf_storage.stride(0), K, BLOCK_K=BLOCK_K
    )
    return q, sf


@triton.jit
def _silu_mul_quant_mxfp4_kernel(
    gateup_ptr,  # (T, 2H) bf16: [0, H) = gate, [H, 2H) = up
    q_ptr,  # (T, H // 2) int8
    sf_ptr,  # (H // 128, T_al) int32 storage
    sf_row_stride,
    H,
    SWIGLU_LIMIT: tl.constexpr,  # 0 disables the clamp
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)
    hc = tl.program_id(1)
    offs = hc * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H
    base = t * (2 * H).to(tl.int64)
    gate = tl.load(gateup_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gateup_ptr + base + H + offs, mask=mask, other=0.0).to(tl.float32)
    if SWIGLU_LIMIT > 0:
        gate = tl.minimum(gate, SWIGLU_LIMIT)
        up = tl.minimum(tl.maximum(up, -SWIGLU_LIMIT), SWIGLU_LIMIT)
    act = gate * tl.sigmoid(gate) * up

    NG: tl.constexpr = BLOCK_H // 32
    absa = tl.reshape(tl.abs(act), (NG, 32))
    amax = tl.max(absa, axis=1)
    sf = tl.maximum(amax / 6.0, 1.0e-4)
    exp = _ceil_ue8m0_exp(sf)
    scale = (exp << 23).to(tl.float32, bitcast=True)
    scale_b = tl.reshape(tl.broadcast_to(scale[:, None], (NG, 32)), (BLOCK_H,))

    code = _fp4_code_hw(act / scale_b)

    c2 = tl.reshape(code, (BLOCK_H // 2, 2))
    lo, hi = tl.split(c2)
    packed = (lo & 0x0F) | ((hi & 0x0F) << 4)
    q_offs = hc * (BLOCK_H // 2) + tl.arange(0, BLOCK_H // 2)
    # int64 row base, same as gateup's: T * (H // 2) can exceed int32.
    tl.store(
        q_ptr + t.to(tl.int64) * (H // 2) + q_offs,
        packed.to(tl.int8),
        mask=q_offs < (H // 2),
    )

    e2d = tl.reshape(exp, (NG // 4, 4))
    sh = tl.arange(0, 4) * 8
    word = tl.sum(e2d << sh[None, :], axis=1)
    w_offs = hc * (NG // 4) + tl.arange(0, NG // 4)
    # Mask the tail block: the last hc program may hold fewer than BLOCK_H
    # valid elements, i.e. fewer than NG // 4 valid scale words.
    tl.store(sf_ptr + w_offs * sf_row_stride + t, word, mask=w_offs < (H // 128))


def silu_mul_quant_mxfp4(
    gateup: torch.Tensor, swiglu_limit: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused SiLU-mul + (1, 32) e2m1/ue8m0 quantization of the MoE gateup
    output, matching the two-step (_legacy_silu_and_mul + quant_mxfp4_group32)
    semantics. swiglu_limit=10 applies the DSV4 gate/up clamps."""
    T, N = gateup.shape
    H = N // 2
    assert H % 128 == 0, f"H={H} must be a multiple of 128"
    q = torch.empty((T, H // 2), device=gateup.device, dtype=torch.int8)
    kb4 = H // 128
    t_al = triton.cdiv(T, 4) * 4
    # zeros, not empty: see the padding-columns note in quant_mxfp4_group32.
    sf_storage = torch.zeros((kb4, t_al), device=gateup.device, dtype=torch.int32)
    sf = sf_storage.transpose(0, 1)[:T, :]
    # Keep the float precision: int() would silently truncate e.g. 10.5 -> 10
    # and diverge from the non-fused activation semantics.
    limit = float(swiglu_limit) if swiglu_limit is not None else 0
    BLOCK_H = 2048 if H >= 2048 else H
    grid = (T, triton.cdiv(H, BLOCK_H))
    _silu_mul_quant_mxfp4_kernel[grid](
        gateup,
        q,
        sf_storage,
        sf_storage.stride(0),
        H,
        SWIGLU_LIMIT=limit,
        BLOCK_H=BLOCK_H,
    )
    return q, sf
