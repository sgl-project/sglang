# SPDX-License-Identifier: Apache-2.0
"""Native MXFP8 (1x32 block, E8M0 scale) quantization for AMD CDNA4 (gfx950).

* per-token MXFP8 activation quant (single fused Triton pass)
* the fp8 grid of the gfx950 native route (``fp8_grid_quant`` and the activation
  wrappers the fused producers emit).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.mxfp8_quant import _mxfp8_quant_kernel

# MXFP8 constants (OCP microscaling: 1x32 block, E8M0 shared scale).
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32
MXFP8_E4M3_MAX = 448.0  # max representable magnitude of float8_e4m3fn


# --------------------------------------------------------------------------- #
# MXFP8 quantization (per-32-block E8M0 scale + FP8-E4M3 values)
# --------------------------------------------------------------------------- #
def _mxfp8_e4m3_quantize_torch(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Naive (reference) MXFP8 quantization.

    For each block of 32 elements along the last dim, compute a shared E8M0
    scale and quantize each element to float8_e4m3fn. The E8M0 exponent is
    rounded *up* -- ``ceil(log2(amax / e4m3_max)) + 127`` -- so the block amax
    stays inside the e4m3 range (no clipping) and the full dynamic range is
    used, matching ``triton_kernels`` ``downcast_to_mxfp`` (ROUND_UP) and the
    SGLang fp8 quant kernels. Returns ``(values [same shape, fp8], scales
    [..., K//32] u8)``.
    """
    assert x.shape[-1] % MXFP8_BLOCK_SIZE == 0
    orig_shape = x.shape
    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE

    x_fp32 = x.to(torch.float32)
    x_blocked = x_fp32.view(*orig_shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    amax = x_blocked.abs().amax(dim=-1)
    amax = amax.clamp(min=torch.finfo(torch.float32).tiny)
    scale_biased = (torch.ceil(torch.log2(amax / MXFP8_E4M3_MAX)) + 127.0).clamp(0, 254)
    scales_uint8 = scale_biased.to(torch.uint8)

    descale = torch.exp2(scale_biased - 127.0)
    x_scaled = (x_blocked / descale.unsqueeze(-1)).clamp(
        -MXFP8_E4M3_MAX, MXFP8_E4M3_MAX
    )
    x_fp8 = x_scaled.view(orig_shape).to(MXFP8_VALUE_DTYPE)

    scales_uint8 = scales_uint8.view(*orig_shape[:-1], num_blocks)
    return x_fp8, scales_uint8


def _mxfp8_e4m3_quantize_triton(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused 2D MXFP8 quant (row-major [M, K//32] UE8M0 scales)."""
    M, K = x.shape
    x = x.contiguous()
    xq = torch.empty((M, K), dtype=MXFP8_VALUE_DTYPE, device=x.device)
    scales = torch.empty(
        (M, K // MXFP8_BLOCK_SIZE), dtype=MXFP8_SCALE_DTYPE, device=x.device
    )
    BLOCK_M = 64
    grid = (triton.cdiv(M, BLOCK_M), K // MXFP8_BLOCK_SIZE)
    _mxfp8_quant_kernel[grid](
        x,
        xq,
        scales,
        M,
        K,
        x.stride(0),
        x.stride(1),
        xq.stride(0),
        xq.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK_M=BLOCK_M,
    )
    return xq, scales


def mxfp8_e4m3_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token MXFP8 quant -> (fp8 values, [.., K//32] uint8 UE8M0 scales).

    Uses the single fused Triton kernel for the common 2D, ``K % 32 == 0`` case
    (activations); falls back to the torch reference otherwise.
    """
    if x.ndim == 2 and x.shape[-1] % MXFP8_BLOCK_SIZE == 0 and x.is_cuda:
        return _mxfp8_e4m3_quantize_triton(x.contiguous())
    return _mxfp8_e4m3_quantize_torch(x)


def mxfp8_operand(
    x: torch.Tensor, scale: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A dense layer's input as 2D MXFP8: fp8 e4m3 [M, K] and ue8m0 [M, K // 32]. A bf16
    x is quantized here; an fp8 x arrives with its scale.
    """
    x2d = x.reshape(-1, x.shape[-1])
    if scale is None:
        return mxfp8_e4m3_quantize(x2d.to(torch.bfloat16))
    assert x2d.dtype == MXFP8_VALUE_DTYPE, x2d.dtype
    return x2d.contiguous(), scale.reshape(-1, scale.shape[-1]).contiguous()


def dequant_mxfp8_to_bf16(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize an MXFP8 tensor (fp8 values + UE8M0 scales) to BF16."""
    x_float = x.to(torch.float32)
    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE
    x_blocked = x_float.view(*x.shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)
    descale = torch.exp2(scales.to(torch.float32) - 127.0)
    dequantized = (x_blocked * descale.unsqueeze(-1)).view(*x.shape)
    return dequantized.to(torch.bfloat16)


# CUDA's per_token_group_quant amax floor (LOCAL_ABSMAX_ABS); mxfp8_gemv_gfx95.cuh hard-codes it too
_FP8_GRID_AMAX_FLOOR = tl.constexpr(1e-10)


class Fp8GridActivation(msgspec.Struct, frozen=True):
    """A bf16 activation already on the fp8 e4m3 grid with a per-32 ue8m0 scale; the bf16-dequant
    linear skips fake_quant_fp8_activation for it, every other consumer unwraps .x. Not a
    tuple: a legacy (fp8, scale) pair check or a row slice must fail loudly."""

    x: torch.Tensor


class Mxfp8Activation(msgspec.Struct, frozen=True):
    """An activation quantized as the CUDA MXFP8 route quantizes it: q fp8 e4m3 [M, K] and
    scale ue8m0 exponent bytes [M, K // 32]; dequantized it is the Fp8GridActivation of
    the same input."""

    q: torch.Tensor
    scale: torch.Tensor


@triton.jit
def fp8_grid_quant(xg):
    """xg is fp32 [G, 32]; returns (q, exp): the fp8 e4m3 codes of each
    group scaled by the smallest power of two >= amax / 448, and that scale's
    ue8m0 exponent per group (the same rule as fp8_grid_round)."""
    amax = tl.max(tl.abs(xg), axis=1)
    amax = tl.maximum(amax, _FP8_GRID_AMAX_FLOOR)
    # Smallest power of two >= amax / 448, on the IEEE bits (exact at powers of two).
    raw = amax * (1.0 / 448.0)
    bits = raw.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    exp = exp + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    scale = (exp << 23).to(tl.float32, bitcast=True)
    scaled = tl.clamp(xg / scale[:, None], -448.0, 448.0)
    return scaled.to(tl.float8e4nv), exp


@triton.jit
def fp8_grid_round(xg):
    """xg is fp32 [G, 32], one ue8m0 group per row; returns it rounded to the
    fp8 e4m3 grid of its group scale (smallest power of two >= amax / 448)."""
    q, exp = fp8_grid_quant(xg)
    return q.to(tl.float32) * (exp << 23).to(tl.float32, bitcast=True)[:, None]


@triton.jit
def _fp8_grid_quant_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    M,
    K,
    stride_xm,
    stride_om,
    stride_sm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EMIT_FP8: tl.constexpr,
):
    """Per-32 ue8m0 fp8 e4m3 quantization, one program per [BLOCK_M, BLOCK_K]: the fp8
    codes and ue8m0 exponents with EMIT_FP8, else their dequantized values."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    xg = tl.reshape(x, (BLOCK_M * (BLOCK_K // 32), 32))
    if EMIT_FP8:
        q, exp = fp8_grid_quant(xg)
        out = tl.reshape(q, (BLOCK_M, BLOCK_K))
        offs_g = pid_k * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)
        tl.store(
            scale_ptr + offs_m[:, None] * stride_sm + offs_g[None, :],
            tl.reshape(exp, (BLOCK_M, BLOCK_K // 32)).to(tl.uint8),
            mask=m_mask[:, None],
        )
    else:
        out = tl.reshape(fp8_grid_round(xg), (BLOCK_M, BLOCK_K))
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_k[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None],
    )


def _launch_fp8_grid_quant(
    x: torch.Tensor, out: torch.Tensor, scale: Optional[torch.Tensor]
) -> None:
    M, K = x.shape
    if M == 0:
        return
    # per-element op, so the tile never changes the result
    if M >= 1024 and K % 512 == 0:
        BLOCK_M, BLOCK_K, num_warps = 32, 512, 8
    else:
        BLOCK_K = 256 if K % 256 == 0 else (128 if K % 128 == 0 else 32)
        BLOCK_M, num_warps = (16 if M >= 1024 else 8), 4
    grid = (triton.cdiv(M, BLOCK_M), K // BLOCK_K)
    _fp8_grid_quant_kernel[grid](
        x,
        out,
        scale if scale is not None else out,
        M,
        K,
        x.stride(0),
        out.stride(0),
        scale.stride(0) if scale is not None else 0,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        EMIT_FP8=scale is not None,
        num_warps=num_warps,
    )


def fake_quant_fp8_activation(x: torch.Tensor) -> torch.Tensor:
    """bf16 [M, K] -> bf16 [M, K] on the fp8 e4m3 grid with a per-32 ue8m0 scale
    (the same rule as the CUDA path's sglang_per_token_group_quant_fp8(scale_ue8m0=True))."""
    assert x.dim() == 2 and x.shape[-1] % 32 == 0, x.shape
    x = x.contiguous()
    out = torch.empty_like(x)
    _launch_fp8_grid_quant(x, out, None)
    return out


def fp8_grid_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """bf16 [M, K] -> (fp8 e4m3 [M, K], ue8m0 exponents uint8 [M, K // 32]) under the rule
    of fake_quant_fp8_activation, whose output they dequantize to exactly."""
    assert x.dim() == 2 and x.shape[-1] % 32 == 0, x.shape
    x = x.contiguous()
    M, K = x.shape
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((M, K // 32), dtype=torch.uint8, device=x.device)
    _launch_fp8_grid_quant(x, q, scale)
    return q, scale


def dequant_block_fp8_weight_to_bf16(
    weight: torch.Tensor, weight_scale: torch.Tensor, block_size: Sequence[int]
) -> torch.Tensor:
    """fp8 e4m3 [N, K] with fp32 power-of-two block scales [ceil(N / bn), K // bk]
    -> bf16 [N, K], exact (an e4m3 value times a power of two fits bf16)."""
    n, k = weight.shape
    block_n, block_k = block_size
    scale = weight_scale.float().repeat_interleave(block_n, dim=0)[:n]
    scale = scale.repeat_interleave(block_k, dim=1)[:, :k]
    return (weight.float() * scale).to(torch.bfloat16)
