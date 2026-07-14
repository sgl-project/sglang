# SPDX-License-Identifier: Apache-2.0
"""Native MXFP8 (1x32 block, E8M0 scale) ops for AMD CDNA4 (gfx950).

  * per-token MXFP8 activation quant (single fused Triton pass)
  * dense GEMM via Triton ``tl.dot_scaled`` (consumes FP8 E4M3 weights + E8M0
    block scales directly, no dequant-to-BF16), lowering to the CDNA4 native MX
    matrix-core ops; ``K % 128 != 0`` falls back to dequant + ``F.linear``.

Replaces the FlyDSL ``v_mfma_scale_f32_32x32x64`` dense path with a single
Triton ``dot_scaled`` GEMM: no load-time weight reformat (fp8 + E8M0 are
consumed as-is) and the activation is MXFP8-quantized in one fused pass.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

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


@triton.jit
def _mxfp8_quant_kernel(
    x_ptr,
    xq_ptr,
    s_ptr,
    M,
    K,
    sxm,
    sxk,
    sqm,
    sqk,
    ssm,
    ssk,
    BLOCK_M: tl.constexpr,
):
    """Per-32-block E8M0 scale + FP8-E4M3 quant, one program per ``[BLOCK_M, 32]``."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)  # which 32-element block along K
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_b * 32 + tl.arange(0, 32)
    m_mask = offs_m < M
    x = tl.load(
        x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)  # [BLOCK_M]
    # Round the E8M0 exponent up (ceil(log2(amax / e4m3_max))) so the block amax
    # stays inside the e4m3 range and the full dynamic range is used.
    sb = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
    descale = tl.exp2(sb - 127.0)
    xq = tl.clamp(x / descale[:, None], -448.0, 448.0).to(xq_ptr.dtype.element_ty)
    tl.store(
        xq_ptr + offs_m[:, None] * sqm + offs_k[None, :] * sqk,
        xq,
        mask=m_mask[:, None],
    )
    tl.store(s_ptr + offs_m * ssm + pid_b * ssk, sb.to(tl.uint8), mask=m_mask)


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


def dequant_mxfp8_to_bf16(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize an MXFP8 tensor (fp8 values + UE8M0 scales) to BF16."""
    x_float = x.to(torch.float32)
    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE
    x_blocked = x_float.view(*x.shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)
    descale = torch.exp2(scales.to(torch.float32) - 127.0)
    dequantized = (x_blocked * descale.unsqueeze(-1)).view(*x.shape)
    return dequantized.to(torch.bfloat16)


# --------------------------------------------------------------------------- #
# Dense MXFP8 linear via Triton tl.dot_scaled (CDNA4 native microscaling)
# --------------------------------------------------------------------------- #
@triton.jit
def _mxfp8_linear_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_xsm,
    stride_xsk,
    stride_wn,
    stride_wk,
    stride_wsn,
    stride_wsk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    xs_ptrs = xs_ptr + offs_m[:, None] * stride_xsm + offs_sk[None, :] * stride_xsk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    ws_ptrs = ws_ptr + offs_n[:, None] * stride_wsn + offs_sk[None, :] * stride_wsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
        xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=0)
        ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=0)
        acc += tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3")
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        xs_ptrs += (BLOCK_K // 32) * stride_xsk
        ws_ptrs += (BLOCK_K // 32) * stride_wsk

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        o_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def _run_mxfp8_linear_kernel(
    x_q: torch.Tensor,  # [M, K] fp8 e4m3
    x_scale: torch.Tensor,  # [M, K//32] uint8 (E8M0)
    w: torch.Tensor,  # [N, K] fp8 e4m3
    w_scale: torch.Tensor,  # [N, K//32] uint8 (E8M0)
    out_dtype: torch.dtype,
) -> torch.Tensor:
    M, K = x_q.shape
    N = w.shape[0]
    out = torch.empty((M, N), dtype=out_dtype, device=x_q.device)
    BLOCK_M, BLOCK_K = 64, 128
    if M <= 512 and (K >= 4096 or (N == 6144 and K in (2048, 3072))):
        BLOCK_N, num_warps = 64, 4
    else:
        BLOCK_N, num_warps = 128, 8
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _mxfp8_linear_kernel[grid](
        x_q,
        x_scale,
        w,
        w_scale,
        out,
        M,
        N,
        K,
        x_q.stride(0),
        x_q.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        w.stride(0),
        w.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return out


def _mxfp8_dot_scaled_linear(
    x: torch.Tensor,  # [M, K] bf16/fp16
    w: torch.Tensor,  # [N, K] fp8 e4m3
    w_scale: torch.Tensor,  # [N, K//32] uint8 (E8M0)
) -> torch.Tensor:
    """bf16/fp16 input -> per-token MXFP8 quant -> dot_scaled GEMM."""
    x_q, x_scale = mxfp8_e4m3_quantize(x)
    return _run_mxfp8_linear_kernel(x_q, x_scale, w, w_scale, x.dtype)


def dot_scaled_mxfp8_blockscaled_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Native dense MXFP8 linear (CDNA4 ``tl.dot_scaled``).

    Consumes FP8 E4M3 ``weight`` + canonical 2D UE8M0 ``weight_scale`` [N, K//32]
    directly. Activations are MXFP8-quantized per token inside the kernel path.
    Drop-in for the SGLang ``w8a8_mxfp8_linear`` callable signature.
    """
    assert weight.dtype == torch.float8_e4m3fn, "MXFP8 weight must be FP8 E4M3."
    assert weight_scale.dtype == torch.uint8, "MXFP8 weight_scale must be UE8M0 uint8."
    assert weight_scale.dim() == 2, (
        "dot_scaled MXFP8 linear expects canonical 2D [N, K//32] weight scales, "
        f"got {weight_scale.dim()}D."
    )

    input_2d = input.view(-1, input.shape[-1]).contiguous()
    output_shape = [*input.shape[:-1], weight.shape[0]]
    if output_dtype is None:
        output_dtype = (
            input_2d.dtype
            if input_2d.dtype in (torch.float16, torch.bfloat16, torch.float32)
            else torch.bfloat16
        )

    m, k = input_2d.shape
    n, k_w = weight.shape
    assert k == k_w, f"{k=} does not match {k_w=}"

    if k % 128 == 0:
        if input_scale is None:
            # Quantize the bf16/fp16 activations per token inside the path.
            x_q, x_scale = mxfp8_e4m3_quantize(input_2d)
            kernel_out_dtype = input_2d.dtype
        else:
            # Activations already MXFP8-quantized by a fused upstream op.
            assert (
                input_2d.dtype == MXFP8_VALUE_DTYPE
            ), "pre-quantized input must be FP8 E4M3 when input_scale is given."
            assert input_scale.dtype == torch.uint8 and input_scale.shape == (
                m,
                k // 32,
            ), "input_scale must be UE8M0 uint8 [M, K//32]."
            x_q, x_scale = input_2d, input_scale
            kernel_out_dtype = output_dtype
        out = _run_mxfp8_linear_kernel(
            x_q, x_scale, weight, weight_scale, kernel_out_dtype
        )
    else:
        # dot_scaled tiling needs K % 128 == 0; dequantize fallback otherwise.
        w_bf16 = dequant_mxfp8_to_bf16(weight, weight_scale)
        if input_scale is not None:
            input_2d = dequant_mxfp8_to_bf16(input_2d, input_scale)
        out = F.linear(input_2d.to(w_bf16.dtype), w_bf16).to(output_dtype)

    if bias is not None:
        out = out + bias
    return out.to(output_dtype).view(*output_shape)
