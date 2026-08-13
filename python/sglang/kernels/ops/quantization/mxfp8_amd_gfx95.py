# SPDX-License-Identifier: Apache-2.0
"""Native MXFP8 (1x32 block, E8M0 scale) ops for AMD CDNA4 (gfx950).

  * per-token MXFP8 activation quant (single fused Triton pass)
  * dense GEMM via Triton ``tl.dot_scaled`` (consumes FP8 E4M3 weights + E8M0
    block scales directly, no dequant-to-BF16), lowering to the CDNA4 native MX
    matrix-core ops; ``K % 128 != 0`` falls back to dequant + ``F.linear``.

The canonical Triton path consumes checkpoint tensors as-is. Exact MiniMax-M3
TP4 signatures may additionally use load-time AITER/FlyDSL preshuffled weights;
unknown signatures and unsupported runtime topologies stay on Triton.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# MXFP8 constants (OCP microscaling: 1x32 block, E8M0 shared scale).
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32
MXFP8_E4M3_MAX = 448.0  # max representable magnitude of float8_e4m3fn

# Per-rank TP4 weight shapes covered by the paired AITER tune table. Runtime
# dispatch still requires an exact (M, N, K, architecture, CU-count) match.
MXFP8_FLYDSL_WEIGHT_SHAPES = frozenset(
    {
        (2304, 6144),  # QKV
        (2560, 6144),  # fused QKV + sparse index
        (6144, 2048),  # attention output
        (6144, 6144),  # dense gate/up
        (6144, 3072),  # dense down
        (1536, 6144),  # shared gate/up
        (6144, 768),  # shared down
    }
)
MXFP8_FLYDSL_M_VALUES = (
    1,
    2,
    4,
    8,
    12,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    8320,
    16384,
)


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


@triton.jit
def _mxfp8_quant_flydsl_scale_kernel(
    x_ptr,
    xq_ptr,
    scale_shuffled_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_qm,
    stride_qk,
    SCALE_K1: tl.constexpr,
    PADDED_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Canonical MXFP8 quant with direct FlyDSL A16W4 scale stores."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_b * 32 + tl.arange(0, 32)
    row_valid = rows < M

    x = tl.load(
        x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xk,
        mask=row_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1.0e-30)
    scale_biased = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    scale_biased = tl.minimum(tl.maximum(scale_biased, 0.0), 254.0)
    descale = tl.exp2(scale_biased - 127.0)
    xq = tl.clamp(x / descale[:, None], -448.0, 448.0).to(
        xq_ptr.dtype.element_ty
    )
    tl.store(
        xq_ptr + rows[:, None] * stride_qm + cols[None, :] * stride_qk,
        xq,
        mask=row_valid[:, None],
    )

    # shuffle_scale_a16w4(src, 1, False):
    # [N1,NPack=2,NLane=16,K1,KPack=2,KLane=4]
    # -> [N1,K1,KLane,NLane,KPack,NPack].
    row_n1 = rows // 32
    row_in_tile = rows % 32
    n_pack = row_in_tile // 16
    n_lane = row_in_tile % 16
    k1 = pid_b // 8
    k_in_tile = pid_b % 8
    k_pack = k_in_tile // 4
    k_lane = k_in_tile % 4
    dst = row_n1 * (SCALE_K1 * 4 * 16 * 2 * 2)
    dst += k1 * (4 * 16 * 2 * 2)
    dst += k_lane * (16 * 2 * 2)
    dst += n_lane * (2 * 2)
    dst += k_pack * 2 + n_pack
    scale_out = tl.where(row_valid, scale_biased, 0.0).to(tl.uint8)
    tl.store(scale_shuffled_ptr + dst, scale_out, mask=rows < PADDED_M)


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


def mxfp8_e4m3_quantize_flydsl(
    x: torch.Tensor,
    scale_shuffled: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize canonically while emitting FlyDSL-ready E8M0 scales."""
    if x.ndim != 2:
        raise ValueError("FlyDSL-layout MXFP8 quant requires a 2D input")
    m, k = x.shape
    if not x.is_cuda or k % 256 != 0:
        raise ValueError("FlyDSL-layout MXFP8 quant requires CUDA and K % 256 == 0")

    x = x.contiguous()
    padded_m = triton.cdiv(m, 32) * 32
    scale_shape = (padded_m, k // MXFP8_BLOCK_SIZE)
    if scale_shuffled is None:
        scale_shuffled = torch.empty(
            scale_shape, dtype=MXFP8_SCALE_DTYPE, device=x.device
        )
    elif (
        scale_shuffled.shape != scale_shape
        or scale_shuffled.dtype != MXFP8_SCALE_DTYPE
        or scale_shuffled.device != x.device
        or not scale_shuffled.is_contiguous()
    ):
        raise ValueError(
            "scale_shuffled must be contiguous uint8 on x.device with "
            f"shape {scale_shape}"
        )

    xq = torch.empty((m, k), dtype=MXFP8_VALUE_DTYPE, device=x.device)
    block_m = 64
    grid = (triton.cdiv(padded_m, block_m), k // MXFP8_BLOCK_SIZE)
    _mxfp8_quant_flydsl_scale_kernel[grid](
        x,
        xq,
        scale_shuffled,
        m,
        k,
        x.stride(0),
        x.stride(1),
        xq.stride(0),
        xq.stride(1),
        SCALE_K1=(k // MXFP8_BLOCK_SIZE) // 8,
        PADDED_M=padded_m,
        BLOCK_M=block_m,
    )
    return xq, scale_shuffled


def mxfp8_e4m3_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token MXFP8 quant -> (fp8 values, [.., K//32] uint8 UE8M0 scales).

    Uses the single fused Triton kernel for the common 2D, ``K % 32 == 0`` case
    (activations); falls back to the torch reference otherwise.
    """
    if x.ndim == 2 and x.shape[-1] % MXFP8_BLOCK_SIZE == 0 and x.is_cuda:
        return _mxfp8_e4m3_quantize_triton(x.contiguous())
    return _mxfp8_e4m3_quantize_torch(x)


def get_flydsl_mxfp8_config(m: int, n: int, k: int) -> Optional[Dict]:
    """Return the paired AITER tune for one exact accepted signature."""
    if (int(n), int(k)) not in MXFP8_FLYDSL_WEIGHT_SHAPES:
        return None
    try:
        from aiter.ops.flydsl import get_mxscale_preshuffle_config
    except ImportError as error:
        raise RuntimeError(
            "AITER/FlyDSL dense MXFP8 requires the paired AITER "
            "v0.1.19.post2 release-port branch."
        ) from error

    return get_mxscale_preshuffle_config(
        int(m), int(n), int(k), a_dtype="fp8", b_dtype="fp8"
    )


def _shuffle_flydsl_activation_scale(
    canonical: torch.Tensor,
    padded: torch.Tensor,
    shuffled: torch.Tensor,
) -> torch.Tensor:
    """Pad canonical [M,K/32] E8M0 scales and write the FlyDSL layout."""
    if (
        canonical.ndim != 2
        or canonical.dtype != MXFP8_SCALE_DTYPE
        or not canonical.is_contiguous()
    ):
        raise ValueError("canonical scale must be contiguous 2D uint8")

    m, scale_k = canonical.shape
    padded_m = (m + 31) // 32 * 32
    expected_shape = (padded_m, scale_k)
    if (
        padded.shape != expected_shape
        or shuffled.shape != expected_shape
        or padded.dtype != MXFP8_SCALE_DTYPE
        or shuffled.dtype != MXFP8_SCALE_DTYPE
        or padded.device != canonical.device
        or shuffled.device != canonical.device
        or not padded.is_contiguous()
        or not shuffled.is_contiguous()
    ):
        raise ValueError(
            "FlyDSL activation-scale buffers must be contiguous uint8 on the "
            f"input device with shape {expected_shape}"
        )
    if scale_k % 8 != 0:
        raise ValueError(
            f"FlyDSL MXFP8 scale K ({scale_k}) must be divisible by 8"
        )

    padded.zero_()
    padded[:m].copy_(canonical)
    n1 = padded_m // 32
    k1 = scale_k // 8
    source = padded.view(1, n1, 2, 16, k1, 2, 4)
    destination = shuffled.view(1, n1, k1, 4, 16, 2, 2)
    destination.copy_(source.permute(0, 1, 4, 6, 3, 5, 2))
    return shuffled


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


def flydsl_mxfp8_blockscaled_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
    *,
    config: Optional[Dict] = None,
    activation_scale_padded: Optional[torch.Tensor] = None,
    activation_scale_shuffled: Optional[torch.Tensor] = None,
    output_buffer: Optional[torch.Tensor] = None,
    splitk_workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run one exact tuned AITER/FlyDSL dense MXFP8 signature.

    ``weight`` and ``weight_scale`` are load-time preshuffled copies. Canonical
    tensors remain owned by the caller for Triton fallback.
    """
    input_2d = input.view(-1, input.shape[-1]).contiguous()
    m, k = input_2d.shape
    n, weight_k = weight.shape
    if k != weight_k:
        raise ValueError(f"input K ({k}) does not match weight K ({weight_k})")
    if weight.dtype != MXFP8_VALUE_DTYPE:
        raise ValueError("FlyDSL MXFP8 weight must be float8_e4m3fn")
    if weight_scale.dtype != MXFP8_SCALE_DTYPE:
        raise ValueError("FlyDSL MXFP8 weight scale must be uint8 E8M0")

    if output_dtype is None:
        output_dtype = (
            input_2d.dtype
            if input_2d.dtype in (torch.float16, torch.bfloat16)
            else torch.bfloat16
        )

    padded_shape = ((m + 31) // 32 * 32, k // MXFP8_BLOCK_SIZE)
    if activation_scale_shuffled is None:
        activation_scale_shuffled = torch.empty(
            padded_shape, dtype=MXFP8_SCALE_DTYPE, device=input.device
        )

    if input_scale is None:
        x_q, activation_scale_shuffled = mxfp8_e4m3_quantize_flydsl(
            input_2d, activation_scale_shuffled
        )
    else:
        if input_2d.dtype != MXFP8_VALUE_DTYPE:
            raise ValueError("pre-quantized FlyDSL input must be FP8 E4M3")
        if (
            input_scale.dtype != MXFP8_SCALE_DTYPE
            or input_scale.shape != (m, k // MXFP8_BLOCK_SIZE)
        ):
            raise ValueError("input_scale must be uint8 E8M0 with shape [M,K/32]")
        x_q = input_2d
        if activation_scale_padded is None:
            activation_scale_padded = torch.empty(
                padded_shape, dtype=MXFP8_SCALE_DTYPE, device=input.device
            )
        _shuffle_flydsl_activation_scale(
            input_scale, activation_scale_padded, activation_scale_shuffled
        )

    selected_config = (
        config if config is not None else get_flydsl_mxfp8_config(m, n, k)
    )
    if selected_config is None:
        raise RuntimeError(f"no accepted FlyDSL MXFP8 config for {(m, n, k)}")

    if output_buffer is None:
        output_buffer = torch.empty(
            (m, n), dtype=output_dtype, device=input.device
        )
    elif (
        output_buffer.shape != (m, n)
        or output_buffer.dtype != output_dtype
        or output_buffer.device != input.device
        or not output_buffer.is_contiguous()
    ):
        raise ValueError(
            f"output_buffer must be contiguous on {input.device} with "
            f"shape {(m, n)} and dtype {output_dtype}"
        )

    split_k = int(selected_config.get("splitK", 1))
    if splitk_workspace is not None and (
        splitk_workspace.shape != (split_k, m, n)
        or splitk_workspace.dtype != torch.float32
        or splitk_workspace.device != input.device
        or not splitk_workspace.is_contiguous()
    ):
        raise ValueError(
            "splitk_workspace must be contiguous float32 on the input device "
            f"with shape {(split_k, m, n)}"
        )

    try:
        from aiter.ops.flydsl import gemm_mxscale_preshuffle
    except ImportError as error:
        raise RuntimeError(
            "AITER/FlyDSL dense MXFP8 requires the paired AITER "
            "v0.1.19.post2 release-port branch."
        ) from error

    gemm_mxscale_preshuffle(
        x_q,
        weight,
        activation_scale_shuffled,
        weight_scale,
        output_buffer,
        a_dtype="fp8",
        b_dtype="fp8",
        config=selected_config,
        require_tuned=True,
        splitk_workspace=splitk_workspace,
    )
    if bias is not None:
        output_buffer.add_(bias)
    return output_buffer.view(*input.shape[:-1], n)
