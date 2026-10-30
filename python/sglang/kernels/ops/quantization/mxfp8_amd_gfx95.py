# SPDX-License-Identifier: Apache-2.0
"""Native MXFP8 (1x32 block, E8M0 scale) quantization for AMD CDNA4 (gfx950).

* per-token MXFP8 activation quant (single fused Triton pass)
* the fp8 grid of the gfx950 native route (``fp8_grid_quant`` and the activation
  wrappers the fused producers emit)
* MiniMax-M3 TP4 FlyDSL path: load-time AITER/FlyDSL preshuffled weights when
  ``--fp8-gemm-backend=aiter``
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

# Per-rank TP4 weight shapes covered by the AITER tune table. Runtime dispatch
# still requires an exact (M, N, K, architecture, CU-count) match.
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
    xq = tl.clamp(x / descale[:, None], -448.0, 448.0).to(xq_ptr.dtype.element_ty)
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


def get_flydsl_mxfp8_config(m: int, n: int, k: int) -> Optional[dict]:
    """Return the paired AITER tune for one exact accepted signature."""
    if (int(n), int(k)) not in MXFP8_FLYDSL_WEIGHT_SHAPES:
        return None
    try:
        from aiter.ops.flydsl import get_mxscale_preshuffle_config
    except ImportError as error:
        raise RuntimeError(
            "AITER FlyDSL MX-scale preshuffle GEMM is unavailable."
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
        raise ValueError(f"FlyDSL MXFP8 scale K ({scale_k}) must be divisible by 8")

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

def flydsl_mxfp8_blockscaled_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
    *,
    config: Optional[dict] = None,
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
        if input_scale.dtype != MXFP8_SCALE_DTYPE or input_scale.shape != (
            m,
            k // MXFP8_BLOCK_SIZE,
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

    selected_config = config if config is not None else get_flydsl_mxfp8_config(m, n, k)
    if selected_config is None:
        raise RuntimeError(f"no accepted FlyDSL MXFP8 config for {(m, n, k)}")

    if output_buffer is None:
        output_buffer = torch.empty((m, n), dtype=output_dtype, device=input.device)
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
        from aiter.ops.flydsl import gemm_a8w8_mxscale_preshuffle_1x32_flydsl
    except ImportError as error:
        raise RuntimeError(
            "AITER FlyDSL MX-scale preshuffle GEMM is unavailable."
        ) from error

    gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
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
