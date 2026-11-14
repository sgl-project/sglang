# SPDX-License-Identifier: Apache-2.0
"""Dense MXFP8 linears for AMD CDNA4 (gfx950).

  * dense GEMM via Triton ``tl.dot_scaled`` (consumes FP8 E4M3 weights + E8M0
    block scales directly, no dequant-to-BF16), lowering to the CDNA4 native MX
    matrix-core ops; ``K % 128 != 0`` falls back to dequant + ``F.linear``.
  * the bf16-dequant linear for shapes the native route does not tile.

Replaces the FlyDSL ``v_mfma_scale_f32_32x32x64`` dense path with a single
Triton ``dot_scaled`` GEMM: no load-time weight reformat (fp8 + E8M0 are
consumed as-is) and the activation is MXFP8-quantized in one fused pass.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
    MXFP8_VALUE_DTYPE,
    dequant_mxfp8_to_bf16,
    fake_quant_fp8_activation,
    mxfp8_e4m3_quantize,
)


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


@triton.jit
def _mxfp8_linear_split_k_kernel(
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
    stride_op,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # k-tiles pid_k, pid_k + SPLIT_K, ... accumulate into out[pid_k] of the fp32 [SPLIT_K, M, N] partials
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_sk = pid_k * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)
    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    xs_ptrs = xs_ptr + offs_m[:, None] * stride_xsm + offs_sk[None, :] * stride_xsk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    ws_ptrs = ws_ptr + offs_n[:, None] * stride_wsn + offs_sk[None, :] * stride_wsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(pid_k, tl.cdiv(K, BLOCK_K), SPLIT_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
        xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=0)
        ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=0)
        acc += tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3")
        x_ptrs += SPLIT_K * BLOCK_K * stride_xk
        w_ptrs += SPLIT_K * BLOCK_K * stride_wk
        xs_ptrs += SPLIT_K * (BLOCK_K // 32) * stride_xsk
        ws_ptrs += SPLIT_K * (BLOCK_K // 32) * stride_wsk

    o_ptrs = (
        out_ptr
        + pid_k * stride_op
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on
    )
    tl.store(o_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@cache_once
def _mxfp8_linear_num_cu(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _pow2_floor(value: int) -> int:
    return 1 << (max(1, value).bit_length() - 1)


# K < 4096 (32 k-tiles at BLOCK_K = 128) was not measured; it keeps the single-pass kernel.
_SPLIT_K_MIN_K_TILES = 32
_SPLIT_K_MAX = 32


def _mxfp8_linear_split_k(
    M: int,
    N: int,
    K: int,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
    num_cu: int,
) -> int:
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    # the split changes the fp32 summation order with M, so batch-invariant mode keeps one pass
    if is_batch_invariant_mode_enabled():
        return 1
    base = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    num_k_tiles = triton.cdiv(K, BLOCK_K)
    if not 0 < base < num_cu or num_k_tiles < _SPLIT_K_MIN_K_TILES:
        return 1
    # aim for ~4 workgroups per CU while keeping >= 4 k-tiles per program
    split_k = _pow2_floor((4 * num_cu) // base)
    return min(split_k, _pow2_floor(num_k_tiles // 4), _SPLIT_K_MAX)


def _run_mxfp8_linear_kernel(
    x_q: torch.Tensor,  # [M, K] fp8 e4m3
    x_scale: torch.Tensor,  # [M, K//32] uint8 (E8M0)
    w: torch.Tensor,  # [N, K] fp8 e4m3
    w_scale: torch.Tensor,  # [N, K//32] uint8 (E8M0)
    out_dtype: torch.dtype,
) -> torch.Tensor:
    M, K = x_q.shape
    N = w.shape[0]
    BLOCK_M, BLOCK_K = 64, 128
    if M <= 512 and (K >= 4096 or (N == 6144 and K in (2048, 3072))):
        BLOCK_N, num_warps = 64, 4
    else:
        BLOCK_N, num_warps = 128, 8

    split_k = _mxfp8_linear_split_k(
        M=M,
        N=N,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_cu=_mxfp8_linear_num_cu(x_q.device),
    )
    if split_k > 1:
        partial = torch.empty((split_k, M, N), dtype=torch.float32, device=x_q.device)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), split_k)
        _mxfp8_linear_split_k_kernel[grid](
            x_q,
            x_scale,
            w,
            w_scale,
            partial,
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
            partial.stride(0),
            partial.stride(1),
            partial.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            SPLIT_K=split_k,
            num_warps=num_warps,
        )
        return partial.sum(0).to(out_dtype)

    out = torch.empty((M, N), dtype=out_dtype, device=x_q.device)
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
            assert input_2d.dtype == MXFP8_VALUE_DTYPE, (
                "pre-quantized input must be FP8 E4M3 when input_scale is given."
            )
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


# at one or two rows aiter's wvSpltK (fixed-order split-K) beats hipBLASLt's small-M kernels
_SKINNY_GEMM_MAX_TOKENS = 2


@cache_once
def _skinny_cu_count() -> int:
    try:
        from aiter.jit.utils.chip_info import get_cu_num
    except ImportError:
        return 0
    return int(get_cu_num())


# both operands are exact in bf16, so only the fp32 sum order differs from the MXFP8 route
def bf16_dequant_blockscaled_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_on_fp8_grid: bool = False,
) -> torch.Tensor:
    """weight is the dequantized bf16 weight (dequant_block_fp8_weight_to_bf16).
    input_on_fp8_grid says a fused upstream kernel (rmsnorm_fake_quant_fp8)
    already put input on the fp8 grid, so the fake-quant here is skipped."""
    assert weight.dtype == torch.bfloat16, weight.dtype
    input_2d = input.view(-1, input.shape[-1])
    if input_on_fp8_grid:
        x = input_2d.to(torch.bfloat16)
    else:
        x = fake_quant_fp8_activation(input_2d.to(torch.bfloat16))
    cu_count = _skinny_cu_count()
    if (
        bias is None
        and x.shape[0] <= _SKINNY_GEMM_MAX_TOKENS
        and cu_count > 0
        and weight.is_contiguous()
        and weight.shape[1] % 8 == 0
    ):
        import aiter

        out = torch.empty(
            (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
        )
        aiter.wvSpltK(weight, x, out, x.shape[0], cu_count)
    else:
        out = F.linear(x, weight, bias)
    return out.view(*input.shape[:-1], weight.shape[0])
