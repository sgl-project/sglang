# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This backward pass is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.
# Modified by NVIDIA: add static FP8 output after activation-dtype rounding.
# Modified by NVIDIA: fuse gated RMS normalization with static FP8 quantization.

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.layernorm_gated import calc_rows_per_block


@triton.jit
def _gated_rmsnorm_static_fp8_kernel(
    X,
    Z,
    W,
    Scale,
    Out,
    M,
    H: tl.constexpr,
    SZT: tl.constexpr,
    SZH: tl.constexpr,
    EPS,
    BR: tl.constexpr,
    COLS_PER_THREAD: tl.constexpr,
):
    rows = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    # Keep the BF16 reference lane width for the variance reduction even
    # though the output stores use FP8 elements.
    cols = tl.max_contiguous(tl.arange(0, 128), COLS_PER_THREAD)
    mask = rows[:, None] < M
    x = tl.load(X + rows[:, None] * 128 + cols[None, :], mask, 0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / 128
    rstd = tl.rsqrt(var + EPS)
    weight = tl.load(W + cols).to(tl.float32)
    z = tl.load(
        Z + (rows[:, None] // H) * SZT + (rows[:, None] % H) * SZH + cols[None, :],
        mask,
        0,
    ).to(tl.float32)
    y = x * rstd[:, None] * weight[None, :]
    y *= z * tl.sigmoid(z)
    # Match the materialized BF16 norm output before checkpoint-scale FP8.
    y = y.to(tl.bfloat16).to(tl.float32)
    scale = tl.load(Scale).to(tl.float32)
    quant = tl.clamp(y * (1.0 / scale), -448.0, 448.0)
    tl.store(Out + rows[:, None] * 128 + cols[None, :], quant, mask)


def gated_rmsnorm_static_fp8(x, z, weight, input_scale, eps=1e-6, out=None):
    """Gated 128-wide RMSNorm followed by static e4m3fn quantization.

    x is contiguous BF16 [T,H,128] or [T*H,128]. z is BF16 [T,H,128]
    with unit last stride; token/head strides may describe a packed projection.
    Weight is BF16/FP32 [128]. input_scale is one FP32 device element, preserved
    unchanged. The result is contiguous e4m3fn [T,H*128]. Normalization precedes
    SiLU gating and its BF16 rounding precedes static FP8 quantization.
    """
    if (
        z.ndim != 3
        or z.shape[-1] != 128
        or z.stride(-1) != 1
        or x.ndim not in (2, 3)
        or x.shape[-1] != 128
        or not x.is_contiguous()
        or x.numel() != z.numel()
        or (x.ndim == 3 and x.shape != z.shape)
    ):
        raise ValueError("Expected contiguous x rows and strided z[T,H,128]")
    if x.dtype != torch.bfloat16 or z.dtype != torch.bfloat16:
        raise ValueError("Gated RMSNorm requires BF16 x and z")
    if (
        weight.shape != (128,)
        or not weight.is_contiguous()
        or weight.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("Expected contiguous BF16/FP32 weight[128]")
    if input_scale.numel() != 1 or input_scale.dtype != torch.float32:
        raise ValueError("Expected one FP32 static scale")
    tokens, heads, _ = z.shape
    shape = (tokens, heads * 128)
    if out is None:
        out = torch.empty(shape, device=x.device, dtype=torch.float8_e4m3fn)
    elif (
        out.shape != shape
        or out.dtype != torch.float8_e4m3fn
        or not out.is_contiguous()
    ):
        raise ValueError("Expected contiguous e4m3fn output[T,H*128]")
    if tokens * heads:
        reference_rows = calc_rows_per_block(tokens * heads, x.device)
        _gated_rmsnorm_static_fp8_kernel[(triton.cdiv(tokens * heads, 16),)](
            x,
            z,
            weight,
            input_scale,
            out,
            tokens * heads,
            H=heads,
            SZT=z.stride(0),
            SZH=z.stride(1),
            EPS=eps,
            BR=16,
            COLS_PER_THREAD=4 if reference_rows == 1 else 8,
            num_warps=4,
        )
    return out
