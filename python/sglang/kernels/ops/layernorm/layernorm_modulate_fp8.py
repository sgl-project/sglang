# SPDX-License-Identifier: Apache-2.0
"""BF16-rounded LayerNorm modulation with dynamic per-row E4M3 output."""

import torch
import triton
import triton.language as tl

from sglang.kernels.numerics import cuda_rsqrtf, div_rn_f32, round_bf16_to_fp32
from sglang.kernels.ops.layernorm.welford import layernorm_welford


@triton.jit
def _layernorm_modulate_fp8_kernel(
    x_ptr,
    scale_ptr,
    q_ptr,
    s_ptr,
    seq_len,
    n_rows,
    scale_row_stride,
    eps,
    D: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_offs = pid * ROWS + tl.arange(0, ROWS)
    row_mask = row_offs < n_rows
    row_base = row_offs * D

    mean, m2 = layernorm_welford(x_ptr, row_base, row_mask, D, ROWS)

    denom = tl.zeros((ROWS, 1), dtype=tl.float32) + D
    rstd = cuda_rsqrtf(div_rn_f32(m2, denom) + eps)  # (ROWS, 1)

    batch = row_offs // seq_len

    cols = tl.arange(0, BLOCK)
    mask = row_mask[:, None] & (cols < D)[None, :]
    x = tl.load(x_ptr + row_base[:, None] + cols[None, :], mask, 0).to(tl.float32)
    normalized = round_bf16_to_fp32(rstd * (x - mean))
    sc = tl.load(
        scale_ptr + batch[:, None] * scale_row_stride + cols[None, :], mask, 0
    ).to(tl.float32)
    modulated = round_bf16_to_fp32(normalized * round_bf16_to_fp32(1.0 + sc))
    amax = tl.max(tl.where(cols[None, :] < D, tl.abs(modulated), 0.0), 1)
    scale = div_rn_f32(amax, 448.0)
    inverse = tl.where(scale == 0.0, 0.0, div_rn_f32(1.0, scale))
    value = tl.minimum(tl.maximum(modulated * inverse[:, None], -448.0), 448.0)
    bits = tl.inline_asm_elementwise(
        "{ .reg .b16 packed; cvt.rn.satfinite.e4m3x2.f32 packed, $1, $1; cvt.u32.u16 $0, packed; }",
        constraints="=r,f",
        args=[value],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    tl.store(
        q_ptr + row_base[:, None] + cols[None, :],
        bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True),
        mask,
    )
    tl.store(s_ptr + row_offs, scale, row_mask)


def layernorm_modulate_fp8(
    x: torch.Tensor, modulation: torch.Tensor, eps: float = 1e-6
):
    """Return E4M3 [B,S,D] and FP32 [B*S,1] dynamic row scales.

    Matches affine-free BF16 LayerNorm followed by BF16 ``(1 + modulation)``
    and multiplication. Inputs and BF16 intermediates must be finite.
    Supports contiguous BF16 x, D divisible by four up to 8192, CUDA SM89+.
    Modulation is BF16 [B,D], contiguous within each row (row padding allowed).
    Zero rows produce zero scale and payload. FP32 divisions are rounded to
    nearest; E4M3 conversion is direct from FP32, avoiding SM89 double rounding.
    """
    if not (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype is torch.bfloat16
        and x.ndim == 3
        and x.is_contiguous()
        and 0 < x.shape[-1] <= 8192
        and x.shape[-1] % 4 == 0
        and modulation.device == x.device
        and modulation.dtype is torch.bfloat16
        and modulation.shape == (x.shape[0], x.shape[-1])
        and modulation.stride(-1) == 1
        and eps > 0
        and torch.cuda.get_device_capability(x.device) >= (8, 9)
    ):
        raise ValueError(
            "layernorm_modulate_fp8 requires contiguous BF16 [B,S,D], "
            "BF16 [B,D] modulation, D divisible by 4 in [4,8192], "
            "positive eps and CUDA SM89+"
        )
    batch, tokens, hidden = x.shape
    quantized = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((batch * tokens, 1), device=x.device, dtype=torch.float32)
    if batch * tokens:
        with torch.cuda.device(x.device):
            _layernorm_modulate_fp8_kernel[(triton.cdiv(batch * tokens, 2),)](
                x,
                modulation,
                quantized,
                scales,
                tokens,
                batch * tokens,
                modulation.stride(0),
                eps,
                hidden,
                2,
                triton.next_power_of_2(hidden),
                num_warps=4,
                enable_fp_fusion=False,
            )
    return quantized, scales
