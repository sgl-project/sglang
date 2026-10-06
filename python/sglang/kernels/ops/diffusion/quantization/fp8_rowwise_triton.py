# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 dynamic rowwise E4M3 quantization with optional zero row padding."""

import torch
import triton
import triton.language as tl


def can_use_fp8_rowwise(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.ndim == 2
        and x.is_contiguous()
        and 0 < x.shape[1] <= 16384
        and torch.cuda.get_device_capability(x.device) >= (8, 9)
    )


@triton.jit
def _fp8_rowwise_kernel(X, Q, S, M: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row.to(tl.int64) * K + col, (row < M) & (col < K), other=0).to(
        tl.float32
    )
    amax = tl.max(tl.abs(x), 0)
    # aten scalar division multiplies by the rounded FP32 reciprocal.
    scale = tl.maximum(amax * (1.0 / 448.0), 1.0e-12)
    # Tensor/tensor division must retain IEEE rounding at FP8 boundaries.
    q = tl.div_rn(x, scale)
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    # On SM89 Triton's default cast can round through FP16. Convert directly
    # from FP32 to avoid double rounding at E4M3 halfway points.
    bits = tl.inline_asm_elementwise(
        "{ .reg .b16 packed; cvt.rn.satfinite.e4m3x2.f32 packed, $1, $1; cvt.u32.u16 $0, packed; }",
        constraints="=r,f",
        args=[q],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    quantized = bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
    tl.store(Q + row.to(tl.int64) * K + col, quantized, col < K)
    tl.store(S + row, scale)


def fp8_rowwise(x: torch.Tensor, row_alignment: int = 1):
    """Return E4M3 ``[ceil(M/alignment)*alignment, K]`` and FP32 row scales.

    Reproduce the FLUX.3 eager chain, including scale >= 1e-12. Padded
    rows contain zeros with scale 1e-12. No intermediate FP32 tensor or
    padded BF16 copy is materialized.
    """
    if not can_use_fp8_rowwise(x):
        raise ValueError(
            "fp8_rowwise requires contiguous 2D FP32/FP16/BF16 on SM89+, K in [1, 16384]"
        )
    if not isinstance(row_alignment, int) or row_alignment <= 0:
        raise ValueError("row_alignment must be a positive integer")
    m, k = x.shape
    padded_m = triton.cdiv(m, row_alignment) * row_alignment
    q = torch.empty((padded_m, k), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((padded_m,), dtype=torch.float32, device=x.device)
    if padded_m:
        with torch.cuda.device(x.device):
            _fp8_rowwise_kernel[(padded_m,)](
                x,
                q,
                scale,
                m,
                k,
                triton.next_power_of_2(k),
                num_warps=4 if k <= 4096 else 8,
                enable_fp_fusion=False,
            )
    return q, scale
