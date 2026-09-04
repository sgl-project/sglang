# SPDX-License-Identifier: Apache-2.0
"""SwiGLU fused with MXFP8 block quantization for the online ``mxfp8`` path.

Reads the fc1 output [rows, 2n] once and writes the e4m3 product (bf16 rounding
after the SiLU and after the product, as eager) together with one E8M0 scale
per 32 columns in the 128x4 swizzled layout cuBLASLt / FlashInfer consume:
``((r // 128) * KP + kb // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 +
kb % 4`` with ``KP = ceil(n / 128)``, rows padded to 128, scales zero where the
padding is. Bit-exact against ``flashinfer.mxfp8_quantize(silu(gate) * up)``;
verified at [1, 256], [333, 512], [2048, 14336], [13000, 28672].
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_mxfp8_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    rows,
    n,
    stride_row,
    KP,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (r < rows)[:, None] & (c < n)[None, :]
    base = x_ptr + r[:, None].to(tl.int64) * stride_row
    gate = tl.load(base + c[None, :], mask=mask, other=0.0).to(tl.float32)
    up = tl.load(base + n + c[None, :], mask=mask, other=0.0).to(tl.float32)
    act = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    prod = (act * up).to(tl.bfloat16).to(tl.float32)
    # one E8M0 scale per 32 columns: exponent of the largest value scaled to e4m3
    blk = tl.reshape(prod, (BLOCK_R, BLOCK_C // 32, 32))
    amax = tl.max(tl.abs(blk), axis=2)
    exp = tl.ceil(tl.log2(tl.maximum(amax, 1e-30) / 448.0))
    exp = tl.where(amax > 0, exp, -127.0)
    exp = tl.minimum(tl.maximum(exp, -127.0), 127.0)
    scale = tl.exp2(exp)
    q = (blk / scale[:, :, None]).to(tl.float8e4nv)
    tl.store(
        q_ptr + r[:, None].to(tl.int64) * n + c[None, :],
        tl.reshape(q, (BLOCK_R, BLOCK_C)),
        mask=mask,
    )
    kb = pid_c * (BLOCK_C // 32) + tl.arange(0, BLOCK_C // 32)
    smask = (r < rows)[:, None] & (kb < tl.cdiv(n, 32))[None, :]
    idx = (
        ((r // 128)[:, None] * KP + (kb // 4)[None, :]) * 512
        + (r % 32)[:, None] * 16
        + ((r % 128) // 32)[:, None] * 4
        + (kb % 4)[None, :]
    )
    tl.store(scale_ptr + idx, (exp + 127.0).to(tl.uint8), mask=smask)


def can_use_silu_mul_mxfp8(hidden: torch.Tensor) -> bool:
    """Row-major bf16 [rows, 2 * n] on CUDA with n a multiple of 32."""
    return (
        hidden.is_cuda
        and hidden.ndim == 2
        and hidden.dtype == torch.bfloat16
        and hidden.stride(-1) == 1
        and hidden.shape[-1] % 64 == 0
        and not torch.compiler.is_compiling()
    )


def silu_mul_mxfp8(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """hidden [rows, 2 * n] bf16 (gate | up) -> (e4m3 [rows, n], E8M0 swizzled
    scales), the ``(input, scales)`` the online mxfp8 linear takes prequantized."""
    if not can_use_silu_mul_mxfp8(hidden):
        raise ValueError(
            "expected a row-major bf16 CUDA [rows, 2 * n] tensor, n % 32 == 0"
        )
    rows, twice = hidden.shape
    n = twice // 2
    kp = -(-(n // 32) // 4)
    rows_pad = -(-rows // 128) * 128
    q = torch.empty(rows, n, dtype=torch.float8_e4m3fn, device=hidden.device)
    scale = torch.zeros(rows_pad * kp * 4, dtype=torch.uint8, device=hidden.device)
    block_r, block_c = 16, 128
    grid = (triton.cdiv(rows, block_r), triton.cdiv(n, block_c))
    with torch.get_device_module().device(hidden.device):
        _silu_mul_mxfp8_kernel[grid](
            hidden,
            q,
            scale,
            rows,
            n,
            hidden.stride(0),
            kp,
            BLOCK_R=block_r,
            BLOCK_C=block_c,
            num_warps=4,
        )
    return q, scale.view(torch.float8_e8m0fnu)


__all__ = ["can_use_silu_mul_mxfp8", "silu_mul_mxfp8"]
