# SPDX-License-Identifier: Apache-2.0
"""Weight-only int8 GEMV for a handful of rows.

``y[M, N] = (x[M, K] @ w[N, K]^T) * scale[N]`` with ``w`` in int8, one fp32
scale per output row, and ``x`` in bf16 or fp16.

A speculative draft model runs once per decode step on the few rows of its
verify window. At that batch its linears are a weight-streaming workload: every
step reads each matrix once, and cuBLAS already streams bf16 close to the DRAM
roofline (1.3-1.5 TB/s measured on RTX PRO 6000). What is left is the number of
bytes per parameter. This kernel reads one byte per parameter instead of two
and keeps the activations in half precision, so nothing needs calibration.

One program owns ``BLOCK_N`` output rows, walks ``K`` once in blocks of
``BLOCK_K`` and lets the tensor cores do the ``16 x BLOCK_K x BLOCK_N``
products in fp16 with fp32 accumulation. Rows are padded to 16 because
``tl.dot`` needs at least 16 on every side.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_M = 16

# (BLOCK_N, BLOCK_K, num_warps). Tuned on RTX PRO 6000 (SM120) over the four
# large linears of a 5120-wide draft: (34816, 5120), (5120, 17408),
# (6144, 5120), (5120, 4096). The same tile won on all of them.
_CONFIG = (64, 256, 8)


@triton.jit
def _int8_weight_only_gemv_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    out_ptr,
    M,
    N,
    K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    # Row bases of w, read as the columns of the transposed tile.
    w_col = w_ptr + offs_n[None, :] * K
    x_row = x_ptr + offs_m[:, None] * K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w_col + offs_k[:, None])
        x = tl.load(x_row + offs_k[None, :], mask=mask_m[:, None], other=0.0)
        acc += tl.dot(x.to(tl.float16), w.to(tl.float16))
    scale = tl.load(scale_ptr + offs_n)
    out = acc * scale[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )


def int8_weight_only_gemv_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Whether :func:`int8_weight_only_gemv` serves ``x[M, K] @ w[N, K].T``.
    Callers keep their dense path when this is False."""
    if x.ndim != 2 or w.ndim != 2:
        return False
    block_n, block_k, _ = _CONFIG
    n, k = w.shape
    return (
        x.device.type == "cuda"
        and w.device == x.device
        and x.dtype in (torch.bfloat16, torch.float16)
        and w.dtype == torch.int8
        and x.shape[0] <= MAX_M
        and x.shape[1] == k
        and x.stride(1) == 1
        and w.stride(1) == 1
        and n % block_n == 0
        and k % block_k == 0
    )


def int8_weight_only_gemv(
    x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``(x @ w.T) * scale`` in ``x.dtype``; the caller guards with
    :func:`int8_weight_only_gemv_supported`."""
    m = x.shape[0]
    n, k = w.shape
    block_n, block_k, num_warps = _CONFIG
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    _int8_weight_only_gemv_kernel[(n // block_n,)](
        x,
        w,
        scale,
        out,
        m,
        n,
        k,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_M=MAX_M,
        num_warps=num_warps,
    )
    return out
