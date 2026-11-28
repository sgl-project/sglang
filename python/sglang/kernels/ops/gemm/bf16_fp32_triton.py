"""BF16 x BF16 -> FP32 GEMM whose bits do not depend on operand addresses.

Selected with ``SGLANG_OPT_BF16_FP32_GEMM_ALGO=triton`` (see ``bf16_fp32.py``).

Why: on SM80 (A100/A800) neither HPC-Ops nor DeepGEMM is available, so these GEMMs
go to cuBLAS, whose FP32-output bits change when an operand is not 16-byte aligned
(a view at an odd element offset). This kernel's bits never depend on where the
operands live. ``benchmark/kernels/gemm/repro_cublas_bf16_fp32_sm80.py`` shows both,
with the time per call of each.

This kernel accumulates in FP32 with ``tl.dot``. GEMMs with a long K and few
output tiles are split along K into a fixed number of slices whose partials are
summed in a fixed order by a second kernel; no floating-point atomics. The tile
and split choice (``_config``) depends only on (M, N, K), never on pointers, so
equal inputs give equal bits. It is not batch-invariant: the tiles and split
change with M.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# A K this long is split when the output has too few tiles to fill the GPU.
SPLIT_MIN_K = 2560


def _config(rows: int, columns: int, k_size: int) -> tuple[int, int, int, int]:
    """(BM, BN, BK, split K) from a sweep on A100 over DeepSeek-V4.1 TP4 shapes
    (routers, indexer and compressor projections, LM head) at M = 1-4096."""
    long_k = k_size >= SPLIT_MIN_K
    if columns <= 128:
        if rows > 1024:
            return 32, 64, 128, 1
        return (16 if rows <= 16 else 32), 32, 128, (8 if long_k and rows <= 64 else 1)
    if columns >= 4096 and rows <= 48:  # LM-head-like: enough N tiles already
        return (16, 64, 64, 1) if rows <= 16 else (32, 64, 128, 5 if long_k else 1)
    if rows <= 16:
        return 16, 32, 128, (8 if long_k else 1)
    if rows <= 96:
        return 32, 128, 128, (8 if long_k else 1)
    if rows <= 256:
        return 64, 128, 128, (8 if long_k else 1)
    if rows <= 1024:
        return 128, 64, 64, (4 if long_k else 1)
    return 128, 64, 64, 1


@triton.jit
def _gemm_kernel(
    a_ptr,
    w_ptr,
    out_ptr,
    a_stride,
    w_stride,
    rows,
    columns,
    k_size,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    m_ok, n_ok = m < rows, n < columns
    acc = tl.zeros((BM, BN), tl.float32)
    # Slice s takes K tiles s, s + SPLIT, s + 2 * SPLIT, ...
    for tile in range(tl.program_id(2), tl.cdiv(k_size, BK), SPLIT):
        k = tile * BK + tl.arange(0, BK)
        k_ok = k < k_size
        a = tl.load(
            a_ptr + m[:, None] * a_stride + k[None, :],
            mask=m_ok[:, None] & k_ok[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + n[:, None] * w_stride + k[None, :],
            mask=n_ok[:, None] & k_ok[None, :],
            other=0.0,
        )
        acc += tl.dot(a, tl.trans(w))
    tl.store(
        out_ptr + tl.program_id(2) * rows * columns + m[:, None] * columns + n[None, :],
        acc,
        mask=m_ok[:, None] & n_ok[None, :],
    )


@triton.jit
def _reduce_kernel(part_ptr, out_ptr, size, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = i < size
    acc = tl.zeros((BLOCK,), tl.float32)
    for part in range(0, SPLIT):  # fixed order
        acc += tl.load(part_ptr + part * size + i, mask=ok, other=0.0)
    tl.store(out_ptr + i, acc, mask=ok)


def linear_bf16_fp32_triton(
    x: torch.Tensor, weight: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``x`` [..., K] times ``weight`` [N, K] -> FP32 [..., N].

    ``out`` is used when it is a contiguous FP32 tensor of the result's shape.
    """
    shape = (*x.shape[:-1], weight.shape[0])
    a = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
    if a.stride(1) != 1:
        a = a.contiguous()
    weight = weight.to(torch.bfloat16)
    if weight.stride(1) != 1:
        weight = weight.contiguous()
    rows, k_size = a.shape
    columns = weight.shape[0]
    assert weight.shape[1] == k_size, (x.shape, weight.shape)
    if (
        out is None
        or out.dtype != torch.float32
        or tuple(out.shape) != shape
        or not out.is_contiguous()
    ):
        out = torch.empty(shape, dtype=torch.float32, device=x.device)
    if rows == 0:
        return out
    bm, bn, bk, split = _config(rows, columns, k_size)
    target = out
    if split > 1:
        target = torch.empty(split, rows, columns, dtype=torch.float32, device=x.device)
    _gemm_kernel[(triton.cdiv(rows, bm), triton.cdiv(columns, bn), split)](
        a,
        weight,
        target,
        a.stride(0),
        weight.stride(0),
        rows,
        columns,
        k_size,
        BM=bm,
        BN=bn,
        BK=bk,
        SPLIT=split,
        num_warps=4,
    )
    if split > 1:
        size = rows * columns
        _reduce_kernel[(triton.cdiv(size, 512),)](
            target, out, size, SPLIT=split, BLOCK=512, num_warps=4
        )
    return out
