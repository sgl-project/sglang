"""Row padding for skinny bf16 GEMMs on SM120, where cuBLAS picks a kernel
4-6x slower than the split-K one below ``M * N * K = 2**21`` elements.
Thresholds measured on an RTX 5090 (``benchmark/kernels/bench_skinny_gemm_pad.py``).
"""

from typing import Callable

import torch

# cuBLAS 13.1 switches to a split-K kernel at M * N * K >= 2**21; re-fit when
# torch's bundled cuBLAS changes.
SM120_SKINNY_GEMM_MIN_ELEMS = 1 << 21
# Below this K the slow kernel is already as fast as the alternative.
SM120_SKINNY_GEMM_MIN_K = 4096
# M == 1 is a GEMV and does not hit the cliff.
SM120_SKINNY_GEMM_MIN_M = 2


def skinny_gemm_pad_rows(
    *,
    m: int,
    n: int,
    k: int,
    min_elems: int = SM120_SKINNY_GEMM_MIN_ELEMS,
    min_k: int = SM120_SKINNY_GEMM_MIN_K,
) -> int:
    """Smallest row count past the cliff for an ``[m, k] x [k, n]`` GEMM, or 0."""
    if m < SM120_SKINNY_GEMM_MIN_M or k < min_k or m * n * k >= min_elems:
        return 0
    return -(-min_elems // (n * k))


def apply_with_padded_rows(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, *, pad_to: int
) -> torch.Tensor:
    m = x.shape[0]
    if not SM120_SKINNY_GEMM_MIN_M <= m < pad_to:
        return fn(x)
    padded = torch.nn.functional.pad(x, (0, 0, 0, pad_to - m))
    return fn(padded)[:m]
