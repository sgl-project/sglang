from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.utils import is_hip, is_npu

if TYPE_CHECKING:
    import torch

_is_npu = is_npu()
_is_hip = is_hip()

# (n, k) -> the largest num_tokens where the tiny GEMM still beats cuBLAS.
# Doubles as the compile-time max_m: one kernel is built per m up to it.
_K3_TINY_GEMM_MAX_TOKENS = {
    (144, 7168): 16,
    (896, 7168): 8,
    (1536, 128): 12,
}

# gfx950: the tiny kernel only wins up to these token counts; above them the
# AITER tuned GEMM rows beat both it and F.linear.
_K3_ROCM_TUNED_TINY_MAX_TOKENS = {
    (144, 7168): 8,
    (1536, 128): 8,
}


def kimi_k3_tiny_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    import torch

    from .tiny_gemm import tiny_gemm_bf16

    m, k = x.shape
    n, _ = w.shape
    max_num_tokens = _K3_TINY_GEMM_MAX_TOKENS.get((n, k))
    if _is_hip and (n, k) in _K3_ROCM_TUNED_TINY_MAX_TOKENS:
        if 0 < m <= _K3_ROCM_TUNED_TINY_MAX_TOKENS[(n, k)]:
            return tiny_gemm_bf16(x, w, max_m=max_num_tokens)
        if m > 0:
            from aiter.tuned_gemm import tgemm

            return tgemm.mm(x, w, None, otype=x.dtype)
    if not _is_npu and max_num_tokens is not None and 0 < m <= max_num_tokens:
        return tiny_gemm_bf16(x, w, max_m=max_num_tokens)
    return torch.nn.functional.linear(x, w)
