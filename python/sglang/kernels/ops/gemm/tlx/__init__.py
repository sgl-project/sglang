"""gfx950 BF16/FP16 GEMM vendored from ``tlx.ops.mm`` in facebookexperimental/triton.

The kernel imports ``triton.language.extra.tlx``. fbtriton ships it; on stock
ROCm Triton, importing ``utlx_plugin`` from the uTLX wheel registers it
(README.md lists a working pairing). Import is lazy, so callers can probe
``mm_supported()`` and fall back without pulling the kernel in. README.md
covers provenance and re-sync.

Coverage: ``x @ weight.T`` for a contiguous ``[N, K]`` weight and contiguous
``x`` of any leading dims, BF16 or FP16, any M/N/K, on gfx950 (the caller gates
the architecture). Rejected: other dtypes, strided operands, empty inputs, and
BF16 single-row inputs (see ``mm_supported``). Bias is not part of the contract.
"""

from __future__ import annotations

import torch

from ._runtime import load_kernel, require_tlx

_MM_DTYPES = (torch.bfloat16, torch.float16)


def mm_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether ``linear(x, weight)`` handles these operands; host-only, never reads device data."""
    if x.dtype not in _MM_DTYPES or weight.dtype != x.dtype:
        return False
    if x.numel() == 0 or not x.is_contiguous() or not weight.is_contiguous():
        return False
    # mm sends bf16 single rows to a packed GEMV built on tlx.amd_packed_dot2, which uTLX lacks.
    return not (x.dtype == torch.bfloat16 and x.numel() == x.shape[-1])


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` for a ``[N, K]`` weight, over any leading dims of ``x``."""
    mm = load_kernel("sglang.kernels.ops.gemm.tlx.mm_gfx950", "mm")
    out = mm(x.reshape(-1, x.shape[-1]), weight.t())
    return out.view(*x.shape[:-1], weight.shape[0])


__all__ = ["linear", "mm_supported", "require_tlx"]
