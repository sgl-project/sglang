"""RMSNorm with an MXFP8 epilogue (CUDA): the bf16 normalized row plus its e4m3 codes and UE8M0 per-32 scales in
FlashInfer's 128x4 swizzled layout: RMSNorm followed by ``mxfp8_quantize(..., is_sf_swizzled_layout=True)``, up to
fp32 summation order. One CTA per row; see ``elementwise/rmsnorm_mxfp8.cuh``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(k: int, early_trigger: bool) -> Module:
    trigger = "true" if early_trigger else "false"
    return load_jit(
        f"rmsnorm_mxfp8_k{k}_{trigger}",
        cuda_files=["elementwise/rmsnorm_mxfp8.cuh"],
        cuda_wrappers=[("run", f"rmsnorm_mxfp8<{k}, {trigger}>")],
        extra_cuda_cflags=["-O3"],
    )


def scale_bytes(num_rows: int, k: int) -> int:
    """Size of the 128x4 scale image for ``num_rows`` rows of width ``k``."""
    return -(-num_rows // 128) * 128 * (k // 32)


def rmsnorm_mxfp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    need_bf16: bool = True,
    early_trigger: bool = False,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """``x`` [M, K] bf16 (K % 256 == 0, K <= 8192) -> ``(y [M, K] bf16 or None, q [M, K] e4m3, scale)``.
    ``early_trigger`` releases the PDL dependents right after the kernel's own PDL wait, for a successor that
    prefetches its weight."""
    m, k = x.shape
    y = torch.empty_like(x) if need_bf16 else None
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(scale_bytes(m, k), dtype=torch.uint8, device=x.device)
    _jit_module(k, early_trigger).run(x, weight, y, q, s, eps)
    return y, q, s
