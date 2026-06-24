from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_fp8_gemv_module() -> Module:
    return load_jit(
        "fp8_gemv",
        cuda_files=["gemm/fp8_gemv.cuh"],
        cuda_wrappers=[("fp8_gemv", "fp8_gemv")],
    )


def can_use_fp8_gemv(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor],
) -> bool:
    """Gate for the native M=1 FP8 GEMV fast path.

    Returns True only for the exact contract the kernel implements; anything else
    must fall back to ``fp8_scaled_mm``. The checks mirror the C++ kernel's
    assumptions (column-major exactly-packed B, 16-byte vector loads) so the fast
    path never claims a shape the kernel cannot serve correctly.
    """
    # Scope: bf16 output, no bias (bias / fp16 fall back to fp8_scaled_mm).
    if bias is not None or out_dtype != torch.bfloat16:
        return False
    if a.dim() != 2 or b.dim() != 2:
        return False
    M, K = a.shape[0], a.shape[1]
    N = b.shape[1]
    if M != 1:  # decode-only fast path
        return False
    if b.shape[0] != K:
        return False
    # Exact dtypes (the kernel reinterprets bytes as fp8_e4m3 / fp32 / bf16).
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        return False
    if scale_a.dtype != torch.float32 or scale_b.dtype != torch.float32:
        return False
    # K%16 (uint4 loads along K); N%8 (bf16 output row is 16-byte aligned).
    if K % 16 != 0 or (N * 2) % 16 != 0:
        return False
    # A row-major; B column-major and EXACTLY packed (kernel addresses B + n*K).
    if a.stride(1) != 1:
        return False
    if b.stride(0) != 1 or b.stride(1) != K:
        return False
    # Measured fall-back region: at large K and N the scalar-fp8 GEMV is
    # instruction-bound and the tensor-core path wins, so defer to fp8_scaled_mm.
    if K >= 4096 and N >= 3072:
        return False
    # Base pointers must be 16-byte aligned for the uint4 loads (a strided/sliced
    # view with a non-16-byte storage offset must fall back).
    if a.data_ptr() % 16 != 0 or b.data_ptr() % 16 != 0:
        return False
    # All operands on one CUDA device.
    if not (a.is_cuda and b.is_cuda and scale_a.is_cuda and scale_b.is_cuda):
        return False
    if not (a.device == b.device == scale_a.device == scale_b.device):
        return False
    # Per-token / per-channel scale ranks.
    if scale_a.numel() != M or scale_b.numel() != N:
        return False
    # Ensure the kernel actually compiles on this platform; otherwise fall back.
    try:
        _jit_fp8_gemv_module()
    except Exception:
        return False
    return True


def fp8_gemv(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Native M=1 FP8 GEMV: out[1, N] = (a[1, K] @ b[K, N]) * scale_a * scale_b.

    Caller must have validated the contract via ``can_use_fp8_gemv``.
    """
    M, N = a.shape[0], b.shape[1]
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    module = _jit_fp8_gemv_module()
    module.fp8_gemv(out, a, b, scale_a, scale_b)
    return out
