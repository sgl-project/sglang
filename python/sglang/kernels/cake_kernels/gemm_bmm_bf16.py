"""Cake BF16 batched matmul (``bmm_bf16(backend="cake")``) via FlashInfer.

FlashInfer entry: ``flashinfer.gemm.bmm_bf16(A, B, out=None, out_dtype=bf16,
backend="cake")`` (``flashinfer/gemm/gemm_base.py``), admission
``_cake_bmm_bf16_requirement``; the generated dispatcher is loaded by
``get_blackwell_bf16_bmm_module(device, backend="cake")`` from the JIT module
``flashinfer.jit.gemm.cake_blackwell_bf16_bmm``. ``"cake"`` is explicit-only
and never chosen by ``backend="auto"``.

Contract at FlashInfer ``46340689a5ab`` (SM100a / SM103a): BF16 3-D ``A [B, M, K]``
contiguous row-major; ``B [B, K, N]`` the exact column-major view with stride
``(K*N, 1, K)`` (``weight.transpose(-2, -1)`` of a contiguous ``[B, N, K]``);
``N % 8 == 0``; ``K in {64, 256, 1024}``; positive ``B, M, N, K``; ``out``
contiguous row-major BF16 / FP16 / FP32 ``[B, M, N]`` that must not overlap the
inputs; 16-byte aligned tensor data. No workspace; the launch is CUDA-graph
capturable once the module is built.

Not supported here (keep the existing SGLang path): other ``K`` values,
row-major ``B``, 2-D inputs, batch broadcasting, SM90 / SM12x.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.cake_kernels._support import (
    BLACKWELL_DATACENTER,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.gemm.gemm_base"
FI_JIT_MODULE = "flashinfer.jit.gemm.cake_blackwell_bf16_bmm"
ARCHS = BLACKWELL_DATACENTER
SUPPORTED_K = (64, 256, 1024)


def supports_bmm_bf16(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> bool:
    """Admission check mirroring ``_cake_bmm_bf16_requirement``; never raises."""
    import torch

    if out_dtype is None:
        out_dtype = torch.bfloat16 if out is None else out.dtype
    if not (
        flashinfer_module_available(FI_MODULE, FI_JIT_MODULE)
        and cuda_tensor_on(A, ARCHS)
        and A.dtype == torch.bfloat16
        and B.dtype == torch.bfloat16
        and A.ndim == 3
        and B.ndim == 3
        and B.device == A.device
        and out_dtype in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return False
    batch, m, k = (int(v) for v in A.shape)
    if int(B.shape[0]) != batch or int(B.shape[1]) != k:
        return False
    n = int(B.shape[2])
    if min(batch, m, n, k) <= 0 or n % 8 or k not in SUPPORTED_K:
        return False
    if not A.is_contiguous() or A.data_ptr() % 16:
        return False
    if tuple(B.stride()) != (k * n, 1, k) or B.data_ptr() % 16:
        return False
    if out is not None and not (
        out.device == A.device
        and out.dtype == out_dtype
        and tuple(out.shape) == (batch, m, n)
        and out.is_contiguous()
        and out.data_ptr() % 16 == 0
    ):
        return False
    return True


def bmm_bf16(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Forward to ``flashinfer.gemm.bmm_bf16(..., backend="cake")``; returns ``out``."""
    import torch
    from flashinfer.gemm import bmm_bf16 as fi_bmm_bf16

    if out_dtype is None:
        out_dtype = torch.bfloat16 if out is None else out.dtype
    return fi_bmm_bf16(A, B, out, out_dtype, backend="cake")
