"""Cake MoE router GEMMs (``M in [1, 16]`` tokens) via FlashInfer.

FlashInfer entries (``flashinfer.gemm.routergemm``, re-exported from
``flashinfer.gemm``): ``mm_M1_16_K7168_N128`` (Mistral Large 3, BF16 out),
``mm_M1_16_K7168_N256`` (DeepSeek-V3, FP32 out) and ``mm_M1_16_K6144_N256``
(GLM-MoE-DSA, FP32 out), each ``(mat_a, mat_b, out, launch_with_pdl=True) -> None``.
Their bodies call ``get_router_gemm_module(backend="cake")``, which returns the
Cake module (``flashinfer.jit.cake_router_gemm.run``; custom ops
``flashinfer::cake_{ml3,dsv3,glm_dsa}_router_gemm_op``) on compute capability
10.0 / 10.3 and the DSv3 CUDA module otherwise. This adapter admits only the
Cake route (SM100a / SM103a); the remaining ``_bf16`` / N384 / N896 /
``tinygemm_bf16`` router entries have no Cake branch at this commit.

Contract at FlashInfer ``46340689a5ab``: ``mat_a`` BF16 row-major ``(M, K)``
with ``1 <= M <= 16``; ``mat_b`` BF16 column-major ``(K, N)`` (``stride(0) == 1``,
i.e. ``weight.t()`` of a contiguous ``[N, K]``); ``out`` row-major ``(M, N)`` of
the entry's output dtype, mutated in place; no workspace; ``launch_with_pdl`` is
a runtime flag. The Cake JIT resolves the architecture from the *current* CUDA
device (``torch.cuda.get_device_capability()``), so ``supports_*`` also requires
the operands to live on the current device.

Not supported here (keep the existing SGLang path): other ``(K, N)`` pairs,
``M > 16``, FP32 / FP16 inputs, row-major ``mat_b``, SM90 / SM107 (FlashInfer
serves those with its DSv3 CUDA kernel, not a Cake kernel), SM12x.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.cake_kernels._support import (
    BLACKWELL_DATACENTER,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.gemm.routergemm"
FI_JIT_MODULE = "flashinfer.jit.cake_router_gemm"
ARCHS = BLACKWELL_DATACENTER
MIN_TOKENS = 1
MAX_TOKENS = 16
# entry name -> (hidden_dim K, num_experts N, out dtype name)
ROUTER_SHAPES = {
    "mm_M1_16_K7168_N128": (7168, 128, "bfloat16"),
    "mm_M1_16_K7168_N256": (7168, 256, "float32"),
    "mm_M1_16_K6144_N256": (6144, 256, "float32"),
}


def _supports_router(
    entry: str, mat_a: torch.Tensor, mat_b: torch.Tensor, out: torch.Tensor
) -> bool:
    import torch

    k, n, out_dtype_name = ROUTER_SHAPES[entry]
    out_dtype = getattr(torch, out_dtype_name)
    if not (
        flashinfer_module_available(FI_MODULE, FI_JIT_MODULE)
        and cuda_tensor_on(mat_a, ARCHS)
        and mat_a.ndim == 2
        and mat_b.ndim == 2
        and out.ndim == 2
        and mat_b.device == mat_a.device
        and out.device == mat_a.device
    ):
        return False
    # The Cake JIT compiles for the current device's architecture.
    current = torch.cuda.current_device()
    if mat_a.device.index not in (None, current):
        return False
    m = int(mat_a.shape[0])
    return (
        mat_a.stride(1) == 1
        and out.stride(1) == 1
        and mat_b.stride(0) == 1
        and MIN_TOKENS <= m <= MAX_TOKENS
        and int(mat_a.shape[1]) == k
        and tuple(mat_b.shape) == (k, n)
        and tuple(out.shape) == (m, n)
        and mat_a.dtype == torch.bfloat16
        and mat_b.dtype == torch.bfloat16
        and out.dtype == out_dtype
    )


def supports_mm_m1_16_k7168_n128(
    mat_a: torch.Tensor, mat_b: torch.Tensor, out: torch.Tensor
) -> bool:
    """Admission check mirroring the FlashInfer contract (BF16 out); never raises."""
    return _supports_router("mm_M1_16_K7168_N128", mat_a, mat_b, out)


def supports_mm_m1_16_k7168_n256(
    mat_a: torch.Tensor, mat_b: torch.Tensor, out: torch.Tensor
) -> bool:
    """Admission check mirroring the FlashInfer contract (FP32 out); never raises."""
    return _supports_router("mm_M1_16_K7168_N256", mat_a, mat_b, out)


def supports_mm_m1_16_k6144_n256(
    mat_a: torch.Tensor, mat_b: torch.Tensor, out: torch.Tensor
) -> bool:
    """Admission check mirroring the FlashInfer contract (FP32 out); never raises."""
    return _supports_router("mm_M1_16_K6144_N256", mat_a, mat_b, out)


def mm_m1_16_k7168_n128(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    """Forward to ``flashinfer.gemm.mm_M1_16_K7168_N128``; writes ``out`` in place."""
    from flashinfer.gemm.routergemm import mm_M1_16_K7168_N128

    mm_M1_16_K7168_N128(mat_a, mat_b, out, launch_with_pdl)


def mm_m1_16_k7168_n256(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    """Forward to ``flashinfer.gemm.mm_M1_16_K7168_N256``; writes ``out`` in place."""
    from flashinfer.gemm.routergemm import mm_M1_16_K7168_N256

    mm_M1_16_K7168_N256(mat_a, mat_b, out, launch_with_pdl)


def mm_m1_16_k6144_n256(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    """Forward to ``flashinfer.gemm.mm_M1_16_K6144_N256``; writes ``out`` in place."""
    from flashinfer.gemm.routergemm import mm_M1_16_K6144_N256

    mm_M1_16_K6144_N256(mat_a, mat_b, out, launch_with_pdl)
