"""Cake ragged BF16 MoE grouped GEMM (forward projection) via FlashInfer.

FlashInfer entries (package ``flashinfer.experimental.cake_moe_grouped_gemm``;
backend ``.cake_backend``, JIT registry ``.cake_jit``; FlashInfer issue #5678):

* ``flashinfer.grouped_mm.grouped_mm_bf16(a, b, m_indptr, out=None,
  out_dtype=bf16, *, backend="cake", tactic=-1)`` -> ``cake_backend.grouped_mm_bf16_cake``
  (stable API opt-in; prepares on every call; warns ``ExperimentalWarning`` once).
* ``cake_moe_grouped_gemm.grouped_gemm_fwd(x, w, offs, out=None)`` -- one-shot
  prepare + launch (``@flashinfer_experimental_api``).
* ``cake_moe_grouped_gemm.prepare_grouped_gemm_fwd(x, w, offs, out=None) ->
  GroupedGemmLaunch`` -- ``launch()`` has no allocation / host sync and is
  CUDA-graph capturable; TMA descriptors are encoded once at prepare (outside
  capture); new ``offs`` values or operand contents may be written into the
  bound tensors between launches (the plan depends only on shapes / SM count).

Contract at FlashInfer ``46340689a5ab`` (sm_100a / sm_103a / sm_107a, compute
capability 10.0 / 10.3 / 10.7 exactly): ``Y[offs[e-1]:offs[e]] =
X[offs[e-1]:offs[e]] @ W[e].T``; ``x`` BF16 ``[sum_m, K]``, ``w`` BF16
``[E, N, K]`` (unit stride along the last dimension), ``offs`` int32 device
``[E]`` cumulative end offsets or ``m_indptr`` ``[E + 1]`` (never read on the
host); ``N % 256 == 0``, ``K % 64 == 0``; ``out`` BF16 ``[sum_m, N]`` with unit
column stride (row padding allowed; the output span must fit 32-bit element
addressing). BF16 output only; ``tactic`` must be ``-1``; bitwise deterministic
(no atomics); rows of ``a`` past ``m_indptr[-1]`` are left untouched.

Not forwarded here: ``grouped_gemm_dgrad`` / ``grouped_gemm_wgrad`` /
``prepare_grouped_gemm_{dgrad,wgrad}`` / ``cake_grouped_mm`` (training only).
Not supported here (keep the existing SGLang path): FP16 / FP32 outputs, FP8 /
FP4 grouped GEMMs, masked (padded-per-expert) layouts, SM90 / SM12x.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from sglang.kernels.cake_kernels._support import (
    SM100,
    SM103,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.experimental.cake_moe_grouped_gemm.cake_backend"
FI_PACKAGE = "flashinfer.experimental.cake_moe_grouped_gemm"
FI_JIT_MODULE = "flashinfer.experimental.cake_moe_grouped_gemm.cake_jit"
FI_STABLE_MODULE = "flashinfer.grouped_mm.core"
SM107 = (10, 7)
ARCHS = (SM100, SM103, SM107)
BLOCK_N = 256
BLOCK_K = 64
INT32_ELEMS = 2147483647


def _fwd_program_registered(device: torch.device) -> bool:
    try:
        from flashinfer.experimental.cake_moe_grouped_gemm.cake_backend import (
            generated_program_available,
        )

        return bool(generated_program_available(device, "fwd"))
    except Exception:
        return False


def supports_grouped_gemm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring the FlashInfer forward contract; never raises.

    ``offs`` may be ``[E]`` end offsets or ``m_indptr`` ``[E + 1]``.
    """
    import torch

    if not (
        flashinfer_module_available(FI_PACKAGE, FI_MODULE, FI_JIT_MODULE)
        and cuda_tensor_on(x, ARCHS)
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and x.ndim == 2
        and w.ndim == 3
        and x.stride(1) == 1
        and w.stride(2) == 1
        and w.device == x.device
        and offs.device == x.device
    ):
        return False
    sum_m, k = (int(v) for v in x.shape)
    num_groups, n = (int(v) for v in w.shape[:2])
    if int(w.shape[2]) != k or n % BLOCK_N or k % BLOCK_K or num_groups < 1:
        return False
    if offs.dtype != torch.int32 or offs.ndim != 1:
        return False
    if offs.numel() not in (num_groups, num_groups + 1):
        return False
    if out is not None:
        if not (
            out.device == x.device
            and tuple(out.shape) == (sum_m, n)
            and out.dtype == torch.bfloat16
            and out.stride(1) == 1
        ):
            return False
        if sum_m * int(out.stride(0)) > INT32_ELEMS:
            return False
    elif sum_m * n > INT32_ELEMS:
        return False
    return _fwd_program_registered(x.device)


def supports_grouped_mm_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    *,
    tactic: int = -1,
) -> bool:
    """Admission check for ``grouped_mm_bf16(backend="cake")``; never raises."""
    import torch

    if out is not None:
        out_dtype = out.dtype
    elif out_dtype is None:
        out_dtype = torch.bfloat16
    if out_dtype != torch.bfloat16 or tactic != -1:
        return False
    return supports_grouped_gemm_fwd(a, b, m_indptr, out)


def grouped_mm_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    *,
    tactic: int = -1,
) -> torch.Tensor:
    """Forward to ``flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")``.

    Prepares on every call; use :func:`prepare_grouped_gemm_fwd` for graphs.
    """
    import torch
    from flashinfer.grouped_mm import grouped_mm_bf16 as fi_grouped_mm_bf16

    if out_dtype is None:
        out_dtype = torch.bfloat16
    return fi_grouped_mm_bf16(
        a, b, m_indptr, out, out_dtype, backend="cake", tactic=tactic
    )


def grouped_gemm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward to ``cake_moe_grouped_gemm.grouped_gemm_fwd``; returns ``out``."""
    from flashinfer.experimental.cake_moe_grouped_gemm import grouped_gemm_fwd as fwd

    return fwd(x, w, offs, out=out)


def prepare_grouped_gemm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> Any:
    """Forward to FlashInfer; returns a ``GroupedGemmLaunch`` (``launch()`` -> ``out``)."""
    from flashinfer.experimental.cake_moe_grouped_gemm.cake_backend import (
        prepare_grouped_gemm_fwd as prepare,
    )

    return prepare(x, w, offs, out=out)


def get_grouped_gemm_launch_class() -> type:
    from flashinfer.experimental.cake_moe_grouped_gemm.cake_backend import (
        GroupedGemmLaunch,
    )

    return GroupedGemmLaunch
