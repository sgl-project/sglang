"""Cake contiguous grouped FP8 GEMM (plain and fused SwiGLU + FP8 quant) via FlashInfer.

FlashInfer entries: ``flashinfer.gemm.prepare_group_gemm_fp8_nt_groupwise_contiguous``
(module ``flashinfer.gemm.cake_grouped_fp8_gemm``, JIT registry
``flashinfer.jit.gemm.cake_grouped_fp8_gemm``) and
``flashinfer.gemm.prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant``
(module ``flashinfer.gemm.cake_grouped_fp8_fused_silu_quant``, JIT registry
``flashinfer.jit.gemm.cake_grouped_fp8_fused_silu_quant``). Both are
prepared-runner factories: the returned object binds tensor *storage* (shapes,
dtypes, addresses) and ``launch()`` submits the kernel(s) on the current stream;
tensor *contents* may change between launches.

Contract at FlashInfer ``46340689a5ab`` (SM100a, compute capability 10.0 only;
SM103/SM120 are rejected by ``SUPPORTED_COMPUTE_CAPABILITIES``):

* plain GEMM: ``a`` E4M3 ``(M, K)`` with ``K % 128 == 0``; ``b`` E4M3
  ``(G, N, K)`` with ``N % 128 == 0``; ``a_scale`` FP32 ``(M, K/128)``;
  ``b_scale`` FP32 ``(G, N/128, K/128)``; ``m_indices`` int32 ``(M,)`` sorted
  nondecreasing (boundaries at any row, empty experts allowed); ``out`` BF16
  ``(M, N)`` (a 2-byte but not 16-byte aligned ``out`` selects the scalar-store
  route). All other tensors 16-byte aligned and contiguous. Same math as
  ``group_gemm_fp8_nt_groupwise_contiguous``.
* fused gate_up GEMM + SwiGLU + per-128-column FP8 quant: ``a`` E4M3 ``(M, K)``
  with ``0 < M <= 8192`` and ``K % 512 == 0``; ``b`` E4M3 ``(G, 2H, K)`` (gate
  rows ``[0, H)`` then up) with ``2H % 256 == 0``; scales as above; every
  *internal* expert boundary of ``m_indices`` must be a multiple of 128 rows
  (only checked with ``validate_indices=True``); outputs ``out_q`` E4M3
  ``(M, H)`` and ``out_s`` FP32 ``(M, H/128)`` = ``max(absmax, 1e-10) / 448``.
  Bitwise equal to CuTe grouped GEMM + ``silu_and_mul`` +
  ``per_token_group_quant_8bit``.

CUDA graphs: the *first* ``launch()`` of either runner initializes private TMA
descriptor storage synchronously and is NOT capturable; later launches submit
only kernels on the current stream and may be captured. ``validate_indices=True``
performs a device synchronization at prepare time.

Not supported here (keep the existing SGLang path): SM90 / SM103 / SM12x
devices, non-E4M3 operands, masked (padded) grouped layouts, outputs other than
BF16 (plain) / E4M3 + FP32 scales (fused), K or N outside the multiples above.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

from sglang.kernels.cake_kernels._support import (
    SM100,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.gemm.cake_grouped_fp8_gemm"
FI_JIT_MODULE = "flashinfer.jit.gemm.cake_grouped_fp8_gemm"
FI_SILU_MODULE = "flashinfer.gemm.cake_grouped_fp8_fused_silu_quant"
FI_SILU_JIT_MODULE = "flashinfer.jit.gemm.cake_grouped_fp8_fused_silu_quant"
ARCHS = (SM100,)
K_BLOCK = 128
B_SCALE_BLOCK_N = 128
SILU_K_MULTIPLE = 512
SILU_N2_MULTIPLE = 256
SILU_MAX_M = 8192
SILU_GROUP_SIZE = 128


@lru_cache(maxsize=None)
def _programs_registered(device_index: int, fused: bool) -> bool:
    """Whether FlashInfer registers generated programs for this device; never raises."""
    try:
        import torch

        device = torch.device("cuda", device_index)
        if fused:
            from flashinfer.gemm.cake_grouped_fp8_fused_silu_quant import (
                is_group_gemm_fp8_nt_groupwise_contiguous_silu_quant_prepared_available as available,
            )
        else:
            from flashinfer.gemm.cake_grouped_fp8_gemm import (
                is_group_gemm_fp8_nt_groupwise_contiguous_prepared_available as available,
            )
        return bool(available(device))
    except Exception:
        return False


def _aligned_contiguous(tensor: torch.Tensor, alignment: int = 16) -> bool:
    return tensor.is_contiguous() and tensor.data_ptr() % alignment == 0


def _common_operands_ok(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    *,
    n_block: int,
) -> bool:
    import torch

    if a.ndim != 2 or b.ndim != 3:
        return False
    m, k = (int(v) for v in a.shape)
    groups, n = (int(v) for v in b.shape[:2])
    if m <= 0 or groups <= 0 or n <= 0 or n % n_block or k <= 0 or k % K_BLOCK:
        return False
    device = a.device
    return (
        a.dtype == torch.float8_e4m3fn
        and _aligned_contiguous(a)
        and b.device == device
        and b.dtype == torch.float8_e4m3fn
        and tuple(b.shape) == (groups, n, k)
        and _aligned_contiguous(b)
        and a_scale.device == device
        and a_scale.dtype == torch.float32
        and tuple(a_scale.shape) == (m, k // K_BLOCK)
        and _aligned_contiguous(a_scale)
        and b_scale.device == device
        and b_scale.dtype == torch.float32
        and tuple(b_scale.shape) == (groups, n // B_SCALE_BLOCK_N, k // K_BLOCK)
        and _aligned_contiguous(b_scale)
        and m_indices.device == device
        and m_indices.dtype == torch.int32
        and tuple(m_indices.shape) == (m,)
        and _aligned_contiguous(m_indices)
    )


def supports_group_gemm_fp8_nt_groupwise_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises.

    Index sortedness / range is a data property FlashInfer checks only with
    ``validate_indices=True``; it is not inspected here.
    """
    import torch

    if not (
        flashinfer_module_available(FI_MODULE, FI_JIT_MODULE)
        and cuda_tensor_on(a, ARCHS)
        and _common_operands_ok(
            a, b, a_scale, b_scale, m_indices, n_block=B_SCALE_BLOCK_N
        )
    ):
        return False
    if out is not None and not (
        out.device == a.device
        and out.dtype == torch.bfloat16
        and tuple(out.shape) == (int(a.shape[0]), int(b.shape[1]))
        and _aligned_contiguous(out, alignment=2)
    ):
        return False
    return _programs_registered(a.device.index, False)


def supports_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    out_q: Optional[torch.Tensor] = None,
    out_s: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises.

    The 128-row alignment of internal expert boundaries is a data property
    FlashInfer checks only with ``validate_indices=True``; not inspected here.
    """
    import torch

    if not (
        flashinfer_module_available(
            FI_MODULE, FI_JIT_MODULE, FI_SILU_MODULE, FI_SILU_JIT_MODULE
        )
        and cuda_tensor_on(a, ARCHS)
        and _common_operands_ok(
            a, b, a_scale, b_scale, m_indices, n_block=SILU_GROUP_SIZE
        )
    ):
        return False
    m, k = (int(v) for v in a.shape)
    n2 = int(b.shape[1])
    if m > SILU_MAX_M or k % SILU_K_MULTIPLE or n2 % SILU_N2_MULTIPLE:
        return False
    h = n2 // 2
    if out_q is not None and not (
        out_q.device == a.device
        and out_q.dtype == torch.float8_e4m3fn
        and tuple(out_q.shape) == (m, h)
        and _aligned_contiguous(out_q)
    ):
        return False
    if out_s is not None and not (
        out_s.device == a.device
        and out_s.dtype == torch.float32
        and tuple(out_s.shape) == (m, h // SILU_GROUP_SIZE)
        and _aligned_contiguous(out_s, alignment=4)
    ):
        return False
    return _programs_registered(a.device.index, True)


def prepare_group_gemm_fp8_nt_groupwise_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    validate_indices: bool = False,
) -> Any:
    """Forward to FlashInfer; returns a ``PreparedGroupGemmFp8NtGroupwiseContiguous``.

    Call ``.launch()`` (or the object itself) to run; it returns the BF16
    ``(M, N)`` output. First launch is not CUDA-graph capturable.
    """
    from flashinfer.gemm.cake_grouped_fp8_gemm import (
        prepare_group_gemm_fp8_nt_groupwise_contiguous as prepare,
    )

    return prepare(
        a, b, a_scale, b_scale, m_indices, out, validate_indices=validate_indices
    )


def prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    out_q: Optional[torch.Tensor] = None,
    out_s: Optional[torch.Tensor] = None,
    *,
    validate_indices: bool = False,
) -> Any:
    """Forward to FlashInfer; returns a ``PreparedGroupGemmFp8NtGroupwiseContiguousSiluQuant``.

    ``.launch()`` returns ``(out_q, out_s)``. The mixed-schedule route is only
    reachable with ``validate_indices=True``. First launch is not CUDA-graph
    capturable.
    """
    from flashinfer.gemm.cake_grouped_fp8_fused_silu_quant import (
        prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant as prepare,
    )

    return prepare(
        a,
        b,
        a_scale,
        b_scale,
        m_indices,
        out_q,
        out_s,
        validate_indices=validate_indices,
    )


def get_prepared_group_gemm_fp8_nt_groupwise_contiguous_class() -> type:
    """Lazy accessor for FlashInfer's runner dataclass (for isinstance / typing)."""
    from flashinfer.gemm.cake_grouped_fp8_gemm import (
        PreparedGroupGemmFp8NtGroupwiseContiguous,
    )

    return PreparedGroupGemmFp8NtGroupwiseContiguous


def get_prepared_group_gemm_fp8_nt_groupwise_contiguous_silu_quant_class() -> type:
    """Lazy accessor for FlashInfer's fused runner dataclass."""
    from flashinfer.gemm.cake_grouped_fp8_fused_silu_quant import (
        PreparedGroupGemmFp8NtGroupwiseContiguousSiluQuant,
    )

    return PreparedGroupGemmFp8NtGroupwiseContiguousSiluQuant
