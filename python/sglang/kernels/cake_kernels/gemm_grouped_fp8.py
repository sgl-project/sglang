"""Cake contiguous grouped FP8 GEMM (plain and fused SwiGLU + FP8 quant) via FlashInfer.

FlashInfer entries: ``flashinfer.gemm.prepare_group_gemm_fp8_nt_groupwise_contiguous``
(module ``flashinfer.gemm.cake_grouped_fp8_gemm``, JIT registry
``flashinfer.jit.gemm.cake_grouped_fp8_gemm``) and
``flashinfer.gemm.prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant``
(module ``flashinfer.gemm.cake_grouped_fp8_fused_silu_quant``, JIT registry
``flashinfer.jit.gemm.cake_grouped_fp8_fused_silu_quant``). Both are
prepared-runner factories: the returned object binds the *weight* operands
(``b``, ``b_scale``) and records the shapes / dtypes / strides / device of the
per-token operands; ``launch()`` submits the kernel(s) on the current stream.

Plain GEMM contract (block-scaled FlashInfer, the ``alignment`` keyword):

* ``a`` E4M3 ``(M, K)`` with ``K % 128 == 0``; ``b`` E4M3 ``(G, N, K)``;
  ``m_indices`` int32 ``(M,)`` nondecreasing over valid rows; ``out`` BF16
  ``(M, N)`` (a 2-byte but not 16-byte aligned ``out`` selects the scalar-store
  route).
* **Block-scaled family** (``a_scale`` and ``b_scale`` both int32 packed UE8M0,
  one int32 = four consecutive 128-wide K-block exponents, byte 0 = lowest
  block): ``a_scale`` ``(M, ceil(K/512))`` with any strides (the DeepGEMM
  MN-major ``(1, M)`` layout included); ``b_scale`` ``(G, N, ceil(K/512))``
  (row-repeated, ``transform_scale_ue8m0``) or ``(G, N/128, ceil(K/512))``;
  ``N % 16 == 0``.  ``-1`` rows of ``m_indices`` are skipped natively (their
  output rows are left untouched; ``fill_padding`` must be ``False``).  Every
  expert's run must start on a multiple of ``alignment`` rows (``alignment`` a
  multiple of 32; multiples of 128 run the fast single-run schedule, other
  values a slower multi-run fallback).
* **FP32 family** (both scales FP32): ``a_scale`` ``(M, K/128)`` and
  ``b_scale`` ``(G, N/128, K/128)`` contiguous, arbitrary values,
  ``N % 128 == 0``; ``-1`` rows only with ``fill_padding=True``.
* ``.launch(a=None, a_scale=None, m_indices=None, out=None)`` rebinds the
  per-token operands (same shapes / dtypes / strides / device as prepared).
  ``prepare`` runs no device work, ``launch`` allocates nothing and is
  CUDA-graph capturable from the first launch.

FlashInfer pins without the ``alignment`` keyword (``0.7.0.post1`` and
earlier, see ``flashinfer_python`` in ``python/pyproject.toml``) only offer the
FP32 family with no ``-1`` support and a non-capturable first launch;
:func:`block_scaled_contract_available` reports which contract is installed and
the ``supports_*`` checks reject int32 scales / ``fill_padding`` / ``alignment``
on the old contract so callers fall back to their default kernel.

Fused gate_up GEMM + SwiGLU + per-128-column FP8 quant (FP32 scales only):
``a`` E4M3 ``(M, K)`` with ``0 < M <= 8192`` and ``K % 512 == 0``; ``b`` E4M3
``(G, 2H, K)`` (gate rows ``[0, H)`` then up) with ``2H % 256 == 0``; every
*internal* expert boundary of ``m_indices`` must be a multiple of 128 rows
(only checked with ``validate_indices=True``); outputs ``out_q`` E4M3 ``(M, H)``
and ``out_s`` FP32 ``(M, H/128)`` = ``max(absmax, 1e-10) / 448``.  Bitwise equal
to CuTe grouped GEMM + ``silu_and_mul`` + ``per_token_group_quant_8bit``.
``validate_indices=True`` performs a device synchronization at prepare time.

Not supported here (keep the existing SGLang path): SM90 / SM103 / SM12x
devices, non-E4M3 operands, mixed FP32 / int32 scale dtypes, masked (padded)
grouped layouts, outputs other than BF16 (plain) / E4M3 + FP32 scales (fused),
K or N outside the multiples above.
"""

from __future__ import annotations

import inspect
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
# Block-scaled (packed UE8M0 int32) family: one int32 packs four K blocks.
UE8M0_BLOCKS_PER_INT32 = 4
BLOCK_SCALED_N_MULTIPLE = 16
ALIGNMENT_MULTIPLE = 32
FAST_ALIGNMENT = 128
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


@lru_cache(maxsize=1)
def block_scaled_contract_available() -> bool:
    """Whether the installed FlashInfer plain-GEMM prepare accepts the
    block-scaled contract (int32 UE8M0 scales, native ``-1`` rows, the
    ``alignment`` / ``fill_padding`` keywords and ``launch(a=..., ...)``
    rebinding).  Detected from the ``alignment`` keyword of
    ``prepare_group_gemm_fp8_nt_groupwise_contiguous``; never raises."""
    try:
        from flashinfer.gemm.cake_grouped_fp8_gemm import (
            prepare_group_gemm_fp8_nt_groupwise_contiguous as prepare,
        )

        params = inspect.signature(prepare).parameters
        return "alignment" in params and "fill_padding" in params
    except Exception:
        return False


@lru_cache(maxsize=None)
def block_scaled_contiguous_available(device_index: int) -> bool:
    """Device-level admission of the block-scaled plain GEMM (no tensors needed):
    the FlashInfer modules are importable with the block-scaled contract, the
    device is one of :data:`ARCHS` and FlashInfer registers programs for it.
    Never raises."""
    try:
        from sglang.kernels.cake_kernels._support import device_in

        return (
            flashinfer_module_available(FI_MODULE, FI_JIT_MODULE)
            and block_scaled_contract_available()
            and device_in(device_index, ARCHS)
            and _programs_registered(device_index, False)
        )
    except Exception:
        return False


def alignment_ok(alignment: Optional[int]) -> bool:
    """``None`` (FlashInfer default) or a positive multiple of 32."""
    return alignment is None or (
        isinstance(alignment, int)
        and alignment > 0
        and alignment % ALIGNMENT_MULTIPLE == 0
    )


def _aligned_contiguous(tensor: torch.Tensor, alignment: int = 16) -> bool:
    return tensor.is_contiguous() and tensor.data_ptr() % alignment == 0


def _scales_ok(
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    m: int,
    n: int,
    k: int,
    groups: int,
    device: torch.device,
    allow_block_scaled: bool,
) -> bool:
    """Both scales FP32 (``(M, K/128)`` / ``(G, N/128, K/128)`` contiguous) or
    both int32 packed UE8M0 (``(M, ceil(K/512))`` any strides /
    ``(G, N, ceil(K/512))`` or ``(G, N/128, ceil(K/512))``)."""
    import torch

    if a_scale.device != device or b_scale.device != device:
        return False
    k_blocks = k // K_BLOCK
    if a_scale.dtype == torch.float32 and b_scale.dtype == torch.float32:
        return (
            tuple(a_scale.shape) == (m, k_blocks)
            and _aligned_contiguous(a_scale)
            and tuple(b_scale.shape) == (groups, n // B_SCALE_BLOCK_N, k_blocks)
            and _aligned_contiguous(b_scale)
        )
    if not (
        allow_block_scaled
        and a_scale.dtype == torch.int32
        and b_scale.dtype == torch.int32
        and block_scaled_contract_available()
    ):
        return False
    cols = -(-k_blocks // UE8M0_BLOCKS_PER_INT32)
    if tuple(a_scale.shape) != (m, cols) or a_scale.data_ptr() % 4:
        return False
    if b_scale.ndim != 3 or int(b_scale.shape[0]) != groups:
        return False
    if int(b_scale.shape[1]) not in (n, n // B_SCALE_BLOCK_N):
        return False
    return int(b_scale.shape[2]) == cols and b_scale.data_ptr() % 4 == 0


def _common_operands_ok(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    *,
    n_block: int,
    allow_block_scaled: bool = False,
) -> bool:
    import torch

    if a.ndim != 2 or b.ndim != 3:
        return False
    m, k = (int(v) for v in a.shape)
    groups, n = (int(v) for v in b.shape[:2])
    if m <= 0 or groups <= 0 or n <= 0 or k <= 0 or k % K_BLOCK:
        return False
    if a_scale.dtype == torch.int32 and allow_block_scaled:
        if n % BLOCK_SCALED_N_MULTIPLE:
            return False
    elif n % n_block:
        return False
    device = a.device
    return (
        a.dtype == torch.float8_e4m3fn
        and _aligned_contiguous(a)
        and b.device == device
        and b.dtype == torch.float8_e4m3fn
        and tuple(b.shape) == (groups, n, k)
        and _aligned_contiguous(b)
        and _scales_ok(
            a_scale,
            b_scale,
            m=m,
            n=n,
            k=k,
            groups=groups,
            device=device,
            allow_block_scaled=allow_block_scaled,
        )
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
    *,
    fill_padding: bool = False,
    alignment: Optional[int] = None,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises.

    int32 (packed UE8M0) scales, ``fill_padding`` and ``alignment`` need the
    block-scaled FlashInfer contract (:func:`block_scaled_contract_available`);
    ``fill_padding`` is only valid for FP32 scales.  Index sortedness / range /
    run alignment are data properties FlashInfer checks only with
    ``validate_indices=True``; they are not inspected here.
    """
    import torch

    if not (
        flashinfer_module_available(FI_MODULE, FI_JIT_MODULE)
        and cuda_tensor_on(a, ARCHS)
        and alignment_ok(alignment)
        and _common_operands_ok(
            a,
            b,
            a_scale,
            b_scale,
            m_indices,
            n_block=B_SCALE_BLOCK_N,
            allow_block_scaled=True,
        )
    ):
        return False
    if (
        fill_padding or alignment is not None
    ) and not block_scaled_contract_available():
        return False
    if fill_padding and a_scale.dtype != torch.float32:
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
    fill_padding: bool = False,
    alignment: int = FAST_ALIGNMENT,
) -> Any:
    """Forward to FlashInfer; returns a ``PreparedGroupGemmFp8NtGroupwiseContiguous``.

    ``.launch(a=None, a_scale=None, m_indices=None, out=None)`` runs the GEMM,
    rebinding the per-token operands when given, and returns the BF16 ``(M, N)``
    output.  ``fill_padding`` / ``alignment`` are only forwarded on the
    block-scaled FlashInfer contract; on the old contract they are not
    expressible and raise ``TypeError`` when set to non-default values
    (``supports_*`` already rejects those requests).
    """
    from flashinfer.gemm.cake_grouped_fp8_gemm import (
        prepare_group_gemm_fp8_nt_groupwise_contiguous as prepare,
    )

    if block_scaled_contract_available():
        return prepare(
            a,
            b,
            a_scale,
            b_scale,
            m_indices,
            out,
            validate_indices=validate_indices,
            fill_padding=fill_padding,
            alignment=alignment,
        )
    if fill_padding or alignment != FAST_ALIGNMENT:
        raise TypeError(
            "installed FlashInfer lacks the block-scaled contiguous grouped FP8 "
            "GEMM contract (fill_padding / alignment keywords); bump the "
            "flashinfer_python pin in python/pyproject.toml"
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
