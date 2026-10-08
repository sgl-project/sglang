"""Cake contiguous grouped FP8 GEMM (plain and fused SwiGLU + FP8 quant) via FlashInfer.

FlashInfer entries: ``flashinfer.gemm.prepare_group_gemm_fp8_nt_groupwise_contiguous``
(module ``flashinfer.gemm.cake_grouped_fp8_gemm``, JIT registry
``flashinfer.jit.gemm.cake_grouped_fp8_gemm``) and
``flashinfer.gemm.prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant``
(module ``flashinfer.gemm.cake_grouped_fp8_fused_silu_quant``, JIT registry
``flashinfer.jit.gemm.cake_grouped_fp8_fused_silu_quant``).  Both are
prepared-runner factories; both run on SM100a (B200) and SM103a (B300).

Plain GEMM contract (FlashInfer main, PR #6048 "block-scaled UE8M0 family with
native -1 padding and per-call operand binding"):

* ``a`` E4M3 ``(M, K)`` contiguous, 16-byte aligned, ``M > 0``, ``K % 128 == 0``;
  ``b`` E4M3 ``(G, N, K)`` contiguous, 16-byte aligned, ``N % 128 == 0``;
  ``m_indices`` int32 ``(M,)`` contiguous, 16-byte aligned, nondecreasing over
  routed rows; ``out`` BF16 ``(M, N)`` contiguous (a 2-byte but not 16-byte
  aligned ``out`` selects the FP32 family's scalar-store route).
* **Block-scaled family** (``a_scale`` and ``b_scale`` both int32 packed UE8M0,
  one int32 = four consecutive 128-wide K-block exponents, byte 0 = lowest
  block): ``a_scale`` ``(M, cols)`` with ``4 * cols >= K / 128``, 16-byte
  aligned, contiguous *or* the transpose of a contiguous tensor (DeepGEMM's
  MN-major ``(1, M)`` layout is read in place); ``b_scale`` ``(G, N, cols)``
  (row-repeated, ``transform_scale_ue8m0``) or ``(G, N/128, cols)``, same
  alignment and layout rule; ``out`` 16-byte aligned.  ``-1`` rows of
  ``m_indices`` are padding: a 32-row sub-block whose leading entry is ``-1``
  is skipped and its output rows left untouched; ``-1`` rows sharing a
  sub-block with routed rows receive finite values of no meaning
  (``fill_padding`` must be ``False``).  Every expert's run must start on a
  multiple of ``alignment`` rows (a positive multiple of 32; multiples of 128
  run the single-run schedule, other values the slower multi-run schedule).
* **FP32 family** (both scales FP32): ``a_scale`` ``(M, K/128)`` and
  ``b_scale`` ``(G, N/128, K/128)`` contiguous, arbitrary values; ``-1`` rows
  only with ``fill_padding=True`` (two small forward-fill kernels per launch).
* ``prepare`` binds the problem shape and the expert weights and runs no
  device work.  ``.launch(a=None, a_scale=None, m_indices=None, out=None)``
  rebinds the per-token operands for that call (same shape / dtype / device /
  packed-scale layout / output alignment class as prepared), allocates
  nothing, submits exactly one kernel (plus the forward-fill pair with
  ``fill_padding``) and is CUDA-graph capturable from the first launch.
  ``.release_prepared_operands()`` drops the references to the per-token
  tensors given at ``prepare`` so a long-lived runner pins no batch buffers;
  afterwards every ``launch`` must pass all four operands.
* ``validate_indices=True`` checks sortedness / range with one device-to-host
  transfer (never inside graph capture).

The ``flashinfer_python`` pin in ``python/pyproject.toml`` (``0.7.0.post1``)
predates this contract: that release only offers the FP32 family, rejects
``-1`` rows, has no ``alignment`` / ``fill_padding`` keywords, no per-call
``launch`` operands and a non-capturable first launch.
:func:`block_scaled_contract_available` detects which contract is installed
and the ``supports_*`` checks reject int32 scales, ``fill_padding`` and
``alignment`` on the old contract so callers fall back to their default kernel.

Fused gate_up GEMM + SwiGLU + per-128-column FP8 quant: FlashInfer main's
version (PR #6049 not merged) still binds *all* operands at ``prepare`` --
``launch()`` takes no arguments -- accepts FP32 scales only and rejects ``-1``
rows, so a caller with DeepGEMM's packed UE8M0 scales or the compact ``-1``
layout cannot use it without staging copies.  Contract: ``a`` E4M3 ``(M, K)``
with ``0 < M <= 8192`` and ``K % 512 == 0``; ``b`` E4M3 ``(G, 2H, K)`` (gate rows
``[0, H)`` then up) with ``2H % 256 == 0``; every *internal* expert boundary of
``m_indices`` a multiple of 128 rows (checked only with ``validate_indices=True``,
one device-to-host transfer); outputs ``out_q`` E4M3 ``(M, H)`` and ``out_s``
FP32 ``(M, H/128)`` = ``max(absmax, 1e-10) / 448``.  Bitwise equal to CuTe
grouped GEMM + ``silu_and_mul`` + ``per_token_group_quant_8bit``.

Not supported here (keep the existing SGLang path): SM90 / SM12x devices,
non-E4M3 operands, mixed FP32 / int32 scale dtypes, masked (padded) grouped
layouts, outputs other than BF16 (plain) / E4M3 + FP32 scales (fused),
K or N outside the multiples above.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

from sglang.kernels.cake_kernels._support import (
    SM100,
    SM103,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.gemm.cake_grouped_fp8_gemm"
FI_JIT_MODULE = "flashinfer.jit.gemm.cake_grouped_fp8_gemm"
FI_SILU_MODULE = "flashinfer.gemm.cake_grouped_fp8_fused_silu_quant"
FI_SILU_JIT_MODULE = "flashinfer.jit.gemm.cake_grouped_fp8_fused_silu_quant"
ARCHS = (SM100, SM103)
K_BLOCK = 128
B_SCALE_BLOCK_N = 128
# Block-scaled (packed UE8M0 int32) family: one int32 packs four K blocks.
UE8M0_BLOCKS_PER_INT32 = 4
PACKED_SCALE_ALIGNMENT = 16
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
    ``alignment`` / ``fill_padding`` keywords, ``launch(a=..., ...)``
    rebinding and ``release_prepared_operands()``).  Detected from the
    ``alignment`` keyword of ``prepare_group_gemm_fp8_nt_groupwise_contiguous``
    and the release method of the prepared class; never raises."""
    try:
        from flashinfer.gemm.cake_grouped_fp8_gemm import (
            PreparedGroupGemmFp8NtGroupwiseContiguous as prepared_cls,
        )
        from flashinfer.gemm.cake_grouped_fp8_gemm import (
            prepare_group_gemm_fp8_nt_groupwise_contiguous as prepare,
        )

        params = inspect.signature(prepare).parameters
        return (
            "alignment" in params
            and "fill_padding" in params
            and callable(getattr(prepared_cls, "release_prepared_operands", None))
        )
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
        and not isinstance(alignment, bool)
        and alignment > 0
        and alignment % ALIGNMENT_MULTIPLE == 0
    )


def _aligned_contiguous(tensor: torch.Tensor, alignment: int = 16) -> bool:
    return tensor.is_contiguous() and tensor.data_ptr() % alignment == 0


def _packed_layout_ok(tensor: torch.Tensor) -> bool:
    """FlashInfer reads packed UE8M0 scales in place when the tensor is contiguous
    or the transpose of a contiguous tensor (DeepGEMM's MN-major layout), with
    16-byte aligned storage."""
    return tensor.data_ptr() % PACKED_SCALE_ALIGNMENT == 0 and (
        tensor.is_contiguous() or tensor.mT.is_contiguous()
    )


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
    both int32 packed UE8M0 (``(M, cols)`` / ``(G, N | N/128, cols)`` with
    ``4 * cols >= K/128``, each contiguous or transpose-contiguous)."""
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
    if a_scale.ndim != 2 or int(a_scale.shape[0]) != m:
        return False
    if UE8M0_BLOCKS_PER_INT32 * int(a_scale.shape[1]) < k_blocks:
        return False
    if not _packed_layout_ok(a_scale):
        return False
    if b_scale.ndim != 3 or int(b_scale.shape[0]) != groups:
        return False
    if int(b_scale.shape[1]) not in (n, n // B_SCALE_BLOCK_N):
        return False
    if UE8M0_BLOCKS_PER_INT32 * int(b_scale.shape[2]) < k_blocks:
        return False
    return _packed_layout_ok(b_scale)


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
    if m <= 0 or groups <= 0 or n <= 0 or k <= 0 or k % K_BLOCK or n % n_block:
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
    ``fill_padding`` is only valid for FP32 scales; the block-scaled family
    needs a 16-byte aligned ``out``.  Index sortedness / range / run alignment
    are data properties FlashInfer checks only with ``validate_indices=True``;
    they are not inspected here.
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
    block_scaled = a_scale.dtype == torch.int32
    if fill_padding and block_scaled:
        return False
    if out is not None and not (
        out.device == a.device
        and out.dtype == torch.bfloat16
        and tuple(out.shape) == (int(a.shape[0]), int(b.shape[1]))
        and _aligned_contiguous(out, alignment=16 if block_scaled else 2)
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

    FP32 scales only; ``-1`` rows are rejected by FlashInfer at ``prepare``
    (``validate_indices=True``) or undefined otherwise.  The 128-row alignment
    of internal expert boundaries is a data property FlashInfer checks only
    with ``validate_indices=True``; not inspected here.
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
    output; ``.release_prepared_operands()`` drops the per-token tensors bound
    here.  ``fill_padding`` / ``alignment`` are only forwarded on the
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

    ``.launch()`` takes no operands (all tensors are bound here) and returns
    ``(out_q, out_s)``.  The mixed-schedule route is only reachable with
    ``validate_indices=True`` (one device-to-host transfer at prepare).
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
