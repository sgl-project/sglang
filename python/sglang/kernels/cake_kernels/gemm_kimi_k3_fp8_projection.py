"""Cake Kimi-K3 ``FP8_PB_WO`` projection GEMMs via FlashInfer.

FlashInfer entries (``flashinfer.gemm.kimi_k3_fp8_projection``, re-exported
from ``flashinfer.gemm``; backend
``flashinfer.experimental.kimi_k3_fp8_projection.cake_backend``, JIT registry
``...kimi_k3_fp8_projection.cake_jit``; flashinfer-ai/flashinfer#4568):

* ``prepare_kimi_k3_fp8_projection_weights(weight, weight_scale, n_valid=None, *,
  splits=None, backend="cake") -> PreparedProjectionWeight`` -- once per weight.
* ``allocate_kimi_k3_fp8_projection_workspace(prepared, M) -> ProjectionWorkspace``
  -- once per ``M``; no launch.
* ``prepare_kimi_k3_fp8_projection(x, prepared, out, workspace, *, backend="cake")
  -> KimiK3Fp8ProjectionRunner`` -- ``launch()`` is allocation-free, has no host
  sync and is CUDA-graph capturable (prepare outside capture; ``x`` is re-read
  on device at every launch).
* ``kimi_k3_fp8_projection(x, prepared, out=None, *, workspace=None,
  backend="cake") -> out`` -- allocating one-shot form (not for capture).

Contract at FlashInfer ``46340689a5ab`` (SM100a / SM103a): ``weight`` contiguous
``float8_e4m3fn [N_pad128, K]`` with ``K % 128 == 0`` and ``N_pad128 % 128 == 0``;
``weight_scale`` FP32 ModelOpt block scale ``[N_pad128/128, 1, K/128, 1]`` or its
2-D view ``[N_pad128/128, K/128]``; ``n_valid`` even in ``(0, N_pad128]``;
``splits`` positive and summing to ``n_valid``. ``x`` contiguous BF16 ``[M, K]``;
``out`` BF16 ``[M, n_valid]`` view with unit column stride, even row stride
``>= n_valid`` and 4-byte alignment (16-byte base + row stride multiple of 8
enables the TMA-store epilogue). Activations are quantized per token in 1x128
blocks to E4M3 with UE8M0 scales (DeepGEMM ``per_token_cast_to_fp8(use_ue8m0=True)``)
and the weight is requantized to UE8M0 scales once (vLLM ``requant_weight_ue8m0``);
single BF16 rounding. The route (persistent 2-CTA GEMM for ``M > 256``, swap-AB
split-K decode or fused in-CTA-quant decode for tabulated families) is selected
from the measured per-architecture decode table and depends on the SM count.
Tolerance of the source contract: ``1e-2 + 1e-2 * |ref| + 2 bf16 ulp``.

Not supported here (keep the existing SGLang path): weights without the 128-row
block padding, non-E4M3 / non-ModelOpt-scale checkpoints, FP16 activations,
non-BF16 outputs, SM90 / SM12x.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

from sglang.kernels.cake_kernels._support import (
    BLACKWELL_DATACENTER,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.gemm.kimi_k3_fp8_projection"
FI_BACKEND_MODULE = "flashinfer.experimental.kimi_k3_fp8_projection.cake_backend"
FI_JIT_MODULE = "flashinfer.experimental.kimi_k3_fp8_projection.cake_jit"
ARCHS = BLACKWELL_DATACENTER
BLOCK = 128


def _modules_available() -> bool:
    return flashinfer_module_available(FI_MODULE, FI_BACKEND_MODULE, FI_JIT_MODULE)


def _programs_registered(device: torch.device, m: Optional[int] = None, prepared=None):
    """FlashInfer's ``generated_program_available``; never raises."""
    try:
        from flashinfer.experimental.kimi_k3_fp8_projection.cake_backend import (
            generated_program_available,
        )

        return bool(generated_program_available(device, m, prepared))
    except Exception:
        return False


def supports_kimi_k3_fp8_projection_weights(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    n_valid: Optional[int] = None,
    *,
    splits: Optional[Sequence[int]] = None,
) -> bool:
    """Admission check mirroring the FlashInfer weight-preparation contract; never raises."""
    import torch

    if not (
        _modules_available()
        and cuda_tensor_on(weight, ARCHS)
        and weight.dtype == torch.float8_e4m3fn
        and weight.ndim == 2
        and weight.is_contiguous()
    ):
        return False
    n_rows, k = (int(v) for v in weight.shape)
    if k % BLOCK or n_rows % BLOCK or n_rows == 0:
        return False
    n_valid = n_rows if n_valid is None else int(n_valid)
    if not (0 < n_valid <= n_rows) or n_valid % 2:
        return False
    if weight_scale.device != weight.device:
        return False
    shape = tuple(weight_scale.shape)
    if shape not in (
        (n_rows // BLOCK, 1, k // BLOCK, 1),
        (n_rows // BLOCK, k // BLOCK),
    ):
        return False
    if splits is not None:
        try:
            parts = tuple(int(s) for s in splits)
        except (TypeError, ValueError):
            return False
        if any(s <= 0 for s in parts) or sum(parts) != n_valid:
            return False
    return _programs_registered(weight.device)


def supports_kimi_k3_fp8_projection(
    x: torch.Tensor,
    prepared: Any,
    out: Optional[torch.Tensor] = None,
    workspace: Any = None,
) -> bool:
    """Admission check for the prepared / one-shot projection; never raises.

    ``out`` / ``workspace`` may be ``None`` for the allocating one-shot form.
    Workspace byte sizing is FlashInfer's; only its shape/dtype/alignment
    contract is mirrored here, plus the exact-route registration check.
    """
    import torch

    try:
        k_w = int(prepared.K)
        n_valid = int(prepared.n_valid)
        device = prepared.device
    except Exception:
        return False
    if not (
        _modules_available()
        and cuda_tensor_on(x, ARCHS)
        and x.device == device
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and int(x.shape[1]) == k_w
        and int(x.shape[0]) >= 1
    ):
        return False
    m = int(x.shape[0])
    if out is not None:
        ldo = int(out.stride(0)) if out.ndim == 2 else 0
        if not (
            out.device == device
            and out.dtype == torch.bfloat16
            and out.ndim == 2
            and tuple(out.shape) == (m, n_valid)
            and out.stride(1) == 1
            and ldo % 2 == 0
            and ldo >= n_valid
            and (out.storage_offset() * 2) % 4 == 0
        ):
            return False
    if workspace is not None:
        try:
            q, sf = workspace
        except (TypeError, ValueError):
            return False
        if not (
            q.device == device
            and tuple(q.shape) == (m, k_w)
            and q.dtype == torch.float8_e4m3fn
            and q.is_contiguous()
            and sf.device == device
            and sf.dtype == torch.uint8
            and sf.ndim == 1
            and sf.is_contiguous()
            and sf.data_ptr() % 256 == 0
        ):
            return False
    return _programs_registered(device, m, prepared)


def prepare_kimi_k3_fp8_projection_weights(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    n_valid: Optional[int] = None,
    *,
    splits: Optional[Sequence[int]] = None,
) -> Any:
    """Forward to FlashInfer; returns a ``PreparedProjectionWeight`` (once per weight)."""
    from flashinfer.gemm.kimi_k3_fp8_projection import (
        prepare_kimi_k3_fp8_projection_weights as prepare,
    )

    return prepare(weight, weight_scale, n_valid, splits=splits, backend="cake")


def allocate_kimi_k3_fp8_projection_workspace(prepared: Any, M: int) -> Any:
    """Forward to FlashInfer; returns a ``ProjectionWorkspace(q, sf)`` for ``M`` rows."""
    from flashinfer.gemm.kimi_k3_fp8_projection import (
        allocate_kimi_k3_fp8_projection_workspace as allocate,
    )

    return allocate(prepared, M)


def prepare_kimi_k3_fp8_projection(
    x: torch.Tensor,
    prepared: Any,
    out: torch.Tensor,
    workspace: Any,
) -> Any:
    """Forward to FlashInfer; returns a ``KimiK3Fp8ProjectionRunner``.

    ``launch()`` returns ``out``; capturable into a CUDA graph (prepare outside).
    """
    from flashinfer.gemm.kimi_k3_fp8_projection import (
        prepare_kimi_k3_fp8_projection as prepare,
    )

    return prepare(x, prepared, out, workspace, backend="cake")


def kimi_k3_fp8_projection(
    x: torch.Tensor,
    prepared: Any,
    out: Optional[torch.Tensor] = None,
    *,
    workspace: Any = None,
) -> torch.Tensor:
    """Forward to FlashInfer; one-shot ``out = bf16(x @ dequant(weight).T)``.

    Allocates ``out`` and the workspace when omitted (not for graph capture).
    """
    from flashinfer.gemm.kimi_k3_fp8_projection import kimi_k3_fp8_projection as run

    return run(x, prepared, out, workspace=workspace, backend="cake")


def get_prepared_projection_weight_class() -> type:
    from flashinfer.experimental.kimi_k3_fp8_projection.cake_backend import (
        PreparedProjectionWeight,
    )

    return PreparedProjectionWeight


def get_projection_workspace_class() -> type:
    from flashinfer.experimental.kimi_k3_fp8_projection.cake_backend import (
        ProjectionWorkspace,
    )

    return ProjectionWorkspace


def get_kimi_k3_fp8_projection_runner_class() -> type:
    from flashinfer.experimental.kimi_k3_fp8_projection.cake_backend import (
        KimiK3Fp8ProjectionRunner,
    )

    return KimiK3Fp8ProjectionRunner
