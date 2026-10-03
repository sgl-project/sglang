"""Cake Kimi-K3 TP12 fused LatentMoE tail via FlashInfer (sm_100a / sm_103a).

FlashInfer entry: ``flashinfer.kimi_k3_tp12_tail`` (experimental API;
implementation in ``flashinfer.experimental.kimi_k3_tp12_tail.cake_backend``,
44 generated modules loaded through ``cake_jit``). Contract at FlashInfer
``46340689a5ab``: exactly twelve ranks of a GB200 / GB300 NVL72 multi-node
NVLink domain; BF16 contiguous tensors on the workspace device:
``routed_partial [M, 3584]``, ``shared_partial [M, 7168]``, ``norm_weight
[3584]``, ``up_weight [7168, 3584]`` (replicated; the rank's row slice is a
view), caller-owned ``out [M, 7168]``, ``1 <= M <= workspace.max_tokens``.
Computes, replicated and bitwise identical on every rank::

    latent = KimiRMSNorm(sum_r routed_partial_r)                 (eps 1e-5)
    out    = BF16(latent @ up_weight.T + sum_r shared_partial_r)

``create_kimi_k3_tp12_tail_workspace`` is collective: it allocates three
FlashInfer MNNVL Lamport workspaces (fabric symmetric memory + multicast,
three rotating buffers) plus slice buffers for up to ``max_tokens`` tokens,
needs an initialised ``torch.distributed`` group (or an explicit
``CommBackend``), and must be ``destroy()``-ed before the process group.
``prepare_kimi_k3_tp12_tail`` validates, selects the route (one-shot
``M <= 16`` / two-shot K1; fused ``k23`` for ``M <= 8``, ``k3_ess`` below
256, persistent at and above) and builds the JIT modules; the returned runner
launches with no allocation or host sync and is CUDA-graph capturable
(prepare outside capture). Every rank must prepare and launch the same ``M``.

The runtime owns the process group and workspace lifetime; ``supports_*``
checks the device architecture, dtype, shapes, the ``world_size`` argument
and FlashInfer module availability.

Not supported here: other hidden / latent sizes, world sizes other than 12,
non-BF16 inputs, ``backend`` values other than ``"cake"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from sglang.kernels.cake_kernels._support import (
    BLACKWELL_DATACENTER,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.kimi_k3_tp12_tail"
FI_BACKEND_MODULE = "flashinfer.experimental.kimi_k3_tp12_tail.cake_backend"
ARCHS = BLACKWELL_DATACENTER
WORLD_SIZE = 12
HIDDEN = 7168
LATENT = 3584


def supports_kimi_k3_tp12_tail(
    routed_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    *,
    world_size: int,
    max_tokens: Optional[int] = None,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises."""
    import torch

    return (
        flashinfer_module_available(FI_MODULE, FI_BACKEND_MODULE)
        and cuda_tensor_on(routed_partial, ARCHS)
        and routed_partial.dtype == torch.bfloat16
        and routed_partial.ndim == 2
        and routed_partial.is_contiguous()
        and routed_partial.shape[0] >= 1
        and routed_partial.shape[1] == LATENT
        and shared_partial.is_cuda
        and shared_partial.device == routed_partial.device
        and shared_partial.dtype == torch.bfloat16
        and shared_partial.is_contiguous()
        and tuple(shared_partial.shape) == (routed_partial.shape[0], HIDDEN)
        and world_size == WORLD_SIZE
        and (max_tokens is None or routed_partial.shape[0] <= max_tokens)
    )


def create_kimi_k3_tp12_tail_workspace(
    *,
    rank: int,
    max_tokens: int = 4096,
    group: Any = None,
    device: Optional[torch.device] = None,
    comm_backend: Any = None,
) -> Any:
    """Forward to FlashInfer; collectively allocates this rank's MNNVL workspace."""
    from flashinfer.kimi_k3_tp12_tail import create_kimi_k3_tp12_tail_workspace

    return create_kimi_k3_tp12_tail_workspace(
        rank=rank,
        max_tokens=max_tokens,
        group=group,
        device=device,
        comm_backend=comm_backend,
        backend="cake",
    )


def prepare_kimi_k3_tp12_tail(
    routed_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    workspace: Any,
) -> Any:
    """Forward to FlashInfer; returns the ``KimiK3Tp12TailRunner`` bound to ``out``."""
    from flashinfer.kimi_k3_tp12_tail import prepare_kimi_k3_tp12_tail

    return prepare_kimi_k3_tp12_tail(
        routed_partial,
        shared_partial,
        norm_weight,
        up_weight,
        out,
        workspace=workspace,
        backend="cake",
    )


def kimi_k3_tp12_tail(
    routed_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    workspace: Any,
) -> torch.Tensor:
    """Forward to FlashInfer; prepare + one launch, returns ``out``."""
    from flashinfer.kimi_k3_tp12_tail import kimi_k3_tp12_tail

    return kimi_k3_tp12_tail(
        routed_partial,
        shared_partial,
        norm_weight,
        up_weight,
        out,
        workspace=workspace,
        backend="cake",
    )
