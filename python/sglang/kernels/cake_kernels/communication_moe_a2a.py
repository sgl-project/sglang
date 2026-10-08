"""Cake MoE expert-parallel all-to-all via FlashInfer (sm_100a / sm_103a).

Two FlashInfer surfaces run the generated Cake all-to-all kernels:

* **Functional throughput all-to-all** (``flashinfer.comm.trtllm_moe_alltoall``
  with ``backend="cake"``; JIT ``flashinfer.jit.comm.gen_moe_alltoall_module(
  "sm100a" | "sm103a")``). Same ABI and payload rules as the TRT-LLM
  all-to-all: ``moe_a2a_get_workspace_size_per_rank`` sizes one rank's slice
  of the symmetric workspace, ``moe_a2a_initialize`` returns the opaque
  metainfo tensor, ``moe_a2a_dispatch`` scatters per-token payload lists to
  the owning expert ranks (optional ``active_rank_mask`` uint64 bitmask and
  EPLB stats), ``moe_a2a_combine`` reduces the top-k expert outputs back to
  the source ranks, ``moe_a2a_sanitize_expert_ids`` invalidates unused slots.
  Every call on a workspace initialised with ``backend="cake"`` must pass
  ``backend="cake"`` (this module pins it); all ranks must select the same
  backend. The module getter resolves the target from the *current* CUDA
  device and raises on other architectures.
* **moe_ep split-comm backend** (``flashinfer.moe_ep.CakeAlltoAll`` /
  ``CakeAlltoAllConfig``, ``@register_communication("cake")``). An
  ``NVLinkOneSidedAlltoAll`` subclass with the same protocol, workspace layout
  and ``dispatch`` / ``combine`` / ``destroy`` interface; only the kernels
  differ. Constructed from a ``BootstrapConfig`` (EP world size / rank /
  optional process group) and ``MoEEpCommParams`` (experts, top-k, max tokens
  per rank, hidden size, dtype); the construction is collective and allocates
  the one-sided NVLink workspace through the bootstrap. This succeeds the
  deprecated ``MoeAlltoAll(backend="cake")`` class, which is not forwarded.

The runtime (``sglang.srt.layers.communication`` / the MoE EP layer) owns the
workspace tensor, process group and bootstrap and must keep one workspace per
live dispatch/combine pair. ``supports_*`` checks the device architecture and
that the installed FlashInfer exposes the Cake backend (the all-to-all module
exists in older releases without the ``backend`` keyword).

Not supported here: ``MoeAlltoAll`` (deprecated class), the legacy
``backend="trtllm"`` kernels, non-Blackwell devices.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Union

from sglang.kernels.cake_kernels._support import (
    BLACKWELL_DATACENTER,
    device_in,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

ARCHS = BLACKWELL_DATACENTER

FI_A2A_MODULE = "flashinfer.comm.trtllm_moe_alltoall"
FI_A2A_JIT_MODULE = "flashinfer.jit.comm"
FI_MOE_EP_MODULE = "flashinfer.moe_ep.backends.split.comm.cake.communication"
FI_MOE_EP_CONFIG_MODULE = "flashinfer.moe_ep.backends.split.comm.cake.config"


def _cuda_device_index(device: Union[torch.device, int, None]) -> Optional[int]:
    """CUDA device index of ``device`` (``None`` = current), or ``None`` if not CUDA."""
    import torch

    if torch.version.cuda is None or not torch.cuda.is_available():
        return None
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    if device.type != "cuda":
        return None
    return torch.cuda.current_device() if device.index is None else device.index


@lru_cache(maxsize=None)
def _moe_a2a_cake_backend_available() -> bool:
    """``True`` when the installed all-to-all module exposes ``backend="cake"``.

    Imports ``flashinfer.comm.trtllm_moe_alltoall`` (no JIT build) and checks
    the ``backend`` keyword and the exact-architecture target resolver.
    """
    if not flashinfer_module_available(FI_A2A_MODULE, FI_A2A_JIT_MODULE):
        return False
    try:
        from flashinfer.comm import trtllm_moe_alltoall as mod

        params = inspect.signature(mod.moe_a2a_initialize).parameters
        return "backend" in params and hasattr(mod, "_moe_alltoall_target")
    except Exception:
        return False


# --------------------------------------------------------------------------
# Functional throughput all-to-all (E2-29 / E2-30)
# --------------------------------------------------------------------------


def supports_moe_a2a(device: Union[torch.device, int, None] = None) -> bool:
    """Admission check for the ``backend="cake"`` all-to-all; never raises."""
    index = _cuda_device_index(device)
    return (
        index is not None
        and _moe_a2a_cake_backend_available()
        and device_in(index, ARCHS)
    )


def moe_a2a_get_workspace_size_per_rank(
    ep_size: int,
    max_num_tokens: int,
    total_dispatch_payload_size_per_token: int,
    combine_payload_size_per_token: int,
    eplb_stats_num_experts: int = 0,
) -> int:
    """Forward to FlashInfer; bytes of one rank's workspace slice (host-only)."""
    from flashinfer.comm.trtllm_moe_alltoall import (
        moe_a2a_get_workspace_size_per_rank,
    )

    return moe_a2a_get_workspace_size_per_rank(
        ep_size,
        max_num_tokens,
        total_dispatch_payload_size_per_token,
        combine_payload_size_per_token,
        eplb_stats_num_experts,
        backend="cake",
    )


def moe_a2a_initialize(
    workspace: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    max_num_tokens: int,
    eplb_stats_num_experts: int = 0,
) -> torch.Tensor:
    """Forward to FlashInfer; returns the opaque metainfo tensor for ``workspace``."""
    from flashinfer.comm.trtllm_moe_alltoall import moe_a2a_initialize

    return moe_a2a_initialize(
        workspace,
        ep_rank,
        ep_size,
        max_num_tokens,
        eplb_stats_num_experts,
        backend="cake",
    )


def moe_a2a_dispatch(
    token_selected_experts: torch.Tensor,
    input_payloads: list[torch.Tensor],
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    runtime_max_tokens_per_rank: int,
    ep_rank: int,
    ep_size: int,
    top_k: int,
    num_experts: int,
    enable_pdl: Optional[bool] = None,
    eplb_local_stats: Optional[torch.Tensor] = None,
    enable_rank_mask: bool = False,
    active_rank_mask: Optional[torch.Tensor] = None,
    *,
    recv_view_cache: Optional[dict] = None,
) -> tuple[list[torch.Tensor], int, Optional[torch.Tensor]]:
    """Forward to FlashInfer; returns ``(output_payloads, combine_payload_offset, eplb_gathered_stats)``."""
    from flashinfer.comm.trtllm_moe_alltoall import moe_a2a_dispatch

    return moe_a2a_dispatch(
        token_selected_experts,
        input_payloads,
        workspace,
        metainfo,
        runtime_max_tokens_per_rank,
        ep_rank,
        ep_size,
        top_k,
        num_experts,
        enable_pdl,
        eplb_local_stats,
        enable_rank_mask,
        active_rank_mask,
        backend="cake",
        recv_view_cache=recv_view_cache,
    )


def moe_a2a_combine(
    payload: torch.Tensor,
    local_num_tokens: int,
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    runtime_max_tokens_per_rank: int,
    ep_rank: int,
    ep_size: int,
    top_k: int,
    combine_payload_offset: int,
    payload_in_workspace: bool = False,
    output_dtype: Optional[torch.dtype] = None,
    output_scales: Optional[torch.Tensor] = None,
    output_scalar_scale: float = 1.0,
    sf_layout: Any = None,
    output: Optional[torch.Tensor] = None,
    *,
    use_low_precision: bool = False,
    enable_pdl: Optional[bool] = None,
    enable_rank_mask: bool = False,
    active_rank_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward to FlashInfer; returns the combined ``[local_num_tokens, ...]`` tensor.

    ``sf_layout=None`` keeps FlashInfer's default (``SfLayout.layout_linear``).
    """
    from flashinfer.comm.trtllm_moe_alltoall import moe_a2a_combine

    kwargs: dict[str, Any] = {}
    if sf_layout is not None:
        kwargs["sf_layout"] = sf_layout
    return moe_a2a_combine(
        payload,
        local_num_tokens,
        workspace,
        metainfo,
        runtime_max_tokens_per_rank,
        ep_rank,
        ep_size,
        top_k,
        combine_payload_offset,
        payload_in_workspace,
        output_dtype,
        output_scales,
        output_scalar_scale,
        output=output,
        use_low_precision=use_low_precision,
        enable_pdl=enable_pdl,
        enable_rank_mask=enable_rank_mask,
        active_rank_mask=active_rank_mask,
        backend="cake",
        **kwargs,
    )


def moe_a2a_sanitize_expert_ids(
    expert_ids: torch.Tensor,
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    ep_rank: int,
    invalid_expert_id: int,
    enable_pdl: Optional[bool] = None,
) -> Any:
    """Forward to FlashInfer; rewrites unused slots of ``expert_ids`` in place."""
    from flashinfer.comm.trtllm_moe_alltoall import moe_a2a_sanitize_expert_ids

    return moe_a2a_sanitize_expert_ids(
        expert_ids,
        workspace,
        metainfo,
        ep_rank,
        invalid_expert_id,
        enable_pdl,
        backend="cake",
    )


# --------------------------------------------------------------------------
# moe_ep split-comm backend (E2-33 / E2-34)
# --------------------------------------------------------------------------


def supports_moe_ep_alltoall(device: Union[torch.device, int, None] = None) -> bool:
    """Admission check for ``flashinfer.moe_ep.CakeAlltoAll``; never raises.

    Checks the Cake modules and the device architecture only; the collective
    platform probe (``CakeAlltoAll.is_platform_supported``) runs at
    construction.
    """
    index = _cuda_device_index(device)
    return (
        index is not None
        and flashinfer_module_available(FI_MOE_EP_MODULE, FI_MOE_EP_CONFIG_MODULE)
        and _moe_a2a_cake_backend_available()
        and device_in(index, ARCHS)
    )


def moe_ep_alltoall_config(**fields: Any) -> Any:
    """Forward to FlashInfer; ``CakeAlltoAllConfig`` (``NVLinkOneSidedConfig`` fields)."""
    from flashinfer.moe_ep.backends.split.comm.cake.config import CakeAlltoAllConfig

    return CakeAlltoAllConfig(**fields)


def moe_ep_alltoall(bootstrap: Any, params: Any, config: Any = None) -> Any:
    """Forward to FlashInfer; collectively constructs a ``CakeAlltoAll``.

    ``bootstrap`` is a ``flashinfer.moe_ep.BootstrapConfig`` (EP world size,
    rank, optional ``process_group``), ``params`` a ``MoEEpCommParams``;
    ``config`` defaults to ``CakeAlltoAllConfig()``. Call ``destroy()`` before
    the process group goes away.
    """
    from flashinfer.moe_ep.backends.split.comm.cake.communication import CakeAlltoAll

    return CakeAlltoAll(bootstrap, params, config)
