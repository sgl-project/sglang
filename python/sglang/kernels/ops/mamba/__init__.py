"""State-space / Mamba kernels (causal conv1d)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import FormatSignature, KernelBackend, KernelSpec

if TYPE_CHECKING:
    import torch

register_kernel(
    KernelSpec(
        op="mamba.causal_conv1d_fwd",
        backend=KernelBackend.AOT,
        target="sgl_kernel.mamba:causal_conv1d_fwd",
        format_signature=FormatSignature(
            in_place=True, description="causal depthwise conv1d forward (prefill)"
        ),
        description="Causal conv1d forward (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="mamba.causal_conv1d_update",
        backend=KernelBackend.AOT,
        target="sgl_kernel.mamba:causal_conv1d_update",
        format_signature=FormatSignature(
            in_place=True, description="causal depthwise conv1d update (decode)"
        ),
        description="Causal conv1d update (sgl_kernel wheel).",
    )
)


def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias_: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    silu_activation: bool,
    pad_slot_id: int,
):
    """Causal depthwise conv1d forward (prefill)."""
    return get_kernel("mamba.causal_conv1d_fwd", KernelBackend.AOT)(
        x,
        weight,
        bias_,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        silu_activation,
        pad_slot_id,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias_: Optional[torch.Tensor],
    silu_activation: bool,
    cache_seqlens: Optional[torch.Tensor],
    conv_state_indices: Optional[torch.Tensor],
    pad_slot_id: int,
):
    """Causal depthwise conv1d update (decode)."""
    return get_kernel("mamba.causal_conv1d_update", KernelBackend.AOT)(
        x,
        conv_state,
        weight,
        bias_,
        silu_activation,
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
    )


__all__ = ["causal_conv1d_fwd", "causal_conv1d_update"]


# Vendored mamba_ssm-derived kernels relocated in Phase 2.5 (RFC #29630).
for _mod, _fn in [
    ("triton_ops.ssd_combined", "mamba_chunk_scan_combined"),
    ("triton_ops.mamba_ssm", "selective_state_update"),
    ("causal_conv1d_triton", "causal_conv1d_fn"),
    ("mamba_state_scatter_triton", "fused_mamba_state_scatter_with_mask"),
]:
    register_kernel(
        KernelSpec(
            op=f"mamba.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.mamba.{_mod}:{_fn}",
        )
    )
del _mod, _fn
