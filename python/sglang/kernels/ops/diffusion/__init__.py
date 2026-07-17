"""Diffusion-model kernels (group-norm+silu, residual-gate-add, qk-norm+rope).

These are JIT CUDA kernels; the wrappers forward to ``sglang.jit_kernel.diffusion``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch
    from torch import nn

_CUDA = frozenset({CapabilityRequirement.CUDA})

register_kernel(
    KernelSpec(
        op="diffusion.apply_group_norm_silu",
        backend=KernelBackend.JIT,
        target="sglang.jit_kernel.diffusion.group_norm_silu:apply_group_norm_silu",
        capabilities=_CUDA,
        format_signature=FormatSignature(description="fused GroupNorm + SiLU"),
        description="Fused group-norm + SiLU (sglang.jit_kernel).",
    )
)
register_kernel(
    KernelSpec(
        op="diffusion.residual_gate_add",
        backend=KernelBackend.JIT,
        target="sglang.jit_kernel.diffusion.residual_gate_add:residual_gate_add_cuda",
        capabilities=_CUDA,
        format_signature=FormatSignature(description="residual + gate * update"),
        description="Fused residual gate-add (sglang.jit_kernel).",
    )
)
register_kernel(
    KernelSpec(
        op="diffusion.fused_inplace_qknorm_rope",
        backend=KernelBackend.JIT,
        target="sglang.jit_kernel.diffusion.qknorm_rope:fused_inplace_qknorm_rope",
        capabilities=_CUDA,
        format_signature=FormatSignature(
            in_place=True, description="fused in-place QK-norm + RoPE"
        ),
        description="Fused QK-norm + RoPE (sglang.jit_kernel).",
    )
)


def apply_group_norm_silu(
    x: torch.Tensor, norm: nn.Module, activation: nn.Module
) -> torch.Tensor:
    """Fused GroupNorm + SiLU (falls back to eager when unsupported)."""
    return get_kernel("diffusion.apply_group_norm_silu", KernelBackend.JIT)(
        x, norm, activation
    )


def residual_gate_add(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Fused ``residual + gate * update``."""
    return get_kernel("diffusion.residual_gate_add", KernelBackend.JIT)(
        residual, update, gate
    )


def fused_inplace_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    eps: float = 1e-6,
    head_dim: int = 0,
    rope_dim: int = 0,
) -> None:
    """Fused in-place QK RMS-norm + RoPE."""
    return get_kernel("diffusion.fused_inplace_qknorm_rope", KernelBackend.JIT)(
        q,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox=is_neox,
        eps=eps,
        head_dim=head_dim,
        rope_dim=rope_dim,
    )


__all__ = [
    "apply_group_norm_silu",
    "residual_gate_add",
    "fused_inplace_qknorm_rope",
]


# Migrated from multimodal_gen (RFC #29630, Phase 2.5).
register_kernel(
    KernelSpec(
        op="diffusion.sparse_linear_attn_fwd",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.diffusion.sparse_linear_attn_kernels:get_block_map",
    )
)
