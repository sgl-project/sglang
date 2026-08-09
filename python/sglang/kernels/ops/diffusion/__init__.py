"""Registered diffusion-model kernels.

Callers import concrete implementations from submodules. This package only
registers kernel specs for the shared selector / namespace discovery path.
"""

from __future__ import annotations

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

_CUDA = frozenset({CapabilityRequirement.CUDA})

register_kernel(
    KernelSpec(
        op="diffusion.apply_group_norm_silu",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.diffusion.group_norm_silu:apply_group_norm_silu",
        capabilities=_CUDA,
        format_signature=FormatSignature(description="fused GroupNorm + SiLU"),
        description="Fused group-norm + SiLU (Triton).",
    )
)
register_kernel(
    KernelSpec(
        op="diffusion.residual_gate_add",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.diffusion.residual_gate_add:residual_gate_add",
        capabilities=_CUDA,
        format_signature=FormatSignature(description="residual + gate * update"),
        description="Fused residual gate-add (sglang.kernels.jit).",
    )
)
register_kernel(
    KernelSpec(
        op="diffusion.fused_inplace_qknorm_rope",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.diffusion.qknorm_rope:fused_inplace_qknorm_rope",
        capabilities=_CUDA,
        format_signature=FormatSignature(
            in_place=True, description="fused in-place QK-norm + RoPE"
        ),
        description="Fused QK-norm + RoPE (sglang.kernels.jit).",
    )
)
# Migrated from multimodal_gen (RFC #29630, Phase 2.5). Hot paths import the
# Triton symbol directly; the registry entry remains for namespace discovery.
register_kernel(
    KernelSpec(
        op="diffusion.sparse_linear_attn_fwd",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.diffusion.sparse_linear_attn_kernels:_attn_fwd",
        capabilities=_CUDA,
        format_signature=FormatSignature(description="sparse linear attention fwd"),
        description="Sparse linear attention forward (Triton).",
    )
)

__all__: list[str] = []
