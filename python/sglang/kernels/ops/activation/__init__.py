"""Fused gated-activation kernels (``act(x[:h]) * x[h:]``).

Each operator is a :class:`~sglang.kernels.fused_op.BaseFusedOp` with a
pure-``torch`` reference (``forward_native``) plus AOT (``sgl_kernel``) and
JIT CUDA backends behind one ``(input, out)`` signature. The JIT backend
additionally accepts ``expert_ids`` / ``expert_step`` — call
``forward_jit`` directly when those are needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op
from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

_ACT_DTYPES = ("float16", "bfloat16")
_CUDA = frozenset({CapabilityRequirement.CUDA})
_HIP = frozenset({CapabilityRequirement.HIP})
# sgl_kernel's gated-activation ops build for CUDA *and* ROCm (production
# imports them from sgl_kernel on both), so the AOT backend spans both devices
# — the canonical OR-semantics case that a device-baked backend name couldn't.
_CUDA_HIP = frozenset({CapabilityRequirement.CUDA, CapabilityRequirement.HIP})
# JIT before AOT to match the production path (srt/layers/activation.py imports
# from sglang.jit_kernel.activation on CUDA); auto-selection must not invert it.
_ACT_PRIORITY = (
    KernelBackend.JIT,
    KernelBackend.AOT,
    KernelBackend.TORCH,
)


class _GatedActivationOp(BaseFusedOp):
    """Shared structure for ``act(x[..., :d]) * x[..., d:]`` operators."""

    # Set by subclasses: sgl_kernel / jit_kernel attr name (same for both).
    kernel_attr: str

    priority = _ACT_PRIORITY
    capabilities = {
        KernelBackend.AOT: _CUDA_HIP,
        KernelBackend.JIT: _CUDA,
    }
    format_signature = FormatSignature(
        supported_dtypes=_ACT_DTYPES,
        description="gated activation; returns tensor",
    )

    def _act(self, gate: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_native(
        self, input: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        d = input.shape[-1] // 2
        result = self._act(input[..., :d]) * input[..., d:]
        if out is None:
            return result
        out.copy_(result)
        return out

    def forward_aot(
        self, input: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        import sgl_kernel

        return getattr(sgl_kernel, self.kernel_attr)(input, out)

    def forward_jit(
        self,
        input: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        expert_ids: Optional[torch.Tensor] = None,
        expert_step: int = 1,
    ) -> torch.Tensor:
        import sglang.jit_kernel.activation as jit_activation

        return getattr(jit_activation, self.kernel_attr)(
            input, out, expert_ids, expert_step
        )


class SiluAndMulOp(_GatedActivationOp):
    """``out = silu(input[..., :d]) * input[..., d:]`` with ``d = input.shape[-1] // 2``.

    Adds an ``AITER`` backend on ``device=HIP``: on ROCm this op has a native
    ``aiter`` kernel (``srt/layers/activation.py`` uses it in production). Note
    the sibling gelu ops below deliberately do *not* register AITER — ROCm/aiter
    coverage is a per-``(op, backend)`` subset, which the decoupled backend/device
    model expresses directly (a device-agnostic ``KernelBackend`` name plus a
    per-backend ``CapabilityRequirement``).
    """

    op = "activation.silu_and_mul"
    kernel_attr = "silu_and_mul"
    # AOT spans CUDA+HIP; JIT is CUDA; AITER is an opt-in HIP path. By priority,
    # CUDA resolves to JIT and HIP resolves to AOT (matching production
    # defaults); AITER is registered and HIP-eligible but sits below AOT, so it
    # is available for explicit/forced selection without changing the default.
    priority = (
        KernelBackend.JIT,
        KernelBackend.AOT,
        KernelBackend.AITER,
        KernelBackend.TORCH,
    )
    capabilities = {
        KernelBackend.AOT: _CUDA_HIP,
        KernelBackend.JIT: _CUDA,
        KernelBackend.AITER: _HIP,
    }
    descriptions = {
        KernelBackend.AOT: "silu_and_mul (sgl_kernel wheel).",
        KernelBackend.JIT: "silu_and_mul (sglang.jit_kernel).",
        KernelBackend.AITER: "silu_and_mul (aiter, ROCm).",
        KernelBackend.TORCH: "silu_and_mul (pure-torch reference).",
    }

    def _act(self, gate: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        return F.silu(gate)

    def forward_aiter(
        self, input: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        import torch
        from aiter import silu_and_mul as _aiter_silu_and_mul

        d = input.shape[-1] // 2
        if out is None:
            out = torch.empty(
                (*input.shape[:-1], d), dtype=input.dtype, device=input.device
            )
        # aiter's ROCm silu_and_mul: (out, input, limit); limit=0.0 = no clamp,
        # matching the standard (unclamped) gated-SiLU used elsewhere.
        _aiter_silu_and_mul(out, input, 0.0)
        return out


class GeluAndMulOp(_GatedActivationOp):
    """``out = gelu(input[..., :d]) * input[..., d:]`` (erf-based GELU)."""

    op = "activation.gelu_and_mul"
    kernel_attr = "gelu_and_mul"
    descriptions = {
        KernelBackend.AOT: "gelu_and_mul (sgl_kernel wheel).",
        KernelBackend.JIT: "gelu_and_mul (sglang.jit_kernel).",
        KernelBackend.TORCH: "gelu_and_mul (pure-torch reference).",
    }

    def _act(self, gate: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        return F.gelu(gate, approximate="none")


class GeluTanhAndMulOp(_GatedActivationOp):
    """``out = gelu_tanh(input[..., :d]) * input[..., d:]`` (tanh-approximated GELU)."""

    op = "activation.gelu_tanh_and_mul"
    kernel_attr = "gelu_tanh_and_mul"
    descriptions = {
        KernelBackend.AOT: "gelu_tanh_and_mul (sgl_kernel wheel).",
        KernelBackend.JIT: "gelu_tanh_and_mul (sglang.jit_kernel).",
        KernelBackend.TORCH: "gelu_tanh_and_mul (pure-torch reference).",
    }

    def _act(self, gate: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        return F.gelu(gate, approximate="tanh")


_SILU_AND_MUL = register_fused_op(SiluAndMulOp(), __name__, "_SILU_AND_MUL")
_GELU_AND_MUL = register_fused_op(GeluAndMulOp(), __name__, "_GELU_AND_MUL")
_GELU_TANH_AND_MUL = register_fused_op(
    GeluTanhAndMulOp(), __name__, "_GELU_TANH_AND_MUL"
)


def silu_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``out = silu(input[..., :d]) * input[..., d:]`` with ``d = input.shape[-1] // 2``."""
    return _SILU_AND_MUL(input, out)


def gelu_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``out = gelu(input[..., :d]) * input[..., d:]``."""
    return _GELU_AND_MUL(input, out)


def gelu_tanh_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``out = gelu_tanh(input[..., :d]) * input[..., d:]``."""
    return _GELU_TANH_AND_MUL(input, out)


__all__ = [
    "SiluAndMulOp",
    "GeluAndMulOp",
    "GeluTanhAndMulOp",
    "silu_and_mul",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
]


# Triton kernel migrated into this group (from layers/triton_ops/softcap);
# registered for inventory. Import it from its module.
for _fn in ("softcap_out", "softcap_inplace_logits"):
    register_kernel(
        KernelSpec(
            op=f"activation.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.activation.softcap:{_fn}",
        )
    )
del _fn
