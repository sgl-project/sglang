"""Fused gated-activation kernels (``act(x[:h]) * x[h:]``).

Public wrappers forward to the AOT ``sgl_kernel`` implementation. The JIT CUDA
backend is registered for inventory; it accepts extra ``expert_ids`` /
``expert_step`` arguments, so reach for it explicitly via ``select_kernel`` when
those are needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

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

_ACT_DTYPES = ("float16", "bfloat16")
_CUDA = CapabilityRequirement(requires_cuda=True)

# op name -> (aot target attr, jit target attr)
_ACTIVATIONS = {
    "silu_and_mul": ("silu_and_mul", "silu_and_mul"),
    "gelu_and_mul": ("gelu_and_mul", "gelu_and_mul"),
    "gelu_tanh_and_mul": ("gelu_tanh_and_mul", "gelu_tanh_and_mul"),
}

for _name, (_aot_attr, _jit_attr) in _ACTIVATIONS.items():
    register_kernel(
        KernelSpec(
            op=f"activation.{_name}",
            backend=KernelBackend.CUDA_AOT,
            target=f"sgl_kernel:{_aot_attr}",
            priority=10,
            format_signature=FormatSignature(
                supported_dtypes=_ACT_DTYPES,
                description="gated activation; returns tensor",
            ),
            description=f"{_name} (sgl_kernel wheel).",
        )
    )
    register_kernel(
        KernelSpec(
            op=f"activation.{_name}",
            backend=KernelBackend.CUDA_JIT,
            target=f"sglang.jit_kernel.activation:{_jit_attr}",
            priority=5,
            capability=_CUDA,
            format_signature=FormatSignature(
                supported_dtypes=_ACT_DTYPES,
                description="gated activation with optional expert_ids/expert_step",
            ),
            description=f"{_name} (sglang.jit_kernel).",
        )
    )

del _name, _aot_attr, _jit_attr


def silu_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``out = silu(input[..., :d]) * input[..., d:]`` with ``d = input.shape[-1] // 2``."""
    return get_kernel("activation.silu_and_mul", KernelBackend.CUDA_AOT)(input, out)


def gelu_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``out = gelu(input[..., :d]) * input[..., d:]``."""
    return get_kernel("activation.gelu_and_mul", KernelBackend.CUDA_AOT)(input, out)


def gelu_tanh_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``out = gelu_tanh(input[..., :d]) * input[..., d:]``."""
    return get_kernel("activation.gelu_tanh_and_mul", KernelBackend.CUDA_AOT)(
        input, out
    )


__all__ = ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul"]
