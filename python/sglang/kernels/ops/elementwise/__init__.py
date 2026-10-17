"""Generic elementwise / fused-pointwise kernels.

Home for cross-cutting pointwise kernels that do not belong to a single
functional group: the fused-pointwise Triton collection (``elementwise``:
sigmoid-mul, gated-activation and fused-rmsnorm variants shared across models)
and the ``add_constant`` JIT reference kernel used by the developer guide.
Individual functions register (or are imported) under the functional op id
they logically belong to.
"""

from typing import TYPE_CHECKING

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

__all__ = ["bf16_add_then_scale"]

register_kernel(
    KernelSpec(
        op="elementwise.bf16_add_then_scale",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.elementwise.xpu_bf16_add_then_scale:bf16_add_then_scale",
        capabilities=frozenset({CapabilityRequirement.XPU}),
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            description="BF16 matrix add rounded before multiplication by a BF16 [1] scalar",
        ),
    )
)


def bf16_add_then_scale(
    x: "torch.Tensor", y: "torch.Tensor", scale: "torch.Tensor"
) -> "torch.Tensor":
    """Two eager BF16 pointwise operations in one XPU pass."""
    from sglang.kernels.ops.elementwise.xpu_bf16_add_then_scale import (
        bf16_add_then_scale as implementation,
    )

    return implementation(x, y, scale)


for _fn in ("hc_combine_gate", "hc_combine_apply", "hc_combine_apply_norm"):
    register_kernel(
        KernelSpec(
            op=f"elementwise.{_fn}",
            backend=KernelBackend.JIT,
            target=f"sglang.kernels.ops.elementwise.hc_combine_decode:{_fn}",
            capabilities=frozenset({CapabilityRequirement.cuda(min_sm=(9, 0))}),
            description="BF16 HC decode with four branches of width 2560 and PDL.",
        )
    )
del _fn


# Public entry points inventoried by logical operator group (RFC #29630).
register_kernel(
    KernelSpec(
        op="elementwise.fused_qwen4_gate_value",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.elementwise.qwen4_gate:fused_qwen4_gate_value",
        capabilities=frozenset({CapabilityRequirement.CUDA}),
    )
)
register_kernel(
    KernelSpec(
        op="elementwise.fused_qwen4_gate_reduce",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.elementwise.qwen4_gate:fused_qwen4_gate_reduce",
        capabilities=frozenset({CapabilityRequirement.CUDA}),
    )
)
register_kernel(
    KernelSpec(
        op="elementwise.row_scale_bf16",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.elementwise.row_scale:row_scale_bf16",
        capabilities=frozenset({CapabilityRequirement.CUDA}),
    )
)
