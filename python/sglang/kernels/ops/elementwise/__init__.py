"""Generic elementwise / fused-pointwise kernels.

Home for cross-cutting pointwise kernels that do not belong to a single
functional group: the fused-pointwise Triton collection (``elementwise``:
sigmoid-mul, gated-activation and fused-rmsnorm variants shared across models)
and the ``add_constant`` JIT reference kernel used by the developer guide.
Individual functions register (or are imported) under the functional op id
they logically belong to.
"""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import CapabilityRequirement, KernelBackend, KernelSpec

__all__ = []


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
