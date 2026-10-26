"""Generic elementwise / fused-pointwise kernels.

Home for cross-cutting pointwise kernels that do not belong to a single
functional group: the fused-pointwise Triton collection (``elementwise``:
sigmoid-mul variants shared across models)
and the ``add_constant`` JIT reference kernel used by the developer guide.
Individual functions register (or are imported) under the functional op id
they logically belong to.
"""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import (
    CUDA_ONLY,
    CapabilityRequirement,
    KernelBackend,
    KernelSpec,
)

__all__ = []

for _fn in ("hc_combine_gate", "hc_combine_apply", "hc_combine_apply_norm"):
    register_kernel(
        KernelSpec(
            op=f"elementwise.{_fn}",
            backend=KernelBackend.JIT,
            target=f"sglang.kernels.ops.elementwise.hc_combine_decode:{_fn}",
            capabilities=frozenset({CapabilityRequirement.cuda(min_sm=(9, 0))}),
        )
    )
del _fn


# Public entry points inventoried by logical operator group (RFC #29630).
register_kernel(
    KernelSpec(
        op="elementwise.fused_qwen4_gate_value",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.elementwise.qwen4_gate:fused_qwen4_gate_value",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="elementwise.fused_qwen4_gate_reduce",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.elementwise.qwen4_gate:fused_qwen4_gate_reduce",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="elementwise.row_scale_bf16",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.elementwise.row_scale:row_scale_bf16",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="elementwise.row_sum",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.elementwise.row_sum:row_sum",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="elementwise.apply_log_scaling_tau",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.elementwise.row_scale:apply_log_scaling_tau",
    )
)
register_kernel(
    KernelSpec(
        op="elementwise.kimi_k3_mla_output_gate",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.elementwise.mla_output_gate:kimi_k3_mla_output_gate",
        capabilities=CUDA_ONLY,
    )
)
