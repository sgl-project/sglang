"""Collective-communication kernels and fused compute/communication kernels.

Stateful operators are imported from their modules so communicator registration,
workspace ownership, and tuning helpers stay together. The registry below only
records metadata; importing this group does not load the implementations.
"""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import CapabilityRequirement, KernelBackend, KernelSpec

__all__ = []

register_kernel(
    KernelSpec(
        op="communication.moe_finalize_shared_gate_all_reduce",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.communication.all_reduce_fusion:moe_finalize_shared_gate_all_reduce",
        capabilities=frozenset({CapabilityRequirement.cuda(min_sm=(9, 0))}),
        description="BF16 routed finalize, FP32 shared gate, and clustered all-reduce.",
    )
)


# Kernels introduced with Kimi-K3, inventoried by logical operator group.
for _mod, _fn in [
    ("all_reduce_residual", "all_reduce_push_res"),
    ("all_reduce_residual", "all_reduce_push_norm"),
    ("all_reduce_residual", "finalize_all_reduce_push_norm"),
    ("all_reduce_residual", "all_reduce_pull_res"),
    ("all_reduce_residual", "all_reduce_pull_norm"),
    ("gemm_ag", "gemm_ag_up_proj"),
    ("sp_collective", "reduce_scatter_res"),
    ("sp_collective", "reduce_scatter_pull"),
    ("sp_collective", "all_gather"),
    ("sp_collective", "all_gather_direct"),
]:
    register_kernel(
        KernelSpec(
            op=f"communication.{_fn}",
            backend=KernelBackend.JIT,
            target=f"sglang.kernels.ops.communication.{_mod}:{_fn}",
            capabilities=frozenset({CapabilityRequirement.CUDA}),
        )
    )
del _mod, _fn
