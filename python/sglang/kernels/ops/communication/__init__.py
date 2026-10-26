"""Collective-communication kernels and fused compute/communication kernels.

Stateful operators are imported from their modules so communicator registration,
workspace ownership, and tuning helpers stay together. The registry below only
records metadata; importing this group does not load the implementations.
"""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import (
    CUDA_ONLY,
    CapabilityRequirement,
    KernelBackend,
    KernelSpec,
)

__all__ = []

register_kernel(
    KernelSpec(
        op="communication.moe_finalize_shared_gate_all_reduce",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.communication.all_reduce_fusion:moe_finalize_shared_gate_all_reduce",
        capabilities=frozenset({CapabilityRequirement.cuda(min_sm=(9, 0))}),
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
    ("attn_res_collective", "attn_res_fused_pull_rs"),
    ("attn_res_collective", "attn_res_fused_direct_ag"),
]:
    register_kernel(
        KernelSpec(
            op=f"communication.{_fn}",
            backend=KernelBackend.JIT,
            target=f"sglang.kernels.ops.communication.{_mod}:{_fn}",
            capabilities=CUDA_ONLY,
        )
    )
del _mod, _fn

# Muse Spark collectives. The MoE finalize op (with its optional fused TP
# all-reduce) is built per layer: inventoried through its class methods.
register_kernel(
    KernelSpec(
        op="communication.muse_spark_all_gather",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.communication.muse_spark_all_gather:all_gather",
        capabilities=frozenset({CapabilityRequirement.cuda(min_sm=(9, 0))}),
    )
)
register_kernel(
    KernelSpec(
        op="communication.muse_spark_all_gather_norm_residual",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.communication.muse_spark_all_gather_norm_residual:all_gather_norm_residual",
        capabilities=frozenset({CapabilityRequirement.cuda(min_sm=(10, 0))}),
    )
)
for _bk in ("torch", "cute_dsl", "helion"):
    register_kernel(
        KernelSpec(
            op="communication.muse_spark_moe_finalize_all_reduce",
            backend=KernelBackend(_bk),
            target="sglang.kernels.ops.communication.muse_spark_moe_finalize_all_reduce:FusedPostExpertNormWeightedSumTRTLLM."
            + ("forward_native" if _bk == "torch" else f"forward_{_bk}"),
            capabilities=frozenset() if _bk == "torch" else CUDA_ONLY,
        )
    )
del _bk

# Blackwell sequence-parallel GEMM collectives (tcgen05 CuTeDSL kernels).
for _fn in ("all_gather_matmul", "matmul_reduce_scatter"):
    register_kernel(
        KernelSpec(
            op=f"communication.{_fn}",
            backend=KernelBackend.CUTE_DSL,
            target=f"sglang.kernels.ops.communication.sp_collective_gemm:{_fn}",
            capabilities=frozenset(
                {CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 9))}
            ),
        )
    )
del _fn
