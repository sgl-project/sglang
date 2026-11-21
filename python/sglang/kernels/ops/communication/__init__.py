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

# Fused all-reduce: cluster push (1-shot / 2-shot) and NVLS pull, named in compute order.
for _mod, _fns in (
    (
        "push",
        (
            "all_reduce_residual_norm",
            "moe_finalize_all_reduce_residual_norm",
            "moe_finalize_shared_gate_all_reduce",
            "all_reduce_mhc_post_combine_norm",
            "moe_finalize_all_reduce_mhc_post",
            "moe_finalize_all_reduce_mhc_post_combine_norm",
            "moe_finalize_all_reduce_mhc_post_combine_norm_quant",
            "all_reduce_norm_prefix",
            "moe_finalize_all_reduce_norm",
            "all_reduce_sandwich_norm",
            "topk_sum_all_reduce_sandwich_norm",
        ),
    ),
    (
        "pull",
        (
            "all_reduce_norm_prefix",
            "all_reduce_norm_residual_norm_quant",
            "all_reduce_norm_residual",
        ),
    ),
):
    for _fn in _fns:
        register_kernel(
            KernelSpec(
                op=f"communication.{_mod}.{_fn}",
                backend=KernelBackend.JIT,
                target=f"sglang.kernels.ops.communication.fused_all_reduce.{_mod}:{_fn}",
                capabilities=frozenset({CapabilityRequirement.cuda(min_sm=(9, 0))}),
            )
        )
del _mod, _fns, _fn


# Kernels introduced with Kimi-K3, inventoried by logical operator group.
for _mod, _fn in [
    ("gemm_ag", "gemm_ag_up_proj"),
    ("moe_front_ag", "moe_front_all_gather"),
    ("sp_collective", "reduce_scatter_res"),
    ("sp_collective", "reduce_scatter_pull"),
    ("sp_collective", "all_gather"),
    ("sp_collective", "all_gather_direct"),
    ("attn_res_collective", "attn_res_fused_pull_rs"),
    ("attn_res_collective", "attn_res_fused_direct_ag"),
    ("gemm_ag", "gather_add3"),
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

register_kernel(
    KernelSpec(
        op="communication.engram_gather_peer",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.communication.engram_gather_peer:engram_gather_peer",
    )
)


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
