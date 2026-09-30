"""Finalize/all-reduce + post + combine, retaining standalone RMSNorm."""

from typing import Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.communication.all_reduce_fusion import (
    default_cluster_size,
    get_registered_comm,
)
from sglang.srt.utils.custom_op import register_custom_op


@cache_once
def _module(world_size, top_k, cluster_size, weight_dtype):
    args = make_cpp_args(
        world_size,
        5120,
        top_k,
        cluster_size,
        is_arch_support_pdl(),
        weight_dtype,
        True,
        False,
        True,
    )
    return load_jit(
        "moe_finalize_all_reduce_mhc_combine",
        *args,
        cuda_files=["distributed/all_reduce_fusion.cuh"],
        cuda_wrappers=[("run", f"MoeFinalizeAllReduceKernel<{args}>::run_mhc_combine")],
    )


@register_custom_op(mutates_args=["out", "mhc_out", "combined"])
def _moe_finalize_all_reduce_mhc_combine(
    world_size: int,
    top_k: int,
    cluster_size: int,
    out: torch.Tensor,
    mhc_out: torch.Tensor,
    combined: torch.Tensor,
    gemm2: torch.Tensor,
    idx: torch.Tensor,
    weights: torch.Tensor,
    shared: Optional[torch.Tensor],
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
) -> None:
    comm = get_registered_comm(world_size)
    assert comm is not None
    _module(world_size, top_k, cluster_size, weights.dtype).run(
        comm,
        out,
        gemm2,
        idx,
        weights,
        shared,
        mhc_out,
        residual,
        post,
        comb,
        pre,
        combined,
    )


def moe_finalize_all_reduce_mhc_combine(
    gemm2,
    idx,
    weights,
    top_k,
    shared,
    residual,
    post,
    comb,
    pre,
    *,
    world_size,
    cluster_size=None,
):
    out = torch.empty(
        (weights.shape[0], 5120), dtype=torch.bfloat16, device=gemm2.device
    )
    mhc_out = torch.empty_like(residual)
    combined = torch.empty_like(out)
    if weights.shape[0]:
        _moe_finalize_all_reduce_mhc_combine(
            world_size,
            top_k,
            cluster_size or default_cluster_size(5120),
            out,
            mhc_out,
            combined,
            gemm2,
            idx,
            weights,
            shared,
            residual,
            post,
            comb,
            pre,
        )
    return out, mhc_out, combined


def all_reduce_mhc_combine(x, residual, post, comb, pre, *, world_size):
    from sglang.kernels.ops.communication.all_reduce_mhc import _identity_routing

    idx, weights = _identity_routing(x.shape[0], x.device)
    return moe_finalize_all_reduce_mhc_combine(
        x, idx, weights, 1, None, residual, post, comb, pre, world_size=world_size
    )
