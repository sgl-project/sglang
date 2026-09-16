"""MoE finalize and TP all-reduce with an HC=4 post-mixing epilogue."""

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
        world_size, 5120, top_k, cluster_size, is_arch_support_pdl(), weight_dtype, True
    )
    return load_jit(
        "moe_finalize_all_reduce_mhc",
        *args,
        cuda_files=["distributed/all_reduce_fusion.cuh"],
        cuda_wrappers=[
            ("run", f"MoeFinalizeAllReduceKernel<{args}>::run_mhc"),
            ("run_norm", f"MoeFinalizeAllReduceKernel<{args}>::run_mhc_norm"),
        ],
    )


@register_custom_op(mutates_args=["out", "mhc_out"])
def _run(
    world_size: int,
    top_k: int,
    cluster_size: int,
    out: torch.Tensor,
    mhc_out: torch.Tensor,
    gemm2: torch.Tensor,
    idx: torch.Tensor,
    weights: torch.Tensor,
    shared: Optional[torch.Tensor],
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
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
    )


def moe_finalize_all_reduce_mhc(
    gemm2,
    idx,
    weights,
    top_k,
    shared,
    residual,
    post,
    comb,
    *,
    world_size,
    cluster_size=None,
):
    out = torch.empty(
        (weights.shape[0], 5120), dtype=torch.bfloat16, device=gemm2.device
    )
    mhc_out = torch.empty_like(residual)
    if weights.shape[0]:
        _run(
            world_size,
            top_k,
            cluster_size or default_cluster_size(5120),
            out,
            mhc_out,
            gemm2,
            idx,
            weights,
            shared,
            residual,
            post,
            comb,
        )
    return out, mhc_out


@register_custom_op(mutates_args=["out", "mhc_out", "normalized"])
def _run_norm(
    world_size: int,
    top_k: int,
    cluster_size: int,
    out: torch.Tensor,
    mhc_out: torch.Tensor,
    normalized: torch.Tensor,
    gemm2: torch.Tensor,
    idx: torch.Tensor,
    weights: torch.Tensor,
    shared: Optional[torch.Tensor],
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> None:
    comm = get_registered_comm(world_size)
    assert comm is not None
    _module(world_size, top_k, cluster_size, weights.dtype).run_norm(
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
        norm_weight,
        eps,
        normalized,
    )


def moe_finalize_all_reduce_mhc_norm(
    gemm2,
    idx,
    weights,
    top_k,
    shared,
    residual,
    post,
    comb,
    pre,
    norm_weight,
    eps,
    *,
    world_size,
    cluster_size=None,
):
    out = torch.empty(
        (weights.shape[0], 5120), dtype=torch.bfloat16, device=gemm2.device
    )
    mhc_out = torch.empty_like(residual)
    normalized = torch.empty_like(out)
    if weights.shape[0]:
        _run_norm(
            world_size,
            top_k,
            cluster_size or default_cluster_size(5120),
            out,
            mhc_out,
            normalized,
            gemm2,
            idx,
            weights,
            shared,
            residual,
            post,
            comb,
            pre,
            norm_weight,
            eps,
        )
    return out, mhc_out, normalized


@cache_once
def _identity_routing(rows, device):
    return (
        torch.arange(rows, device=device, dtype=torch.int32),
        torch.ones(rows, 1, device=device, dtype=torch.float32),
    )


def all_reduce_mhc_norm(x, residual, post, comb, pre, norm_weight, eps, *, world_size):
    idx, weights = _identity_routing(x.shape[0], x.device)
    return moe_finalize_all_reduce_mhc_norm(
        x,
        idx,
        weights,
        1,
        None,
        residual,
        post,
        comb,
        pre,
        norm_weight,
        eps,
        world_size=world_size,
    )


@cache_once
def _quant_module(world_size, top_k, cluster_size, weight_dtype):
    args = make_cpp_args(
        world_size,
        5120,
        top_k,
        cluster_size,
        is_arch_support_pdl(),
        weight_dtype,
        True,
        True,
    )
    return load_jit(
        "moe_finalize_all_reduce_mhc_quant",
        *args,
        cuda_files=["distributed/all_reduce_fusion.cuh"],
        cuda_wrappers=[("run", f"MoeFinalizeAllReduceKernel<{args}>::run_mhc_quant")],
    )


@register_custom_op(
    mutates_args=["out", "mhc_out", "normalized", "quantized", "scales"]
)
def _run_quant(
    world_size: int,
    top_k: int,
    cluster_size: int,
    out: torch.Tensor,
    mhc_out: torch.Tensor,
    normalized: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    gemm2: torch.Tensor,
    idx: torch.Tensor,
    weights: torch.Tensor,
    shared: Optional[torch.Tensor],
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> None:
    comm = get_registered_comm(world_size)
    assert comm is not None
    _quant_module(world_size, top_k, cluster_size, weights.dtype).run(
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
        norm_weight,
        eps,
        normalized,
        quantized,
        scales,
    )


def moe_finalize_all_reduce_mhc_quant(
    gemm2,
    idx,
    weights,
    top_k,
    shared,
    residual,
    post,
    comb,
    pre,
    norm_weight,
    eps,
    *,
    world_size,
    cluster_size=None,
):
    rows = weights.shape[0]
    assert 0 < rows <= 8
    out = torch.empty((rows, 5120), dtype=torch.bfloat16, device=gemm2.device)
    mhc_out = torch.empty_like(residual)
    normalized = torch.empty_like(out)
    quantized = torch.empty_like(out, dtype=torch.float8_e4m3fn)
    scales = torch.empty(160 * 128, device=gemm2.device, dtype=torch.uint8)
    _run_quant(
        world_size,
        top_k,
        cluster_size or default_cluster_size(5120),
        out,
        mhc_out,
        normalized,
        quantized,
        scales,
        gemm2,
        idx,
        weights,
        shared,
        residual,
        post,
        comb,
        pre,
        norm_weight,
        eps,
    )
    return out, mhc_out, normalized, quantized, scales
