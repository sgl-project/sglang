"""MoE finalize and TP all-reduce with an HC=4 post-mixing epilogue."""

from typing import Optional, Tuple

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
    require_cluster_launch_arch,
)
from sglang.srt.utils.custom_op import register_custom_op

# The kernel is built for this width only (static_assert in all_reduce_fusion.cuh).
_MHC_HIDDEN_DIM = 5120


@cache_once
def _jit_mhc_module(world_size, top_k, cluster_size, weight_dtype):
    require_cluster_launch_arch()
    args = make_cpp_args(
        world_size,
        _MHC_HIDDEN_DIM,
        top_k,
        cluster_size,
        is_arch_support_pdl(),
        weight_dtype,
        True,
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
def _moe_finalize_all_reduce_mhc_op(
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
    _jit_mhc_module(world_size, top_k, cluster_size, weights.dtype).run(
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
    gemm2: torch.Tensor,
    idx: torch.Tensor,
    weights: torch.Tensor,
    top_k: int,
    shared: Optional[torch.Tensor],
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    world_size: int,
    cluster_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deferred MoE finalize -> push all-reduce -> HC=4 post mixing.

    Same finalize inputs as :func:`all_reduce_fusion.moe_finalize_all_reduce`;
    ``residual`` is ``[T, 4, hidden]`` bf16, ``post`` ``[T, 4]`` and ``comb``
    ``[T, 4, 4]`` fp32. Returns ``(reduced [T, hidden], mhc_out [T, 4, hidden])``.
    """
    out = torch.empty(
        (weights.shape[0], _MHC_HIDDEN_DIM), dtype=torch.bfloat16, device=gemm2.device
    )
    mhc_out = torch.empty_like(residual)
    if weights.shape[0]:
        _moe_finalize_all_reduce_mhc_op(
            world_size,
            top_k,
            cluster_size or default_cluster_size(_MHC_HIDDEN_DIM),
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
def _moe_finalize_all_reduce_mhc_norm_op(
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
    _jit_mhc_module(world_size, top_k, cluster_size, weights.dtype).run_norm(
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
    gemm2: torch.Tensor,
    idx: torch.Tensor,
    weights: torch.Tensor,
    top_k: int,
    shared: Optional[torch.Tensor],
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    world_size: int,
    cluster_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`moe_finalize_all_reduce_mhc` plus the HC=4 pre-collapse
    (``pre`` ``[T, 4]`` fp32) and RMSNorm of the collapsed row. Returns
    ``(reduced, mhc_out, normalized [T, hidden])``.
    """
    out = torch.empty(
        (weights.shape[0], _MHC_HIDDEN_DIM), dtype=torch.bfloat16, device=gemm2.device
    )
    mhc_out = torch.empty_like(residual)
    normalized = torch.empty_like(out)
    if weights.shape[0]:
        _moe_finalize_all_reduce_mhc_norm_op(
            world_size,
            top_k,
            cluster_size or default_cluster_size(_MHC_HIDDEN_DIM),
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


def all_reduce_mhc_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plain all-reduce of ``x`` with the mHC + norm epilogue: the finalize
    kernel driven with identity routing (top_k = 1, unit weights)."""
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
def _jit_mhc_quant_module(world_size, top_k, cluster_size, weight_dtype):
    require_cluster_launch_arch()
    args = make_cpp_args(
        world_size,
        _MHC_HIDDEN_DIM,
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
def _moe_finalize_all_reduce_mhc_quant_op(
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
    _jit_mhc_quant_module(world_size, top_k, cluster_size, weights.dtype).run(
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
    gemm2: torch.Tensor,
    idx: torch.Tensor,
    weights: torch.Tensor,
    top_k: int,
    shared: Optional[torch.Tensor],
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    world_size: int,
    cluster_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`moe_finalize_all_reduce_mhc_norm` plus fp8 e4m3 quantization of
    the normalized row with ue8m0 group scales (rows <= 8). Returns
    ``(reduced, mhc_out, normalized, quantized, scales)``.
    """
    rows = weights.shape[0]
    assert 0 < rows <= 8
    out = torch.empty(
        (rows, _MHC_HIDDEN_DIM), dtype=torch.bfloat16, device=gemm2.device
    )
    mhc_out = torch.empty_like(residual)
    normalized = torch.empty_like(out)
    quantized = torch.empty_like(out, dtype=torch.float8_e4m3fn)
    # ue8m0 scale layout: one byte per 32-wide group, rows padded to 128
    scales = torch.empty(
        (_MHC_HIDDEN_DIM // 32) * 128, device=gemm2.device, dtype=torch.uint8
    )
    _moe_finalize_all_reduce_mhc_quant_op(
        world_size,
        top_k,
        cluster_size or default_cluster_size(_MHC_HIDDEN_DIM),
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
