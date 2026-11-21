"""Fused push all-reduce (``csrc/distributed/fused_all_reduce/push/``, one file per epilogue).

Thin ops over the push kernels, named in compute order (prologue, ``all_reduce``, epilogue; the plain load and plain
store are left out). Each takes the communicator and its launch configuration explicitly (``cluster_size``, and
``two_shot`` where both algorithms exist); choosing them is the caller's job. One row per cluster of
``cluster_size`` CTAs, one 16-byte vector per thread, so ``hidden_size / 8 / cluster_size`` must be whole warps, at
most 1024 threads.

- MiniMax M3: :func:`all_reduce_residual_norm`, :func:`moe_finalize_all_reduce_residual_norm` (1-shot or 2-shot)
- Qwen4 decode: :func:`moe_finalize_shared_gate_all_reduce`
- DeepSeek V4: :func:`all_reduce_mhc_post_combine_norm`, :func:`moe_finalize_all_reduce_mhc_post`,
  :func:`moe_finalize_all_reduce_mhc_post_combine_norm`, :func:`moe_finalize_all_reduce_mhc_post_combine_norm_quant`
- Kimi K3: :func:`all_reduce_norm_prefix`, :func:`moe_finalize_all_reduce_norm`
- Beam 1.0: :func:`all_reduce_sandwich_norm`, :func:`topk_sum_all_reduce_sandwich_norm` (multicast when the plane has
  a multicast VA)

The reduce is an fp32 sum of the ranks' bf16 partials in rank order, rounded to bf16; every epilogue starts from that
bf16 row. An empty batch returns empty outputs without a launch where the op allows it (every rank of an idle batch
skips the call together); the gated, quantized and Kimi K3 ops need rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.kernels.ops.communication.all_reduce import Communicator


@cache_once
def _module(
    pattern: str, fn: str, world_size: int, hidden_size: int, cluster_size: int, *args
) -> Module:
    """``fn<world_size, hidden_size, cluster_size, pdl, *args>`` of ``fused_all_reduce/push/<pattern>.cuh`` exported
    as ``run``."""
    if not current_arch().is_sm_at_least(9):
        raise RuntimeError(
            "fused all-reduce cluster kernels require CUDA SM90 or newer"
        )
    cpp_args = make_cpp_args(
        world_size, hidden_size, cluster_size, is_arch_support_pdl(), *args
    )
    return load_jit(
        f"fused_all_reduce_push_{fn}",
        *cpp_args,
        cuda_files=[f"distributed/fused_all_reduce/push/{pattern}.cuh"],
        cuda_wrappers=[("run", f"host::distributed::push::{fn}<{cpp_args}>")],
    )


# MiniMax M3


def all_reduce_residual_norm(
    comm: Communicator,
    partial: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    cluster_size: int,
    two_shot: bool,
    residual_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual_out = bf16(allreduce(partial) + residual)``, ``out = rmsnorm(residual_out)``.

    :returns: ``(out, residual_out)``; ``residual_out`` defaults to ``residual`` (updated in place).
    """
    out = torch.empty_like(partial)
    residual_out = residual if residual_out is None else residual_out
    if partial.shape[0] == 0:
        return out, residual_out
    _module(
        "residual_rms_norm",
        "all_reduce_residual_norm",
        comm.world_size,
        partial.shape[1],
        cluster_size,
        two_shot,
    ).run(comm, out, partial, residual, residual_out, norm_weight, float(eps))
    return out, residual_out


def moe_finalize_all_reduce_residual_norm(
    comm: Communicator,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    cluster_size: int,
    two_shot: bool,
    residual_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The deferred MoE finalize, plus ``shared_output`` when given, as this rank's partial, then the residual +
    RMSNorm epilogue of :func:`all_reduce_residual_norm`; top-k from ``expert_weights [T, K]``.

    :returns: ``(out, residual_out)``; ``residual_out`` defaults to ``residual`` (updated in place).
    """
    out = torch.empty_like(residual)
    residual_out = residual if residual_out is None else residual_out
    if expert_weights.shape[0] == 0:
        return out, residual_out
    _module(
        "residual_rms_norm",
        "moe_finalize_all_reduce_residual_norm",
        comm.world_size,
        residual.shape[1],
        cluster_size,
        two_shot,
        expert_weights.shape[1],
        expert_weights.dtype,
    ).run(
        comm,
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        residual,
        residual_out,
        norm_weight,
        float(eps),
    )
    return out, residual_out


# Qwen4 decode


def moe_finalize_shared_gate_all_reduce(
    comm: Communicator,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: torch.Tensor,
    shared_gate: torch.Tensor,
    *,
    cluster_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``out = allreduce(bf16(bf16(shared_gate * shared_output) + bf16(routed)))``, ``shared_gate`` fp32 per
    token."""
    out = torch.empty_like(shared_output) if out is None else out
    _module(
        "store",
        "moe_finalize_shared_gate_all_reduce",
        comm.world_size,
        shared_output.shape[1],
        cluster_size,
        expert_weights.shape[1],
        expert_weights.dtype,
    ).run(
        comm,
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        shared_gate,
    )
    return out


# DeepSeek V4 (HC = 4)


def all_reduce_mhc_post_combine_norm(
    comm: Communicator,
    partial: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    cluster_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(reduced, mhc_out, normalized)``."""
    num_tokens, hidden_size = partial.shape
    out = torch.empty_like(partial)
    mhc_out = torch.empty(
        num_tokens, 4, hidden_size, dtype=torch.bfloat16, device=partial.device
    )
    normalized = torch.empty_like(partial)
    if num_tokens == 0:
        return out, mhc_out, normalized
    _module(
        "mhc_post",
        "all_reduce_mhc_post_combine_norm",
        comm.world_size,
        hidden_size,
        cluster_size,
    ).run(
        comm,
        out,
        partial,
        mhc_out,
        residual,
        post,
        comb,
        pre,
        norm_weight,
        float(eps),
        normalized,
    )
    return out, mhc_out, normalized


def _mhc(
    comm: Communicator,
    norm: bool,
    quant: bool,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: Optional[torch.Tensor],
    norm_weight: Optional[torch.Tensor],
    eps: float,
    cluster_size: int,
    scales: Optional[torch.Tensor] = None,
) -> Any:  # should be tuple with 2 or 3 elements
    """``scales`` (quant only): the scale tile to write, every byte of it (padding rows zeroed); allocated if None."""
    num_tokens, hidden_size = expert_weights.shape[0], residual.shape[-1]
    device = gemm2_out.device
    out = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    mhc_out = out.new_empty(num_tokens, 4, hidden_size)
    normalized = torch.empty_like(out) if norm else None
    quantized = (
        torch.empty(num_tokens, hidden_size, dtype=torch.float8_e4m3fn, device=device)
        if quant
        else None
    )
    if quant and scales is None:
        scales = torch.empty(hidden_size // 32 * 128, dtype=torch.uint8, device=device)
    if num_tokens == 0 and not quant:
        return tuple(t for t in (out, mhc_out, normalized) if t is not None)
    _module(
        "mhc_post",
        "moe_finalize_all_reduce_mhc_post",
        comm.world_size,
        hidden_size,
        cluster_size,
        expert_weights.shape[1],
        expert_weights.dtype,
        norm,
        quant,
    ).run(
        comm,
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        mhc_out,
        residual,
        post,
        comb,
        pre,
        norm_weight,
        float(eps),
        normalized,
        quantized,
        scales,
    )
    return tuple(
        t for t in (out, mhc_out, normalized, quantized, scales) if t is not None
    )


def moe_finalize_all_reduce_mhc_post(
    comm: Communicator,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    cluster_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(reduced, mhc_out)``."""
    return _mhc(
        comm,
        False,
        False,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        residual,
        post,
        comb,
        None,
        None,
        0.0,
        cluster_size,
    )


def moe_finalize_all_reduce_mhc_post_combine_norm(
    comm: Communicator,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    cluster_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(reduced, mhc_out, normalized)``."""
    return _mhc(
        comm,
        True,
        False,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        residual,
        post,
        comb,
        pre,
        norm_weight,
        eps,
        cluster_size,
    )


def moe_finalize_all_reduce_mhc_post_combine_norm_quant(
    comm: Communicator,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    cluster_size: int,
    scales: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(reduced, mhc_out, normalized, quantized, scales)``; at most 128 tokens (one scale tile). ``scales``:
    the tile to write, every byte of it (padding rows zeroed); allocated if None."""
    return _mhc(
        comm,
        True,
        True,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        residual,
        post,
        comb,
        pre,
        norm_weight,
        eps,
        cluster_size,
        scales,
    )


# Kimi K3


def all_reduce_norm_prefix(
    comm: Communicator,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    num_norm_rows: int,
    cluster_size: int,
) -> torch.Tensor:
    """In place: allreduce, then RMSNorm over the first ``num_norm_rows`` rows of ``x`` viewed as ``[-1, H]``, ``H``
    the norm weight's size (the latent width)."""
    hidden_size = norm_weight.shape[0]
    _module(
        "rms_norm", "all_reduce_norm_prefix", comm.world_size, hidden_size, cluster_size
    ).run(comm, x.view(-1, hidden_size), norm_weight, float(eps), num_norm_rows)
    return x


def moe_finalize_all_reduce_norm(
    comm: Communicator,
    out: torch.Tensor,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    cluster_size: int,
) -> torch.Tensor:
    """``out = rmsnorm(allreduce(finalize))`` on every row of ``out`` viewed as ``[-1, H]``, ``H`` the norm weight's
    size; top-k from ``expert_weights``."""
    hidden_size = norm_weight.shape[0]
    _module(
        "rms_norm",
        "moe_finalize_all_reduce_norm",
        comm.world_size,
        hidden_size,
        cluster_size,
        expert_weights.shape[1],
        expert_weights.dtype,
    ).run(
        comm,
        out.view(-1, hidden_size),
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        norm_weight,
        float(eps),
    )
    return out


# Beam 1.0


def _sandwich_norm(
    comm: Communicator,
    fn: str,
    partial: torch.Tensor,
    residual: torch.Tensor,
    post_norm_weight: torch.Tensor,
    next_norm_weight: Optional[torch.Tensor],
    eps: float,
    scales: tuple[float, float],
    cluster_size: int,
    *prologue_args,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    combined = torch.empty_like(residual)
    normalized = torch.empty_like(residual) if next_norm_weight is not None else None
    if partial.shape[0] == 0:
        return combined, normalized
    top_k = (partial.shape[1],) if partial.ndim == 3 else ()
    _module(
        "sandwich_norm", fn, comm.world_size, partial.shape[-1], cluster_size, *top_k
    ).run(
        comm,
        partial,
        residual,
        post_norm_weight,
        next_norm_weight,
        combined,
        normalized,
        float(eps),
        float(scales[0]),
        float(scales[1]),
        *prologue_args,
    )
    return combined, normalized


def all_reduce_sandwich_norm(
    comm: Communicator,
    partial: torch.Tensor,
    residual: torch.Tensor,
    post_norm_weight: torch.Tensor,
    next_norm_weight: Optional[torch.Tensor],
    eps: float,
    scales: tuple[float, float],
    *,
    cluster_size: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``x = allreduce(partial)`` of ``partial [T, H]``, then ``combined = bf16(bf16(rmsnorm(x) * post_norm_weight *
    x_scale) + bf16(residual * residual_scale))`` with ``(x_scale, residual_scale) = scales``, and with
    ``next_norm_weight`` also ``normalized = rmsnorm(combined) * next_norm_weight``.

    :returns: ``(combined, normalized)``; ``normalized`` is None without ``next_norm_weight``.
    """
    assert partial.ndim == 2, partial.shape
    return _sandwich_norm(
        comm,
        "all_reduce_sandwich_norm",
        partial,
        residual,
        post_norm_weight,
        next_norm_weight,
        eps,
        scales,
        cluster_size,
    )


def topk_sum_all_reduce_sandwich_norm(
    comm: Communicator,
    partial: torch.Tensor,
    residual: torch.Tensor,
    post_norm_weight: torch.Tensor,
    next_norm_weight: Optional[torch.Tensor],
    eps: float,
    scales: tuple[float, float],
    *,
    routed_scale: float,
    cluster_size: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """:func:`all_reduce_sandwich_norm` on ``bf16(routed_scale * sum_k partial[:, k])`` of ``partial [T, K, H]`` (the
    routing-weighted expert rows, summed in fp32).

    :returns: ``(combined, normalized)``; ``normalized`` is None without ``next_norm_weight``.
    """
    assert partial.ndim == 3, partial.shape
    return _sandwich_norm(
        comm,
        "topk_sum_all_reduce_sandwich_norm",
        partial,
        residual,
        post_norm_weight,
        next_norm_weight,
        eps,
        scales,
        cluster_size,
        float(routed_scale),
    )
