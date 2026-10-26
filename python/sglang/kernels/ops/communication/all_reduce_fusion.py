"""Fused deferred-MoE finalize + 1shot push all-reduce [+ RMSNorm] (bf16).

The base entry point, :func:`moe_finalize_all_reduce`, wraps
``csrc/distributed/all_reduce_fusion.cuh``::

    out[t] = allreduce( sum_k expert_weights[t, k] * gemm2_out[idx[t*top_k + k]]
                        (+ shared_output[t]) )            # then, optionally,
    out[t] = out[t] * rsqrt(mean(out[t]^2) + eps) * norm_weight

The rank-local finalize (the trtllm-gen ``do_finalize=False`` triple, see
``moe_runner/flashinfer_trtllm.py``) is computed in registers and pushed
straight into every peer's CustomAllReduceV2 push slot, so it never
materializes; ``idx == -1`` slots (EP: non-local expert) contribute nothing.
Small-batch only: the whole ``[T, hidden]`` bf16 row view must fit one push
slot (checked C++-side; :func:`fits_push_slot` lets callers pre-check).

The base entry point needs :func:`register_comm` once per process and keys on
``world_size`` alone. :func:`moe_finalize_shared_gate_all_reduce` takes the
caller's communicator directly and applies an FP32 shared gate after rounding
the routed values to BF16, before the collective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.kernels.ops.communication.all_reduce import Communicator


# Storage plane: the CustomAllReduceV2 Communicator (push plane only)

# Norm-epilogue launch policy, hidden 6144 (TP4 on B300, TP8 on H200): 1shot below
# TWO_SHOT_MIN_TOKENS[world_size], then 2shot with the first ``(max_rows, vecs_per_thread,
# cluster_size)`` entry that covers the batch. One vector per thread is faster while
# the per-row clusters fit one wave; the finalize kernel uses more registers, so it
# runs out of residency at fewer rows than the plain one.
TWO_SHOT_MIN_TOKENS = {4: 48, 8: 8}
_TWO_SHOT_GEOMETRY = {
    False: ((272, 1, 4), (None, 2, 3)),
    True: ((208, 1, 4), (232, 1, 3), (None, 2, 3)),
}

_COMM_MAP: dict[int, Communicator] = {}


def register_comm(comm: Communicator) -> None:
    """Register the CustomAllReduceV2 communicator whose push plane the fused
    kernel stages through. ``world_size`` is the whole key, so at most one
    communicator per size may be registered in a process.
    """
    prev = _COMM_MAP.get(comm.world_size)
    assert prev is None or prev is comm, (
        f"a different communicator is already registered for world_size="
        f"{comm.world_size}; these ops key only on world_size, so two groups of "
        f"the same size cannot coexist in one process"
    )
    _COMM_MAP[comm.world_size] = comm


def get_registered_comm(world_size: int) -> Optional[Communicator]:
    return _COMM_MAP.get(world_size)


# Geometry

_VEC_ELEMS = 8  # bf16 per 16B vector = per thread
_MAX_CLUSTER_SIZE = 8  # portable cluster size limit


def valid_cluster_sizes(hidden_dim: int) -> list[int]:
    """Cluster sizes the kernel can be built for at this hidden width: whole
    16B vectors per row, whole warps per block, <= 1024 threads, <= 8 blocks."""
    if hidden_dim % _VEC_ELEMS != 0:
        return []
    row_vecs = hidden_dim // _VEC_ELEMS
    return [
        c
        for c in range(1, _MAX_CLUSTER_SIZE + 1)
        if row_vecs % c == 0 and (row_vecs // c) % 32 == 0 and row_vecs // c <= 1024
    ]


@cache_once
def default_cluster_size(hidden_dim: int) -> int:
    if hidden_dim % 1024 == 0 and hidden_dim <= 8192:
        return hidden_dim // 1024
    if hidden_dim % 512 == 0 and hidden_dim <= 3584:
        return hidden_dim // 512
    candidates = valid_cluster_sizes(hidden_dim)
    if not candidates:
        raise ValueError(
            f"hidden_dim={hidden_dim} has no valid cluster geometry (needs a "
            f"multiple of {_VEC_ELEMS * 32} bf16)"
        )
    # closest to 128 threads per block, larger block on ties
    return min(candidates, key=lambda c: (abs(hidden_dim // _VEC_ELEMS // c - 128), c))


def fits_push_slot(max_push_size: int, num_tokens: int, hidden_dim: int) -> bool:
    """Whether a ``[num_tokens, hidden_dim]`` bf16 row view fits one push slot
    (``CustomAllReduceV2.max_push_size``)."""
    return 0 < num_tokens * hidden_dim * 2 <= max_push_size


# JIT module: one per (world_size, hidden_dim, top_k, cluster_size, weight_dtype); the
# shared-add and norm variants are compiled into it and picked at call time.


def require_cluster_launch_arch() -> None:
    if is_hip_runtime() or get_jit_cuda_arch().major < 9:
        raise RuntimeError(
            "fused all-reduce cluster kernels require CUDA SM90 or newer"
        )


@cache_once
def _jit_module(
    world_size: int,
    hidden_dim: int,
    top_k: int,
    cluster_size: int,
    weight_dtype: torch.dtype,
) -> Module:
    require_cluster_launch_arch()
    assert cluster_size in valid_cluster_sizes(hidden_dim), (
        f"cluster_size={cluster_size} is not valid for hidden_dim={hidden_dim}; "
        f"choose from {valid_cluster_sizes(hidden_dim)}"
    )
    args = make_cpp_args(
        world_size, hidden_dim, top_k, cluster_size, is_arch_support_pdl(), weight_dtype
    )
    return load_jit(
        "moe_finalize_all_reduce",
        *args,
        cuda_files=["distributed/all_reduce_fusion.cuh"],
        cuda_wrappers=[
            (name, f"MoeFinalizeAllReduceKernel<{args}>::{name}")
            for name in (("run_plain",) if top_k == 0 else ("run", "run_residual_norm"))
        ],
    )


@register_custom_op(mutates_args=["out"])
def _moe_finalize_all_reduce_op(
    world_size: int,
    hidden_dim: int,
    top_k: int,
    cluster_size: int,
    out: torch.Tensor,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    norm_weight: Optional[torch.Tensor],
    norm_eps: float,
    prefetch_metadata: bool,
) -> None:
    comm = _COMM_MAP.get(world_size)
    assert comm is not None, (
        f"no communicator registered for world_size={world_size}; call "
        "all_reduce_fusion.register_comm(comm.obj) first"
    )
    _jit_module(world_size, hidden_dim, top_k, cluster_size, expert_weights.dtype).run(
        comm,
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        norm_weight,
        norm_eps,
        prefetch_metadata,
    )


def moe_finalize_all_reduce(
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    top_k: int,
    shared_output: Optional[torch.Tensor] = None,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: Optional[float] = None,
    *,
    world_size: int,
    hidden_dim: int,
    cluster_size: Optional[int] = None,
    prefetch_metadata: bool = False,
) -> torch.Tensor:
    """Deferred MoE finalize [+ shared add] -> 1shot push all-reduce [-> RMSNorm].

    :param gemm2_out: ``[P, hidden_dim]`` bf16, trtllm-gen permuted / padded rows.
    :param expanded_idx_to_permuted_idx: ``[T * top_k]`` int32, ``-1`` = dropped slot.
    :param expert_weights: ``[T, top_k]`` bf16 or fp32; any routed scaling factor is
                           already folded in (nothing is rescaled here).
    :param shared_output: optional ``[T, hidden_dim]`` bf16 added before the reduce.
    :param norm_weight: optional ``[hidden_dim]`` bf16 RMSNorm weight; with
                        ``norm_eps`` it turns on the fused norm epilogue.
    :param prefetch_metadata: read the plane's phase counter and the routing
                              metadata before the PDL wait; valid only when the
                              preceding kernel is not an all-reduce on the same
                              plane and the producers of
                              ``expanded_idx_to_permuted_idx`` /
                              ``expert_weights`` are complete. Defaults to False.
    :returns: a new ``[T, hidden_dim]`` bf16 tensor (not in place).
    """
    num_tokens = expert_weights.shape[0]
    assert expert_weights.dtype in (torch.bfloat16, torch.float32)
    assert expert_weights.shape[1] == top_k, (expert_weights.shape, top_k)
    assert (norm_weight is None) == (norm_eps is None), (
        "norm_weight and norm_eps must be given together"
    )
    out = torch.empty(
        num_tokens, hidden_dim, dtype=torch.bfloat16, device=gemm2_out.device
    )
    if num_tokens == 0:  # nothing staged: no phase flip on any rank, stays in step
        return out
    _moe_finalize_all_reduce_op(
        world_size,
        hidden_dim,
        top_k,
        cluster_size or default_cluster_size(hidden_dim),
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        norm_weight,
        float(norm_eps) if norm_eps is not None else 0.0,
        prefetch_metadata,
    )
    return out


@register_custom_op(mutates_args=["out", "residual_out"])
def _moe_finalize_all_reduce_residual_norm_op(
    world_size: int,
    hidden_dim: int,
    top_k: int,
    cluster_size: int,
    out: torch.Tensor,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    norm_weight: torch.Tensor,
    norm_eps: float,
    residual: torch.Tensor,
    residual_out: torch.Tensor,
    weight_offset: float,
    two_shot: bool,
    vecs_per_thread: int,
) -> None:
    _jit_module(
        world_size, hidden_dim, top_k, cluster_size, expert_weights.dtype
    ).run_residual_norm(
        _COMM_MAP[world_size],
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        norm_weight,
        norm_eps,
        residual,
        residual_out,
        weight_offset,
        two_shot,
        vecs_per_thread,
    )


@register_custom_op(mutates_args=["out", "residual_out"])
def _all_reduce_residual_norm_op(
    world_size: int,
    hidden_dim: int,
    cluster_size: int,
    out: torch.Tensor,
    input: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    norm_eps: float,
    residual: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    weight_offset: float,
    two_shot: bool,
    vecs_per_thread: int,
) -> None:
    _jit_module(world_size, hidden_dim, 0, cluster_size, torch.bfloat16).run_plain(
        _COMM_MAP[world_size],
        out,
        input,
        norm_weight,
        norm_eps,
        residual,
        residual_out,
        weight_offset,
        two_shot,
        vecs_per_thread,
    )


def _norm_launch(
    world_size: int,
    num_tokens: int,
    hidden_dim: int,
    finalize: bool,
    two_shot: Optional[bool],
    vecs_per_thread: Optional[int],
    cluster_size: Optional[int],
) -> tuple[bool, int, int]:
    """``(two_shot, vecs_per_thread, cluster_size)``; explicit arguments win."""
    if two_shot is None:
        two_shot = num_tokens >= TWO_SHOT_MIN_TOKENS.get(world_size, 48)
    if vecs_per_thread is None and cluster_size is None and two_shot:
        vecs_per_thread, cluster_size = next(
            (vecs, cluster)
            for max_rows, vecs, cluster in _TWO_SHOT_GEOMETRY[finalize]
            if max_rows is None or num_tokens <= max_rows
        )
        if (hidden_dim // _VEC_ELEMS) % (cluster_size * vecs_per_thread * 32) != 0:
            vecs_per_thread, cluster_size = None, None
    return (
        two_shot,
        vecs_per_thread or 1,
        cluster_size or default_cluster_size(hidden_dim),
    )


def moe_finalize_all_reduce_residual_norm(
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    top_k: int,
    shared_output: Optional[torch.Tensor],
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    *,
    world_size: int,
    hidden_dim: int,
    gemma: bool = False,
    residual_out: Optional[torch.Tensor] = None,
    cluster_size: Optional[int] = None,
    two_shot: Optional[bool] = None,
    vecs_per_thread: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`moe_finalize_all_reduce` whose norm epilogue first adds ``residual``.

    :returns: ``(normed, residual_out)``; ``residual_out`` defaults to ``residual``
              (updated in place, like ``fused_add_rmsnorm``).
    """
    num_tokens = expert_weights.shape[0]
    out = torch.empty(
        num_tokens, hidden_dim, dtype=torch.bfloat16, device=gemm2_out.device
    )
    residual_out = residual if residual_out is None else residual_out
    if num_tokens == 0:
        return out, residual_out
    two_shot, vecs_per_thread, cluster_size = _norm_launch(
        world_size,
        num_tokens,
        hidden_dim,
        True,
        two_shot,
        vecs_per_thread,
        cluster_size,
    )
    _moe_finalize_all_reduce_residual_norm_op(
        world_size,
        hidden_dim,
        top_k,
        cluster_size,
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        norm_weight,
        float(norm_eps),
        residual,
        residual_out,
        1.0 if gemma else 0.0,
        two_shot,
        vecs_per_thread,
    )
    return out, residual_out


def all_reduce_residual_norm(
    input: torch.Tensor,
    residual: Optional[torch.Tensor],
    norm_weight: Optional[torch.Tensor],
    norm_eps: Optional[float],
    *,
    world_size: int,
    gemma: bool = False,
    residual_out: Optional[torch.Tensor] = None,
    cluster_size: Optional[int] = None,
    two_shot: Optional[bool] = None,
    vecs_per_thread: Optional[int] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """1shot push all-reduce of a ``[T, hidden]`` bf16 partial sum [-> residual add] [-> RMSNorm].

    :returns: ``(out, residual_out)``; ``out`` is normalized when ``norm_weight`` is given,
              ``residual_out`` defaults to ``residual`` (updated in place).
    """
    assert (norm_weight is None) == (norm_eps is None)
    assert residual is None or norm_weight is not None
    num_tokens, hidden_dim = input.shape
    out = torch.empty_like(input)
    if residual is not None and residual_out is None:
        residual_out = residual
    if num_tokens == 0:
        return out, residual_out
    if norm_weight is None:
        two_shot, vecs_per_thread = bool(two_shot), 1
        cluster_size = cluster_size or default_cluster_size(hidden_dim)
    else:
        two_shot, vecs_per_thread, cluster_size = _norm_launch(
            world_size,
            num_tokens,
            hidden_dim,
            False,
            two_shot,
            vecs_per_thread,
            cluster_size,
        )
    _all_reduce_residual_norm_op(
        world_size,
        hidden_dim,
        cluster_size,
        out,
        input,
        norm_weight,
        float(norm_eps) if norm_eps is not None else 0.0,
        residual,
        residual_out,
        1.0 if gemma else 0.0,
        two_shot,
        vecs_per_thread,
    )
    return out, residual_out


@cache_once
def _jit_shared_gate_module(
    world_size: int,
    hidden_dim: int,
    top_k: int,
    cluster_size: int,
    weight_dtype: torch.dtype,
) -> Module:
    require_cluster_launch_arch()
    assert cluster_size in valid_cluster_sizes(hidden_dim)
    args = make_cpp_args(
        world_size,
        hidden_dim,
        top_k,
        cluster_size,
        is_arch_support_pdl(),
        weight_dtype,
        False,  # kMhc
        False,  # kQuant
        False,  # kCollapse
        True,  # kGatedShared
    )
    return load_jit(
        "moe_finalize_shared_gate_all_reduce",
        *args,
        cuda_files=["distributed/all_reduce_fusion.cuh"],
        cuda_wrappers=[("run", f"MoeFinalizeAllReduceKernel<{args}>::run_shared_gate")],
    )


def moe_finalize_shared_gate_all_reduce(
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: torch.Tensor,
    shared_gate: torch.Tensor,
    comm: Communicator,
    *,
    cluster_size: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Finalize routed BF16 values, gate the shared branch, then all-reduce.

    ``shared_gate`` is FP32 and is applied after the routed BF16 rounding.
    The caller owns the communicator and joins all input-producing streams.
    """
    _, hidden_dim = shared_output.shape
    if out is None:
        out = torch.empty_like(shared_output)
    _jit_shared_gate_module(
        comm.world_size,
        hidden_dim,
        expert_weights.shape[1],
        cluster_size or default_cluster_size(hidden_dim),
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
