"""Fused deferred-MoE finalize + 1shot push all-reduce [+ RMSNorm] (bf16).

One entry point, :func:`moe_finalize_all_reduce`, over
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

The un-normed result is what DeepSeek-V4.1 consumes (its consumer is the mHC
post-split, not an RMSNorm), so ``norm_weight=None`` is the primary
configuration; ``norm_weight`` + ``norm_eps`` give the K3-style fused norm.

Geometry: one thread-block cluster per token row plus a bumper cluster that
keeps the plane's phase counters uniform; ``cluster_size`` blocks share a
row (``hidden / cluster_size`` dims each). :func:`default_cluster_size` holds
the tuned default per hidden size and can be overridden per call.

Needs :func:`register_comm` once per process (the CustomAllReduceV2
``Communicator``); the ops key on ``world_size`` alone, like the K3 ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.kernels.ops.communication.all_reduce import Communicator


# Storage plane: the CustomAllReduceV2 Communicator (push plane only)

_COMM_MAP: dict[int, Communicator] = {}


def register_comm(comm: Communicator) -> None:
    """Register the CustomAllReduceV2 communicator whose push plane the fused
    kernel stages through.

    ``world_size`` is the whole key (the custom op takes nothing else), so at
    most one communicator per size may be registered in a process; a second
    group of the same size would silently inherit the first one's peer
    pointers and the symptom would be a hang, hence the assert.
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


# JIT module: one per (world_size, hidden_dim, top_k, cluster_size); the
# shared-add and norm variants are compiled into it and picked at call time.


@cache_once
def _jit_module(
    world_size: int, hidden_dim: int, top_k: int, cluster_size: int
) -> Module:
    assert cluster_size in valid_cluster_sizes(hidden_dim), (
        f"cluster_size={cluster_size} is not valid for hidden_dim={hidden_dim}; "
        f"choose from {valid_cluster_sizes(hidden_dim)}"
    )
    args = make_cpp_args(
        world_size, hidden_dim, top_k, cluster_size, is_arch_support_pdl()
    )
    return load_jit(
        "moe_finalize_all_reduce",
        *args,
        cuda_files=["distributed/all_reduce_fusion.cuh"],
        cuda_wrappers=[("run", f"MoeFinalizeAllReduceKernel<{args}>::run")],
    )


def compile_moe_finalize_all_reduce(
    world_size: int, hidden_dim: int, top_k: int, cluster_size: Optional[int] = None
) -> None:
    """Warm the JIT module (tests / benches precompile in parallel)."""
    _jit_module(
        world_size, hidden_dim, top_k, cluster_size or default_cluster_size(hidden_dim)
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
    _jit_module(world_size, hidden_dim, top_k, cluster_size).run(
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
    :param expert_weights: ``[T, top_k]`` bf16; any routed scaling factor is
                           already folded in (nothing is rescaled here).
    :param shared_output: optional ``[T, hidden_dim]`` bf16 added before the reduce.
    :param norm_weight: optional ``[hidden_dim]`` bf16 RMSNorm weight; with
                        ``norm_eps`` it turns on the fused norm epilogue.
    :param prefetch_metadata: let the kernel read the plane's phase counter and
                              the routing metadata before its PDL wait. Under
                              PDL the kernel may start as soon as the preceding
                              kernel *triggers*, and nothing earlier in the
                              stream is guaranteed complete until the wait: so
                              this is only valid when the preceding kernel is
                              not an all-reduce on the same plane AND the
                              producers of ``expanded_idx_to_permuted_idx`` /
                              ``expert_weights`` are known complete (a chain of
                              early-triggering kernels such as the TRT-LLM MoE
                              GEMMs is not). Defaults to False (wait first).
    :returns: a new ``[T, hidden_dim]`` bf16 tensor (not in place).
    """
    num_tokens = expert_weights.shape[0]
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
