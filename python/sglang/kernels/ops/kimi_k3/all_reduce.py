"""K3 MNNVL fused all-reduce (bf16): zero-copy AR and AR+RMSNorm.

Four entry points over ``csrc/kimi_k3/comm/ar_fusion.cuh``, spanning two
algorithm families x two epilogues:

============  =========================  ==================================
              res (+ optional residual)  norm (fused RMSNorm on the latent)
============  =========================  ==================================
push (1shot)  :func:`all_reduce_push_res`   :func:`all_reduce_push_norm`
pull (2shot)  :func:`all_reduce_pull_res`   :func:`all_reduce_pull_norm`
============  =========================  ==================================

* **push** — 1shot multicast-push. Works on ANY contiguous bf16 tensor
  (input is read and written in place); reuses the CustomAllReduceV2 push
  plane, whose multicast base the plane itself carries.
  Best for small messages. Needs :func:`register_comm`.
* **pull** — low-SM NVLS 2shot ON the input, which must be allocated from
  multicast-bound symmetric memory (the caller passes its multicast VA):
  reduce-scatter + broadcast in place.
  Launch geometry defaults to :data:`RES_TUNING` /
  :data:`NORM_TUNING` and can be overridden per call via ``num_blocks`` /
  ``unroll`` (``num_blocks`` must be uniform across ranks per call).
  Barriers reuse the CustomAllReduceV2 barrier plane (same reservation
  protocol as the generic pull kernels, signaled via one multicast red).

Epilogue contracts: the ``res`` residual must be identical on every rank (a
fully reduced tensor such as the attn-res prefix sum) or absent; the
``norm`` input is the K3 latent|shared MoE buffer ([N, 3584] latent then
[N, 7168] shared, contiguous — the row layout is derived and hardcoded
C++-side).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

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


@cache_once
def _jit_module(world_size: int) -> Module:
    args = make_cpp_args(world_size, is_arch_support_pdl())
    cls = f"AllReduceFusionKernel<{args}>"
    return load_jit(
        "kimi_k3_all_reduce",
        *args,
        cuda_files=["kimi_k3/comm/ar_fusion.cuh"],
        cuda_wrappers=[
            ("push_res", f"{cls}::push_res"),
            ("push_norm", f"{cls}::push_norm"),
            ("pull_res", f"{cls}::pull_res"),
            ("pull_norm", f"{cls}::pull_norm"),
            ("finalize_push_norm", f"{cls}::finalize_push_norm"),
        ],
        extra_cuda_cflags=["-O3"],
    )


# Storage plane: the CustomAllReduceV2 Communicator


_COMM_MAP: dict[int, Communicator] = {}


def register_comm(comm: Communicator) -> None:
    """Register the CustomAllReduceV2 communication planes.

    The push kernels use the push plane, the pull kernels the barrier plane;
    both carry their own multicast base, so nothing else has to be threaded
    through here.
    """
    # world_size is the whole key, so at most one communicator per size can be
    # registered in a process. That matches how these ops are called -- the custom
    # ops below take world_size and nothing else, so a second group of the same
    # size would silently inherit the first one's peer pointers and semaphores,
    # and the symptom would be a hang or corruption rather than an error. Assert
    # it instead of letting the overwrite happen; widening to per-group handles
    # means changing the custom-op signatures, which is a separate change.
    prev = _COMM_MAP.get(comm.world_size)
    assert prev is None or prev is comm, (
        f"a different communicator is already registered for world_size="
        f"{comm.world_size}; these ops key only on world_size, so two groups of "
        f"the same size cannot coexist in one process"
    )
    _COMM_MAP[comm.world_size] = comm


class PullTuning(NamedTuple):
    num_blocks: int
    unroll: int  # 2, 4, 8, or 16 (every width is compiled into the module)


class PullTuningTable(NamedTuple):
    bands: tuple[tuple[int, PullTuning], ...]
    fallback: PullTuning

    def lookup(self, nbytes: int) -> PullTuning:
        for max_bytes, tuning in self.bands:
            if nbytes <= max_bytes:
                return tuning
        return self.fallback


_KB, _MB = 1024, 1024 * 1024

RES_TUNING = PullTuningTable(
    bands=(
        (128 * _KB, PullTuning(num_blocks=1, unroll=8)),
        (4 * _MB, PullTuning(num_blocks=16, unroll=8)),
    ),
    fallback=PullTuning(num_blocks=4, unroll=8),
)

NORM_TUNING = PullTuningTable(
    bands=(
        (512 * _KB, PullTuning(num_blocks=2, unroll=4)),
        (2 * _MB, PullTuning(num_blocks=12, unroll=2)),
    ),
    fallback=PullTuning(num_blocks=24, unroll=2),
)


def _resolve_tuning(
    table: PullTuningTable,
    *,
    nbytes: int,
    num_blocks: Optional[int],
    unroll: Optional[int],
) -> PullTuning:
    """Per-size tuned config, with explicit overrides taking precedence
    (C++-side, the block count is clamped to the semaphore capacity)."""
    tuned = table.lookup(nbytes)
    return PullTuning(
        num_blocks=num_blocks or tuned.num_blocks,
        unroll=unroll or tuned.unroll,
    )


# Custom ops, one per C++ entry point


@register_custom_op(mutates_args=["x"])
def _push_res_op(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> None:
    _jit_module(world_size).push_res(_COMM_MAP[world_size], x.view(-1), residual)


@register_custom_op(mutates_args=["x"])
def _push_norm_op(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_norm_rows: int,
) -> None:
    _jit_module(world_size).push_norm(
        _COMM_MAP[world_size], x.view(-1), weight, eps, num_norm_rows
    )


@register_custom_op(mutates_args=["out"])
def _finalize_push_norm_op(
    world_size: int,
    out: torch.Tensor,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    _jit_module(world_size).finalize_push_norm(
        _COMM_MAP[world_size],
        out.view(-1),
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        weight,
        eps,
    )


@register_custom_op(mutates_args=["x"])
def _pull_res_op(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    input_mc_ptr: int,
    num_blocks: int,
    unroll: int,
) -> None:
    _jit_module(world_size).pull_res(
        _COMM_MAP[world_size],
        x.view(-1),
        residual,
        input_mc_ptr,
        num_blocks,
        unroll,
    )


@register_custom_op(mutates_args=["x"])
def _pull_norm_op(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_norm_rows: int,
    input_mc_ptr: int,
    num_blocks: int,
    unroll: int,
) -> None:
    _jit_module(world_size).pull_norm(
        _COMM_MAP[world_size],
        x.view(-1),
        weight,
        eps,
        num_norm_rows,
        input_mc_ptr,
        num_blocks,
        unroll,
    )


def all_reduce_push_res(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """In-place ``x = allreduce(x) [+ residual]`` via 1shot multicast push.

    ``x`` may be any contiguous bf16 CUDA tensor whose byte size fits a slot
    of the registered push plane. Call :func:`register_comm` once beforehand.
    """
    residual_ = residual.view(-1) if residual is not None else None
    _push_res_op(world_size, x, residual_)
    return x


def all_reduce_push_norm(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    num_norm_rows: int,
) -> torch.Tensor:
    """In-place allreduce via 1shot multicast push + RMSNorm over the first
    ``num_norm_rows`` rows of ``x`` viewed as [numel / 3584, 3584]."""
    _push_norm_op(world_size, x, weight, eps, num_norm_rows)
    return x


def finalize_all_reduce_push_norm(
    world_size: int,
    out: torch.Tensor,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Deferred MoE finalize + 1shot push all-reduce + RMSNorm on EVERY row.

    ``out`` ([T, 3584] bf16) is output-only; each rank's partial latent
    (``sum_k expert_weights[t, k] * gemm2_out[idx[t*16 + k]]``, -1 slots
    skipped) is computed during the multicast staging pass from the
    trtllm-gen deferred-finalize triple (``do_finalize=False``) and never
    materializes in global memory. top_k is fixed to 16 (K3)."""
    _finalize_push_norm_op(
        world_size,
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        weight,
        eps,
    )
    return out


def all_reduce_pull_res(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    input_mc_ptr: int,
    num_blocks: Optional[int] = None,
    unroll: Optional[int] = None,
) -> torch.Tensor:
    """In-place ``x = allreduce(x) [+ residual]`` via low-SM NVLS 2shot.

    ``x`` MUST be allocated from multicast-bound symmetric memory and
    ``input_mc_ptr`` must be its multicast VA (it varies per call, unlike the
    barrier plane's own multicast base). Call :func:`register_comm` once
    beforehand.
    """
    tuning = _resolve_tuning(
        RES_TUNING,
        nbytes=x.numel() * x.element_size(),
        num_blocks=num_blocks,
        unroll=unroll,
    )
    residual_ = residual.view(-1) if residual is not None else None
    _pull_res_op(
        world_size, x, residual_, input_mc_ptr, tuning.num_blocks, tuning.unroll
    )
    return x


def all_reduce_pull_norm(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    num_norm_rows: int,
    input_mc_ptr: int,
    num_blocks: Optional[int] = None,
    unroll: Optional[int] = None,
) -> torch.Tensor:
    """In-place allreduce via low-SM NVLS 2shot + RMSNorm over the first
    ``num_norm_rows`` rows of ``x`` viewed as [numel / 3584, 3584]; ``x``
    must live in multicast-bound symmetric memory."""
    tuning = _resolve_tuning(
        NORM_TUNING,
        nbytes=x.nbytes,
        num_blocks=num_blocks,
        unroll=unroll,
    )
    _pull_norm_op(
        world_size,
        x,
        weight,
        eps,
        num_norm_rows,
        input_mc_ptr,
        tuning.num_blocks,
        tuning.unroll,
    )
    return x
