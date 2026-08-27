"""K3 MNNVL fused all-reduce (bf16): zero-copy AR and AR+RMSNorm.

The plain ``res`` pair runs on the shared NVLink collectives, which carry the
residual through the same reduction; only the ``norm`` epilogues still need
``csrc/kimi_k3/comm/ar_fusion.cuh``:

============  =========================  ==================================
              res (+ optional residual)  norm (fused RMSNorm on the latent)
============  =========================  ==================================
push (1shot)  :func:`all_reduce_push_res`   :func:`all_reduce_push_norm`
pull (2shot)  :func:`all_reduce_pull_res`   :func:`all_reduce_pull_norm`
============  =========================  ==================================

* **push** -- 1shot multicast-push. Works on ANY contiguous bf16 tensor
  (input is read and written in place); reuses the CustomAllReduceV2 push
  plane, whose multicast base the plane itself carries.
  Best for small messages. Needs :func:`comm.register`.
* **pull** -- low-SM NVLS 2shot ON the input, which must be allocated from
  multicast-bound symmetric memory (the caller passes its multicast VA):
  reduce-scatter + broadcast in place.
  The ``res`` block count is chosen for the size unless a call overrides it,
  which the side-stream caller does to keep the collective off the SMs its
  GEMMs want; ``norm`` still resolves geometry from :data:`NORM_TUNING`.
  Barriers reuse the CustomAllReduceV2 barrier plane.

Epilogue contracts: the ``res`` residual must be identical on every rank (a
fully reduced tensor such as the attn-res prefix sum) or absent; the
``norm`` input is the K3 latent|shared MoE buffer ([N, 3584] latent then
[N, 7168] shared, contiguous -- the row layout is derived and hardcoded
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
from sglang.kernels.ops.communication import nvlink_comm as nvl
from sglang.kernels.ops.kimi_k3 import comm as k3_comm
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(world_size: int) -> Module:
    args = make_cpp_args(world_size, is_arch_support_pdl())
    cls = f"AllReduceFusionKernel<{args}>"
    return load_jit(
        "kimi_k3_all_reduce",
        *args,
        cuda_files=["kimi_k3/comm/ar_fusion.cuh"],
        cuda_wrappers=[
            ("push_norm", f"{cls}::push_norm"),
            ("pull_norm", f"{cls}::pull_norm"),
            ("finalize_push_norm", f"{cls}::finalize_push_norm"),
        ],
        extra_cuda_cflags=["-O3"],
    )


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


# 16 bytes is the collectives' vector, so viewing a flat buffer as [N/8, 8]
# keeps the "any contiguous bf16 tensor" contract while giving them the [rows,
# hidden] shape they match on. The row width does not affect the all-reduce --
# measured within 0.1% of viewing at the real hidden size.
_VEC_ELEMS = 8


def _as_rows(x: torch.Tensor) -> torch.Tensor:
    return x.view(-1, _VEC_ELEMS)


@register_custom_op(mutates_args=["x"])
def _push_res_op(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> None:
    rows = _as_rows(x)
    nvl.all_reduce_push(
        k3_comm.get(world_size),
        rows,
        rows,
        None if residual is None else _as_rows(residual),
    )


@register_custom_op(mutates_args=["x"])
def _push_norm_op(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_norm_rows: int,
) -> None:
    _jit_module(world_size).push_norm(
        k3_comm.get(world_size), x.view(-1), weight, eps, num_norm_rows
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
        k3_comm.get(world_size),
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
) -> None:
    rows = _as_rows(x)
    nvl.all_reduce_pull(
        k3_comm.get(world_size),
        rows,
        rows,
        None if residual is None else _as_rows(residual),
        in_mc_ptr=input_mc_ptr,
        out_mc_ptr=input_mc_ptr,
        num_blocks_hint=num_blocks,
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
        k3_comm.get(world_size),
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
    of the registered push plane. Call :func:`comm.register` once beforehand.
    """
    _push_res_op(world_size, x, residual)
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
    num_blocks: int = 0,
) -> torch.Tensor:
    """In-place ``x = allreduce(x) [+ residual]`` via low-SM NVLS 2shot.

    ``x`` MUST be allocated from multicast-bound symmetric memory and
    ``input_mc_ptr`` must be its multicast VA (it varies per call, unlike the
    barrier plane's own multicast base). Call :func:`comm.register` once
    beforehand.
    """
    _pull_res_op(world_size, x, residual, input_mc_ptr, num_blocks)
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
