"""Fused NVLS pull all-reduce (``csrc/distributed/fused_all_reduce/pull/``, one file per epilogue).

2-shot over a multicast-bound symmetric buffer: every rank ``multimem.ld_reduce``s the rows of its shard (the
reduce-scatter happens in the switch), runs the epilogue on them, and multicasts what every rank needs (the
all-gather). The partials must live in symmetric memory; the caller passes their multicast VA. Barriers use the
communicator's pull plane: every rank must launch the same ``num_blocks``.

Thin ops named in compute order (``all_reduce``, then the epilogue). Each takes the communicator and its launch
configuration explicitly (``num_blocks``, ``rows_per_group``); choosing them is the caller's job. One CTA of
``hidden_size / 8`` threads per row, one 16-byte vector each.

- Kimi K3: :func:`all_reduce_norm_prefix`
- Gemma 4: :func:`all_reduce_norm_residual_norm_quant` (every site but the last), :func:`all_reduce_norm_residual`
  (the last layer), rounded as Hugging Face's layer
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

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.kernels.ops.communication.all_reduce import Communicator


@cache_once
def _module(pattern: str, fn: str, hidden_size: int) -> Module:
    """``fn<hidden_size, pdl>`` of ``fused_all_reduce/pull/<pattern>.cuh`` exported as ``run``: one CTA of
    ``hidden_size / 8`` threads per row, one 16-byte vector each, so ``hidden_size`` is a multiple of 256 up to 8192."""
    assert hidden_size % 256 == 0 and hidden_size <= 8192, (
        f"unsupported hidden_size {hidden_size}"
    )
    cpp_args = make_cpp_args(hidden_size, is_arch_support_pdl())
    return load_jit(
        f"fused_all_reduce_pull_{fn}",
        *cpp_args,
        cuda_files=[f"distributed/fused_all_reduce/pull/{pattern}.cuh"],
        cuda_wrappers=[("run", f"host::distributed::pull::{fn}<{cpp_args}>")],
    )


# Kimi K3


def all_reduce_norm_prefix(
    comm: Communicator,
    x: torch.Tensor,
    x_mc: int,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    num_norm_rows: int,
    num_blocks: int,
    rows_per_group: int,
) -> torch.Tensor:
    """In place: allreduce, then RMSNorm over the first ``num_norm_rows`` rows of ``x`` viewed as ``[-1, H]``, ``H``
    the norm weight's size; ``x_mc`` is ``x``'s multicast VA. The rows are split evenly over the ranks;
    ``rows_per_group`` is 1, 2, 4 or 8."""
    hidden_size = norm_weight.shape[0]
    _module("rms_norm", "all_reduce_norm_prefix", hidden_size).run(
        comm,
        x.view(-1, hidden_size),
        x_mc,
        norm_weight,
        float(eps),
        num_norm_rows,
        num_blocks,
        rows_per_group,
    )
    return x


# Gemma 4: the residual stream stays row-sharded; rank r owns rows [row0, row0 + rows) of every partial


def all_reduce_norm_residual_norm_quant(
    comm: Communicator,
    partial: torch.Tensor,
    partial_mc: int,
    residual: torch.Tensor,
    post_norm_weight: torch.Tensor,
    next_norm_weight: torch.Tensor,
    scalar: Optional[torch.Tensor],
    eps: float,
    out: torch.Tensor,
    out_mc: int,
    scales: torch.Tensor,
    scales_mc: int,
    *,
    row0: int,
    rows: int,
    num_blocks: int,
    rows_per_group: int,
) -> None:
    """``a = bf16(rmsnorm(allreduce(partial)) * post_norm_weight)``, ``v = bf16(a + residual)``, times ``scalar``
    (bf16) when given, into ``residual`` (this rank's ``[rows, H]``), then ``out``, ``scales`` = the per-128 FP8 quant
    of ``bf16(rmsnorm(v) * next_norm_weight)`` for every row (``scales [T, H / 128]`` fp32, any strides, e.g.
    DeepGEMM's MN-major layout)."""
    hidden_size = partial.shape[1]
    _module(
        "rms_norm_residual", "all_reduce_norm_residual_norm_quant", hidden_size
    ).run(
        comm,
        partial,
        partial_mc,
        residual,
        post_norm_weight,
        next_norm_weight,
        scalar,
        float(eps),
        out,
        out_mc,
        scales,
        scales_mc,
        row0,
        rows,
        num_blocks,
        rows_per_group,
    )


def all_reduce_norm_residual(
    comm: Communicator,
    partial: torch.Tensor,
    partial_mc: int,
    residual: torch.Tensor,
    post_norm_weight: torch.Tensor,
    scalar: Optional[torch.Tensor],
    eps: float,
    out: torch.Tensor,
    out_mc: int,
    *,
    row0: int,
    rows: int,
    num_blocks: int,
    rows_per_group: int,
) -> None:
    """The last layer: ``v = bf16(bf16(rmsnorm(allreduce(partial)) * post_norm_weight) + residual)``, times
    ``scalar`` (bf16) when given, into ``residual`` (this rank's ``[rows, H]``) and, as bf16, into ``out`` for every
    row (``out`` may be ``partial``)."""
    hidden_size = partial.shape[1]
    _module("rms_norm_residual", "all_reduce_norm_residual", hidden_size).run(
        comm,
        partial,
        partial_mc,
        residual,
        post_norm_weight,
        scalar,
        float(eps),
        out,
        out_mc,
        row0,
        rows,
        num_blocks,
        rows_per_group,
    )
