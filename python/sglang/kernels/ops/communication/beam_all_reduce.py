# SPDX-License-Identifier: Apache-2.0
"""Beam BF16 all-reduce with FP32 sandwich normalization and rounded residuals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from sglang.kernels.ops.communication.all_reduce import Communicator


@cache_once
def _module(world_size: int, hidden_size: int, cluster: int, topk: int):
    return load_jit(
        "beam_all_reduce_norm",
        str(world_size),
        str(hidden_size),
        str(cluster),
        str(topk),
        cuda_files=["distributed/beam_all_reduce.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"BeamAllReduceNorm<{world_size}, {hidden_size}, {cluster}, {topk}>::run",
            )
        ],
        # Round every multiply and add separately, as the unfused norm does.
        extra_cuda_cflags=["--fmad=false"],
    )


def _cluster_size(world_size: int, rows: int) -> int:
    # Measured on 8xH200: 5-CTA clusters are faster at TP8 up to 16 rows; one CTA
    # per row is faster at TP2 and TP4.
    return 5 if world_size == 8 and rows <= 16 else 1


def beam_all_reduce_norm(
    comm: Communicator,
    branch: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    next_weight: Optional[torch.Tensor],
    eps: float,
    scales: tuple[float, float],
    *,
    routed_scale: float = 1.0,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """Return `(normalized, combined)` using a caller-owned V2 communicator.

    All ranks must call with identical shapes/epilogue and serialize use of the
    communicator. Inputs and weights are contiguous BF16 CUDA tensors; the
    complete row view must fit the communicator's push plane and counter array.
    A three-dimensional branch contains eight already routing-weighted expert
    outputs per token. Their FP32 sum is multiplied by routed_scale and rounded
    to BF16 before rank reduction. A two-dimensional branch is already combined.
    The post-norm, each scaled residual term, their sum, and the optional next
    norm retain separate BF16 rounding boundaries.
    """
    topk = branch.shape[1] if branch.ndim == 3 else 1
    if branch.ndim not in (2, 3) or branch.shape[-1] != 5120 or topk not in (1, 8):
        raise ValueError("Beam fusion requires [tokens, 5120] or [tokens, 8, 5120]")
    combined = torch.empty_like(residual)
    normalized = torch.empty_like(residual) if next_weight is not None else None
    if branch.shape[0]:
        _module(
            comm.world_size,
            branch.shape[-1],
            _cluster_size(comm.world_size, branch.shape[0]),
            topk,
        ).run(
            comm,
            branch,
            residual,
            post_weight,
            next_weight,
            combined,
            normalized,
            eps,
            *scales,
            routed_scale,
        )
    return normalized, combined
