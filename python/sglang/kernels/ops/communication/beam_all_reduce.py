"""Beam BF16 all-reduce with FP32 sandwich normalization and rounded residuals."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _module(world_size: int, hidden_size: int, cluster: int = 1, topk: int = 1):
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
        extra_cuda_cflags=["--fmad=false"],
    )


def _cluster_size(world_size: int, rows: int, topk: int) -> int:
    # TP8 benefits from distributing the smallest rows across SMs; at larger
    # batches a single CTA per row avoids cluster synchronization overhead.
    if world_size == 8:
        return 5 if rows <= (16 if topk == 8 else 8) else 1
    return 5 if world_size == 2 and rows == 1 and topk == 8 else 1


def beam_all_reduce_norm(
    comm, branch, residual, post_weight, next_weight, eps, scales, *, routed_scale=1.0
):
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
            _cluster_size(comm.world_size, branch.shape[0], topk),
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
