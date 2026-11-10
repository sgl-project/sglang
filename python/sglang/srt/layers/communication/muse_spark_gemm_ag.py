"""Column-parallel bf16 GEMM + all-gather for a replicated projection at decode M.

``out = x @ weight.T (+ b) (+ c)`` where ``x`` is identical on every TP rank
(e.g. an all-reduced tensor) and every rank needs the full result. Each rank
runs the bf16 GEMM dispatcher on its contiguous ``N / world`` row block of the
weight (reading ``1 / world`` of the bytes), then one kernel multicast-stores the
slice into every peer's CustomAllReduceV2 push workspace and Lamport-spins the
full rows together. Any TP world size; needs the plane's multicast mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.communication.muse_spark_all_gather import (
    MAX_TOKENS,
    all_gather,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )


def reserve_projection_workspace(comm: Optional[CustomAllReduceV2], n: int) -> None:
    """Reserve shared push storage during model construction, before capture.

    Keep the all-reduce thresholds and counter/launch geometry unchanged.
    The gather strides over its output using the existing counter capacity.
    """
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    if (
        isinstance(comm, CustomAllReduceV2)
        and not comm.disabled
        and comm.has_multicast
        and comm.world_size in (4, 8)
        and n in (4096, 8192)
    ):
        comm.reserve_push_capacity(MAX_TOKENS * n * 2)


def gemm_local(
    comm: CustomAllReduceV2, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """This rank's ``[M, N / world]`` column slice of ``x @ weight.T``."""
    from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

    n_local = weight.shape[0] // comm.world_size
    w_slice = weight[comm.rank * n_local : (comm.rank + 1) * n_local]
    return bf16_gemm_dispatch(x.contiguous(), w_slice, None)


def gemm_ag(
    comm: CustomAllReduceV2,
    x: torch.Tensor,
    weight: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    c: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``x @ weight.T (+ b) (+ c)``; ``weight`` is the FULL replicated [N, K]."""
    local = gemm_local(comm, x, weight)
    if out is None:
        out = torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)
    return all_gather(comm, local, out, b, c)
