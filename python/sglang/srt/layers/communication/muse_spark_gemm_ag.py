"""Column-parallel bf16 GEMM + all-gather for a replicated projection at decode M.

``out = x @ weight.T (+ b) (+ c)`` where ``x`` is identical on every TP rank
(e.g. an all-reduced tensor) and every rank needs the full result. Each rank
runs the bf16 GEMM dispatcher on its contiguous ``N / world`` row block of the
weight (reading ``1 / world`` of the bytes), then one kernel multicast-stores the
slice into every peer's push slots of the TP symmetric workspace and Lamport-spins
the full rows together. Any TP world size; needs the plane's multicast mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.communication.muse_spark_all_gather import (
    MAX_TOKENS,
    all_gather,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.symm_workspace import (
        WorkspaceLease,
    )

# The Muse Spark collectives' lease on the TP group's symmetric workspace, and the
# gathered projection's slot declared before it is taken.
_LEASE: Optional[WorkspaceLease] = None
_PROJECTION_SLOT_BYTES = 0


def muse_workspace() -> Optional[WorkspaceLease]:
    """The Muse Spark collectives' lease on the TP workspace.

    Every Muse collective (projection all-gather, unpool gather, finalize all-reduce)
    stages through one slot: the TP custom all-reduce's, grown to the gathered
    projection's when :func:`reserve_projection_workspace` declared one.
    """
    global _LEASE
    if _LEASE is None:
        from sglang.srt.distributed.device_communicators.symm_workspace import (
            WorkspaceRequest,
            request_group_workspace,
            v2_push_slot_bytes,
        )
        from sglang.srt.runtime_context import get_parallel

        tp_group = get_parallel().tp_group
        v2_slot = v2_push_slot_bytes(tp_group.cpu_group)
        if get_parallel().tp_size <= 1 or v2_slot is None:
            return None
        _LEASE = request_group_workspace(
            tp_group,
            WorkspaceRequest(
                owner="muse_spark",
                push_slot_bytes=max(v2_slot, _PROJECTION_SLOT_BYTES),
            ),
        )
    return _LEASE


def reserve_projection_workspace(n: int) -> None:
    """Declare the gathered projection's slot during model construction.

    Callers gate on the architecture. A larger slot than the lease already taken
    declares it again on the next :func:`muse_workspace` call.
    """
    global _LEASE, _PROJECTION_SLOT_BYTES
    from sglang.srt.distributed.device_communicators.symm_workspace import (
        is_multicast_supported,
    )
    from sglang.srt.runtime_context import get_parallel

    tp_group = get_parallel().tp_group
    # Only the shapes gemm_ag_supported routes past the small-projection bound.
    if (
        get_parallel().tp_size in (4, 8)
        and n in (4096, 8192)
        and is_multicast_supported(tp_group.cpu_group, tp_group.device)
    ):
        _PROJECTION_SLOT_BYTES = max(_PROJECTION_SLOT_BYTES, MAX_TOKENS * n * 2)
        if (
            _LEASE is not None
            and _LEASE.request.push_slot_bytes < _PROJECTION_SLOT_BYTES
        ):
            _LEASE = None


def gemm_local(
    comm: WorkspaceLease, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """This rank's ``[M, N / world]`` column slice of ``x @ weight.T``."""
    from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

    n_local = weight.shape[0] // comm.world_size
    w_slice = weight[comm.rank * n_local : (comm.rank + 1) * n_local]
    return bf16_gemm_dispatch(x.contiguous(), w_slice, None)


def gemm_ag(
    comm: WorkspaceLease,
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
