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
# FlashInfer's gather + RMSNorm + NVFP4 quant per (lease, n): its plane and width.
_NORM_QUANT_OPS: dict = {}


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


def _weight_slice(comm: WorkspaceLease, weight: torch.Tensor) -> torch.Tensor:
    n_local = weight.shape[0] // comm.world_size
    return weight[comm.rank * n_local : (comm.rank + 1) * n_local]


def gemm_local(
    comm: WorkspaceLease, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """This rank's ``[M, N / world]`` column slice of ``x @ weight.T``."""
    from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

    return bf16_gemm_dispatch(x.contiguous(), _weight_slice(comm, weight), None)


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


def _norm_quant_op(comm: WorkspaceLease, n: int, eps: float):
    """FlashInfer's gather + norm + quant on ``comm``'s push plane, one per (lease, n),
    built collectively and compiled for ``eps`` on first use (the warmup runs before
    CUDA graph capture)."""
    # Leases are not hashable; holding the lease in the value keeps its id unique.
    _, op = _NORM_QUANT_OPS.get((id(comm), n), (None, None))
    if op is None:
        from flashinfer.comm.mnnvl_cutedsl_ag import (
            MNNVLCuteDSLAllGatherNormQuantWorkspace,
            PushPlane,
        )

        from sglang.srt.runtime_context import get_parallel

        slot_bytes = comm.capacity.push_slot_bytes
        op = MNNVLCuteDSLAllGatherNormQuantWorkspace(
            comm.world_size,
            comm.rank,
            slot_bytes // (n * 2),
            n,
            group=get_parallel().tp_group.device_group,
            push_plane=PushPlane(
                buffer=comm.data.push_slots[comm.rank],
                multicast_address=comm.data.push_mc,
                epoch=comm.control.epoch,
                half_bytes=comm.world_size * slot_bytes,
                atom_bytes=4,  # the push plane's LamportTrait word
            ),
        )
        _NORM_QUANT_OPS[(id(comm), n)] = (comm, op)
    op.compile(eps=eps)
    return op


def gemm_ag_norm_fp4_quant(
    comm: WorkspaceLease,
    x: torch.Tensor,
    weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Static NVFP4 ``(fp4 [M, N / 2] uint8, sf [M, N / 16] e4m3)`` of RMSNorm(``x @ weight.T``).

    The slice GEMM stores straight into the push plane when its kernel has a peer-store
    epilogue (same shapes and dispatch on every rank); otherwise FlashInfer publishes it.
    """
    from flashinfer.comm.mnnvl_cutedsl_ag import all_gather_rmsnorm_fp4_quantize

    from sglang.srt.layers.quantization.unquant import bf16_gemm_peer_store

    op = _norm_quant_op(comm, weight.shape[0], eps)  # before any rank pushes
    pushed = bf16_gemm_peer_store(
        x.contiguous(),
        _weight_slice(comm, weight),
        multicast_ptr=comm.data.push_mc,
        epoch_ptr=comm.control.epoch.data_ptr() + 4,  # EpochCounter phase word
        slot_bytes=comm.capacity.push_slot_bytes,
        rank=comm.rank,
        world_size=comm.world_size,
        layout="gather",
    )
    fp4, sf = all_gather_rmsnorm_fp4_quantize(
        None if pushed else gemm_local(comm, x, weight),
        norm_weight,
        global_scale.reshape(1),
        op,
        eps=eps,
        num_tokens=x.shape[0],
    )
    return fp4, sf.view(torch.float8_e4m3fn)
