"""TP all-reduce fused with the gated HC write-back and the next grouped norm.

Wraps ``csrc/distributed/hc_all_reduce_combine_norm.cuh``: one kernel in place of
the CustomAllReduceV2 1shot push all-reduce of a ``[rows, 2560]`` BF16 partial
sum followed by ``hc_combine_apply_norm`` (4 branches of 2560), bit-identical to
that pair. The kernel stages through the push plane of the communicator the
unfused all-reduce would have used.
"""

from __future__ import annotations

import logging
import os
import time
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

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.kernels.ops.communication.all_reduce import Communicator

logger = logging.getLogger(__name__)

HC_COUNT = 4
HIDDEN = 2560
# Rows the bitwise claim was checked at; the kernel itself takes up to 16.
MAX_ROWS = 4
# When the next kernel (the next read's mix) may launch and prefetch: "entry", "wait" (once
# the producer is complete) or "reduced" (the stock all-reduce's point); all give the same bits.
_TRIGGERS = {"entry": 0, "wait": 1, "reduced": 2}
_TRIGGER = _TRIGGERS[os.environ.get("SGLANG_HC_AR_COMBINE_NORM_TRIGGER", "wait")]


@cache_once
def _jit_module(world_size: int) -> Module:
    if is_hip_runtime() or get_jit_cuda_arch().major < 9:
        raise RuntimeError("the fused HC all-reduce needs CUDA SM90 or newer clusters")
    args = make_cpp_args(world_size, is_arch_support_pdl())
    return load_jit(
        "hc_all_reduce_combine_norm",
        *args,
        cuda_files=["distributed/hc_all_reduce_combine_norm.cuh"],
        cuda_wrappers=[("run", f"HcAllReduceCombineNorm<{args}>::run")],
    )


def warmup(world_size: int) -> None:
    """Build (or load) the module outside the serving path."""
    _jit_module(world_size)


_WARMED: set = set()


def warmup_on_all_ranks(*, group, world_sizes) -> None:
    """Build (or load) the module for each of ``world_sizes`` on every rank of
    ``group``, a GroupCoordinator whose ranks construct the same layers, and
    raise on all of them if any rank failed. A rank without the module would
    fail its first fused call while its peers poll for its push."""
    key = (group.unique_name, tuple(sorted({w for w in world_sizes if w > 1})))
    if key in _WARMED:
        return
    _WARMED.add(key)
    error = None
    start = time.perf_counter()
    try:
        for world_size in key[1]:
            _jit_module(world_size)
    except Exception as exc:  # noqa: BLE001 -- reported below, on every rank
        error = exc
    import torch.distributed as dist

    failed = torch.zeros(group.world_size, dtype=torch.int32, device="cpu")
    failed[group.rank_in_group] = int(error is not None)
    dist.all_reduce(failed, group=group.cpu_group)
    if failed.any():
        ranks = failed.nonzero().flatten().tolist()
        raise RuntimeError(
            f"SGLANG_HC_AR_COMBINE_NORM: the fused kernel failed to build on "
            f"rank(s) {ranks} of {group.unique_name} (world sizes {list(key[1])})"
            + (f": {error!r}" if error is not None else "")
            + "; unset SGLANG_HC_AR_COMBINE_NORM to run the unfused all-reduce"
        ) from error
    logger.info(
        "SGLANG_HC_AR_COMBINE_NORM: fused kernel ready on every rank of %s, "
        "world sizes %s (%.1f s)",
        group.unique_name,
        list(key[1]),
        time.perf_counter() - start,
    )


def one_shot_push_plane(coordinator, x: torch.Tensor) -> Optional[object]:
    """The CustomAllReduceV2 whose 1shot push kernel ``coordinator.all_reduce(x)``
    would launch, or None when that call takes any other path (the fused kernel is
    bit-identical only to that one)."""
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    ca = coordinator.ca_comm
    hpu, xpu = coordinator.hpu_communicator, coordinator.xpu_communicator
    if (
        coordinator.world_size == 1
        or not isinstance(ca, CustomAllReduceV2)
        or ca.disabled
        or x.is_cpu
        or torch.compiler.is_compiling()
        or (hpu is not None and not hpu.disabled)
        or (xpu is not None and not xpu.disabled)
    ):
        return None
    mscclpp = coordinator.pymscclpp_comm
    use_mscclpp = mscclpp is not None and mscclpp.should_mscclpp_allreduce(x)
    if use_mscclpp or not ca.should_custom_ar(x):
        return None
    # GroupCoordinator.all_reduce tries FlashInfer before the out-of-place resolver.
    if coordinator._can_use_flashinfer_allreduce(x):
        return None
    method = coordinator._resolve_outplace_all_reduce_method(
        x, should_use_pymscclpp_allreduce=use_mscclpp
    )
    if method != "ca" or not ca.uses_one_shot_push(x.numel() * x.element_size()):
        return None
    if x.shape[0] > ca.config.num_push_blocks:
        return None
    return ca


def hc_all_reduce_combine_norm(
    comm: Communicator,
    block_partial: torch.Tensor,
    residual: torch.Tensor,
    partials: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    trigger: int = _TRIGGER,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``hc_combine_apply_norm(all_reduce(block_partial), residual, partials,
    norm_weight, eps)`` in one kernel; returns ``(combined, normalized)``.

    block_partial: [rows, 2560] bf16, this rank's partial sum; residual:
    [rows, 10240] bf16; partials: [rows, 8, 4] fp32 gate partials; norm_weight:
    [10240] bf16. Every rank must call with the same rows, in the same order
    relative to the other kernels on the plane.
    """
    combined = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    _jit_module(comm.world_size).run(
        comm,
        block_partial,
        residual,
        partials,
        norm_weight,
        combined,
        normalized,
        float(eps),
        trigger,
    )
    return combined, normalized
