from __future__ import annotations

import functools
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_forward,
    get_parallel,
    max_prefill_buffer_tokens,
)

_COMM_BLOCKS = {
    (2, 20): 64,
    (2, 16): 80,
    (4, 20): 64,
    (4, 16): 64,
}
_STAT_MMA_BLOCKS = {
    (2, 20): 80,
    (2, 16): 64,
    (4, 20): 80,
    (4, 16): 80,
}
_STAT_RED_BLOCKS = 4
_NUM_BF16_SLICES = 2  # bf16x2 covers ~16 mantissa bits, strictly above DG's tf32 (10)


def row_split(total_rows: int, rank: int, world_size: int) -> tuple[int, int]:
    """This rank's ``(first row, row count)``. Mirrored in the boundary kernel."""
    avg, rem = divmod(total_rows, world_size)
    return rank * avg + min(rank, rem), avg + (rank < rem)


def can_use_sp(forward_mode, total_rows: int) -> bool:
    """Whether this forward shards the residual: enabled, prefill, and long enough.

    Every predicate derives from the arguments, so this stays correct inside a CUDA
    graph replay; a flag written during capture would be stale.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

    if (
        not envs.SGLANG_OPT_DSV41_MHC_SP_FUSION.get()
        or forward_mode != ForwardMode.EXTEND
        or get_is_capture_mode()
    ):
        return False
    # The kernel's counters assume every rank owns at least one row.
    min_tokens = max(
        get_parallel().tp_group.world_size,
        envs.SGLANG_OPT_DSV41_MHC_SP_FUSION_MIN_TOKENS.get(),
    )
    return total_rows >= min_tokens


# NOTE: typically, `total_rows` are identical within 1 layer; cache to save overhead
@functools.lru_cache(maxsize=8)
def _local_rows(total_rows: int) -> tuple[int, int]:
    """This rank's ``(first row, row count)`` of a batch of ``total_rows``."""
    tp_group = get_parallel().tp_group
    return row_split(total_rows, tp_group.rank_in_group, tp_group.world_size)


@functools.lru_cache(maxsize=8)
def _shard_rows(total_rows: int) -> tuple[int, ...]:
    tp_group = get_parallel().tp_group
    return tuple(
        row_split(total_rows, rank, tp_group.world_size)[1]
        for rank in range(tp_group.world_size)
    )


def shard(rows: torch.Tensor, total_rows: int) -> torch.Tensor:
    """This rank's view of a replicated ``[total_rows, ...]`` tensor; not a collective."""
    assert rows.shape[0] == total_rows
    offset, count = _local_rows(total_rows)
    return rows[offset : offset + count]


def gather(rows: torch.Tensor, total_rows: int) -> torch.Tensor:
    """All-gather a sharded ``[local rows, ...]`` tensor back to ``[total_rows, ...]``."""
    tp_group = get_parallel().tp_group
    assert rows.shape[0] == _local_rows(total_rows)[1]
    sizes = _shard_rows(total_rows)
    output = rows.new_empty((total_rows, *rows.shape[1:]))
    tp_group.all_gatherv(rows.contiguous(), sizes=sizes, output=output)
    return output


def get_active_rows() -> Optional[int]:
    """The batch's total rows while the residual runs sharded; None outside that region."""
    return get_forward().sp_mhc_rows


def get_alloc_context(device: torch.device):
    from sglang.srt.distributed.symmetric_memory import symmetric_context

    return symmetric_context(device)


def _pick_grid(world_size: int, split_k: int, num_sms: int) -> tuple[int, int]:
    """``(communication blocks, statistics mma blocks)``, checked against the device.

    The mma count must divide by ``split_k`` -- anything else is an nvcc compile error.
    """
    mma = _STAT_MMA_BLOCKS[(world_size, split_k)]
    mma -= mma % split_k
    comm = _COMM_BLOCKS[(world_size, split_k)]
    total = comm + mma + _STAT_RED_BLOCKS
    assert total <= num_sms, (
        f"the mHC SP boundary wants {total} co-resident blocks at world size "
        f"{world_size} ({comm} communication, {mma} statistics mma, "
        f"{_STAT_RED_BLOCKS} statistics reduce) and this device has {num_sms} SMs"
    )
    return comm, mma


def init_workspace(max_num_rows: Optional[int] = None) -> None:
    """Hand the kernel the all-reduce pull plane's semaphores and its tile counters."""
    import sglang.kernels.ops.communication.mhc_sp_fusion as op

    if op._COMM is not None:
        return
    if max_num_rows is None:
        max_num_rows = max_prefill_buffer_tokens()

    comm = get_parallel().tp_group.ca_comm
    op.init_mhc_workspace(comm.obj, max_num_rows)


def fusion(
    partial: torch.Tensor,
    total_rows: int,
    residual: torch.Tensor,
    norm_weight: Optional[torch.Tensor],  # None drops the combine/norm/AG tail
    eps: float,
    *,
    hc_w: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    num_comm_blocks: Optional[int] = None,
    stat_mma_blocks: Optional[int] = None,
    stat_red_blocks: int = _STAT_RED_BLOCKS,
    rms_eps: float = 1e-6,
    hc_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(new residual, next input, pre, post, comb)`` for one sublayer boundary.

    ``partial`` is a symmetric-memory tensor holding this rank's partial sublayer
    output for every row; the coefficients are produced by the statistics CTAs of the
    same launch that consumes them. The next input is written back into ``partial``,
    so it is consumed: a second boundary run against it would reduce this one's
    output. With ``norm_weight=None`` nothing is written to it at all.
    """
    import torch.distributed._symmetric_memory as torch_symm_mem

    from sglang.kernels.ops.communication.mhc_sp_fusion import (
        BEST_SPLIT_K,
        mhc_sp_fusion,
    )

    tp_group = get_parallel().tp_group
    # A collective on first sight, a cached handle after; `partial` must be the whole
    # allocation or the addresses would not name its first byte.
    handle = torch_symm_mem.rendezvous(partial, tp_group.device_group)
    num_sms = torch.cuda.get_device_properties(residual.device).multi_processor_count
    hc_w = hc_w[:_NUM_BF16_SLICES]
    comm, mma = _pick_grid(tp_group.world_size, BEST_SPLIT_K[hc_w.shape[0]], num_sms)
    num_comm_blocks = comm if num_comm_blocks is None else num_comm_blocks
    stat_mma_blocks = mma if stat_mma_blocks is None else stat_mma_blocks
    row_offset, _ = _local_rows(total_rows)
    residual_out = torch.empty_like(residual)
    num_rows, streams = residual.shape[0], residual.shape[1]
    opts = {"dtype": torch.float32, "device": residual.device}
    pre = torch.empty(num_rows, streams, **opts)
    post = torch.empty(num_rows, streams, **opts)
    comb = torch.empty(num_rows, streams, streams, **opts)
    mhc_sp_fusion(
        residual,
        residual_out,
        post,
        comb,
        pre,
        norm_weight,
        # y and x alias on purpose; the kernel allows it for a bf16 gather.
        y_mc=handle.multicast_ptr,
        x_mc=handle.multicast_ptr,
        y_ptrs=handle.buffer_ptrs,
        x_ptrs=handle.buffer_ptrs,
        world_size=tp_group.world_size,
        row_offset=row_offset,
        total_rows=total_rows,
        eps=eps,
        comm_blocks=num_comm_blocks,
        hc_w=hc_w,
        hc_scale=hc_scale,
        hc_base=hc_base,
        stat_mma_blocks=stat_mma_blocks,
        stat_red_blocks=stat_red_blocks,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
    )
    return residual_out, partial, pre, post, comb
