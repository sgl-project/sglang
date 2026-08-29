"""Hand cached-but-free allocator blocks back to the driver when it runs dry.

Not everything that needs device memory goes through the torch caching
allocator: HSA/HIP queue resources, Triton module loads (``cuModuleLoadData``),
and RCCL buffers all draw on what the driver still holds. The allocator only
grows -- once a block is reserved it stays reserved -- so a long-running server
can starve the driver while sitting on tens of GB of free blocks.

Measured on Kimi-K3 tp8/dcp8 at CONC=48: driver-side free oscillates down to
2.2 GB while ``memory_reserved - memory_allocated`` holds 30-35 GB. A Triton
module load needs 2 MB; an HSA queue allocation that fails aborts the process
with ``HSA_STATUS_ERROR_OUT_OF_RESOURCES``.

``--mem-fraction-static`` does NOT control this: its slack comes out of the KV
pool, and the allocator absorbs it again during serving. Lowering it from 0.85
to 0.8 halved the KV pool and made the abort arrive sooner, not later.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_GB = 1 << 30
_reclaim_count = 0


def reclaim_if_low() -> None:
    """Release cached free blocks when driver-side memory is nearly gone.

    Cheap enough for a per-forward call: the free-memory query is a driver call,
    and ``empty_cache`` only runs below the threshold.
    """
    threshold_gb = envs.SGLANG_DEVICE_HEADROOM_RECLAIM_GB.get()
    if threshold_gb <= 0 or not torch.cuda.is_available():
        return

    # Releasing blocks mid-capture would invalidate the graph's own allocations.
    if torch.cuda.is_current_stream_capturing():
        return

    free_before = torch.cuda.mem_get_info()[0]
    if free_before >= threshold_gb * _GB:
        return

    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.empty_cache()
    free_after = torch.cuda.mem_get_info()[0]

    global _reclaim_count
    _reclaim_count += 1
    logger.warning(
        "Device headroom low: %.2f GB free, reclaimed %.2f GB from the caching "
        "allocator (now %.2f GB free; reserved %.2f GB, allocated %.2f GB). "
        "Reclaim #%d.",
        free_before / _GB,
        (free_after - free_before) / _GB,
        free_after / _GB,
        reserved / _GB,
        allocated / _GB,
        _reclaim_count,
    )
