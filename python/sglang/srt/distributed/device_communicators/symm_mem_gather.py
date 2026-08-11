"""One-sided fixed-shape gather over torch symmetric memory.

Each rank stores its row into every peer's buffer, then waits on a barrier. No
communicator takes part, so unlike an all-gather this needs no ordering against
the forward's collectives.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# A stuck peer raises instead of spinning forever.
_BARRIER_TIMEOUT_MS = 10_000
# A peer's stores for round N+1 land before its barrier(N+1) returns, so one
# region would be overwritten while a slower rank still reads round N.
_NUM_SLOTS = 2


class SymmMemGather:
    """Allocated and rendezvoused once: a symmetric operand must keep its
    address for its whole lifetime and resolve to the same (region, offset) on
    every rank, which a per-forward pool allocation does not satisfy."""

    def __init__(
        self,
        world_size: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        group_name: str,
    ):
        from torch._C._distributed_c10d import _SymmetricMemory

        # Outside inference mode on purpose: a region created inside it is an
        # inference tensor and rejects the in-place peer stores below.
        with torch.inference_mode(False):
            region = _SymmetricMemory.empty_strided_p2p(
                (_NUM_SLOTS * world_size * width,),
                [1],
                dtype,
                device,
                group_name,
            ).view(_NUM_SLOTS, world_size, width)
        self._handle = _SymmetricMemory.rendezvous(region)
        self._region = region
        self._world_size = world_size
        self._width = width
        self._slot = 0
        rank = self._handle.rank
        # A peer row is a tensor view of that peer's memory; writing it is a
        # store that never blocks on the peer.
        self._peer_rows = [
            [
                self._handle.get_buffer(peer, (_NUM_SLOTS, world_size, width), dtype)[
                    slot
                ][rank]
                for peer in range(world_size)
            ]
            for slot in range(_NUM_SLOTS)
        ]
        logger.info(
            "Symmetric-memory DP gather active: world=%d width=%d slots=%d",
            world_size,
            width,
            _NUM_SLOTS,
        )

    def gather(self, local_row: torch.Tensor) -> torch.Tensor:
        """Return the (world_size, width) gathered rows for this round."""
        slot = self._slot
        self._slot = (slot + 1) % _NUM_SLOTS
        for row in self._peer_rows[slot]:
            row.copy_(local_row)
        self._handle.barrier(0, _BARRIER_TIMEOUT_MS)
        return self._region[slot]


def maybe_create_symm_mem_gather(
    world_size: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    group_name: str,
) -> Optional[SymmMemGather]:
    """Build a gatherer, or return None when symmetric memory is unusable."""
    try:
        return SymmMemGather(world_size, width, dtype, device, group_name)
    except Exception as e:
        logger.warning(
            "Symmetric-memory DP gather unavailable (%s: %s); falling back.",
            type(e).__name__,
            e,
        )
        return None
