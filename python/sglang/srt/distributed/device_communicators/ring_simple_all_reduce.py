"""ring_simple for CustomAllReduceV2 on ROCm gfx1250."""

from typing import List, Optional, Tuple

import torch

from sglang.kernels.ops.communication.ring_simple import RingPlane, custom_ring_simple

from .configs.custom_all_reduce_v2 import AllReduceConfig, Range

# a channel double-buffers its chunks: two 2 MB slices
RING_SIMPLE_BUFFER_SIZE = 4 * 1024 * 1024
_SLOTS = 2
_SEMAPHORE_BYTES = 128  # sizeof(RingSemaphore)


class RingSimpleAllReduce:
    """ring_simple on ROCm gfx1250: a ring all-reduce over a plane of its own.

    Each block runs one ring (channel) over a contiguous share of the tensor:
    every chunk goes to the next rank only, through a FIFO of chunk-sized slices
    in that rank's memory, behind 64-bit counters.
    """

    @staticmethod
    def plan(config: AllReduceConfig, buffer_size: int) -> Tuple[AllReduceConfig, int]:
        """``(config, slot bytes)``; a slot size of 0 disables it.

        The plane exists only where the tuned table picks ring_simple; otherwise
        its ranges are cleared so no size selects it.
        """
        uses_ring = any(
            h.ring_simple.max_bytes > 0 for h in (config.graph, config.eager)
        )
        slot_bytes = 0
        if uses_ring and config.num_ring_simple_channels and buffer_size > 0:
            slot_bytes = (buffer_size // _SLOTS + 15) // 16 * 16
        if not slot_bytes:
            config = config._replace(
                graph=config.graph._replace(ring_simple=Range(0, 0)),
                eager=config.eager._replace(ring_simple=Range(0, 0)),
            )
        return config, slot_bytes

    @staticmethod
    def workspace_bytes(*, num_channels: int, slot_bytes: int) -> int:
        """Per rank: ``[num_channels semaphores | num_channels FIFOs]``."""
        return num_channels * (_SEMAPHORE_BYTES + _SLOTS * slot_bytes)

    def __init__(
        self,
        *,
        slabs: List[torch.Tensor],
        offset: int,
        rank: int,
        world_size: int,
        num_channels: int,
        slot_bytes: int,
    ) -> None:
        """The plane lives in ``[offset, offset + workspace_bytes)`` of ``slabs``,
        which must start zeroed."""
        sem_end = offset + num_channels * _SEMAPHORE_BYTES
        self.fifo_shape = [num_channels, _SLOTS, slot_bytes]
        fifo_end = sem_end + num_channels * _SLOTS * slot_bytes
        self._plane: Optional[RingPlane] = RingPlane(
            rank,
            world_size,
            workspaces=[s[sem_end:fifo_end].view(self.fifo_shape) for s in slabs],
            semaphores=[
                s[offset:sem_end].view(num_channels, _SEMAPHORE_BYTES) for s in slabs
            ],
        )

    def all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        return custom_ring_simple(self._plane, input)

    def close(self) -> None:
        # drop the views; the slabs belong to the caller
        self._plane = None
