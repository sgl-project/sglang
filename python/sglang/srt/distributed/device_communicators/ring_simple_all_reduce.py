"""ring_simple for CustomAllReduceV2 on ROCm gfx1250."""

import logging
from typing import Optional, Tuple

import torch
from torch.distributed import ProcessGroup

from sglang.kernels.ops.communication.ring_simple import RingPlane, custom_ring_simple

from .configs.custom_all_reduce_v2 import AllReduceConfig, Range
from .symm_workspace import SymmetricBuffer

logger = logging.getLogger(__name__)

# a channel double-buffers its chunks: two 2 MB slices
RING_SIMPLE_BUFFER_SIZE = 4 * 1024 * 1024
_SLOTS = 2
_SEMAPHORE_BYTES = 128  # sizeof(RingSemaphore)


class RingSimpleAllReduce:
    """ring_simple on ROCm gfx1250: a ring all-reduce over a plane of its own.

    Each block runs one ring (channel) over a contiguous share of the tensor:
    every chunk goes to the next rank only, through a FIFO of chunk-sized slices
    in that rank's memory, behind 64-bit counters. The ring keeps each SoC's
    partitions next to each other: they share the SoC's fabric bandwidth, so
    with DPX partners apart every hop crosses the fabric and the ring runs
    1.5-1.8x slower from 64 MB.
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
        group: ProcessGroup,
        device: torch.device,
        rank: int,
        world_size: int,
        num_channels: int,
        slot_bytes: int,
    ) -> None:
        """The plane lives on a symmetric buffer of its own, which starts zeroed."""
        from .gfx1250_fabric import probe_soc_ring_order

        order = probe_soc_ring_order(group, device)
        self._buffer = SymmetricBuffer(
            group,
            device,
            self.workspace_bytes(num_channels=num_channels, slot_bytes=slot_bytes),
        )
        # the plane's ranks are ring positions
        slabs = [self._buffer.slabs[r] for r in order]
        sem_bytes = num_channels * _SEMAPHORE_BYTES
        self.fifo_shape = [num_channels, _SLOTS, slot_bytes]
        fifo_bytes = num_channels * _SLOTS * slot_bytes
        self._plane: Optional[RingPlane] = RingPlane(
            order.index(rank),
            world_size,
            workspaces=[
                s[sem_bytes : sem_bytes + fifo_bytes].view(self.fifo_shape)
                for s in slabs
            ],
            semaphores=[
                s[:sem_bytes].view(num_channels, _SEMAPHORE_BYTES) for s in slabs
            ],
        )
        if rank == 0:
            logger.info(
                "Custom all-reduce v2 ring_simple: %.2f MB, channels = %d, slot = %d B, "
                "order = %s",
                self._buffer.slabs[rank].numel() / (1 << 20),
                num_channels,
                slot_bytes,
                order,
            )

    def all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        return custom_ring_simple(self._plane, input)

    def close(self) -> None:
        self._plane = None  # the views go before the memory
        self._buffer.close()
