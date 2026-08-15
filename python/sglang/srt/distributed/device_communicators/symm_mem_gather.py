"""One-sided fixed-shape gather over torch symmetric memory."""

import logging
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_BARRIER_TIMEOUT_MS = 10_000
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

        if width != 7 or dtype != torch.int64:
            raise ValueError("symmetric DP metadata gather requires 7 int64 fields")

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
        self._slot = 0
        self._stream = torch.cuda.Stream(device=device)
        self._staging = torch.zeros(width, dtype=dtype, device=device)
        self._host_in = torch.zeros(width, dtype=dtype).pin_memory()
        self._host_out = torch.zeros(world_size, width, dtype=dtype).pin_memory()
        self._host_ready = torch.empty(world_size, dtype=torch.uint32).pin_memory()
        self._ready = torch.empty(world_size, dtype=torch.uint32, device=device)
        self._row_snapshot = torch.empty(world_size, width, dtype=dtype, device=device)
        self._generation = 0
        rank = self._handle.rank
        self._peer_rows = [
            [
                self._handle.get_buffer(peer, (_NUM_SLOTS, world_size, width), dtype)[
                    slot
                ][rank]
                for peer in range(world_size)
            ]
            for slot in range(_NUM_SLOTS)
        ]

        with torch.inference_mode(False):
            marker_region = _SymmetricMemory.empty_strided_p2p(
                (_NUM_SLOTS * world_size,),
                [1],
                torch.uint32,
                device,
                group_name,
            ).view(_NUM_SLOTS, world_size)
            marker_region.zero_()
        torch.cuda.current_stream(device).synchronize()
        self._marker_region = marker_region
        self._marker_handle = _SymmetricMemory.rendezvous(marker_region)
        self._peer_row_ptrs = [
            torch.tensor(
                [row.data_ptr() for row in self._peer_rows[slot]],
                dtype=torch.uint64,
                device=device,
            )
            for slot in range(_NUM_SLOTS)
        ]
        self._peer_marker_ptrs = [
            torch.tensor(
                [
                    self._marker_handle.get_buffer(
                        peer, (_NUM_SLOTS, world_size), torch.uint32
                    )[slot, rank].data_ptr()
                    for peer in range(world_size)
                ],
                dtype=torch.uint64,
                device=device,
            )
            for slot in range(_NUM_SLOTS)
        ]
        torch.cuda.current_stream(device).synchronize()
        logger.info(
            "Symmetric-memory DP gather active: world=%d width=%d slots=%d",
            world_size,
            width,
            _NUM_SLOTS,
        )

    def gather(self, local_row_cpu: torch.Tensor) -> torch.Tensor:
        """Host row in, (world_size, width) host rows out."""
        slot = self._slot
        self._slot = (slot + 1) % _NUM_SLOTS

        self._host_in.copy_(local_row_cpu)
        with torch.cuda.stream(self._stream):
            self._staging.copy_(self._host_in, non_blocking=True)
            from sglang.srt.distributed.device_communicators.symm_mem_marker import (
                copy_row_and_publish,
            )

            self._generation = self._generation % 0xFFFFFFFF + 1
            copy_row_and_publish(
                self._peer_row_ptrs[slot],
                self._peer_marker_ptrs[slot],
                self._staging,
                self._generation,
            )

        from sglang.srt.distributed.device_communicators.symm_mem_marker import (
            snapshot_rows_acquire,
        )

        deadline = time.monotonic() + _BARRIER_TIMEOUT_MS / 1000
        while True:
            with torch.cuda.stream(self._stream):
                snapshot_rows_acquire(
                    self._region[slot],
                    self._marker_region[slot],
                    self._row_snapshot,
                    self._ready,
                    self._generation,
                )
                self._host_ready.copy_(self._ready, non_blocking=True)
                self._host_out.copy_(self._row_snapshot, non_blocking=True)
            self._stream.synchronize()
            if self._host_ready.all().item():
                return self._host_out
            if time.monotonic() >= deadline:
                raise TimeoutError("symmetric-memory completion marker timeout")


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
