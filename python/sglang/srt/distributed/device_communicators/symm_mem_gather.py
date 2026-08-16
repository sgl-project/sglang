"""One-sided fixed-shape gather over torch symmetric memory."""

import logging
import time
from typing import Optional

import torch

from sglang.srt.utils.nvtx_utils import (
    NVTX_SCHEDULER_ENABLED,
    scheduler_nvtx_range,
)

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

        # Keep the default gather method untouched. Only opt-in processes bind
        # the instrumented host path; that path uses the same CUDA operations,
        # stream, and synchronization order as gather().
        from sglang.srt.environ import envs

        if envs.SGLANG_SYMM_MEM_DP_SYNC_TELEMETRY.get():
            from sglang.srt.distributed.device_communicators.symm_mem_gather_telemetry import (
                SymmMemGatherTelemetry,
                register_symm_mem_gather_telemetry,
            )

            self._telemetry = SymmMemGatherTelemetry(
                world_size=world_size,
                group_rank=rank,
                max_records=envs.SGLANG_SYMM_MEM_DP_SYNC_TELEMETRY_MAX_RECORDS.get(),
            )
            self._host_ready_numpy = self._host_ready.numpy()
            self._host_out_numpy = self._host_out.numpy()
            register_symm_mem_gather_telemetry(self._telemetry)
            self.gather = self._gather_with_telemetry

    def gather(self, local_row_cpu: torch.Tensor) -> torch.Tensor:
        """Host row in, (world_size, width) host rows out."""
        slot = self._slot
        self._slot = (slot + 1) % _NUM_SLOTS

        self._generation = self._generation % 0xFFFFFFFF + 1
        generation = self._generation
        rank = self._handle.rank

        if NVTX_SCHEDULER_ENABLED:
            trace = scheduler_nvtx_range(
                f"scheduler.pd.symm_gather.gen={generation}.rank={rank}"
            )
        else:
            trace = scheduler_nvtx_range("")

        with trace:
            return self._gather(slot, generation, rank, local_row_cpu)

    def _gather(
        self,
        slot: int,
        generation: int,
        rank: int,
        local_row_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Execute one gather with profile-only generation/ready-mask markers."""

        self._host_in.copy_(local_row_cpu)
        with torch.cuda.stream(self._stream):
            self._staging.copy_(self._host_in, non_blocking=True)
            from sglang.srt.distributed.device_communicators.symm_mem_marker import (
                copy_row_and_publish,
            )

            copy_row_and_publish(
                self._peer_row_ptrs[slot],
                self._peer_marker_ptrs[slot],
                self._staging,
                generation,
            )

        from sglang.srt.distributed.device_communicators.symm_mem_marker import (
            snapshot_rows_acquire,
        )

        deadline = time.monotonic() + _BARRIER_TIMEOUT_MS / 1000
        poll_index = 0
        while True:
            poll_index += 1
            with torch.cuda.stream(self._stream):
                snapshot_rows_acquire(
                    self._region[slot],
                    self._marker_region[slot],
                    self._row_snapshot,
                    self._ready,
                    generation,
                )
                self._host_ready.copy_(self._ready, non_blocking=True)
                self._host_out.copy_(self._row_snapshot, non_blocking=True)
            self._stream.synchronize()
            if NVTX_SCHEDULER_ENABLED:
                ready_mask = sum(
                    int(bool(value)) << peer
                    for peer, value in enumerate(self._host_ready.tolist())
                )
                with scheduler_nvtx_range(
                    "scheduler.pd.symm_poll."
                    f"gen={generation}.rank={rank}.poll={poll_index}."
                    f"mask=0x{ready_mask:x}"
                ):
                    pass
            if self._host_ready.all().item():
                return self._host_out
            if time.monotonic() >= deadline:
                raise TimeoutError("symmetric-memory completion marker timeout")

    def _gather_with_telemetry(self, local_row_cpu: torch.Tensor) -> torch.Tensor:
        """Opt-in gather path with host-only observations after existing syncs."""
        gather_start_ns = time.perf_counter_ns()
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

        record = self._telemetry.begin(
            generation=self._generation,
            slot=slot,
            gather_start_ns=gather_start_ns,
            local_row=local_row_cpu.numpy(),
        )
        from sglang.srt.distributed.device_communicators.symm_mem_marker import (
            snapshot_rows_acquire,
        )

        deadline = time.monotonic() + _BARRIER_TIMEOUT_MS / 1000
        while True:
            poll_begin_ns = time.perf_counter_ns()
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
            sync_begin_ns = time.perf_counter_ns()
            self._stream.synchronize()
            sync_done_ns = time.perf_counter_ns()
            if record is not None:
                ready_mask = self._telemetry.note_poll(
                    record,
                    ready=self._host_ready_numpy,
                    poll_begin_ns=poll_begin_ns,
                    sync_begin_ns=sync_begin_ns,
                    sync_done_ns=sync_done_ns,
                )
                all_ready = ready_mask == (1 << self._telemetry.world_size) - 1
            else:
                all_ready = self._host_ready.all().item()
            if all_ready:
                if record is not None:
                    self._telemetry.finish(
                        record,
                        gather_done_ns=time.perf_counter_ns(),
                        peer_rows=self._host_out_numpy,
                    )
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
