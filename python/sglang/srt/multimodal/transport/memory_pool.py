"""Shared stream-ordered lifecycle for GPU multimodal feature pools."""

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.multimodal.transport.admission import MmMemoryRequestTooLarge
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

CONTROL_WORD_BYTES = 4
DATA_ALIGNMENT = 256
DEFAULT_MAX_INFLIGHT_SLICES = 4096


def get_mm_feature_consumer_count() -> int:
    """Number of scheduler ranks receiving one routed multimodal request."""
    parallel = get_parallel()
    if parallel.attn_dp_enabled:
        return parallel.tp_size // parallel.num_dp_ranks
    return parallel.tp_size


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _driver_modules():
    from cuda.bindings import driver as cuda

    from sglang.srt.utils.cuda_vmm_utils import check_drv

    return cuda, check_drv


def stream_wait_value32(
    device_id: int, address: int, value: int, transport_name: str
) -> None:
    cuda, check_drv = _driver_modules()
    stream = torch.cuda.current_stream(device_id)
    check_drv(
        cuda.cuStreamWaitValue32(stream.cuda_stream, address, value, 0),
        f"cuStreamWaitValue32(mm {transport_name})",
    )


def stream_write_value32(
    device_id: int, address: int, value: int, transport_name: str
) -> None:
    cuda, check_drv = _driver_modules()
    stream = torch.cuda.current_stream(device_id)
    check_drv(
        cuda.cuStreamWriteValue32(stream.cuda_stream, address, value, 0),
        f"cuStreamWriteValue32(mm {transport_name})",
    )


def resolve_consumer_rank(
    total_consumer_count: int,
    consumer_rank: Optional[int] = None,
    transport_name: str = "GPU",
) -> int:
    if total_consumer_count == 1:
        return 0
    if consumer_rank is None:
        try:
            # DP groups are contiguous; attention/DCP subgroup ranks can alias
            rank = int(get_parallel().tp_rank) % total_consumer_count
        except Exception as exc:
            raise RuntimeError(
                f"Cannot resolve the {transport_name} consumer rank before "
                "parallel state initialization"
            ) from exc
    else:
        rank = int(consumer_rank)
    if not 0 <= rank < total_consumer_count:
        raise RuntimeError(
            f"{transport_name} consumer rank {rank} is outside "
            f"[0, {total_consumer_count})"
        )
    return rank


class StreamOrderedPoolConsumerMixin:
    """Ready/wait/ack protocol for stream-ordered GPU feature proxies."""

    def _init_stream_ordered_consumer(
        self,
        *,
        ready_byte_offset: int,
        ack_byte_offset: int,
        generation: int,
        total_consumer_count: int,
        transport_name: str,
    ) -> None:
        if total_consumer_count <= 0:
            raise ValueError("total_consumer_count must be positive")
        self.ready_byte_offset = ready_byte_offset
        self.ack_byte_offset = ack_byte_offset
        self.generation = generation
        self.total_consumer_count = total_consumer_count
        self.transport_name = transport_name
        self._consumer_acknowledged = False

    def _wait_until_ready(self, base_address: int, device_id: int) -> None:
        stream_wait_value32(
            device_id,
            base_address + self.ready_byte_offset,
            self.generation,
            self.transport_name,
        )

    def _acknowledge_on_stream(
        self,
        base_address: int,
        device_id: int,
        consumer_count: int,
        consumer_rank: Optional[int] = None,
    ) -> None:
        if self._consumer_acknowledged:
            return
        if consumer_count == self.total_consumer_count:
            consumer_ranks = range(self.total_consumer_count)
        elif consumer_count == 1:
            consumer_ranks = (
                resolve_consumer_rank(
                    self.total_consumer_count,
                    consumer_rank,
                    self.transport_name,
                ),
            )
        else:
            raise ValueError(
                f"{self.transport_name} acknowledgements support one consumer "
                "or the complete consumer group, got "
                f"{consumer_count}/{self.total_consumer_count}"
            )

        for rank in consumer_ranks:
            stream_write_value32(
                device_id,
                base_address + self.ack_byte_offset + rank * CONTROL_WORD_BYTES,
                self.generation,
                self.transport_name,
            )
        self._consumer_acknowledged = True


@dataclass(frozen=True)
class PoolLease:
    start: int
    end: int
    nbytes: int
    slot: int
    generation: int
    ready_byte_offset: int
    ack_byte_offset: int


class StreamOrderedMmFeaturePool:
    """Bounded GPU pool with generation-safe producer/consumer leases."""

    on_capacity_change = None

    def __init__(
        self,
        *,
        memory_size: int,
        byte_tensor: torch.Tensor,
        base_address: int,
        device_id: int,
        consumer_count: int,
        recycle_interval: float,
        transport_name: str,
        max_inflight_slices: int = DEFAULT_MAX_INFLIGHT_SLICES,
    ) -> None:
        if memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if consumer_count <= 0:
            raise ValueError("consumer_count must be positive")
        if max_inflight_slices <= 0:
            raise ValueError("max_inflight_slices must be positive")
        if recycle_interval <= 0:
            raise ValueError("recycle_interval must be positive")
        if (
            not byte_tensor.is_cuda
            or byte_tensor.device.index != device_id
            or byte_tensor.dtype != torch.uint8
            or not byte_tensor.is_contiguous()
            or byte_tensor.numel() < memory_size
        ):
            raise ValueError(
                "byte_tensor must be a sufficiently large contiguous uint8 tensor "
                f"on cuda:{device_id}"
            )

        self.memory_size = memory_size
        self.byte_tensor = byte_tensor
        self.base_address = base_address
        self.device_id = device_id
        self.consumer_count = consumer_count
        self.control_words_per_slot = 1 + consumer_count
        self.max_inflight_slices = max_inflight_slices
        self.transport_name = transport_name
        control_bytes = (
            max_inflight_slices * self.control_words_per_slot * CONTROL_WORD_BYTES
        )
        self.data_start = align_up(control_bytes, DATA_ALIGNMENT)
        if memory_size <= self.data_start:
            raise ValueError(
                f"{transport_name} pool is too small after control metadata: "
                f"pool={memory_size}, control={self.data_start}"
            )

        control_word_count = max_inflight_slices * self.control_words_per_slot
        self._control_words = (
            byte_tensor[: control_word_count * CONTROL_WORD_BYTES]
            .view(torch.int32)
            .view(max_inflight_slices, self.control_words_per_slot)
        )
        self._control_words.zero_()
        torch.cuda.synchronize(device_id)

        self._available_ranges = [(self.data_start, memory_size)]
        self._available_slots = list(reversed(range(max_inflight_slices)))
        self._slot_generations = [0] * max_inflight_slices
        self._occupied: dict[int, PoolLease] = {}
        self._cancelling: set[int] = set()
        self._lock = threading.Lock()
        self.on_capacity_change = None
        self._pool_error = None
        self._recycle_interval = recycle_interval
        self._recycler_stop_event = threading.Event()
        self._recycle_thread = threading.Thread(
            target=self._recycle_loop,
            name=f"{transport_name}MmFeaturePoolRecycler",
            daemon=True,
        )
        self._recycle_thread.start()

    @property
    def usable_size(self) -> int:
        return self.memory_size - self.data_start

    def capacity_snapshot(self):
        """Allocator-only diagnostics; never query or synchronize CUDA here."""
        with self._lock:
            free = sum(end - start for start, end in self._available_ranges)
            return {
                "occupied_bytes": self.usable_size - free,
                "free_bytes": free,
                "largest_free_bytes": max(
                    (end - start for start, end in self._available_ranges), default=0
                ),
                "free_slots": len(self._available_slots),
            }

    @property
    def active_lease_count(self) -> int:
        with self._lock:
            return len(self._occupied)

    def _allocate_locked(self, nbytes: int) -> Optional[PoolLease]:
        allocation_bytes = align_up(nbytes, DATA_ALIGNMENT)
        candidates = [
            (end - start, index, start, end)
            for index, (start, end) in enumerate(self._available_ranges)
            if end - start >= allocation_bytes
        ]
        if not candidates or not self._available_slots:
            return None
        slot = self._available_slots[-1]
        generation = self._slot_generations[slot] + 1
        if generation > 0x7FFFFFFF:
            raise RuntimeError(f"{self.transport_name} pool slot generation exhausted")
        _, index, start, end = min(candidates)
        self._available_ranges.pop(index)
        if start + allocation_bytes < end:
            self._available_ranges.append((start + allocation_bytes, end))
        self._available_slots.pop()
        self._slot_generations[slot] = generation
        ready_byte_offset = slot * self.control_words_per_slot * CONTROL_WORD_BYTES
        lease = PoolLease(
            start=start,
            end=start + allocation_bytes,
            nbytes=nbytes,
            slot=slot,
            generation=generation,
            ready_byte_offset=ready_byte_offset,
            ack_byte_offset=ready_byte_offset + CONTROL_WORD_BYTES,
        )
        self._occupied[slot] = lease
        return lease

    def try_reserve_batch(self, sizes: list[int]) -> Optional[list[PoolLease]]:
        """Reserve all ranges and control slots, or leave no partial ownership."""
        required = sum(align_up(size, DATA_ALIGNMENT) for size in sizes)
        if required > self.usable_size or len(sizes) > self.max_inflight_slices:
            raise MmMemoryRequestTooLarge(
                f"CUDA IPC publication needs {required} bytes/{len(sizes)} slots; "
                f"capacity is {self.usable_size} bytes/{self.max_inflight_slices} slots"
            )
        with self._lock:
            if self._pool_error is not None:
                raise RuntimeError("CUDA IPC feature pool failed") from self._pool_error
            if self._recycler_stop_event.is_set():
                raise RuntimeError("CUDA IPC feature pool is closing")
            leases = []
            complete = False
            try:
                for size in sizes:
                    lease = self._allocate_locked(size)
                    if lease is None:
                        return None
                    leases.append(lease)
                complete = True
                return leases
            finally:
                if not complete:
                    for lease in leases:
                        self._release_locked(lease)
                    self._merge_ranges_locked()

    def release_reserved_batch(self, leases: list[PoolLease]) -> None:
        """Release reservations only before copy starts or after stream completion."""
        with self._lock:
            for lease in leases:
                self._release_locked(lease)
            self._merge_ranges_locked()
        if self.on_capacity_change is not None:
            self.on_capacity_change()

    def copy_reserved_batch(
        self, tensors: list[torch.Tensor], leases: list[PoolLease]
    ) -> list[torch.Tensor]:
        destinations = []
        try:
            for tensor, lease in zip(tensors, leases, strict=True):
                _, destination = self.copy_tensor(tensor, lease=lease)
                destinations.append(destination)
            torch.cuda.current_stream(self.device_id).synchronize()
            return destinations
        except BaseException:
            # cancellation/failure cannot make bytes reusable while an enqueued
            # copy still touches them; a failed CUDA sync poisons the pool
            try:
                torch.cuda.current_stream(self.device_id).synchronize()
            except BaseException as error:
                self._pool_error = error
                raise
            self.release_reserved_batch(leases)
            raise

    def _release_locked(self, lease: PoolLease) -> None:
        active_lease = self._occupied.get(lease.slot)
        if active_lease != lease:
            raise RuntimeError(
                f"Cannot release inactive {self.transport_name} pool lease "
                f"(slot={lease.slot}, generation={lease.generation})"
            )
        del self._occupied[lease.slot]
        self._available_slots.append(lease.slot)
        self._available_ranges.append((lease.start, lease.end))

    def _merge_ranges_locked(self) -> None:
        merged = []
        for start, end in sorted(self._available_ranges):
            if merged and merged[-1][1] == start:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        self._available_ranges = merged

    def _recycle_ready_leases(self) -> None:
        with self._lock:
            leases = list(self._occupied.values())
        if not leases:
            return
        slot_indices = torch.tensor(
            [lease.slot for lease in leases],
            dtype=torch.long,
            device=f"cuda:{self.device_id}",
        )
        expected = torch.tensor(
            [lease.generation for lease in leases],
            dtype=torch.int32,
            device=f"cuda:{self.device_id}",
        ).unsqueeze(1)
        completed = (
            (self._control_words.index_select(0, slot_indices) == expected)
            .all(dim=1)
            .cpu()
            .tolist()
        )
        reclaimed = False
        with self._lock:
            for lease, is_complete in zip(leases, completed):
                if (
                    is_complete
                    and self._occupied.get(lease.slot) is lease
                    and lease.slot not in self._cancelling
                ):
                    self._release_locked(lease)
                    reclaimed = True
            self._merge_ranges_locked()
        if reclaimed and self.on_capacity_change is not None:
            self.on_capacity_change()

    def _recycle_loop(self) -> None:
        torch.cuda.set_device(self.device_id)
        while not self._recycler_stop_event.is_set():
            try:
                with torch.cuda.device(self.device_id):
                    self._recycle_ready_leases()
            except Exception:
                logger.warning(
                    "%s multimodal pool recycle failed",
                    self.transport_name,
                    exc_info=True,
                )
            self._recycler_stop_event.wait(self._recycle_interval)

    def copy_tensor(
        self, tensor: torch.Tensor, *, lease: Optional[PoolLease] = None
    ) -> tuple[Optional[PoolLease], Optional[torch.Tensor]]:
        if not tensor.is_cuda:
            raise ValueError(f"{self.transport_name} requires a CUDA tensor")
        source = tensor.contiguous()
        nbytes = source.numel() * source.element_size()
        if nbytes == 0:
            raise ValueError(f"{self.transport_name} cannot transport an empty tensor")
        reserved = lease is not None
        if not reserved:
            with self._lock:
                lease = self._allocate_locked(nbytes)
        if lease is None:
            return None, None

        try:
            with torch.cuda.device(self.device_id):
                destination = self.byte_tensor[lease.start : lease.start + lease.nbytes]
                destination.copy_(
                    source.view(torch.uint8).reshape(-1), non_blocking=True
                )
                stream_write_value32(
                    self.device_id,
                    self.base_address + lease.ready_byte_offset,
                    lease.generation,
                    self.transport_name,
                )
        except Exception:
            if not reserved:
                try:
                    torch.cuda.current_stream(self.device_id).synchronize()
                except BaseException as error:
                    self._pool_error = error
                    raise
                self.release_reserved_batch([lease])
            raise
        return lease, destination

    def cancel_lease(
        self,
        *,
        ready_byte_offset: int,
        ack_byte_offset: int,
        generation: int,
    ) -> None:
        """Acknowledge every consumer for a lease that was not dispatched."""
        slot_stride = self.control_words_per_slot * CONTROL_WORD_BYTES
        if (
            ready_byte_offset % slot_stride != 0
            or ack_byte_offset != ready_byte_offset + CONTROL_WORD_BYTES
        ):
            raise RuntimeError(f"Invalid {self.transport_name} pool lease offsets")
        slot = ready_byte_offset // slot_stride

        with self._lock:
            lease = self._occupied.get(slot)
            if (
                lease is None
                or lease.generation != generation
                or lease.ready_byte_offset != ready_byte_offset
                or lease.ack_byte_offset != ack_byte_offset
            ):
                raise RuntimeError(
                    f"Cannot cancel inactive {self.transport_name} pool lease "
                    f"(slot={slot}, generation={generation})"
                )

            if slot in self._cancelling:
                return
            self._cancelling.add(slot)

        # keep the lease pinned, not the allocator lock, across driver calls
        try:
            with torch.cuda.device(self.device_id):
                stream_wait_value32(
                    self.device_id,
                    self.base_address + ready_byte_offset,
                    generation,
                    self.transport_name,
                )
                for rank in range(self.consumer_count):
                    stream_write_value32(
                        self.device_id,
                        self.base_address + ack_byte_offset + rank * CONTROL_WORD_BYTES,
                        generation,
                        self.transport_name,
                    )
        except BaseException as error:
            self._pool_error = error
            raise
        else:
            with self._lock:
                self._cancelling.remove(slot)

    def shutdown(self) -> None:
        self._recycler_stop_event.set()
        if self._recycle_thread.is_alive():
            self._recycle_thread.join()
