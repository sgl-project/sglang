"""Host-tier copies of the per-token capturer rows (routed experts, indexer
top-k) for the hierarchical cache.

A capturer keeps one row per device KV slot, and get_topk reads a request's
rows through req_to_token. A HiCache load-back puts a prefix into new device
slots, so without these copies the prefix rows read back would be whatever
those slots held before. Every HiCache write copies the rows of its device
slots to its host slots, and every load-back copies them into the new device
slots.

The slot indices reach the CPU with a non-blocking copy on the transfer's own
stream, right after the KV copy is submitted. A write's rows are copied when
its ack is processed, before the device slots can be freed. A load's rows are
copied when its ack is processed, or earlier when get_topk reads rows.
"""

import collections
import logging
from typing import List, Optional

import msgspec
import torch

from sglang.srt.state_capturer.base import BaseHostCache, BaseTopkCapturer
from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

logger = logging.getLogger(__name__)


class _RowCopy(msgspec.Struct):
    event: torch.cuda.Event
    device_indices_cpu: torch.Tensor
    host_indices_cpu: torch.Tensor
    # Keeps the device index tensor alive until the copy has run.
    device_indices: torch.Tensor


class HostTopkRows:
    def __init__(self, host_size: int, capturers: List[BaseTopkCapturer], device: str):
        self._capturers = capturers
        self._host_caches = [
            BaseHostCache(
                num_tokens=host_size,
                num_layers=c.num_layers,
                topk_size=c.topk_size,
                name=f"{c.host_cache.name}_hicache",
                device=device,
            )
            for c in capturers
        ]
        self._writes: collections.deque = collections.deque()
        self._loads: collections.deque = collections.deque()
        for capturer in capturers:
            capturer.add_pre_read_hook(self.apply_loads)

    @staticmethod
    def create(host_size: int, device: str) -> Optional["HostTopkRows"]:
        capturers = [
            c
            for c in (get_global_experts_capturer(), get_global_indexer_capturer())
            if c is not None
        ]
        return (
            HostTopkRows(host_size=host_size, capturers=capturers, device=device)
            if capturers
            else None
        )

    @staticmethod
    def _copy_indices(
        device_indices: torch.Tensor, host_indices: torch.Tensor, stream
    ) -> _RowCopy:
        device_indices_cpu = torch.empty(
            device_indices.shape, dtype=device_indices.dtype, pin_memory=True
        )
        with torch.cuda.stream(stream):
            device_indices_cpu.copy_(device_indices, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        return _RowCopy(
            event=event,
            device_indices_cpu=device_indices_cpu,
            host_indices_cpu=host_indices.cpu(),
            device_indices=device_indices,
        )

    def on_write(self, device_indices, host_indices, stream) -> None:
        """A device-to-host KV write was submitted on stream."""
        self._writes.append(
            self._copy_indices(
                device_indices=device_indices, host_indices=host_indices, stream=stream
            )
        )

    def on_load(self, device_indices, host_indices, stream) -> None:
        """A host-to-device KV load into new device slots was submitted on stream."""
        self._loads.append(
            self._copy_indices(
                device_indices=device_indices, host_indices=host_indices, stream=stream
            )
        )

    def apply_writes(self, count: Optional[int] = None) -> None:
        """Copy device-slot rows to host slots for the oldest count writes
        (all when None)."""
        while self._writes and (count is None or count > 0):
            op = self._writes.popleft()
            op.event.synchronize()
            for host, capturer in zip(self._host_caches, self._capturers):
                host.buffer[op.host_indices_cpu] = capturer.host_cache.buffer[
                    op.device_indices_cpu
                ]
            if count is not None:
                count -= 1

    def apply_loads(self, count: Optional[int] = None) -> None:
        """Copy host-slot rows into the new device slots for the oldest count
        loads (all when None)."""
        while self._loads and (count is None or count > 0):
            op = self._loads.popleft()
            op.event.synchronize()
            for host, capturer in zip(self._host_caches, self._capturers):
                capturer.host_cache.buffer[op.device_indices_cpu] = host.buffer[
                    op.host_indices_cpu
                ]
            if count is not None:
                count -= 1

    def reset(self) -> None:
        self._writes.clear()
        self._loads.clear()
