"""Host-tier copies of the routed-experts capturer rows for the hierarchical
cache.

The routed-experts capturer keeps one row per device KV slot, and get_topk
reads a request's rows through req_to_token. A HiCache load-back puts a prefix
into new device slots, so without these copies the prefix rows read back would
be whatever those slots held before. Every HiCache write copies the rows of its
device slots to its host slots, and every load-back copies them into the new
device slots.

The slot indices reach the CPU with a non-blocking copy on the transfer
engine's index stream, so they never wait behind earlier KV copies. Each
transfer's TopkRowCopy rides on its HiCacheAck. A write's rows are copied when
its ack is processed, before the device slots can be freed. Under overlap
scheduling a finished request is cached while the next forward still writes
its last slot, whose row lands only at that forward's finalize(); a write
issued then copies its rows after that finalize instead. A load's rows are
copied when its ack is processed, or earlier when get_topk reads rows.
"""

import collections
from typing import Any, Optional

import msgspec
import torch

from sglang.srt.state_capturer.base import BaseTopkCapturer
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer
from sglang.srt.utils import get_device_module

device_module = get_device_module()


class TopkRowCopy(msgspec.Struct):
    # A device_module.Event recorded after the index copy.
    event: Any
    device_indices_cpu: torch.Tensor
    host_indices_cpu: torch.Tensor
    # Keeps the device index tensor alive until the copy has run.
    device_indices: torch.Tensor
    # The capturer's latest output when the transfer was issued: a write's
    # rows are final once it has landed.
    last_output: int = 0
    applied: bool = False


class HostTopkRows:
    def __init__(self, host_size: int, capturer: BaseTopkCapturer, device: str):
        # The global capturer is set once at model load and owns (and frees)
        # the host-tier buffer.
        self._capturer = capturer
        self._host_tier = capturer.alloc_host_tier_cache(
            num_tokens=host_size, device=device
        )
        # Submitted loads in ack order; get_topk applies the ones not yet acked.
        self._pending_loads: collections.deque = collections.deque()
        # Acked writes, in ack order, whose rows have not all landed yet.
        self._pending_writes: collections.deque = collections.deque()
        self._num_landed_outputs = 0
        capturer.add_pre_read_hook(self.apply_pending_loads)
        capturer.add_finalize_hook(self._on_outputs_landed)

    @staticmethod
    def create(host_pool: Any, device: str) -> Optional["HostTopkRows"]:
        capturer = get_global_experts_capturer()
        if capturer is None:
            return None
        # Host slot ids span the logical size (DCP and page-interleaved pools
        # give each physical row several logical slots).
        return HostTopkRows(
            host_size=host_pool.logical_size, capturer=capturer, device=device
        )

    def copy_indices(
        self, device_indices: torch.Tensor, host_indices: torch.Tensor, stream
    ) -> TopkRowCopy:
        """Copy the slot indices of a transfer to the host on stream (the
        transfer engine's index stream), after the current stream that
        produced them."""
        device_indices_cpu = torch.empty(
            device_indices.shape, dtype=device_indices.dtype, pin_memory=True
        )
        stream.wait_stream(device_module.current_stream())
        with device_module.stream(stream):
            device_indices_cpu.copy_(device_indices, non_blocking=True)
            event = device_module.Event()
            event.record(stream)
        return TopkRowCopy(
            event=event,
            device_indices_cpu=device_indices_cpu,
            host_indices_cpu=host_indices.cpu(),
            device_indices=device_indices,
            last_output=self._capturer.num_outputs,
        )

    def add_load(self, row_copy: TopkRowCopy) -> None:
        self._pending_loads.append(row_copy)

    def finish_write(self, row_copy: TopkRowCopy) -> None:
        """Copy the rows of the written device slots to their host slots, now
        or once the output in flight when the write was issued has landed.
        The device slots are freed only after the ack, so a forward reusing
        them is launched (and finalized) after that output."""
        if row_copy.last_output <= self._num_landed_outputs:
            self._copy_write_rows(row_copy)
        else:
            self._pending_writes.append(row_copy)

    def _copy_write_rows(self, row_copy: TopkRowCopy) -> None:
        row_copy.event.synchronize()
        self._host_tier.buffer[row_copy.host_indices_cpu] = (
            self._capturer.host_cache.buffer[row_copy.device_indices_cpu]
        )

    def _on_outputs_landed(self, num_landed: int) -> None:
        self._num_landed_outputs = max(self._num_landed_outputs, num_landed)
        while (
            self._pending_writes
            and self._pending_writes[0].last_output <= self._num_landed_outputs
        ):
            self._copy_write_rows(self._pending_writes.popleft())

    def finish_load(self, row_copy: TopkRowCopy) -> None:
        """Copy the rows of the loaded host slots into the new device slots."""
        if not row_copy.applied:
            row_copy.event.synchronize()
            self._capturer.host_cache.buffer[row_copy.device_indices_cpu] = (
                self._host_tier.buffer[row_copy.host_indices_cpu]
            )
            row_copy.applied = True
        while self._pending_loads and self._pending_loads[0].applied:
            self._pending_loads.popleft()

    def apply_pending_loads(self) -> None:
        for row_copy in list(self._pending_loads):
            self.finish_load(row_copy)

    def reset(self) -> None:
        self._pending_loads.clear()
        self._pending_writes.clear()
