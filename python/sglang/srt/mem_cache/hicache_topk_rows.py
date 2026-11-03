"""Host-tier copies of the routed-experts capturer rows for the hierarchical
cache.

The routed-experts capturer keeps one row per device KV slot, and get_topk
reads a request's rows through req_to_token. A HiCache load-back puts a prefix
into new device slots, so without these copies the prefix rows read back would
be whatever those slots held before. Every HiCache write copies the rows of its
device slots to its host slots, and every load-back copies them into the new
device slots.

The slot indices reach the CPU with a non-blocking copy on the transfer's own
stream, issued before the KV copy. Each transfer's TopkRowCopy rides on its
HiCacheAck. A write's rows are copied when its ack is processed, before the
device slots can be freed. A load's rows are copied when its ack is processed,
or earlier when get_topk reads rows.
"""

import collections
import logging
from typing import Optional

import msgspec
import torch

from sglang.srt.state_capturer.base import BaseTopkCapturer
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

logger = logging.getLogger(__name__)


class TopkRowCopy(msgspec.Struct):
    event: torch.cuda.Event
    device_indices_cpu: torch.Tensor
    host_indices_cpu: torch.Tensor
    # Keeps the device index tensor alive until the copy has run.
    device_indices: torch.Tensor
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
        capturer.add_pre_read_hook(self.apply_pending_loads)

    @staticmethod
    def create(
        host_size: int, device: str, has_storage_backend: bool
    ) -> Optional["HostTopkRows"]:
        capturer = get_global_experts_capturer()
        if capturer is None:
            return None
        if has_storage_backend:
            logger.warning(
                "Routed experts are not stored in the HiCache storage backend: "
                "prefix pages loaded from storage return stale routed-experts rows."
            )
        return HostTopkRows(host_size=host_size, capturer=capturer, device=device)

    @staticmethod
    def copy_indices(
        device_indices: torch.Tensor, host_indices: torch.Tensor, stream
    ) -> TopkRowCopy:
        """Copy the slot indices of a transfer to the host on stream, after the
        current stream that produced them. Call it before submitting the KV
        copy, so the indices do not wait for the KV."""
        device_indices_cpu = torch.empty(
            device_indices.shape, dtype=device_indices.dtype, pin_memory=True
        )
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            device_indices_cpu.copy_(device_indices, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        return TopkRowCopy(
            event=event,
            device_indices_cpu=device_indices_cpu,
            host_indices_cpu=host_indices.cpu(),
            device_indices=device_indices,
        )

    def add_load(self, row_copy: TopkRowCopy) -> None:
        self._pending_loads.append(row_copy)

    def finish_write(self, row_copy: TopkRowCopy) -> None:
        """Copy the rows of the written device slots to their host slots."""
        row_copy.event.synchronize()
        self._host_tier.buffer[row_copy.host_indices_cpu] = (
            self._capturer.host_cache.buffer[row_copy.device_indices_cpu]
        )

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
