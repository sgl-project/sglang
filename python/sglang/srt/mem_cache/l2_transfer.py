from __future__ import annotations

import logging
from functools import cache
from typing import Any, Callable, NamedTuple, Optional

import torch

from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


@cache
def _timing_events_supported() -> bool:
    try:
        device_module.Event(enable_timing=True)
        return True
    except (TypeError, NotImplementedError):
        logger.warning(
            "%s.Event does not support timing; L2 transfer timing is disabled",
            device_module.__name__,
        )
        return False


def make_timing_event_pair():
    timing_enabled = _timing_events_supported()
    kwargs = {"enable_timing": True} if timing_enabled else {}
    return device_module.Event(**kwargs), device_module.Event(**kwargs), timing_enabled


class L2Transfer(NamedTuple):
    host_pool: Any
    device_pool: Any
    host_indices: torch.Tensor
    device_indices: torch.Tensor
    layer_mapper: Optional[Callable[[int], Optional[int]]] = None
    is_draft: bool = False


from sglang.srt.kv_compression.types import BufferDrainError
from sglang.srt.mem_cache.l2_completion import RestoreTransferResult, TransferCompletion


class L2TransferEngine:
    """Runs resolved device↔host transfers without owning cache state."""

    def __init__(self, io_backend: str):
        self.io_backend = io_backend
        self.device_to_host_stream = device_module.Stream()
        self.host_to_device_stream = device_module.Stream()

    def submit_device_to_host(
        self,
        transfers: list[L2Transfer],
        *,
        async_state=None,
        page_refs=None,
        node_ids=(),
    ) -> TransferCompletion:
        if async_state is not None:
            if len(transfers) != 1 or page_refs is None:
                raise ValueError(
                    "Async FULL L2 requires one pool and explicit page identities"
                )
            transfer = transfers[0]
            return async_state.submit_backup(
                transfer.host_indices,
                page_refs,
                transfer.device_indices,
                self._start_event(None),
                node_ids,
            )
        start_event = self._start_event(None)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        with device_module.stream(self.device_to_host_stream):
            start_event.wait(self.device_to_host_stream)
            ack_start.record()
            for transfer in transfers:
                transfer.host_pool.backup_from_device_all_layer(
                    transfer.device_pool,
                    transfer.host_indices,
                    transfer.device_indices,
                    self.io_backend,
                )
            ack_finish.record()
            self._record_stream(transfers, self.device_to_host_stream)
        return TransferCompletion(ack_start, ack_finish, timing_enabled)

    def submit_host_to_device(
        self,
        transfers: list[L2Transfer],
        *,
        transfer_layer_id_max: int,
        start_event=None,
        on_layer_done=None,
    ) -> TransferCompletion:
        start_event = self._start_event(start_event)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        primary = transfers[0] if transfers else None
        with device_module.stream(self.host_to_device_stream):
            start_event.wait(self.host_to_device_stream)
            ack_start.record()
            for layer_id in range(transfer_layer_id_max):
                for transfer in transfers:
                    local_layer_id = (
                        transfer.layer_mapper(layer_id)
                        if transfer.layer_mapper is not None
                        else layer_id
                    )
                    if local_layer_id is None or (
                        transfer is not primary
                        and transfer.layer_mapper is None
                        and layer_id >= transfer.host_pool.layer_num
                    ):
                        continue
                    transfer.host_pool.load_to_device_per_layer(
                        transfer.device_pool,
                        transfer.host_indices,
                        transfer.device_indices,
                        local_layer_id,
                        self.io_backend,
                        is_draft=transfer.is_draft,
                    )
                if on_layer_done is not None:
                    on_layer_done(layer_id)
            ack_finish.record()
            self._record_stream(transfers, self.host_to_device_stream)
        return TransferCompletion(ack_start, ack_finish, timing_enabled)

    def submit_async_restore(self, state, leases, indices, fence_stream=None):
        import time

        enqueued = time.perf_counter()
        ready = self._start_event(None)
        fence = None
        if fence_stream is not None:
            fence = device_module.Event()
            fence.record(fence_stream)

        def restore():
            started = time.perf_counter()
            # These counters belong to the same serial executor as this task.
            gpu_keys = (
                "restore_copy_gpu_ms",
                "decompress_gpu_ms",
                "writeback_gpu_ms",
            )
            before_gpu = sum(state.runtime.stats[k] for k in gpu_keys)
            try:
                state.runtime.stream.wait_event(ready)
                if fence is not None:
                    state.runtime.stream.wait_event(fence)
                state.runtime.restore(leases, indices)
            except BaseException:
                # Fence submission can fail before runtime.restore owns cleanup.
                try:
                    state.runtime.drain()
                except BufferDrainError as exc:
                    state.runtime.quarantine(exc, (leases, indices))
                    raise
                raise
            pages = [lease.future.result() for lease in leases]
            return RestoreTransferResult(
                pages=len(indices),
                actual_bytes=sum(page.nbytes for page in pages),
                logical_bytes=sum(page.raw_bytes for page in pages),
                queue_seconds=started - enqueued,
                execution_seconds=time.perf_counter() - started,
                gpu_seconds=(sum(state.runtime.stats[k] for k in gpu_keys) - before_gpu)
                / 1000.0,
            )

        return TransferCompletion(future=state.runtime.submit(restore, priority=0))

    @staticmethod
    def _start_event(start_event):
        if start_event is None:
            start_event = device_module.Event()
            start_event.record()
        return start_event

    @staticmethod
    def _record_stream(transfers: list[L2Transfer], stream) -> None:
        for transfer in transfers:
            for indices in (transfer.host_indices, transfer.device_indices):
                if indices.is_cuda:
                    indices.record_stream(stream)
