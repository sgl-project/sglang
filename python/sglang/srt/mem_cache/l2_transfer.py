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


class TransferCompletion(NamedTuple):
    start_event: Any
    finish_event: Any
    timing_enabled: bool


class PrefetchHandle:
    """Handle for on-demand per-layer H2D loading.

    Only layer 0 is loaded in submit; subsequent layers are triggered
    during forward via trigger_layer(), so H2D overlaps with attention
    compute instead of being issued as a single bulk transfer.
    """

    def __init__(
        self,
        engine: "L2TransferEngine",
        transfers: list[L2Transfer],
        layer_num: int,
        ack_finish_event,
        on_layer_done,
    ):
        self._engine = engine
        self._transfers = transfers
        self._layer_num = layer_num
        self._ack_finish_event = ack_finish_event
        self._on_layer_done = on_layer_done
        self._next_layer = 1  # layer 0 already loaded

    def trigger_layer(self, layer_id: int) -> None:
        if layer_id < self._next_layer:
            return
        if layer_id >= self._layer_num:
            return
        self._next_layer = layer_id + 1
        primary = self._transfers[0] if self._transfers else None
        with device_module.stream(self._engine.host_to_device_stream):
            for transfer in self._transfers:
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
                    self._engine.io_backend,
                    is_draft=transfer.is_draft,
                )
            if self._on_layer_done is not None:
                self._on_layer_done(layer_id)
            if layer_id == self._layer_num - 1:
                self._ack_finish_event.record()

    def drain(self) -> None:
        """Record completion events for any layers not yet triggered."""
        if self._next_layer >= self._layer_num:
            return
        with device_module.stream(self._engine.host_to_device_stream):
            for layer_id in range(self._next_layer, self._layer_num):
                if self._on_layer_done is not None:
                    self._on_layer_done(layer_id)
            self._ack_finish_event.record()
        self._next_layer = self._layer_num


class L2TransferEngine:
    """Runs resolved device↔host transfers without owning cache state."""

    def __init__(self, io_backend: str):
        self.io_backend = io_backend
        self.device_to_host_stream = device_module.Stream()
        self.host_to_device_stream = device_module.Stream()
        self._prefetch_handle: Optional[PrefetchHandle] = None

    def submit_device_to_host(self, transfers: list[L2Transfer]) -> TransferCompletion:
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
        layer_num: int,
        start_event=None,
        on_layer_done=None,
        on_demand: bool = False,
    ) -> TransferCompletion:
        start_event = self._start_event(start_event)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        primary = transfers[0] if transfers else None

        if on_demand:
            with device_module.stream(self.host_to_device_stream):
                start_event.wait(self.host_to_device_stream)
                ack_start.record()
                for transfer in transfers:
                    local_layer_id = (
                        transfer.layer_mapper(0)
                        if transfer.layer_mapper is not None
                        else 0
                    )
                    if local_layer_id is None or (
                        transfer is not primary
                        and transfer.layer_mapper is None
                        and 0 >= transfer.host_pool.layer_num
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
                    on_layer_done(0)
                if layer_num <= 1:
                    ack_finish.record()
                self._record_stream(transfers, self.host_to_device_stream)
            self._prefetch_handle = PrefetchHandle(
                self, transfers, layer_num, ack_finish, on_layer_done
            )
            return TransferCompletion(ack_start, ack_finish, timing_enabled)

        with device_module.stream(self.host_to_device_stream):
            start_event.wait(self.host_to_device_stream)
            ack_start.record()
            for layer_id in range(layer_num):
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
        self._prefetch_handle = None
        return TransferCompletion(ack_start, ack_finish, timing_enabled)

    @property
    def prefetch_handle(self) -> Optional[PrefetchHandle]:
        return self._prefetch_handle

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
